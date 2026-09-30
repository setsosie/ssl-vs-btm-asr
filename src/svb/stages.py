"""The stage ledger: what a run has finished, so it can be resumed.

A run is hours of training cut into stages — phase 0, one expert per language,
the merge, one evaluation per language, one transfer per held-out language. On
a scheduler with a wall-clock limit a run does not reliably finish in one job,
and a run that must start again from nothing each time it is interrupted is a
run that never finishes. ``stages.json`` records each stage as it completes;
``svb run --resume`` reads it back and skips what is done.

Four rules keep a resumed run honest:

* **A stage is reused only with the files it left behind.** A record whose
  checkpoint or predictions are gone is not a result.
* **Nothing after a re-run stage is reused.** Every later stage was computed
  from the old version of it: experts branched from a phase 0 that has since
  been replaced would be merged against a base they did not come from. Stages
  always execute in one order, so everything after the first stage that
  actually runs is run again.
* **A record is withdrawn before its stage is run again**, on disk. The trainer
  writes a checkpoint at the start of training, so a re-run killed partway
  leaves a file where the old record says one should be; were the record still
  there, the next resume would take that half-trained file for the finished one.
* **A run that is not resuming starts an empty ledger**, so records from an
  earlier attempt cannot be picked up by a later ``--resume``.

And one keeps two runs from being one: **a run directory belongs to one process
at a time.** Two jobs on the same cell — a resubmission while the first is still
running is all it takes — would train the same stage into the same checkpoint
file and leave a ledger describing whichever wrote last. :func:`run_lock` takes
the directory for the life of the process, and a second process is refused
before it has written anything.

Each stage is seeded from the run seed and its own name before it runs
(:func:`svb.seeding.seed_stage`), so a stage draws the same stream whether the
stages before it ran in this process or were read back from the ledger. On a GPU
that makes a resumed run equivalent in distribution to an uninterrupted one, not
identical to it — nothing here is bit-reproducible, resumed or not.

The ledger also records each stage's wall-clock time. That is the only
measurement of what a run costs, and the larger tiers are budgeted from it.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .seeding import seed_stage

LEDGER = "stages.json"
LOCK = ".lock"


class RunLockedError(RuntimeError):
    """Another process holds the run directory."""


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Hold a run directory for this process, or refuse.

    An advisory ``flock``, because the kernel drops it when the process dies: a
    job killed by the scheduler cannot leave a lock behind for someone to clear
    by hand. A filesystem that does not implement it gets a warning and no
    protection, which is better than a run that cannot start.

    Raises:
        RunLockedError: If another process holds the directory.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(run_dir / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunLockedError(
                f"{run_dir} is in use by another process. Two jobs on one run would train "
                "the same stages into the same files; wait for the first to finish, or "
                "cancel it, before submitting this cell again."
            ) from None
        except OSError as exc:
            warnings.warn(
                f"{run_dir}: this filesystem does not support locking ({exc}), so nothing "
                "prevents a second job from running on the same directory",
                UserWarning,
                stacklevel=3,
            )
        yield
    finally:
        os.close(fd)


def finished_stages(run_dir: Path) -> list[str]:
    """Names of the stages a run directory's ledger records as finished."""
    path = run_dir / LEDGER
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("stages", {}))


@dataclass
class StageOutput:
    """What a stage hands back: its record, and the files that make it real."""

    #: JSON-serializable. This is what a resumed run gets instead of re-running.
    result: dict[str, Any]
    #: Files the stage wrote that later stages or the reporting commands read.
    artifacts: Sequence[Path] = field(default_factory=list)


class StageLedger:
    """Runs stages in order, recording each and reusing what a resume allows."""

    def __init__(
        self,
        run_dir: Path,
        seed: int,
        resume: bool = False,
        invocation: dict[str, Any] | None = None,
    ) -> None:
        """Open the ledger for one invocation of a run.

        Args:
            run_dir: The run's directory; artifact paths are stored relative to
                it, so a results tree can be moved and still be resumed.
            seed: The run seed each stage's seed is derived from.
            resume: Reuse finished stages from an existing ledger. Without it
                any existing ledger is discarded.
            invocation: What identifies this process — git SHA, dirty flag,
                timestamp. Every invocation that contributed to a run is kept,
                so a result assembled across a code change says so.
        """
        self.run_dir = run_dir
        self.seed = seed
        self.path = run_dir / LEDGER
        self._records: dict[str, dict[str, Any]] = {}
        self._invocations: list[dict[str, Any]] = []
        self._visited: list[str] = []
        self._ran_a_stage = False
        if resume and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = payload.get("stages", {})
            self._invocations = payload.get("invocations", [])
        self._invocations.append({**(invocation or {}), "resumed": bool(self._records)})
        self._write()

    def run(self, name: str, stage: Callable[[], StageOutput]) -> dict[str, Any]:
        """Run ``stage`` unless a finished, intact record of it can be reused.

        Returns the stage's result either way, so the caller does not need to
        know which happened.
        """
        if name in self._visited:
            raise ValueError(f"stage {name!r} was already run in this invocation")
        self._visited.append(name)

        record = self._records.get(name)
        if record is not None and not self._ran_a_stage and self._intact(record):
            print(f"[svb] stage {name}: done in an earlier invocation, reusing it")
            return dict(record["result"])

        # From here on everything is recomputed, so this stage's old record and
        # the records of stages not yet reached describe results that are about
        # to be replaced. They go from the file now, not when the stage
        # finishes: a job killed in between must not leave them to be believed.
        self._ran_a_stage = True
        kept = {k: v for k, v in self._records.items() if k in self._visited and k != name}
        if len(kept) != len(self._records):
            self._records = kept
            self._write()

        stage_seed = seed_stage(self.seed, name)
        started = time.monotonic()
        output = stage()
        self._records[name] = {
            "result": output.result,
            "artifacts": [self._relative(path) for path in output.artifacts],
            "seed": stage_seed,
            "wall_seconds": time.monotonic() - started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "invocation": len(self._invocations) - 1,
        }
        self._write()
        return dict(output.result)

    def summary(self) -> dict[str, Any]:
        """The run's stages and invocations, for ``results.json``."""
        return {
            "stages": {
                name: {
                    "seed": record["seed"],
                    "wall_seconds": record["wall_seconds"],
                    "invocation": record["invocation"],
                }
                for name, record in self._records.items()
            },
            "invocations": self._invocations,
        }

    def _intact(self, record: dict[str, Any]) -> bool:
        return all((self.run_dir / path).exists() for path in record.get("artifacts", []))

    def _relative(self, path: Path) -> str:
        return os.path.relpath(path, self.run_dir)

    def _write(self) -> None:
        """Replace the ledger atomically.

        A job killed mid-write must leave the previous ledger, not half of a new
        one: a truncated file would fail to parse and take every finished stage
        with it.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {"invocations": self._invocations, "stages": self._records}
        scratch = self.path.with_suffix(".json.tmp")
        scratch.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(scratch, self.path)
