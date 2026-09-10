"""Emit the full (arm, scale, seed) job list in the paper's run order.

Smallest-scale-first, cheapest-arm-first, so partial completion still yields
complete lower scales (graceful degradation). Pipe into your scheduler:

    python scripts/run_matrix.py | while read cmd; do sbatch scripts/slurm/run_one.sbatch $cmd; done

Every line this prints costs a scheduler slot, so a scale whose preset is still
a placeholder is left out rather than queued: `get_preset` raises on it, and the
jobs would be a queue of guaranteed failures. The omission goes to stderr, which
the pipeline above does not read, so the reader is told without the note ending
up in the job list.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

import yaml

from svb.data.registry import get_preset

SCALES = ["3", "16", "64"]
ARMS = ["A_ssl", "B_btm_ssl", "C_btm_scratch"]  # A<B<C by cost


def default_configs_dir() -> Path:
    """`configs/` beside this script, not relative to the working directory.

    A submit-time helper runs from wherever the scheduler put the job, which is
    rarely the checkout root.
    """
    return Path(__file__).resolve().parents[1] / "configs"


def populated_scales(configs_dir: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Split the scales into those that can run and those that cannot.

    Returns the runnable scale names in order, and ``(scale, reason)`` for each
    one left out. A missing preset file raises ``FileNotFoundError`` while an
    empty one raises ``ValueError``; both mean "not runnable", and a sweep whose
    job is to report the gap must survive either.
    """
    kept: list[str] = []
    skipped: list[tuple[str, str]] = []
    for scale in SCALES:
        try:
            get_preset(scale, configs_dir=configs_dir)
        except (ValueError, FileNotFoundError) as exc:
            skipped.append((scale, str(exc)))
        else:
            kept.append(scale)
    return kept, skipped


def main(
    configs_dir: Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> None:
    configs_dir = configs_dir or default_configs_dir()
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    seeds = yaml.safe_load((configs_dir / "seeds.yaml").read_text(encoding="utf-8"))["seeds"]
    scales, skipped = populated_scales(configs_dir)

    for scale, reason in skipped:
        err.write(f"[svb] scale {scale} omitted from the matrix: {reason}\n")
    if not scales:
        raise SystemExit("[svb] no populated preset: there is nothing to submit")

    for scale in scales:
        for arm in ARMS:
            for seed in seeds:
                out.write(f"--arm {arm} --scale {scale} --seed {seed}\n")


if __name__ == "__main__":
    main()
