"""Emit the full (arm, scale, seed) job list in the paper's run order.

Smallest-scale-first, cheapest-arm-first, so partial completion still yields
complete lower scales (graceful degradation). Pipe into your scheduler:

    python scripts/run_matrix.py | while read cmd; do sbatch scripts/slurm/run_one.sbatch $cmd; done

``--scale``, ``--arm`` and ``--n-seeds`` narrow the list — `--scale 3` is the
whole of the small study. This script is the only reader of `configs/seeds.yaml`:
the Makefile and the Slurm launcher get their seeds from its output, so there is
one list and no copy of it to drift. ``--n-seeds`` takes a prefix of that list
rather than a choice from it, which keeps a reduced study a subset of the full
one and leaves nobody deciding which seeds to keep.

Every line this prints costs a scheduler slot, so a scale whose preset is still
a placeholder is left out rather than queued: `get_preset` raises on it, and the
jobs would be a queue of guaranteed failures. The omission goes to stderr, which
the pipeline above does not read, so the reader is told without the note ending
up in the job list.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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


def load_seeds(configs_dir: Path, n_seeds: int | None = None) -> list[int]:
    """The committed seed list, or its first ``n_seeds`` entries."""
    seeds: list[int] = yaml.safe_load((configs_dir / "seeds.yaml").read_text(encoding="utf-8"))[
        "seeds"
    ]
    if n_seeds is None:
        return seeds
    if not 1 <= n_seeds <= len(seeds):
        raise SystemExit(
            f"[svb] --n-seeds {n_seeds}: configs/seeds.yaml holds {len(seeds)} seed(s). "
            "More runs need more seeds drawn and committed, not invented here."
        )
    return seeds[:n_seeds]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--scale", action="append", choices=SCALES, help="only this scale; repeatable"
    )
    parser.add_argument("--arm", action="append", choices=ARMS, help="only this arm; repeatable")
    parser.add_argument(
        "--n-seeds",
        dest="n_seeds",
        type=int,
        default=None,
        help="use only the first N seeds of configs/seeds.yaml (default: all)",
    )
    return parser


def main(
    configs_dir: Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    argv: Sequence[str] = (),
) -> None:
    configs_dir = configs_dir or default_configs_dir()
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(list(argv))

    seeds = load_seeds(configs_dir, args.n_seeds)
    scales, skipped = populated_scales(configs_dir)

    for scale, reason in skipped:
        # A scale the caller named is a request, not a sweep: leaving it out
        # quietly would hand back an empty list that looks like a finished one.
        if args.scale and scale in args.scale:
            raise SystemExit(f"[svb] scale {scale} cannot run: {reason}")
        err.write(f"[svb] scale {scale} omitted from the matrix: {reason}\n")
    if args.scale:
        scales = [scale for scale in scales if scale in args.scale]
    if not scales:
        raise SystemExit("[svb] no populated preset: there is nothing to submit")

    arms = [arm for arm in ARMS if not args.arm or arm in args.arm]
    for scale in scales:
        for arm in arms:
            for seed in seeds:
                out.write(f"--arm {arm} --scale {scale} --seed {seed}\n")


if __name__ == "__main__":
    main(argv=sys.argv[1:])
