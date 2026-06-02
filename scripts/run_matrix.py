"""Emit the full (arm, scale, seed) job list in the paper's run order.

Smallest-scale-first, cheapest-arm-first, so partial completion still yields
complete lower scales (graceful degradation). Pipe into your scheduler:

    python scripts/run_matrix.py | while read cmd; do sbatch scripts/slurm/run_one.sbatch $cmd; done
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

SCALES = ["3", "16", "64"]
ARMS = ["A_ssl", "B_btm_ssl", "C_btm_scratch"]  # A<B<C by cost


def main() -> None:
    seeds = yaml.safe_load((Path("configs/seeds.yaml")).read_text())["seeds"]
    for scale in SCALES:
        for arm in ARMS:
            for seed in seeds:
                sys.stdout.write(f"--arm {arm} --scale {scale} --seed {seed}\n")


if __name__ == "__main__":
    main()
