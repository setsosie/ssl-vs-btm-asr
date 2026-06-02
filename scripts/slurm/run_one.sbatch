#!/bin/bash
#SBATCH --job-name=svb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-%j.out
# Single (arm, scale, seed) run on one GPU. Args are forwarded to `svb run`:
#   sbatch scripts/slurm/run_one.sbatch --arm A_ssl --scale 3 --seed 0
# Independent jobs run in parallel across GPUs (no DDP within a job).
set -euo pipefail

: "${XEUS_CHECKPOINT:?set XEUS_CHECKPOINT to the espnet/xeus checkpoint path}"
: "${CV_ROOT:?set CV_ROOT to the extracted Common Voice 25 directory}"

uv run svb run "$@" --config configs/base.yaml --device cuda \
    --merge-strategy "${MERGE_STRATEGY:-average}"
