#!/bin/bash
#SBATCH --job-name=svb
#SBATCH --gres=gpu:1
# One core for the main process on top of train.num_workers (8 in
# configs/base.yaml), so the loader workers do not share a core with it.
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Written to the submit directory, which always exists. A results-relative path
# fails the job before it starts when that directory does not exist yet, and it
# would drift from the results themselves whenever SVB_RESULTS_ROOT is set.
# Redirect with `sbatch --output=...` if the logs belong somewhere else.
#SBATCH --output=slurm-svb-%j.out
# Single (arm, scale, seed) run on one GPU. Args are forwarded to `svb run`:
#   sbatch scripts/slurm/run_one.sbatch --arm A_ssl --scale 3 --seed 0
# Independent jobs run in parallel across GPUs (no DDP within a job).
#
# NOTE: --cpus-per-task above must exceed train.num_workers in the config.
# They are currently both 8, which leaves the main process sharing a core with
# a loader worker. Either raise --cpus-per-task to 9 or lower num_workers to 7.
set -euo pipefail

: "${XEUS_CHECKPOINT:?set XEUS_CHECKPOINT to the espnet/xeus checkpoint path}"
: "${CV_ROOT:?set CV_ROOT to the extracted Common Voice 25 directory}"
# Every arm ends in held-out transfer, and the per-language text statistics are
# written before any training starts, so a run without this dies either way.
# Failing here costs a second instead of a scheduler slot.
: "${OPENSLR_ROOT:?set OPENSLR_ROOT to the fetch_openslr.py extraction root (make fetch)}"

# --merge-strategy only means something for the BTM arms, so read the arm out
# of the forwarded arguments rather than passing a strategy to arm A.
arm=""
args=("$@")
for i in "${!args[@]}"; do
    case "${args[i]}" in
        --arm) arm="${args[i + 1]:-}" ;;
        --arm=*) arm="${args[i]#--arm=}" ;;
    esac
done

merge_args=()
case "$arm" in
    B_btm_ssl | C_btm_scratch) merge_args=(--merge-strategy "${MERGE_STRATEGY:-average}") ;;
esac

# "$@" goes last: argparse takes the last occurrence of an option, so a caller
# passing --device cpu must be able to override the default below.
uv run svb run \
    --config configs/base.yaml \
    --device cuda \
    ${merge_args[@]+"${merge_args[@]}"} \
    "$@"
