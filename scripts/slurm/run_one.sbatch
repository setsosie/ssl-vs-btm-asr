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
#
# NOTE: --cpus-per-task above must exceed train.num_workers in the config.
# They are currently both 8, which leaves the main process sharing a core with
# a loader worker. Either raise --cpus-per-task to 9 or lower num_workers to 7.
set -euo pipefail

: "${XEUS_CHECKPOINT:?set XEUS_CHECKPOINT to the espnet/xeus checkpoint path}"
: "${CV_ROOT:?set CV_ROOT to the extracted Common Voice 25 directory}"

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
