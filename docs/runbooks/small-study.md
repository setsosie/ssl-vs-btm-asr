# Runbook: the small study

Exact commands to take [the small study](../small-study.md) from a fresh clone
to `tables/3_arms.md`. Run them from the repository root, in order. Steps 1–4
are gates: each one is cheap, and each catches something that would otherwise
surface hours into a GPU job.

Nothing here has been executed against real audio yet. Where a step has never
been run, it says so.

## 0. Environment

```bash
uv sync --frozen --extra dev
make check                       # ruff, mypy, the CPU suite
```

`make check` includes an end-to-end `svb run` of all three arms against a
stand-in encoder, reading real mp3 and wav files. If it is green, the pipeline's
plumbing works on this machine.

```bash
export CV_ROOT=/path/to/common_voice_25        # contains en/ ja/ hi/
export OPENSLR_ROOT=/path/to/openslr           # written by `make fetch`
export XEUS_CHECKPOINT=/path/to/xeus_checkpoint.pth
export SVB_RESULTS_ROOT=/path/to/scratch/svb-results
```

Put `SVB_RESULTS_ROOT` on scratch. Every run keeps its checkpoints there: one
per language per stage, each the size of the encoder.

## 1. Data

```bash
make fetch                       # the four held-out corpora, ~5 GB
make data                        # every language reachable, with split sizes
uv run svb data-stats --scope 3
uv run svb data-stats --scope heldout --tables-dir tables/heldout
```

(Two output directories because both calls write `data_durations.md`, and the
second would replace the first.)

Common Voice is not fetched by any script: it is distributed through Mozilla
Data Collective behind an account, so download and extract `en`, `ja` and `hi`
yourself. Only those three are needed for this study.

`tables/data_durations.md` should show `clip_durations.tsv` as the duration
source for the Common Voice languages, and its hours should agree with the
table in [the protocol](../small-study.md#data). If it says `audio headers`, the release
did not ship that file, and the hours cap will read every clip's header instead
— correct, and slow on English.

## 2. Encoder cross-check

Never run. Needs a separate Python 3.10 environment; see
[`espnet-crosscheck.md`](../espnet-crosscheck.md) for why.

```bash
uv venv --python 3.10 .venv-espnet
uv pip install --python .venv-espnet -r scripts/espnet-crosscheck/requirements.txt

.venv-espnet/bin/python scripts/crosscheck_espnet.py \
    --checkpoint "$XEUS_CHECKPOINT" --device cuda \
    --out docs/espnet-crosscheck-report.json
```

Exit code 0 is within tolerance. Commit the report. **If this fails, stop**:
every later step would be measuring a different encoder from the published one.

## 3. Smoke: one run, real encoder, minutes of audio

A throwaway run with the caps turned down and one epoch per stage, to a results
root of its own so it cannot be mistaken for a study run.

```bash
cat > /tmp/svb-smoke.yaml <<'YAML'
train:
  max_train_hours: 0.05
  max_val_hours: 0.02
  phase0_epochs: 1
  expert_epochs: 1
  finetune_epochs: 1
  patience: 1
text:
  min_char_count: 1
YAML

CELL=$(uv run python scripts/run_matrix.py --scale 3 --arm B_btm_ssl --n-seeds 1)
uv run svb run $CELL --device cuda \
    --config configs/base.yaml --config /tmp/svb-smoke.yaml \
    --results-root "$SVB_RESULTS_ROOT/smoke"
```

It still decodes every test split in full, so expect the evaluation stages to
dominate. It passes if `results.json` is written; the error rates mean nothing.
What to read: GPU memory at batch size 8, and that no language warns about an
unreadable clip.

## 4. Timing: one real seed of arm B

Arm B is the cheaper of the two SSL arms and exercises every stage.

```bash
CELL=$(uv run python scripts/run_matrix.py --scale 3 --arm B_btm_ssl --n-seeds 1)
STUDY=small sbatch scripts/slurm/run_one.sbatch $CELL
```

When it finishes, `stages.json` in the run directory holds `wall_seconds` per
stage. Scale those by the epoch counts in [the budget](../small-study.md#budget)
to price the other fourteen runs before submitting them. If a job hits its time
limit, submit the same line again: it resumes from the last finished stage.

Resume is per stage, not per epoch. A single stage longer than the job's time
limit can never finish, and the longest stage in this study is arm A's English
fine-tune (50 epochs over 10 hours). Read its projected length off phase 0's
`wall_seconds` here — phase 0 covers 27 hours per epoch — and raise
`#SBATCH --time` in `scripts/slurm/run_one.sbatch` before step 5 if it does not
fit with room to spare.

This run is a study run and is kept.

## 5. The study

```bash
uv run python scripts/run_matrix.py --scale 3 | while read -r cell; do
    STUDY=small sbatch scripts/slurm/run_one.sbatch $cell
done
```

Fifteen jobs, one GPU each, cheapest arm first. The cell from step 4 is
submitted again here and finishes immediately, because every stage is already
recorded. Add `--n-seeds 3` to `run_matrix.py` if compute forces fewer runs, and
say so in the table caption.

To resubmit only what is unfinished — **once no job of the study is still
queued or running** (`squeue -u "$USER" -n svb` is empty), because a cell whose
job is still running has no `results.json` yet either:

```bash
uv run python scripts/run_matrix.py --scale 3 | while read -r cell; do
    set -- $cell                                   # --arm A --scale S --seed N
    [ -f "$SVB_RESULTS_ROOT/$2/$4/seed$6/results.json" ] \
        || STUDY=small sbatch scripts/slurm/run_one.sbatch $cell
done
```

A second job on a cell that is still running exits at once with "in use by
another process" rather than training over the first — the run directory is
locked — so the mistake costs a scheduler slot, not a result. To throw a cell
away and redo it, pass `--restart` to `svb run` explicitly; nothing in this
runbook does that.

Without a scheduler, on one machine:

```bash
make study STUDY=small RESUME=1
```

## 6. Tables

No GPU needed; everything is recomputed from the prediction sidecars.

```bash
uv run svb compare --scale 3                       # tables/3_arms.{md,json}
for arm in A_ssl B_btm_ssl C_btm_scratch; do
    uv run svb aggregate --arm $arm --scale 3      # per-arm mean ± std
    uv run svb analyze --arm $arm --scale 3        # per-run intervals
done
```

`tables/3_arms.md` is the result. Before quoting it:

- Every arm lists the same seeds.
- Under `data` in any two `results.json` files, `subset_sha1` matches per
  language — every run trained on the same utterances.
- `invocations` in each `results.json` lists one `git_sha`, and `git_dirty` is
  false. A run resumed across a code change records both commits.
- `unk_rate` for Japanese in `text_stats.json`, which is a floor under its
  character error rate.
