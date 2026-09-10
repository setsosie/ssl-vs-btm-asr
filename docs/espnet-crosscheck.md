# The ESPnet cross-check

`src/svb/model/xeus_standalone.py` re-expresses the XEUS encoder in PyTorch so
the published checkpoint loads without ESPnet. Every number this repository
produces comes out of that reimplementation.

That is a risk worth naming. An earlier version of this same file had
E-Branchformer forward deviations from the reference that invalidated training
results before anyone noticed — the encoder ran, loss went down, and the
features were wrong. A reimplementation is not verified by the fact that it
loads a checkpoint without complaining.

`scripts/crosscheck_espnet.py` is what retires that risk: it loads one
checkpoint into both the reference ESPnet model and the standalone one, runs the
same audio through each, and compares the features.

**It has not been run.** Doing so needs the published XEUS checkpoint and, in
practice, a GPU. Until the report described below exists, the encoder's fidelity
to the reference is an assumption, and any write-up should say so.

## What the check proves, and what it does not

It proves that on the audio it was given, the two implementations produce the
same final-layer features to within a tolerance. That covers the architecture,
the weight mapping from the checkpoint, and the forward pass.

It does not prove the reimplementation is correct on audio unlike the test
cases, and it is not a claim about training. It is a necessary condition, not a
sufficient one.

Two cases are run, because they answer different questions.

**`single`** — one utterance, no padding. Does the architecture match at all? A
failure here is a structural bug: wrong layer order, wrong weight mapping, a
missing residual.

**`padded_batch`** — two utterances, one of them 7000 samples padded out to
16000. The encoder's two time-axis convolutions do not mask padded frames. That
is deliberate and matches the reference, which the published weights were
pretrained under, so masking here would diverge from the checkpoint rather than
correct it. The consequence is that an utterance's encoding depends slightly on
what shares its batch, which is why evaluation batches by length (see the
protocol section of the README).

This case reports a delta **per utterance**, not just per batch. A single
batch-wide maximum cannot say which utterance it came from, and the whole point
is to find out whether disagreement concentrates in the padded one — which is
where a masking divergence would appear. Measured rather than assumed.

Only frames inside an utterance's own output length are compared. Frames past it
are padding in both encoders and are sliced off before the CTC head sees them,
so a difference there could not reach any reported number.

## Running it

The reference environment is separate from the project's and cannot be merged
into it; `scripts/espnet-crosscheck/requirements.txt` explains why at length.
Short version: the fork needs `numpy<1.24` and Python ≤3.10, which this project
is not and cannot be.

```bash
# 1. Build the reference environment, once. Python 3.10, its own venv.
uv venv --python 3.10 .venv-espnet
uv pip install --python .venv-espnet -r scripts/espnet-crosscheck/requirements.txt

# 2. Run the check from the repository root.
.venv-espnet/bin/python scripts/crosscheck_espnet.py \
    --checkpoint "$XEUS_CHECKPOINT" \
    --device cuda \
    --out docs/espnet-crosscheck-report.json
```

On a CPU-only machine, add `--torch-backend cpu` to the `uv pip install` and
pass `--device cpu`. The check is small — two short utterances — so CPU is
viable if slow.

The script is run from the repository root and loads the standalone encoder from
its source file rather than from an installed package, because `svb` is not
installable in that environment. Running it from elsewhere will not find the
encoder.

Exit codes: `0` both cases within tolerance, `1` outside it, `2` the reference
could not be loaded at all — most likely the environment was not built.

## Reading the result

```
checkpoint : xeus_checkpoint.pth
device     : cuda   tolerance: 0.001

  single        max|Δ| = 3.100e-06   per-utterance: [3.100e-06]
  padded_batch  max|Δ| = 4.700e-06   per-utterance: [3.100e-06, 4.700e-06]

PASS  (worst max|Δ| = 4.7e-06)
```

Deltas at the 1e-6 level are floating-point accumulation and mean the
implementations agree. Deltas near or above the 1e-3 default tolerance mean they
do not, and the per-utterance column says where to look: a delta confined to the
padded utterance points at padding handling, while one present in both points at
the architecture.

A shape mismatch is reported instead of a delta and never passes, whatever the
tolerance. There is no number to compare, and reporting a zero there would read
as perfect agreement.

## What to do with the report

**Commit the passing report as `docs/espnet-crosscheck-report.json`.**

One location, in `docs/` rather than under `results/`, for a practical reason:
`results/` is gitignored, and this file is the durable evidence for a claim the
write-up makes. Its value is that a reader can check it without access to the
GPU box or the checkpoint.

The report carries the checkpoint's SHA-256 alongside the numbers. That field is
load-bearing: "the two implementations agree" is a claim about a specific set of
weights, and a report that does not pin which ones cannot be checked against a
later run or against someone else's download. It also records the torch, numpy
and ESPnet versions, since the reference model is whatever that fork commit
builds under those.

Re-run the check and replace the report whenever `xeus_standalone.py` changes in
any way that could affect the forward pass. A stale report is worse than none:
it attests to code that is no longer there.
