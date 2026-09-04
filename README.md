# ssl-vs-btm-asr

[![ci](https://github.com/setsosie/ssl-vs-btm-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/setsosie/ssl-vs-btm-asr/actions/workflows/ci.yml)

**Is it self-supervised pretraining or the Branch-Train-Merge pipeline that
carries low-resource multilingual ASR?**

A multi-seed ablation on the [XEUS](https://arxiv.org/abs/2407.00837) encoder.
Three training conditions, compared across 3 / 16 / 64 languages and on four
held-out Indic languages the model never saw:

| Arm | Init | Pipeline | Question it answers |
|-----|------|----------|---------------------|
| **A. SSL-only** | XEUS SSL | per-language fine-tune | value of SSL alone |
| **B. BTM-on-SSL** | XEUS SSL | phase 0 → experts → merge | does BTM add anything on top of SSL? |
| **C. BTM-from-scratch** | random | phase 0 → experts → merge | does BTM work *without* SSL? |

Each cell is run over the seeds in `configs/seeds.yaml` and reported as mean ±
standard deviation, with utterance-level bootstrap intervals and paired
permutation tests available from the per-utterance predictions every run writes.

## Status

**No GPU run has been executed against this code.** Everything below is
implemented and covered by 371 CPU tests; none of it has yet produced a number
from real audio. Two other gaps are open by design rather than by oversight:

- `configs/scales/64.yaml` is a placeholder and lists no languages. Loading it
  raises rather than training on nothing, so the 64-language tier cannot be run
  until the list is finalized.
- Odia is not in the held-out set. See [Held-out data](#held-out-data) below —
  the set is **four** languages, and any write-up should say four.

## Quickstart

```bash
uv sync --frozen --extra dev              # install exactly what uv.lock pins
export OPENSLR_ROOT=/path/to/openslr      # where held-out corpora are extracted
make fetch                                # download + extract them (~5 GB)
export CV_ROOT=/path/to/common_voice_25   # you download this one yourself
make data                                 # what is reachable, and how big
export XEUS_CHECKPOINT=/path/to/xeus_checkpoint.pth
svb run --arm A_ssl --scale 3 --seed 0 --config configs/base.yaml
```

Then read the results back:

```bash
svb aggregate --arm A_ssl --scale 3            # WER, CER and primary, mean ± std
svb aggregate --arm A_ssl --scale 3 --json     # the same, machine-readable
svb analyze --arm A_ssl --scale 3 --compare-to B_btm_ssl   # intervals + paired test
svb data-stats --scope all --min-train-hours 50            # audio hours per language
make exp ARM=A_ssl SCALE=3                     # every seed for one cell
make tables SCALE=3 ARM=A_ssl COMPARE_TO=B_btm_ssl
```

Arms A and B need the XEUS SSL checkpoint (`espnet/xeus` on the Hugging Face
Hub); set `XEUS_CHECKPOINT` or put the path in `configs/base.yaml`. Arm C is
randomly initialized and needs no checkpoint.

## Data

Two public corpora, both read from a local directory. Nothing downloads at load
time and nothing goes through the Hugging Face Hub. Full details in
[`docs/data.md`](docs/data.md).

**Training and in-distribution evaluation** use Common Voice 25. Since October
2025 Common Voice is distributed only through
[Mozilla Data Collective](https://commonvoice.mozilla.org/en/datasets), behind
an account and a terms acceptance no script can give, so you download and
extract it yourself and point `CV_ROOT` at the result.

### Held-out data

**Held-out transfer** uses four crowdsourced Indic read-speech corpora from
openslr.org — Malayalam (SLR63), Marathi (SLR64), Telugu (SLR66) and Gujarati
(SLR78) — all CC-BY-SA-4.0, fetched by `scripts/fetch_openslr.py`. They are
never in any training mix.

Odia is deliberately excluded. It is SLR103, the Multilingual and Code-Switching
ASR Challenge corpus, not one of the crowdsourced sets; it is published under a
Microsoft Research Open Data licence that has not been read; and it ships its
own train/test split, so the split derivation used for the other four must not
be applied to it. It stays commented out in `configs/scales/heldout.yaml` until
that licence is reviewed. **The held-out set is four languages.**

These four corpora ship no train/validation/test partition, so one is derived at
load time from a stable hash of the speaker id — speaker-disjoint, so test
speakers are unseen in training. Because whole speakers are indivisible and
Marathi has only nine of them, realised proportions drift from the nominal
80/10/10. Every run records the policy it used and a digest of the exact test
utterance list beside the number.

## Text normalization

Every transcript passes through one versioned normalizer, `svb-norm-1`, applied
at exactly three places — vocabulary construction, training targets, and both
sides of every score — so training and evaluation cannot drift apart on text
policy. It does NFKC, case folding, punctuation and symbol removal, and
script-specific handling of Arabic vocalization and Malayalam chillu letters,
while keeping combining marks and intra-word apostrophes. The full rule list and
the reasoning behind each rule are in
[`docs/normalization.md`](docs/normalization.md).

**These numbers are not comparable to the author's earlier results on this
material.** That earlier work applied no normalization at evaluation time: its
tokenizer case-folded training targets and emitted lowercase hypotheses while
scoring against raw mixed-case references, and punctuation was modelled and
retained on both sides. Every capitalized reference word was therefore scored as
a substitution, and every punctuation slip as a word error, so those error rates
should be read as upper bounds. The inflation is largely a common term across
the arms being compared, so comparisons *within* that earlier work remain
informative — but it is not exactly additive, because normalization forgives an
error class that a weaker system commits more often, which compresses
differences between arms. Results produced under different policy hashes must
not be pooled.

## Evaluation protocol

- **Greedy CTC decode.** No beam search, no language model.
- **Length-sorted batching.** The encoder's two time-axis convolutions do not
  mask padded frames, matching the ESPnet reference the published weights were
  pretrained under, so an utterance's encoding depends slightly on what shares
  its batch. Evaluating in descending length order makes batch membership a
  function of the split and the batch size rather than of row order on disk.
  Results are comparable only between runs that used the same batch size.
- **WER and CER are both always computed.** Which is *primary* follows the
  `word_boundary` field of the language's preset entry, because whitespace
  tokenization of a script written without spaces yields one token per sentence.
  Each run checks that declaration against its own transcripts and warns when
  they disagree.
- **No truncation at test time.** Training truncates long audio and drops the
  transcripts that no longer fit; evaluation does neither, because a clipped
  waveform scored against its full reference manufactures deletions.
- **Empty references are excluded and counted.** An utterance whose reference
  normalizes to nothing contributes its whole hypothesis to the numerator and
  nothing to the denominator. Both the exclusion and its count are reported.
- **Per-utterance sidecars.** Every evaluation writes its reference/hypothesis
  pairs, so intervals and paired tests are recomputed from disk at no GPU cost.

`svb aggregate` reports spread across seeds. `svb analyze` reports what the test
set leaves uncertain: a bootstrap percentile interval per language, and a
one-sided paired permutation test between two arms at the same seed. These are
different quantities; a five-seed standard deviation is not a confidence
interval. The paired test verifies both runs hold identical references and
refuses otherwise.

## Provenance

Every run writes a self-contained directory —
`resolved_config.yaml`, `env.json`, `text_stats.json`, `vocab.json`,
`results.json` and the prediction sidecars. `env.json` carries the git SHA and
whether the tree was dirty, the `uv.lock` digest, library versions, the GPU and
the Unicode version. `text_stats.json` carries the per-language evidence for the
normalization policy, including the unknown-character rate, which is an
irreducible floor under that language's error rate. Held-out results carry the
split policy and a digest of the test utterance list.

Layout and how the reporting commands consume it:
[`docs/results.md`](docs/results.md). Tables in `tables/` are generated by
`make tables` and `svb data-stats`, never hand-typed.

## Reproducibility notes

Things a reader should know before quoting a number:

- **Checkpoints are selected on validation loss, not WER.** The two diverge for
  CTC, and the selected checkpoint is not necessarily the best-scoring one.
- **Merging is full-state-dict by default**, CTC head included. `merge_head:
  false` (or `--no-merge-head`) restricts it to the encoder, which is the
  ablation for how much of any merging penalty lives in the head.
- **Runs are not bit-reproducible on GPU.** CTC's backward pass has no
  deterministic CUDA kernel — requesting one raises rather than selecting one —
  so a fixed seed does not give a fixed result. That is why the study reports
  several seeds and a spread rather than a single number.
- **Effective batch size is 16** with the shipped config (8 × 2 accumulation
  steps) on a single GPU. Any comparison to numbers produced at a different
  effective batch should say so.
- **The encoder is a fork-free reimplementation**, not the reference
  implementation. See Acknowledgements.

## Layout

```
src/svb/
    model/    XEUS encoder + CTC head, character vocabulary
    text/     the svb-norm-1 normalizer and its per-corpus statistics
    data/     Common Voice and OpenSLR local loaders, collation
    train/    the CTC trainer
    btm/      phase 0, expert branching, merging
    merge/    average, TIES, DARE-TIES
    eval/     greedy decode, scoring, held-out transfer
    stats/    seed aggregation, bootstrap, paired permutation
    report/   the aggregate / analyze / data-stats commands
```

## Development

```bash
make install     # uv sync --frozen --extra dev
make check       # everything CI runs: ruff, format, mypy, pytest
make precommit   # the git hooks, over the whole tree
```

## Acknowledgements

- **XEUS** — Chen et al., *Towards Robust Speech Representation Learning for
  Thousands of Languages* ([arXiv:2407.00837](https://arxiv.org/abs/2407.00837));
  weights at `espnet/xeus`. All credit for the model and the weights belongs to
  its authors; this repository only loads and fine-tunes them.
- **ESPnet** — the toolkit XEUS is built with. The reference XEUS load path is
  *not* part of this repository: it requires a fork of ESPnet pinned to
  `numpy<1.24`, which cannot be installed on the Python version this project
  targets, so `model/xeus_standalone.py` reimplements the encoder in PyTorch
  alone and mirrors the reference's behaviour — including its unmasked
  convolutions — rather than depending on it.
- Merge strategies follow Ilharco et al. (task arithmetic), Yadav et al. (TIES)
  and Yu et al. (DARE).
- The XEUS loader is adapted from the author's FLAIME project.

## Licence

MIT — see [LICENSE](LICENSE). The corpora carry their own licences: Common Voice
under its release terms, and the four OpenSLR held-out sets under CC-BY-SA-4.0,
which requires attribution and share-alike on derivatives.
