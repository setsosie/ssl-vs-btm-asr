# The small study

**This is a protocol, written before any run. No result exists yet.** It fixes
what will be run, on what data, under which seeds, and how the outcome will be
read, so that none of those can be chosen after the numbers are in.

It is the smallest experiment in this repository that answers the question in
its name, and it is sized to finish.

## Question

On a multilingual encoder, is it the self-supervised pretraining or the
Branch-Train-Merge pipeline that carries low-resource ASR?

| Contrast | What it asks | Role |
|---|---|---|
| **A − B** (SSL-only vs BTM-on-SSL) | Does BTM add anything on top of SSL? | **primary** |
| **B − C** (BTM-on-SSL vs BTM-from-scratch) | How much of BTM's result is the SSL initialisation? | secondary |
| **A − C** (SSL-only vs BTM-from-scratch) | Can the pipeline stand in for pretraining? | secondary |

One contrast is primary because one is what the title asks and because three
unadjusted tests are not one. The other two are reported with the same
machinery and read as supporting evidence.

## Design

- **Arms**: A, B and C as defined in the [README](../README.md).
- **Training languages**: the 3-language preset — English, Japanese, Hindi
  ([`configs/scales/3.yaml`](../configs/scales/3.yaml)).
- **Held-out languages**: Malayalam, Marathi, Telugu, Gujarati
  ([`configs/scales/heldout.yaml`](../configs/scales/heldout.yaml)). Four.
- **Seeds**: every seed in [`configs/seeds.yaml`](../configs/seeds.yaml), for
  every arm. Fifteen runs.
- **Settings**: [`configs/base.yaml`](../configs/base.yaml) with
  [`configs/studies/small.yaml`](../configs/studies/small.yaml) laid over it.
  The overlay changes two values and nothing else.

### The comparison the claim rests on

**A − B on held-out transfer, macro-averaged over the four languages, word
error rate.** Every arm fine-tunes each held-out language under the same epoch
ceiling from its own starting point, so this comparison is matched on the
budget of its last stage — not on everything before it, which is the treatment,
and not on where early stopping ends each arm. That is one pre-specified test. The per-language rows beneath it, the
two secondary contrasts and the in-distribution section are reported without
adjustment and are read as breakdowns, not as further tests.

The in-distribution comparison is reported and is secondary. Arm A fine-tunes
each language for 50 epochs; a BTM language sees 15 epochs of joint training and
9 of its own expert. A difference there is partly a difference in budget, and
reading it as the effect of the pipeline alone overstates the case against BTM.

## Data

Every training and validation split is capped in hours. Without a cap the preset
cannot be run: Common Voice 25 has about 2,700 trainable hours of English, and
arm A would pass them through the encoder fifty times per seed.

| Language | Trainable | Used for training | Validation | Used for validation | Test |
|---|---:|---:|---:|---:|---:|
| English | 2,703.6 h | 10 h | 24.0 h | 1 h | 24.0 h, whole |
| Japanese | 349.5 h | 10 h | 11.2 h | 1 h | 11.2 h, whole |
| Hindi | 7.0 h | 7.0 h, whole | 3.9 h | 1 h | 4.7 h, whole |
| held-out ×4 | a few hours each | whole | under 1 h each | whole | whole |

Hours are from [`languages.md`](languages.md), which is generated from Common
Voice 25's own release statistics; `svb data-stats` measures them from the
corpus on disk, held-out languages included, and each run records exactly what
its cap kept under `data` in `results.json`.

The subset is drawn by hash ([`svb.data.hours`](../src/svb/data/hours.py)):

- **The same utterances for every arm and every seed.** The run seed does not
  enter the selection. Seed spread therefore measures initialisation and data
  order, and the arms are not separated by which ten hours each was given. Each
  run records a digest of its subset, so this is checkable from the results.
- **Nested.** The 5-hour subset is a prefix of the 10-hour one, so a later study
  at a larger cap extends this one.
- **The test split is never capped.**

This makes the study a statement about *these* ten hours of each language, and
the intervals below are conditional on them: the subset is fixed, so nothing in
the study says how much a different ten hours would have moved the numbers. It
does not say what happens with the full corpora either.

## Seeds

Five, drawn from the operating system's entropy source and committed before any
run existed. They are not re-drawn, extended or reordered. If compute forces
fewer, the first three are used and the table caption says so, quoting the
three-seed coverage from the table under [Statistics](#statistics).

The same five values seed every arm. The bootstrap treats the arms' runs as
unpaired, so any correlation a shared seed induces between arms — the transfer
stage shuffles its data in the same order under both — makes the interval
somewhat conservative, partly offsetting the undercoverage measured below. It
is not modelled and not claimed.

Runs are not bit-reproducible on a GPU — CTC's backward pass has no
deterministic CUDA kernel — which is why there are several seeds at all. Within
a run each stage is seeded from the run seed and the stage's name, so a stage
draws the same stream whether the run was interrupted or not.

## Budget

Audio passed through the encoder in training, per run, at the shipped epoch
counts with 27 training hours in distribution (10 + 10 + 7):

| | In-distribution training | Transfer training |
|---|---:|---:|
| Arm A | 50 × 27 h = 1,350 h | 50 × *T* |
| Arms B, C | 15 × 27 h + 9 × 27 h = 648 h | 50 × *T* |

*T* is the four held-out training splits together. Validation adds 3 hours per
in-distribution epoch. These are ceilings: arm A's fine-tune and the transfer
stage can stop early, phase 0 and the experts cannot (their patience is at or
above their epoch count).

**How long that takes is not known**, because nothing has been run. Each run
writes every stage's wall-clock time to `stages.json`; one seed of arm B is the
cheapest way to measure it, and the rest of the study should be scheduled from
that measurement and not from a guess.

## Statistics

Produced by `svb compare --scale 3`, into `tables/3_arms.{md,json}`:

- Per arm and language: mean ± sample standard deviation across seeds.
- Per pair of arms: the difference in mean error rate with a 95% interval from a
  bootstrap that resamples each arm's seeds and each language's test utterances
  together (the Multi-Bootstrap of Sellam et al., 2022). It carries run-to-run
  and test-set variation in one interval; a seed standard deviation carries only
  the first and an utterance bootstrap only the second.
- The macro row of the transfer section for A − B is the pre-specified test.
  Every other row is a breakdown and is not adjusted for multiplicity.

**The interval is anti-conservative at this many seeds**, and the amount is
measured rather than guessed. `scripts/interval_coverage.py` simulates two
arms with no true difference, seed effects shared across a run's languages, and
binomial utterance noise, and calls the real statistic:

| regime | seeds/arm | per-language coverage | macro coverage |
|---|---:|---:|---:|
| 1 language, utterance noise dominant | 5 | 0.94 | 0.94 |
| 1 language, seed noise dominant | 5 | 0.92 | 0.92 |
| 4-language macro, seed noise dominant | 3 | 0.87 | 0.84 |
| 4-language macro, seed noise dominant | 5 | 0.88 | 0.87 |
| 4-language macro, seed noise dominant | 10 | 0.93 | 0.92 |
| 4-language macro, utterance noise dominant | 5 | 0.93 | 0.89 |

Nominal 95%, 300 replicates each, so every cell carries a Monte-Carlo error of
about ±0.02 and another seed of the script moves it by that much. So with five
seeds a `p < 0.05` on the primary contrast is roughly one-in-ten evidence under
the null, not one-in-twenty, and with three seeds it is nearer one-in-six. The
regimes are stylised and bracket the study rather than model it; the table is
regenerated with the script, not edited.

`svb aggregate` and `svb analyze` remain the per-arm and per-run views.

## Before the first run

1. **The encoder has been checked against the reference implementation**
   ([`espnet-crosscheck.md`](espnet-crosscheck.md)). It is a reimplementation,
   and every number here comes out of it.
2. **The data is staged** and `make data` reports every language reachable.
3. **One seed of one arm has run end to end** on the real encoder. The CPU
   suite runs the whole pipeline against a stand-in encoder, which covers the
   plumbing and not the model.

Commands for all of it: [`runbooks/small-study.md`](runbooks/small-study.md).

## What this study cannot say

- **Three training languages.** The merging-versus-typology question needs the
  16- and 64-language presets and is out of scope here.
- **Read speech only**, prompted and crowdsourced. A result here is about read
  speech.
- **Hindi has 7.0 trainable hours**, under the 50-hour rule the larger presets
  apply. Its per-language number belongs beside that figure.
- **"Held out" means held out of the supervised pipeline.** XEUS was pretrained
  on unlabelled audio that includes the four held-out languages.
- **Five seeds is few.** The interval covers a true null about 87–94% of the
  time, not 95%; the table above gives the figures. Ten seeds would bring it to
  about 92–93%.
- **Test utterances are resampled as independent.** A Common Voice test set
  holds many clips per speaker and per sentence, and the held-out sets few
  speakers, so the utterance half of the interval is also somewhat too narrow.
  The prediction sidecars carry no speaker id, so a speaker-clustered bootstrap
  would need them added first.
- **The held-out four are fixed, and all Indic.** The macro interval treats
  them as the population; it is a statement about these four languages, and
  Marathi shares its script with training-language Hindi.
- **Hyperparameters are inherited**, from a search run at a different effective
  batch size and on none of the held-out languages.
- **The vocabulary floor bites harder on capped data.** `min_char_count: 5` was
  set for full corpora. In ten hours of Japanese a real character can occur
  fewer than five times, in which case it is evicted and becomes an error no
  arm can avoid. The rate is recorded per language as `unk_rate` in
  `text_stats.json` and is identical across arms, so it moves absolute numbers
  and not the contrasts — but it should be read before the Japanese number is
  quoted.

## Open decisions

Set here as defaults; each is a judgement and is cheap to change before the
first run and expensive after it.

| Decision | Set to | Alternative |
|---|---|---|
| Training cap | 10 h per language | 5 h is nested inside it; a larger cap needs the budget re-derived |
| Validation cap | 1 h per language | Uncapped validation costs more than training for English |
| Same subset for every seed | yes | A per-seed subset would fold data sampling into seed spread |
| Seeds | 5 | The first 3, stated in the caption |
