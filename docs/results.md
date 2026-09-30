# Run artifacts

One `svb run` writes one directory. Everything needed to read, check or
re-analyse that run is inside it, so a result never depends on remembering how
it was produced.

```
results/<arm>/<scale>/seed<N>/
    resolved_config.yaml     every setting the run actually used
    env.json                 git SHA, library and Unicode versions, GPU
    text_stats.json          what normalization did, per language and split
    vocab.json               the character vocabulary and its policy
    stages.json              which stages have finished, for --resume
    results.json             the metrics; written last, so it marks a finished run
    predictions/<lang>.json  per-utterance references and hypotheses
    transfer/<lang>/
        predictions.json     the same, for a held-out language
        best.pt              the adapted checkpoint
```

The root is `./results` by default, `$SVB_RESULTS_ROOT` if set, and whatever
`--results-root` says if passed. Checkpoints also land here — `phase0/`,
`experts/expert_<lang>/`, `merged/merged_<strategy>.pt` for the BTM arms and
`finetune/<lang>/` for arm A — and are gitignored along with the rest of
`results/`. The JSON is small, reproducible from the checkpoints, and the part
worth keeping.

## What each file answers

**`resolved_config.yaml`** is the configuration after the YAML and the
command-line flags have been layered, not the file that was passed in. It
carries the normalization policy, its hash, and a hash of the normalizer module
itself, so a policy edited without a version bump is still detectable.

**`env.json`** carries the git SHA and whether the tree was dirty, the SHA-256
of `uv.lock`, the resolved library versions, the GPU, and the Unicode version —
`unicodedata` decides both character categories and case folding, so two Python
builds can normalize one transcript differently. A SHA alone does not identify
the code when the tree was dirty, which is why the flag sits beside it.

**`results.json`** holds `arm`, `scale`, `seed`, the resolved `languages` and
`heldout_langs`, and two metric sections. Every language reports `wer`, `cer`,
`n` and `n_empty_refs`. Held-out languages add `split_policy` and
`test_files_sha1`, because those corpora ship no partition: which utterances
were scored is part of the number, and the digest pins it.

It also holds a `training` section, one entry per stage — `phase0`,
`expert_<lang>` for the BTM arms, `finetune_<lang>` for arm A, and
`transfer_<lang>` for every arm's held-out fine-tunes. Each records the
best validation loss and the epoch it came from, how many epochs actually ran,
the checkpoint it selected, and two counts that separate the split from what the
model saw:
`n_dropped_unalignable` (pairs whose transcript is longer than the encoder's
frame budget, which CTC cannot align) and `n_at_audio_guard` (utterances the
training-time truncation guard clipped). Both are counted over the first epoch,
which is the size of the effect on the split.

`data` records what an hours cap kept of each language's training and validation
audio — rows and hours available and selected, and a digest of the selection —
or says that no cap was set. See `data.md`.

`stages` lists every stage in the order it ran, with the seed it was given, its
wall-clock time, and which invocation ran it. `invocations` lists every process
that contributed to the run, with its git SHA and dirty flag. A run that was
never interrupted has one; a run resumed across a code change has two SHAs, and
that is visible here rather than only in a scheduler log.

## Stages and resuming

A run is a fixed sequence of stages, and `stages.json` records each as it
finishes:

| Arm | Stages, in order |
|---|---|
| A | `finetune_<lang>`, `eval_<lang>` per training language; then `transfer_<lang>` per held-out language |
| B, C | `phase0`; `expert_<lang>` per language; `merge`; `eval_<lang>` per language; then `transfer_<lang>` per held-out language |

`svb run --resume` skips the stages already recorded and continues;
`svb run --restart` discards them and starts clean. A run directory that holds
finished stages refuses to run without one or the other, because reusing them by
accident mixes two attempts and discarding them by accident throws away
GPU-days. Four rules keep a resumed run honest:

- A stage is reused only if the files it wrote — its checkpoint, its predictions
  — are still there.
- Once any stage has been run again, every later stage is too. They were
  computed from the version that was replaced.
- A record is withdrawn from the ledger before its stage is run again. The
  trainer writes a checkpoint as soon as it starts, so a re-run killed partway
  leaves a file where the old record says a finished one is.
- A run is not resumed under a different configuration, or over data that no
  longer builds the same vocabulary. The first check compares against the
  `resolved_config.yaml` the earlier invocation wrote and names the settings
  that differ; `train.num_workers` is exempt, since it changes how fast a run
  goes and not what it computes. The second compares the vocabulary built now
  with `vocab.json`, because the config cannot see the corpus.

A run directory belongs to one process at a time: `svb run` holds a lock on it
(`.lock`, an advisory `flock`, released by the kernel when the process dies) and
a second process is refused before it has written anything. Two jobs on one
cell would train the same stages into the same files.

Each stage is seeded from the run seed and its own name, so a stage draws the
same random stream whether the stages before it ran in this process or were read
back. A resumed run is therefore equivalent in distribution to an uninterrupted
one. It is not bit-identical to it, and neither are two uninterrupted runs.

`results.json` is removed when a run starts and written when it finishes, so its
presence means the run is complete. The reporting commands count a seed only if
it has one.

**`text_stats.json`** is the evidence for the normalization policy, per language
and split: utterance and character counts before and after, how many utterances
normalized to nothing, the median whitespace tokens per utterance beside the
declared `word_boundary`, and the unknown-character rate against the vocabulary
that language is actually scored with. See `normalization.md`.

**`predictions/<lang>.json`** is what makes the statistics possible after the
fact. It holds `pairs` — the reference and hypothesis for every scored utterance
— so intervals and paired tests are recomputed from disk at no GPU cost. A
corpus-level error rate cannot be resampled; these pairs can.

In-distribution sidecars sit in `predictions/`, held-out ones in
`transfer/<lang>/predictions.json`. The two are written by different code paths
and `svb analyze` reads both.

## How the reporting commands consume it

`svb aggregate` reads `results.json` from every seed of one `(arm, scale)` and
reports mean ± standard deviation across seeds. It reads the per-language
`word_boundary` back out of `text_stats.json` rather than from today's presets,
so a preset edited after a run cannot retroactively change which metric that
run's numbers were chosen under.

The macro-average is built the same way a single language is: the languages are
averaged within each seed, and those per-seed values are then averaged across
seeds. Its `±` is therefore run-to-run spread, the same quantity as every
per-language row's. The dispersion *between* the languages is a different number
and is reported separately, under its own name, never as an error bar — two
languages thirty points apart that each move two points between seeds have a
seed spread of about one and a half, not twenty-one. The macro is unweighted by
utterance count: the question is how a system does across languages, and a
corpus-weighted mean is dominated by whichever language shipped the most audio.
A language absent from any seed is excluded from the macro and named in the
output, because a mean whose membership changes between seeds is not comparable
seed to seed. In the JSON, `std_is` records which quantity `std` is,
`per_seed` carries the values it was computed from, and
`spread_across_languages` is the between-language number.
`svb analyze` reads the sidecars and reports what the test set leaves uncertain:
a bootstrap percentile interval per language, and, with `--compare-to`, a
one-sided paired permutation test against another arm at the same seed. It
writes `tables/<scale>_wer.{md,json}` and `tables/<scale>_cer.{md,json}`.

These are different quantities and the split is deliberate. Seed spread measures
what retraining moves; a bootstrap interval measures what the particular test
set contributes. A five-seed standard deviation is not a confidence interval.

The paired test checks that both runs hold identical references before comparing
them, and refuses otherwise. Two runs under different normalization policies
derive different reference strings from the same corpus, and pairing those would
compare two systems across two test sets while looking well-formed.

`svb data-stats` does not read runs at all. It reads the corpora and reports
audio hours per language and split, into `tables/data_durations.{md,json}`. See
`data.md`.

## tables/

Generated, never hand-typed. `make tables` regenerates the analysis tables and
`svb data-stats` the data ones. Every number in `tables/` traces to a file under
`results/` or to a corpus on disk, which is the property that makes a write-up
checkable rather than trusted.
