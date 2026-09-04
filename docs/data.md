# Data

Everything is read from a local directory. Nothing downloads at load time.

| Role | Corpus | Env var | Fetched by |
|---|---|---|---|
| Training / in-distribution eval | Common Voice 25 | `CV_ROOT` | you, manually |
| Training / in-distribution eval | 18 other public corpora | `CORPORA_ROOT` | `scripts/prepare_<corpus>.py` |
| Held-out transfer | OpenSLR Indic (SLR63, 64, 66, 78) | `OPENSLR_ROOT` | `scripts/fetch_openslr.py` |

Which language comes from which corpus, with hours and licences, is in
[`languages.md`](languages.md); the corpus records themselves are in
[`../configs/corpora.yaml`](../configs/corpora.yaml).

Check what is reachable and how big each split is with `python scripts/check_data.py`.
It reads transcripts and manifests only and never decodes audio.

For how much *audio* each language contributes rather than how many utterances,
run `svb data-stats`. It reports train, dev and test hours per language, flags
anything under a `--min-train-hours` threshold, and writes
`tables/data_durations.{md,json}` for the data appendix. It decodes no audio
either: Common Voice durations come from the release's own `clip_durations.tsv`
where one is shipped, and OpenSLR durations from WAV headers. Utterance counts
are not a proxy for hours — sentence length varies by an order of magnitude
across Common Voice languages.

Its Common Voice training hours follow `--cv-train-source`, so the audit prices
the rows a run will actually read rather than a split it may not use. Under the
default it also reports what the speaker guard removed, per language.

## Common Voice 25 (`CV_ROOT`)

Since October 2025 Common Voice is distributed only through
[Mozilla Data Collective](https://commonvoice.mozilla.org/en/datasets); the
`mozilla-foundation/common_voice_*` datasets on the Hub are empty. Download and
extract v25 yourself (account and terms acceptance required), then point
`CV_ROOT` at the extraction root:

```
$CV_ROOT/<lang>/train.tsv  dev.tsv  test.tsv  validated.tsv
$CV_ROOT/<lang>/clips/<file>.mp3
```

`<lang>` is the Common Voice language code, which is also the `hf_config` field
of a `commonvoice` entry in `configs/scales/*.yaml`. The `hf_dataset` field on
those entries is **not** a Hub id — it is a provenance label for the release
(`common_voice_25`) that gets recorded in run metadata. Both field names are
historical; renaming them would touch every preset entry.

The loader reads the `path`, `sentence` and `client_id` columns. mp3 decoding
goes through torchaudio's ffmpeg backend, so install ffmpeg if clips fail to
load.

### Split policy

Evaluation is the official `dev.tsv` and `test.tsv`, always, under either
training source. Training is one of two, set by `train.cv_train_source` in
`configs/base.yaml` and recorded in every run's `resolved_config.yaml`:

| `cv_train_source` | Training rows |
|---|---|
| `validated_minus_eval` (default) | every validated clip not in dev or test, and not by a dev or test speaker |
| `train` | the official `train.tsv` |

**Why the default is not `train.tsv`.** The `train`/`dev`/`test` partition is
not a partition of `validated`. Quoting the release's
[own documentation](https://github.com/common-voice/cv-dataset/tree/main/datasets/scripted-speech):

> `validated` -- clips with two or more validations where `up_votes` >
> `down_votes`

> We use the Corpora Creator tool to parse through metadata to generate train,
> dev, and test sets. The Corpora Creator eliminates duplication in clips and
> maximizes for speaker diversity.

> Note that total clips in these sets will most probably not add up to the total
> validated clips because of this limitation.

Corpora Creator splits a frame holding at most one clip per sentence, so
`train.tsv` is roughly one recording per sentence and about a third of the
validated audio across the release. Every further recording of a sentence that
was already covered sits in `validated.tsv` and in no split at all. Counted on
`train.tsv`, only 24 Common Voice 25 locales reach 50 training hours; counted on
validated minus dev and test, 44 do. See [`languages.md`](languages.md).

**The speaker guard.** Subtracting dev and test by clip path is not enough. In
Corpora Creator (`src/corporacreator/corpus.py`, blob
`da6233c3d76d13627e569763cd6cdcfb32540b80`) the split label is assigned one
whole `client_id` at a time, so train, dev and test are speaker-disjoint from
each other — but the labels are assigned over the deduplicated frame, and the
test split is then truncated with `.head(test_size)`. Both leave validated clips
in no split, and some belong to the dev and test speakers. So the loader also
drops every validated clip whose `client_id` appears in dev or test.

The release documentation promises nothing about speakers across splits, which
is why this is a filter rather than an assertion. A release that ships no
`client_id` column is refused rather than trained on without the guard.

`svb data-stats --cv-train-source validated_minus_eval` reports per language how
many clips, how many distinct speakers and how many hours the guard removed,
into `tables/data_durations.{md,json}`. That is the number to look at before
committing a large run: it is the price of speaker-disjointness, and the release
statistics cannot predict it because they carry no speaker information.

Two properties follow from the label-per-speaker rule and are asserted in the
tests rather than assumed: `validated_minus_eval` is a **superset** of
`train.tsv`, and no training clip shares a speaker with dev or test.

**Licence.** Mozilla releases Common Voice under
[CC0](https://creativecommons.org/publicdomain/zero/1.0/) — see the
[Mozilla Foundation release notes](https://www.mozillafoundation.org/en/blog/common-voice-18-dataset-release/).
The terms you accept at download time govern your use; confirm them against the
release you actually downloaded rather than against this file.

## The other corpora (`CORPORA_ROOT`)

Common Voice cannot reach 64 languages at 50 training hours — it reaches 44 — so
the large tier draws on eighteen more corpora. They ship in about ten different
shapes: per-utterance XML, JSONL with word-level alignments, Kaldi-ish flat
directories, STM and VTT, one `.txt` per `.wav`. Ten dataset classes would be
ten places to get a split wrong.

So each corpus is converted **once**, by a preparer, into one layout that a
single loader reads:

```
$CORPORA_ROOT/<corpus id>/<language>/manifest.tsv
$CORPORA_ROOT/<corpus id>/<language>/<audio, wherever the manifest says>
```

`manifest.tsv` is tab-separated with a header and exactly five columns:

| Column | Meaning |
|---|---|
| `utt_id` | stable, unique within the language; orders the rows, digests the test set, and is the fallback grouping key |
| `path` | audio path relative to the language directory |
| `text` | the transcript, already extracted from whatever the corpus shipped |
| `speaker` | speaker id, or empty |
| `split` | `train`, `validation`, `test`, or empty |

Two rules make the split describable afterwards. **Either every row carries a
split or none does** — a manifest that is part shipped and part derived is
refused, because the resulting partition could not be written down in a results
file. And **an empty `speaker` on any row** drops that whole language to an
utterance-level split, because a partition that is speaker-disjoint for most
rows is not speaker-disjoint. Every run records which policy it got, and a
digest of the exact test utterance list, the same way the held-out set does.

Where a corpus ships no usable split, `svb.data.splits` derives one — the same
speaker-disjoint 80/10/10 derivation the held-out four use, now shared rather
than duplicated. Two corpora ship an incomplete one and re-derive: Kannada has
no dev split, and Zeroth's shipped test is 1.2 hours, under the 2-hour bar. In
both cases the published division is not the one a run uses, and a write-up has
to say so.

### The preparer contract

A preparer fetches what `configs/corpora.yaml` lists and writes the manifest.
It may do anything to get there; it owns the per-corpus knowledge, which is why
that knowledge is not in the loader. The shared parts — resumable download,
integrity checking, zip and tar extraction, and a Hugging Face snapshot path —
are in `scripts/corpus_fetch.py`, which stays standard library only except on
the Hub path so a preparer can run on a machine with no project install.

```bash
export CORPORA_ROOT=/path/to/corpora
python scripts/prepare_google_crowdsourced.py --root $CORPORA_ROOT   # jv su si ne bn
python scripts/prepare_nchlt.py --root $CORPORA_ROOT --langs zu
```

**All ten are implemented. None has been run against a real archive.** Every
format claim below was established by reading a zip central directory or a tar
header chain over HTTP Range, or by fetching a repository's own metadata — real
evidence, but not the same as an ingest. Each has a page in
[`corpora/`](corpora/) recording what was read and where.

| Preparer | Languages | Split | Notes |
|---|---|---|---|
| [`prepare_google_crowdsourced.py`](../scripts/prepare_google_crowdsourced.py) | jv su si ne bn | derived | [`corpora/`](corpora/) — sixteen shards each, one `utt_spk_text.tsv` |
| [`prepare_nchlt.py`](../scripts/prepare_nchlt.py) | zu xh nso ts ve | test shipped, dev derived | [`nchlt.md`](corpora/nchlt.md) |
| [`prepare_parlaspeech.py`](../scripts/prepare_parlaspeech.py) | hr | shipped | [`parlaspeech.md`](corpora/parlaspeech.md) |
| [`prepare_taltech.py`](../scripts/prepare_taltech.py) | et | shipped | [`taltech.md`](corpora/taltech.md) |
| [`prepare_ksc.py`](../scripts/prepare_ksc.py) | kk | shipped | [`slr102_ksc_kazakh.md`](corpora/slr102_ksc_kazakh.md) |
| [`prepare_samromur.py`](../scripts/prepare_samromur.py) | is | shipped | [`slr112_samromur.md`](corpora/slr112_samromur.md) |
| [`prepare_zeroth.py`](../scripts/prepare_zeroth.py) | ko | re-derived | [`slr40_zeroth_korean.md`](corpora/slr40_zeroth_korean.md) |
| [`prepare_kannada_mile.py`](../scripts/prepare_kannada_mile.py) | kn | re-derived | [`slr126_kannada.md`](corpora/slr126_kannada.md) |
| [`prepare_tibmd.py`](../scripts/prepare_tibmd.py) | bo | derived | [`slr124_tibetan.md`](corpora/slr124_tibetan.md) |
| [`prepare_armenian.py`](../scripts/prepare_armenian.py) | hy | derived, utterance level | [`slr160_armenian.md`](corpora/slr160_armenian.md) |

Three of them cost real disk. **TalTech is the expensive one: budget about
320 GB of peak.** Its 159 GB tar and its extracted tree exist at the same time,
and its half-hour recordings are then cut at their transcript bounds into
roughly 600,000 small WAVs — the manifest addresses whole files, so long-form
audio has to be cut somewhere and it is cut here. ParlaSpeech needs about 116 GB
of slices plus about the same extracted, and each NCHLT language is a 4.6–5.1 GB
zip.

### Integrity

**Three corpora publish a checksum and are checked against it:** NCHLT, per
bitstream from the DSpace API; ParlaSpeech, per file from its METS record; and
Zeroth, at `resources/40/checksum.md5`. A mismatch stops the fetch — a corrupt
or substituted archive has no useful handling — and `fetch_manifest.json`
records the value matched under `checksum_verified`.

Every other corpus publishes nothing. For those, the length is checked against
`Content-Length` where the server sends one, the archive is read end to end, and
the recorded SHA-256 is good only for comparing one fetch against another.
`checksum_verified` is null there, which is the difference between an archive
that was verified and one that was merely digested.

Three preparers go further and check the corpus rather than the bytes: NCHLT and
ParlaSpeech and TalTech each re-read the licence field at fetch time and refuse
to ingest a corpus whose licence has changed, and ParlaSpeech refuses a manifest
whose segment counts differ from the published 380,836 / 500 / 513 / 22,076. A
licence copied into a config file is a claim about the past.

### Licences, and a caveat about the mix

Every corpus's verbatim licence name and URL is in `configs/corpora.yaml`, read
on its own source page. The set mixes CC0, CC BY 2.0, 3.0 and 4.0 and
CC BY-SA 4.0; ShareAlike over a training corpus is not settled law with respect
to model weights, and the paper should take a position rather than say nothing.
Anything behind a gate, a form or an email request is excluded on principle,
however large; `languages.md` lists what that ruled out.

**The mix is less domain-diverse than 64 languages sounds.** The additions are
almost entirely read prompts or parliamentary speech. NCHLT's prompts are
scripted and its published test side is an 8-speaker suite. A result that
improves on read speech should not be reported as improving on speech.

## OpenSLR held-out Indic (`OPENSLR_ROOT`)

Four crowdsourced multi-speaker read-speech corpora, held out of every training
mix in this repository and absent from the training vocabulary until the CTC
head is expanded for transfer.

They are **not** unseen by the encoder. XEUS was pretrained on unlabelled audio
from over four thousand languages, these among them, so "held out" means held
out of the supervised pipeline rather than a language the model has never heard.
The transfer experiment measures how far fine-tuning drives each starting point
down on a language the pipeline never supervised, not zero-shot generalization
to an unheard one. Any write-up should say it that way.

Fetch them once:
```bash
python scripts/fetch_openslr.py --root $OPENSLR_ROOT
python scripts/fetch_openslr.py --root $OPENSLR_ROOT --langs marathi   # one language
```

The script needs only the standard library and PyYAML, so it can stage data on a
machine that has no project install. Downloads resume, complete archives are
skipped, and each language ends up in one flat directory:

```
$OPENSLR_ROOT/SLR63/line_index_female.tsv
$OPENSLR_ROOT/SLR63/line_index_male.tsv
$OPENSLR_ROOT/SLR63/mlf_02879_01795762363.wav ...
$OPENSLR_ROOT/SLR63/manifest.json
```

| Language | SLR | Archives | Index files |
|---|---|---|---|
| Malayalam | 63 | `ml_in_female.zip`, `ml_in_male.zip` | `line_index_female.tsv`, `line_index_male.tsv` |
| Marathi | 64 | `mr_in_female.zip` | `line_index.tsv` |
| Telugu | 66 | `te_in_female.zip`, `te_in_male.zip` | `line_index_female.tsv`, `line_index_male.tsv` |
| Gujarati | 78 | `gu_in_female.zip`, `gu_in_male.zip` | `line_index_female.tsv`, `line_index_male.tsv` |

Marathi is the odd one out: female speakers only, one index, and only nine
speakers in total. That asymmetry is worth a sentence in any write-up.

Each index is two tab-separated columns with no header, `FileID` then
transcript, and the audio for a row is `<FileID>.wav` beside it. Audio is 16-bit
mono PCM WAV at 48 kHz, read with `soundfile` and resampled to 16 kHz, so this
path needs no mp3 decoder and no ffmpeg.

Every archive is flat and ships its own index member named `line_index.tsv`.
Extracting a language's male and female archives into one directory would
therefore clobber it, so the fetch script writes each archive's index under the
name the config declares. That is why `archives` and `index_files` in
`configs/scales/heldout.yaml` are parallel lists.

**Integrity.** openslr.org publishes no checksums for these resources. An
archive is accepted when its length matches the server's `Content-Length` — when
the server sends one — and a full zip CRC pass succeeds; a failing archive is
deleted rather than extracted. Where there is no `Content-Length` the length
check cannot run and the CRC is the only evidence, which the fetch says out loud
rather than reporting the download as fully verified. The SHA-256 of each archive
is recorded in `manifest.json` for later reference, not compared against a
published value.

The downloaded archives themselves are deleted once extracted; pass
`--keep-archives` to `scripts/fetch_openslr.py` to keep them under
`$OPENSLR_ROOT/.archives/` and make a re-extraction free.
**Licence.** All four are
[CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/), stated on each
`openslr.org/<n>/` page. Attribute the corpora and share derivatives alike.

### Split policy

These corpora ship no train/validation/test split, so one is derived at load
time. It is derived from a stable SHA-1 rather than by shuffling with a seed:
a library's shuffle is only reproducible for a pinned version of that library,
which is not something a results file records.

The split is **speaker-disjoint** at a nominal 80/10/10, whenever the FileIDs
carry a speaker field. They have the form `<corpus-prefix>_<speaker>_<utterance>`
— in SLR63's female index, 2103 utterances carry only 24 distinct middle fields
(160, 142, 140 … rows each) while every third field is unique, which is what
identifies the middle field as a speaker. Test speakers are then never seen in
training, which is what a transfer number should measure.

That inspection was done on SLR63's female index only. The other three corpora
are documented the same way on openslr.org and are expected to match, but the
loader does not assume it: it parses the ids at load time and reports which
policy it ended up with, so a corpus that turns out to be shaped differently
says so rather than quietly producing a split of another kind.
Speakers are ordered by SHA-1 of the speaker key, then each is given to whichever
split has the largest remaining shortfall. Bucketing by hash alone would be far
too lumpy at these sizes: Malayalam has 42 speakers across both archives and
Marathi only 9, and one speaker can be a large fraction of a small corpus — in
SLR63's female index the biggest contributes 160 of 2103 rows. Because whole
speakers are indivisible, the realised proportions drift from 80/10/10 — more so
the fewer speakers a language has.

That shortfall rule on its own can leave a split with nothing in it, which one
dominant speaker in a nine-speaker corpus is enough to do. So a second pass
fills any empty split: the split holding the most speakers donates its smallest
one, until none is empty. Reading the shortfall rule alone would suggest an
empty test split is reachable for Marathi; it is not, and the difference is this
pass rather than luck.

If **any** FileID does not parse that way, or a language has fewer than three
speakers, the split degrades to utterance level for that whole language. That
case carries an obvious caveat: the same speaker then appears in train and test,
so the result is not speaker-independent.

Which policy was used is never left to be assumed. `check_data.py` prints it per
language, and every run records it — together with `test_files_sha1`, the SHA-1
of the test FileID list — under `transfer.<language>` in its `results.json`. So
a held-out number always sits beside both the policy that produced it and a
digest pinning the exact set of utterances scored.
### Odia — decision pending

Odia is deliberately **not** part of the held-out set. It is kept as a commented
block in `configs/scales/heldout.yaml` and differs from the four above in three
ways that all need a decision first:

- It is SLR103, the Multilingual and Code-Switching ASR Challenge (MUCS 2021)
  sub-task 1 corpus, not one of the crowdsourced read-speech sets.
- Its licence is a Microsoft Research Open Data licence, linked from
  `openslr.org/103/`, **not** CC-BY-SA-4.0. It has not been read. Read it before
  redistributing anything derived from it or committing its transcripts.
- It ships its own train/test split, so the 80/10/10 derivation above must not
  be applied to it.

There is also a mechanical obstacle, which is worth knowing before anyone spends
a download on it: `scripts/fetch_openslr.py` handles zip archives with a
`line_index*.tsv` member, and Odia ships `.tar.gz` archives with
`transcription_*.txt`. Enabling the commented block as written downloads the
archives and then fails on the first one as an unreadable zip. Enabling Odia
therefore needs a tar path and a different index parser as well as the licence
decision.

Until that is resolved the held-out set is four languages, and any write-up
should say four.
