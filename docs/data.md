# Data

Two public corpora, both read from a local directory. Nothing in this repository
downloads at load time and nothing goes through the Hugging Face Hub.

| Role | Corpus | Env var | Fetched by |
|---|---|---|---|
| Training / in-distribution eval | Common Voice 25 | `CV_ROOT` | you, manually |
| Held-out transfer | OpenSLR Indic (SLR63, 64, 66, 78) | `OPENSLR_ROOT` | `scripts/fetch_openslr.py` |

Check what is reachable and how big each split is with `bash scripts/check_data.sh`.
It reads transcripts and manifests only and never decodes audio.

For how much *audio* each language contributes rather than how many utterances,
run `svb data-stats`. It reports train, dev and test hours per language, flags
anything under a `--min-train-hours` threshold, and writes
`tables/data_durations.{md,json}` for the data appendix. It decodes no audio
either: Common Voice durations come from the release's own `clip_durations.tsv`
where one is shipped, and OpenSLR durations from WAV headers. Utterance counts
are not a proxy for hours — sentence length varies by an order of magnitude
across Common Voice languages.

## Common Voice 25 (`CV_ROOT`)

Since October 2025 Common Voice is distributed only through
[Mozilla Data Collective](https://commonvoice.mozilla.org/en/datasets); the
`mozilla-foundation/common_voice_*` datasets on the Hub are empty. Download and
extract v25 yourself (account and terms acceptance required), then point
`CV_ROOT` at the extraction root:

```
$CV_ROOT/<lang>/train.tsv  dev.tsv  test.tsv
$CV_ROOT/<lang>/clips/<file>.mp3
```

`<lang>` is the Common Voice language code, which is also the `hf_config` field
of a `commonvoice` entry in `configs/scales/*.yaml`. The `hf_dataset` field on
those entries is **not** a Hub id — it is a provenance label for the release
(`common_voice_25`) that gets recorded in run metadata. Both field names are
historical; renaming them would touch every preset entry.

The loader reads the `path` and `sentence` columns and uses the official
`train`/`dev`/`test` split as shipped. mp3 decoding goes through torchaudio's
ffmpeg backend, so install ffmpeg if clips fail to load.

**Licence.** Mozilla releases Common Voice under
[CC0](https://creativecommons.org/publicdomain/zero/1.0/) — see the
[Mozilla Foundation release notes](https://www.mozillafoundation.org/en/blog/common-voice-18-dataset-release/).
The terms you accept at download time govern your use; confirm them against the
release you actually downloaded rather than against this file.

## OpenSLR held-out Indic (`OPENSLR_ROOT`)

Four crowdsourced multi-speaker read-speech corpora, held out of every training
mix. Fetch them once:

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

Which policy was used is never left to be assumed. `check_data.sh` prints it per
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
