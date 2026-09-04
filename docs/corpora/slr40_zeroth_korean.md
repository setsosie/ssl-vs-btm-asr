# Zeroth-Korean — `slr40_zeroth_korean`

| | |
|---|---|
| Language | Korean (`ko`) |
| Source page | <https://www.openslr.org/40/> |
| Archive | `zeroth_korean.tar.gz` (10 G) |
| Preparer | `scripts/prepare_zeroth.py` |
| Size | 51.6 h train (22,263 utterances, 105 speakers), 1.2 h test (457 utterances, 10 speakers) |
| Audio | FLAC |

## Licence

**Attribution 4.0 International (CC BY 4.0)**, verbatim from the source page and
from `resources/40/info.txt`. <https://creativecommons.org/licenses/by/4.0/>

## Download

```bash
python scripts/prepare_zeroth.py --root $CORPORA_ROOT
```

**This is the one corpus in this group that publishes a checksum.**
`resources/40/checksum.md5` gives

```
8e0a4268bb8e80db3773c331025ef1e2  zeroth_korean.tar.gz
```

so the archive is verified against a published value rather than merely digested
for later comparison, and a mismatch deletes it.

## Verified layout

Read directly off the archive's first tar members:

```
AUDIO_INFO
train_data_01/<SCRIPTID>/<SPEAKERID>/<SPEAKERID>_<SCRIPTID>_<n>.flac
train_data_01/<SCRIPTID>/<SPEAKERID>/<SPEAKERID>_<SCRIPTID>.trans.txt
test_data_01/...
```

`AUDIO_INFO` is pipe-separated with the header

```
SPEAKERID|NAME|SEX|SCRIPTID|DATASET
104|...|m|003|test_data_01
106|...|f|003|train_data_01
```

The transcripts are LibriSpeech-shaped: one `.trans.txt` per
speaker-and-script directory, each line `<utterance id> <transcript>`, for
example

```
149_003_0164 우리는 위기의 진앙이 아닌 안전지대가 됐다는 말까지 나올 정도였다
```

This settles two fields the corpus record had as unconfirmed: the audio is FLAC,
and a speaker id is present both in `AUDIO_INFO` and in the path.

## Split policy: **re-derived, speaker-disjoint 80/10/10**

The shipped partition is speaker-disjoint but its test side is **1.2 hours**,
under this project's 2-hour evaluation bar. So the preparer clears `split` on
every row and the shared derivation in `svb.data.splits` repartitions all 115
speakers. The shipped test speakers are not discarded; they enter the same pool.

**This costs roughly 20% of the published training hours.** 80/10/10 over the
52.8 h total leaves about 42 h of training against the published 51.6 h, with
the balance in validation and test — both then comfortably over the bar. The
published 51.6/1.2 figures no longer describe the splits a run uses, and a
write-up has to say the split was re-derived rather than quote them.

Because there is no published figure left to quote, the preparer measures the
realised hours per derived split and records them in `fetch_manifest.json` under
`derived_hours`. Durations come out of each FLAC's STREAMINFO header, which
costs 26 bytes a file and no decoder.

## Caveats

- **`AUDIO_INFO` carries contributors' personal names** in its `NAME` column.
  The preparer reads that file only for `SPEAKERID` and `DATASET`; the name
  never reaches the manifest, the fetch record or a log, and a test asserts it.
  The numeric `SPEAKERID` is the only speaker identity carried.
- The archive also ships a pretrained language model, a lexicon and a
  morpheme segmenter. None of that is read; only the FLAC and the `.trans.txt`
  files are.
- The sample rate is not stated on the page and is read from the files.

## Citation

> Zeroth Project, Goodatlas. <https://github.com/goodatlas/zeroth>
