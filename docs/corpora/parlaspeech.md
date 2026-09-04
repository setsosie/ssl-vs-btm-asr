# ParlaSpeech-HR v1.0 (Croatian)

One language, `hr`, from CLARIN.SI handle `11356/1494`, fetched by
[`scripts/prepare_parlaspeech.py`](../../scripts/prepare_parlaspeech.py).

```bash
export CORPORA_ROOT=/path/to/corpora
python scripts/prepare_parlaspeech.py --root $CORPORA_ROOT
```

The repository labels the item `PUB` — publicly available — and every bitstream
serves to an anonymous client. No login, no form, no email request.

## What is downloaded

Resolved from the item's METS record at
`https://www.clarin.si/repository/xmlui/metadata/handle/11356/1494/mets.xml`
rather than from hard-coded URLs, because that document carries a published MD5
and byte size for every file:

| File | Bytes | MD5 | What it is |
|---|---|---|---|
| `ParlaSpeech-HR.v1.0.jsonl` | 712,525,343 | `271ef6589623facd86527b1e05b740f4` | every segment's metadata |
| `ParlaSpeech-HR.v1.0.txt` | 1,097 | `71a7479a87e107510c99bc2602e1076e` | the README quoted below |
| `ParlaSpeech-HR.flac.tgz.0` | 52,428,800,000 | `84076b62f51eb1da9870c1f6c4da436b` | audio, slice 0 |
| `ParlaSpeech-HR.flac.tgz.1` | 52,428,800,000 | `8123e76721d437837a2439dd662a973b` | audio, slice 1 |
| `ParlaSpeech-HR.flac.tgz.2` | 20,321,039,259 | `cd8e71d1d93a3b89d10a208c288c824e` | audio, slice 2 |

Each is checked against its published MD5 and deleted on a mismatch.

**The three `.tgz.N` are byte slices of one gzip stream, not three archives.**
The first two are exactly 50,000 MiB, and slices 1 and 2 do not begin with a
gzip header — their first bytes are `02 f6 52 d7` and `39 1d db 90`. Only slice 0
starts `1f 8b`. So they are read back to back through one reader and the tar is
extracted from that stream, which also spares the disk a joined 116 GB copy.

**Disk.** About 116 GB of slices plus about 116 GB of extracted FLAC. Gzip over
FLAC compresses almost nothing, so the extracted tree is close to the archive
size rather than smaller. The slices are deleted after extraction unless
`--keep-archives` is passed.

## Licence

Verbatim, from `dc.rights` in the item's METS record:

> Creative Commons - Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

with `dc.rights.uri` `https://creativecommons.org/licenses/by-sa/4.0/` and
`dc.rights.label` `PUB`. The preparer reads that field at fetch time and refuses
to ingest a corpus whose licence no longer contains "Attribution-ShareAlike 4.0".

ShareAlike over a training corpus is not settled law with respect to model
weights. That is a position the paper has to take rather than pass over; see
[`../data.md`](../data.md).

## Layout

**Audio.** One flat tar of FLAC, member names exactly the `path` field of the
JSONL, so the index and the archive line up without any mapping:

```
seg.qNpeHxO0WzA_1967.6-1986.11.flac
seg.-k1z8behXXg_14758.42-14778.39.flac
seg.a9Jz6KAkBFs_2680.84-2700.52.flac
```

FLAC at **16 kHz, mono, 16-bit**, read out of the first file's STREAMINFO block.
The item page states no sample rate, which is why `configs/corpora.yaml` lists it
under `unverified`; the preparer reads the rate it actually finds and records it
in `fetch_manifest.json` rather than assuming one.

**Transcripts.** One JSON object per line. From the README the repository ships
beside the data, verbatim:

```
path: name of the file with the segment recording
orig_file: name of the original file harvested from YouTube
start: second when the segment starts in the original file
end: second when the segment ends in the original file
words: list of words from the original transcript
word_start_times: relative time references (in seconds) to each word
norm_words: list of words normalized with an imperfect rule-based normaliser
norm_words_start_times: relative time references (in seconds) to each word in the
                        normalized transcript
utterance_id_start: ID of the utterance in the ParlaMint 2.1 corpus where the
                    segment starts
utterance_id_end: ID of the utterance in the ParlaMint 2.1 corpus where the
                  segment ends
speaker_info: list of speaker attributes from ParlaMint 2.1, if single speaker
              (null otherwise)
split: either "train", "dev", or "test", or "null" if multiple speakers
```

`speaker_info`, when present, is an object rather than a list:

```json
{"Speaker_role": "Regular", "Speaker_type": "MP", "Speaker_party": "SDP",
 "Speaker_party_name": "Klub Socijaldemokratske partije Hrvatske",
 "Party_status": "Opposition", "Speaker_name": "Maras, Gordan",
 "Speaker_gender": "M", "Speaker_birth": "1974"}
```

`Speaker_name` is the manifest's `speaker` column. It is the only stable speaker
key the release carries, and these are members of parliament named in published
proceedings.

**`text` is joined from `words`, not from `norm_words`.** This repository
normalizes transcripts under its own per-language policy at collate time. Folding
the corpus's own "imperfect rule-based normaliser" in first would make Croatian
the one language whose text had been through two normalizers, which is the kind
of asymmetry that quietly moves a WER.

## Split policy

All three splits ship and are used as shipped. From the item page:

> The dataset is divided into a training, a development, and a testing subset.
> Development data consist of 500 segments coming from the 5 most frequent
> speakers, with the goal of not losing speaker variety on dev data. Test data
> consist of 513 segments that come from 3 male (258 segments) and 3 female
> speakers (255 segments). There are no segments coming from the 6 test speakers
> in the two remaining subsets. The 22,076 instances not having speaker
> information are not assigned to any of the three subsets. The remaining 380,836
> instances form the training set.

Those counts are **checked, not just recorded**: the preparer refuses to write a
manifest whose segment counts differ from 380,836 / 500 / 513 / 22,076. A
silently different corpus is the failure that survives all the way into a results
table.

The corpus calls its middle split `dev`; the manifest layer calls it
`validation`, and the preparer renames it. Writing `dev` through would be
rejected by the loader as an unknown split value — deliberately, because that is
the mistake a preparer author makes from habit.

**The 22,076 multi-speaker segments are dropped.** They carry neither a speaker
nor a split. Keeping them would break the manifest layer's rule that either every
row carries a split or none does, and an empty speaker on any row drops the whole
language to an utterance-level split. `fetch_manifest.json` records how many were
dropped and why.

**Hours are measured, not copied.** The repository publishes dev and test as
segment counts rather than hours, which is what `configs/corpora.yaml` records as
unverified. Every record carries `start` and `end`, so the preparer sums the
segment bounds per split and writes the realised hours into
`fetch_manifest.json`. No audio is decoded to get them.

## Caveats

- **Parliamentary speech.** Spontaneous, but one domain and one register.
  ParlaSpeech, ARTUR, RixVox, NPSC and VoxPopuli are all parliament; together
  with the read-prompt corpora, the 64-language mix is far less domain-diverse
  than its language count suggests.
- **Six test speakers, five dev speakers.** The dev set is deliberately drawn
  from the five *most frequent* speakers, so it is not a random sample of the
  speaker population and is easier than an average held-out speaker would be.
- **Transcripts are parliamentary records aligned to YouTube audio**, not
  verbatim annotations of what was said. The corpus derives from ParlaMint 2.1
  proceedings.
- **Speaker names are personal names.** They are public record for members of
  parliament and are what the release ships; the manifest carries them because
  there is no other stable speaker key.
- **A newer release exists.** The item page states this one "is replaced by a
  newer submission", `http://hdl.handle.net/11356/1914`. v1.0 is what
  `configs/corpora.yaml` records and what these numbers describe; moving to the
  newer one is a decision, not a bump.

## Citation

> Nikola Ljubešić, Danijel Koržinek, Peter Rupnik, Ivo-Pavao Jazbec, Vuk
> Batanović, Lenka Bajčetić and Bojan Evkoski, *ASR training dataset for Croatian
> ParlaSpeech-HR v1.0*, Jožef Stefan Institute, 2022.
> http://hdl.handle.net/11356/1494

Referenced by https://aclanthology.org/2022.parlaclarin-1.16.
