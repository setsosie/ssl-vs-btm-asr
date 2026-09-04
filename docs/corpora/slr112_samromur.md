# Samrómur 21.05 — `slr112_samromur`

| | |
|---|---|
| Language | Icelandic (`is`) |
| Source page | <https://www.openslr.org/112/> |
| Archive | `samromur_21.05.tgz` (7.0 G) |
| Preparer | `scripts/prepare_samromur.py` |
| Size | 145 h, 100,000 utterances, 8,392 speakers |
| Audio | FLAC, 16 kHz, 16-bit linear PCM, 1 channel |

## Licence

**CC BY 4.0**, verbatim from the source page and from `resources/112/info.txt`.
<https://creativecommons.org/licenses/by/4.0/>

## Download

```bash
python scripts/prepare_samromur.py --root $CORPORA_ROOT
```

openslr.org publishes no checksum for this resource; integrity is the byte
length against `Content-Length` plus a full read of every tar member, with the
SHA-256 recorded for later comparison.

## Verified layout

Read off the archive's first tar members:

```
dev/003040/003040-0022916.flac
dev/001277/001277-0009138.flac
```

The top-level directory is the split, the directory under it is the speaker id,
and the file stem is `{speaker_ID}-{utterance_ID}` — matching what the page
states: "folders that correspond to speaker IDs, and the audio files inside use
the following naming convention: {speaker_ID}-{utterance_ID}.flac".

So the split, the speaker and the utterance id all come out of the path and none
of a row's identity is inferred from metadata.

## Split policy: **shipped, passed through**

The source page states the corpus is **"split into train, dev, and test subsets
with no speaker overlap"**. That is the cleanest partition of any corpus in this
set, and re-deriving it would replace a published speaker-disjoint split with a
worse one. The preparer fills `split` from the top-level directory, mapping the
corpus's `dev` to the loader's `validation`, and derives nothing.

The Hugging Face repackaging of this corpus reports the subsets as train
114 h 34 m, test 15 h 51 m and validation 15 h 16 m.

## Caveats

- **The metadata file is not named on the source page.** It says only that the
  corpus "is distributed with a metadata file with a detailed information on
  each utterance and speaker", and the transcript is the read prompt, which is
  in that file rather than in the path. So the preparer finds it by shape: any
  tab-separated file in the extracted tree whose header carries both an
  utterance-id column and a text column. What it found, including the full
  column list, goes in `fetch_manifest.json`. The Hugging Face repackaging
  documents `audio_id`, `speaker_id`, `gender`, `age`, `duration` and
  `normalized_text` at `corpus/files/metadata_{train,dev,test}.tsv`, but that is
  a different packaging from the OpenSLR tgz, so those names are looked for
  rather than assumed. If the archive names its columns differently, extend
  `ID_COLUMNS` / `TEXT_COLUMNS` in the preparer.
- **Check on first run that all three splits are present.** The page describes
  three subsets, but `resources/112/info.txt` calls the tgz "whole corpus
  (includes dev and train sets)" and does not mention test. If a split directory
  is missing the preparer refuses rather than writing a manifest the loader
  accepts and whose test side scores on nothing. `--derive-splits` is the
  documented way out: it clears the column and derives a speaker-disjoint
  partition over whatever did ship, at the cost of no longer using the published
  one, which a write-up would then have to say.
- Read prompts, not spontaneous speech. A result that improves here should not
  be reported as improving on speech generally.

## Citation

> Samrómur 21.05, Language and Voice Lab, Reykjavík University.
> <https://samromur.is/>
