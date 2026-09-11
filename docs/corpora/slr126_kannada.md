# IISc-MILE Kannada ASR Corpus — `slr126_kannada`

| | |
|---|---|
| Language | Kannada (`kn`) |
| Source page | <https://www.openslr.org/126/> |
| Archives | `mile_kannada_train.tar.gz` (22 G), `mile_kannada_test.tar.gz` (6.1 G) |
| Preparer | `scripts/prepare_kannada_mile.py` |
| Size | ~350 h, 915 speakers |
| Audio | ".wav file recordings (16 KhZ, 16 bit, mono, PCM format)" |

## Licence

**Attribution 2.0 Generic (CC BY 2.0)**, verbatim from the source page and from
`resources/126/info.txt`. <https://creativecommons.org/licenses/by/2.0/>

The only CC BY 2.0 corpus in this set, and the only one on a 2.x licence at all,
which is worth a line in any licence table.

## Download

```bash
python scripts/prepare_kannada_mile.py --root $CORPORA_ROOT
python scripts/prepare_kannada_mile.py --root $CORPORA_ROOT --splits test
```

Both archives are fetched by default and pooled. openslr.org publishes no
checksum for this resource; integrity is the byte length against
`Content-Length` plus a full read of every tar member, with the SHA-256 recorded
for later comparison.

## Verified layout

Read off both archives' tar members. Each carries its own top-level split
directory, so they cannot collide and the layout is kept rather than flattened:

```
train/audio_files/MILE_03_SP_3415_UTT_0038.wav
train/trans_files/MILE_03_SP_3415_UTT_0038.txt
test/audio_files/MILE_03_SP_3386_UTT_0066.wav
test/trans_files/MILE_03_SP_3386_UTT_0066.txt
```

Matching the page: "each folder contains two subfolders named audio_files and
trans_files", with ".txt files in UTF-8 Unicode text corresponding to each audio
file". The transcript is the whole file, one line, whitespace collapsed — for
example `ಇವುಗಳಿಗಾಗಿ ಏಳುನೂರು ಕೋಟಿ ರುಗೆ ಒಪ್ಪಿಗೆ ನೀಡಲಾಗಿದೆ ಎಂದು ಸಚಿವರು ತಿಳಿಸಿದರು`.

**Speaker ids are present**, which the corpus record had as unconfirmed: the file
stem is `MILE_03_SP_<speaker>_UTT_<n>`, so everything before `_UTT_` identifies
the speaker. Confirmed both against the archive members and against ESPnet's
`egs2/kn_openslr126/asr1/local/data_prep.py`, which derives it the same way:

```python
def speaker_of(utt):
    # MILE_03_SP_2496_UTT_0116 -> MILE_03_SP_2496
    return utt.split("_UTT_")[0]
```

## Split policy: **re-derived, speaker-disjoint 80/10/10**

Train and test ship; **there is no dev split**. The loader takes a partition
whole or derives one whole — a manifest that is part shipped and part derived
could not be written down in a results file — so the preparer clears `split` on
every row and the shared derivation produces all three.

Nothing is discarded: both shipped archives enter the same pool, and the
per-archive counts are recorded in `fetch_manifest.json` as `shipped_counts`.
**The published train/test division is not the one a run uses**, and a write-up
has to say the split was re-derived rather than describe the corpus's own.

## Caveats

- ESPnet's recipe for this corpus guards against an extra wrapping directory
  around `audio_files`, so the preparer searches the tree rather than assuming a
  fixed depth. The archives as published do not need that, but a re-packaging
  might.
- A stem that does not carry `_UTT_` gets no speaker and is counted under
  `no_speaker` rather than given an invented one. Any such row drops the whole
  language to an utterance-level split, so that count is worth reading.
- Read speech recorded "in a noise-free recording environment with high quality
  USB microphones" — clean and prompted, not spontaneous.

## Citation

> A. Madhavaraj, Pilar Bharathi and G. Ramakrishnan A. IISc-MILE Kannada ASR
> Corpus, Indian Institute of Science, 2022. Two accompanying arXiv papers cover
> subword techniques and knowledge-driven grammar modelling for Kannada ASR;
> both are linked from the source page.
