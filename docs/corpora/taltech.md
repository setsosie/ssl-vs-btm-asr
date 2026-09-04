# TalTech Estonian Speech Dataset 1.0

One language, `et`, from Tallinn University of Technology, fetched by
[`scripts/prepare_taltech.py`](../../scripts/prepare_taltech.py).

```bash
export CORPORA_ROOT=/path/to/corpora
python scripts/prepare_taltech.py --root $CORPORA_ROOT
```

This is the most expensive preparer in the repository, and not because of the
download. The audio is long form — recordings run half an hour and more — while
the manifest layer addresses whole files with no offset or duration column. So
the preparer **cuts every recording at its transcript's segment bounds** and
writes one WAV per utterance.

## What is downloaded

One tar, no registration, no API and no checksum:

| File | Bytes | URL |
|---|---|---|
| `taltech-asr-speech-dataset-1.0.tar` | 159,267,522,560 | `https://cs.taltech.ee/staff/tanel.alumae/data/est-pub-asr-data/taltech-asr-speech-dataset-1.0.tar` |

The server sends `Accept-Ranges: bytes` and a `Content-Length`, so the download
resumes and its length is checked. Nothing else can be checked: the publisher
posts no checksum, so `fetch_manifest.json` says
`"tar read end to end; the publisher publishes no checksum"` rather than
reporting a digest as a verification. The SHA-256 is recorded for comparing one
fetch against another, which is all it is good for here.

**Disk.** About 159 GB of tar, about the same again extracted, and roughly
130 GB of cut segments. Without `--keep-archives` the tar is deleted as soon as
extraction finishes and each long-form recording as soon as its segments are
written, so the steady state is close to the segments alone — but the peak is
still the tar and the extracted tree at the same time. Budget for it.

## Licence

Verbatim, from the source page:

> **Licence**
>
> CC BY-SA 4.0 DEED: https://creativecommons.org/licenses/by-sa/4.0/
>
> Copyright of the audio material in the dataset belongs to corresponding
> parties.

That second sentence sits directly under the ShareAlike tag and is worth reading
before redistributing anything derived from this corpus. There is no API here and
no checksum, so the page is the only record of the terms — and it is the same
host the 159 GB comes from. The preparer fetches it first and refuses to
continue if it no longer says `CC BY-SA 4.0`.

## Layout

Walked over HTTP Range rather than downloaded: the tar's header chain was
followed from byte 0 and sampled through to the end. The shape is

```
<split>/transcriptions/stm/<subset>/<recording>.stm
<split>/transcriptions/vtt/<subset>/<recording>.vtt
<split>/transcriptions/trs/<subset>/<recording>.trs
<split>/audio/<subset>/<recording>.wav
```

with `<split>` one of `train`, `dev`, `test`, all three of which are present, and
`<subset>` one of the collections the page names: ERR2020, aktuaalne2021,
intervjuukorpus, paevakaja, MKK, aktuaalne-kaamera, er-uudised, jutusaated,
konverentsid, veebiseminarid, Riigikogu_salvestused, podcastid. Real members,
quoted from the walk:

```
train/transcriptions/vtt/aktuaalne2021/ringvaade-2056-323841.vtt
train/transcriptions/stm/er-uudised/20051206-1430ER_uudised_2005_12_06_14_00.stm
train/audio/aktuaalne2021/aktuaalne-kaamera-ilm-nadal-322855.wav     74,092,622 bytes
dev/transcriptions/stm/paevakaja/paevakaja-2015-03-09.stm
test/audio/er-uudised/20051227-1130ER_uudised_2005_12_27_11_00.wav
```

Some names need GNU `@LongLink` entries, so a few path components are long enough
to be worth guarding against; the preparer shortens any component over 200 bytes
and appends a digest so two long names cannot collide.

**Only the STM tree is read.** The three transcript formats carry the same
content — the page says the Transcriber `trs` files "are pre-converted to STM and
VTT formats" — and STM is the one that puts the speaker and the segment bounds on
a single line. VTT and TRS are left in the archive rather than written to disk.
A few members under `.../transcriptions/stm/` carry a `.vtt` extension, so the
selection is by suffix as well as by directory; parsing WEBVTT as STM would
produce silence rather than an error.

**Audio format.** 16 kHz, 16-bit, mono PCM WAV, decoded from the RIFF header of a
sampled member:

```
RIFF....WAVEfmt \x10\x00\x00\x00  \x01\x00 \x01\x00  \x80>\x00\x00  \x00}\x00\x00  \x02\x00 \x10\x00
                fmt size 16       PCM      1 chan     16000 Hz       32000 B/s     align 2, 16 bits
```

The page states no format at all, which is why `configs/corpora.yaml` lists
`audio_format`, `sample_rate_hz` and `speaker_ids` under `unverified`. All three
are now confirmed. Nothing is resampled here and the rate found is recorded.

## The STM format

From a real file in the archive, `train/transcriptions/stm/er-uudised/`:

```
;; Exported from .../er-uudised/20051206-1430ER_uudised....trs using local/trs2stm.py
20051206-1430ER... 1 inter_segment_gap 0.0 5.409 <o,f0,>
20051206-1430ER... 1 20051206-1430ER..._Vallo_Kelmsaar 5.409 9.175 <o,f0,> Kell on kaks, te kuulete Eesti Raadio uudiseid, stuudios on Vallo Kelmsaar.
20051206-1430ER... 1 20051206-1430ER..._Urmas_Paet 36.768 46.937 <o,f0,> Vene minister nimetas veel kord seda, et nad näeksid meeleldi, et tuleks alustada siis ikkagi uusi kõnelusi, uusi läbirääkimisi, noh meie, me ei näe selleks põhjust.
```

Six whitespace-separated fields then the transcript: recording id, channel,
speaker, start and end in seconds, and an angle-bracketed label. Rows are dropped
when they carry no transcript — that is `inter_segment_gap`, the silence between
segments — and when the transcript is the NIST marker
`IGNORE_TIME_SEGMENT_IN_SCORING`.

`utt_id` is `<subset>_<recording>_<start_ms>-<end_ms>`. The subset has to be in
it: recording names repeat across subsets in the real archive —
`71_ID117_352639` appears under both ERR2020 and intervjuukorpus — and `utt_id`
must be unique within a language.

## Split policy

All three splits ship and are used exactly as shipped; nothing is derived. The
split of a row is its top-level directory, with `dev` renamed to `validation`
because that is what the manifest layer calls it. The page's own figures are
1334 hours train, 21 dev, 23 test; the preparer measures the realised hours from
the segment bounds and records them beside the data, because what reaches the
manifest is the transcribed segments rather than the whole audio.

**The shipped split cannot be claimed to be speaker-disjoint.** Speaker ids are
*per recording*: the recording id is a prefix of every speaker id in its own STM,
so the same person in two recordings is two speaker ids. Nothing in the release
identifies a person across recordings. That does not affect this manifest, whose
split comes from the corpus rather than from the speaker column, but any claim
that train and test share no speaker would be unsupported.

## Caveats

- **Non-professional transcription with documented errors.** The page says "Most
  of the material has been transcribed by non-professional transcribers and there
  is a significant amount of errors, that shouldn't however exceed 5% of the
  words." A WER on this corpus has that floor built into it.
- **Not all of the training audio is speech.** The page: "Training split contains
  1334 hours of audio in total, but not all of it is speech (there are also
  segments containing music, etc)." Dropping the `inter_segment_gap` rows removes
  the labelled silence but not music inside a transcribed segment.
- **Long-form, spontaneous, broadcast-heavy.** 1066 of the hours are broadcast,
  237 conference and lecture, 31 parliament. Segmentation is the publisher's,
  through its own transcript boundaries, not a VAD of ours.
- **Segments that run past the end of their recording are dropped** and counted
  in `fetch_manifest.json` as `segments_past_end_of_recording`. An empty WAV would
  fail the loader mid-run instead of failing here.
- **The cut is lossless but not free.** Frames are copied verbatim at the source
  rate, width and channel count. The result is roughly 600,000 small WAV files;
  a filesystem with a small inode budget will notice.

## Citation

> Tanel Alumäe, Joonas Kalda, Külliki Bode, and Martin Kaitsa. 2023. Automatic
> Closed Captioning for Estonian Live Broadcasts. In Proceedings of the 24th
> Nordic Conference on Computational Linguistics (NoDaLiDa), pages 492–499,
> Tórshavn, Faroe Islands. University of Tartu Library.
> https://aclanthology.org/2023.nodalida-1.49

Laboratory of Language Technology, Tallinn University of Technology, funded by
the national programme for Estonian language technology.
