# Armenian Speech Crowdsourcing Data — `slr160_armenian`

| | |
|---|---|
| Language | Armenian (`hy`) |
| Source page | <https://www.openslr.org/160/> |
| Archive | `armenian_speech_crowdsourcing_data.tar.gz` (5.6 G) |
| Preparer | `scripts/prepare_armenian.py` |
| Size | 70 h |
| Audio | WAV; sample rate not stated on the page, read from the files |

## Licence

**CC BY 4.0**, verbatim from the source page and from `resources/160/info.txt`.
<https://creativecommons.org/licenses/by/4.0/>

The text was contributed by Yerevan City Magazine, which "agreed to share
portions of its published text archive for open use in language technology
research and development"; the recordings were collected through the Toloka
crowdsourcing platform.

## Download

```bash
python scripts/prepare_armenian.py --root $CORPORA_ROOT
```

openslr.org publishes no checksum for this resource; integrity is the byte
length against `Content-Length` plus a full read of every tar member, with the
SHA-256 recorded for later comparison.

## Verified layout

Read off the archive's own members:

```
armenian_speech_crowdsourcing_data/
    pitched.jsonl
    pitched/eu.<uuid>.wav
```

`pitched.jsonl` is one JSON object per line carrying **exactly three keys**, read
off the real file:

```json
{"text": "Լավ էլ ապրում են իրանց համար։", "audio_filepath": "/data/pitched/eu.6a770d68-46a2-4288-95d5-fe7bff083609.wav", "duration": 5.3333125}
```

`audio_filepath` is an absolute path from the machine that built the corpus, so
only its file name is used and the audio is resolved under the extracted tree.
`duration` is summed into `indexed_hours` in `fetch_manifest.json`.

## Split policy: **derived, utterance level — not speaker-independent**

**There is no speaker field, and there is not meant to be one.** The source page
states that "to protect the privacy of contributors and prevent potential
misuse, the voices themselves were anonymized so that they could not be easily
identified or matched to individual speakers". `pitched.jsonl` bears that out: it
carries text, audio path and duration and nothing else.

So `speaker` is empty on every row, the shared derivation in `svb.data.splits`
falls back to an utterance-level 80/10/10, and the loader reports
`split_policy == "utterance"`. This corrects the corpus record, which had
`speaker_ids: unconfirmed`: they are absent.

**A result on Armenian is therefore not speaker-independent.** The same
contributor can appear in train and test, so the number is optimistic relative to
the speaker-disjoint numbers the other corpora here produce. That belongs in any
write-up that reports it, and the fetch record carries the reason so it travels
with the data.

## Caveats

- Not one of the Google crowdsourced OpenSLR sets, despite also being
  crowdsourced read speech on OpenSLR. Those five ship sixteen
  `asr_<language>_*.zip` shards and a three-column `utt_spk_text.tsv` whose
  middle column is the speaker — the one field this corpus does not have — so
  there is nothing shared with `scripts/prepare_google_crowdsourced.py`.
- Read speech, segmented into 3–15 second utterances and verified for reading
  accuracy before inclusion.

## Citation

> Nikolay Karpov, Sofia Kostandian, Nune Tadevosyan, Alexan Ayrapetyan, Andrei
> Andrusenko, Ara Yeroyan, Mher Yerznkanyan and Vitaly Lavrukhin. "From Scarcity
> to Sufficiency: Speech Recognition Pipeline for Zero-resource Language."
> *Interspeech 2025*, pp. 4303–4307.
> <https://doi.org/10.21437/Interspeech.2025-950>
