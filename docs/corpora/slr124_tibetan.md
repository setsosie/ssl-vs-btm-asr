# TIBMD@MUC — `slr124_tibetan`

| | |
|---|---|
| Language | Tibetan (`bo`) |
| Source page | <https://www.openslr.org/124/> |
| Archive | `Tibetan_speech_data.tgz` (29 G) |
| Preparer | `scripts/prepare_tibmd.py` |
| Size | 84.33 h, 87 native speakers, 10,390 unique sentences |
| Audio | WAV, "sampled at 16KHz, 16bit quantization accuracy" |

Three dialects: Ü-Tsang 52.53 h, Kham 5.87 h, Amdo 25.93 h. The corpus contains
3,006 unique Tibetan characters.

## Licence

**CC BY-SA 4.0**, verbatim from the source page and from `resources/124/info.txt`.
<https://creativecommons.org/licenses/by-sa/4.0/>

ShareAlike, unlike the other five corpora in this group. ShareAlike over a
training corpus is not settled law with respect to model weights, and a paper
should take a position rather than say nothing.

## Download

```bash
python scripts/prepare_tibmd.py --root $CORPORA_ROOT
```

openslr.org publishes no checksum for this resource; integrity is the byte
length against `Content-Length` plus a full read of every tar member, with the
SHA-256 recorded for later comparison.

## Verified layout

Read off the archive's own members:

```
Tibetan mutil-dialect data/
    Amdo Dialect/
        Pastoral areas dialect/
            Aba pastoral dialect/
                Speaker 65/
                    Reading/
                        wav/阿坝牧区_crzm_A523.wav
```

`Tibetan mutil-dialect data` is the archive's own spelling. Paths carry spaces
and non-Latin characters throughout.

**Speaker ids are present**, which the corpus record had as unconfirmed: every
recording sits under a `Speaker <N>` directory, with the dialect a `... Dialect`
component nearer the top and a task level (`Reading`) between them.

Two traps in matching those, both real:

- The archive root itself contains the word "dialect", so a substring match takes
  the root for the dialect on every path in the corpus. The match is anchored at
  the end of a path component instead.
- The sub-area directories under a dialect (`Pastoral areas dialect`, `Aba
  pastoral dialect`) also end in the word, so the *first* matching component
  wins.
- The depth between the dialect and the speaker differs across the three dialect
  trees, so both are located by name rather than by position.

The speaker key is prefixed with its dialect (`Amdo_Dialect-speaker_65`). The
corpus is not stated to number its 87 speakers globally, and merging two
speakers would put one voice in train and test.

## Split policy: **derived, speaker-disjoint 80/10/10**

The corpus ships no partition, so `split` is empty on every row and the shared
derivation in `svb.data.splits` produces all three, disjoint by speaker.

The **dialect travels in an extra `dialect` manifest column**. The loader reads
the manifest with `csv.DictReader` and only requires its five columns to be
present, so an extra one stays with the data rather than being pushed into a side
file, and per-dialect breakdowns stay possible downstream.

## Caveats

- **The transcript convention is not verified.** The source page documents none,
  and probing the archive did not settle it: 444 MB of decompressed prefix — 561
  consecutive wav files under a single speaker — contained no non-audio member at
  all, so it is certainly *not* one `.txt` beside each `.wav`. Rather than guess,
  the preparer discovers it after extraction, accepting either

  - one transcript file per utterance, matched by stem, anywhere in the tree; or
  - an index file whose lines start with an utterance stem.

  If neither yields a match it refuses and reports the non-audio files it did
  find, with their suffixes and examples, so one run establishes the real
  convention instead of a guess failing quietly. Whichever fired is recorded in
  `fetch_manifest.json` as `transcript_convention`. **Expect to read that report
  on the first real run**, and extend `discover_transcripts` if the archive uses
  a third shape.
- **Tibetan is written without word boundaries**, so the preset entry carries
  `word_boundary: false` and character error rate is the primary metric.
- The dialects are very unevenly sized (Kham is 5.87 h against Ü-Tsang's 52.53),
  so an aggregate Tibetan number is dominated by Ü-Tsang.
- Recording conditions vary: indoors, using voice recorders, laptops and mobile
  phones.

## Citation

> Yue Zhao et al. "An open speech resource for Tibetan multi-dialect and
> multitask recognition." *International Journal of Computational Science and
> Engineering*, 22(2/3):297–304, 2020. TIBMD@MUC, Minzu University of China.
