# Kazakh Speech Corpus (KSC) — `slr102_ksc_kazakh`

| | |
|---|---|
| Language | Kazakh (`kk`) |
| Source page | <https://www.openslr.org/102/> |
| Archive | `ISSAI_KSC_335RS_v1.1_flac.tar.gz` (19 G) |
| Preparer | `scripts/prepare_ksc.py` |
| Size | ~332 h, 153,000+ utterances |
| Audio | FLAC |

## Licence

**Attribution 4.0 International (CC BY 4.0)**, verbatim from the source page and
from `resources/102/info.txt`. <https://creativecommons.org/licenses/by/4.0/>

Attribution is required; there is no ShareAlike condition.

## Download

```bash
python scripts/prepare_ksc.py --root $CORPORA_ROOT
```

openslr.org publishes no checksum for this resource, so the archive is accepted
on its byte length against the server's `Content-Length` plus a full read of
every tar member, and its SHA-256 is recorded for later comparison rather than
checked against anything.

## Verified layout

```
ISSAI_KSC_335RS_v1.1_flac/
    Audios_flac/<uttID>.flac
    Transcriptions/<uttID>.txt
    Meta/{train,dev,test}.csv
```

`Audios_flac/` and the opaque `<uttID>` names were read directly off the
archive's first tar members. The rest comes from the corpus author's own ESPnet
recipe, `egs2/ksc/asr1/local/prepare_data.sh` (ISSAI, Yerbolat Khassanov — first
author of the KSC paper), which builds one data directory per Meta file:

```bash
for filename in $KSC/ISSAI_KSC_335RS_v1.1_flac/Meta/*; do
  dir=data/$(basename "$filename" .csv)
  ...
      echo "$uttID $(cat $KSC/ISSAI_KSC_335RS_v1.1_flac/Transcriptions/${uttID}.txt)" >> ${dir}/text
      echo "$uttID $KSC/ISSAI_KSC_335RS_v1.1_flac/Audios_flac/${uttID}.flac" >> ${dir}/wav.scp
      echo "$uttID $uttID" >> ${dir}/utt2spk
```

## Split policy: **shipped, passed through**

`Meta/train.csv`, `Meta/dev.csv` and `Meta/test.csv` are the partition, and the
manifest fills `split` from whichever Meta file an utterance is listed in. `dev`
becomes the loader's `validation`. Nothing is derived.

This corrects the corpus record, which had `ships_split: none` — the source page
says nothing about a split, so the record was not wrong about the page, only
about the archive.

The Meta files are named `.csv` but the reference recipe reads them on
whitespace (`IFS=" " read -r uttID others`), so the preparer sniffs the delimiter
from the header rather than assuming one, and records the column names in
`fetch_manifest.json`.

## Caveats

- **No speaker id, so the split's speaker-disjointness is unknown.** The
  reference recipe writes `utt2spk` as `uttID uttID` — every utterance is its own
  speaker — which is what a corpus with no speaker mapping looks like. The
  preparer leaves `speaker` empty and fills it only from a Meta column literally
  named as a speaker id; inferring one from the shape of the values would claim a
  property the corpus never established. A write-up should say the split is the
  published one and that whether it is speaker-disjoint is not established.
- The sample rate is not stated on the page and is read from the files.
- KSD (SLR140, 554 h, CC BY-SA 3.0) is the larger alternative, at 22/44 kHz and
  so needing resampling. KSC is preferred here for being 4.0-licensed.

## Citation

> Yerbolat Khassanov, Saida Mussakhojayeva, Almas Mirzakhmetov, Alen Adiyev,
> Mukhamet Nurpeiissov and Huseyin Atakan Varol. "A Crowdsourced Open-Source
> Kazakh Speech Corpus and Initial Speech Recognition Baseline." *Proceedings of
> the 16th Conference of the European Chapter of the Association for
> Computational Linguistics*, 2021, pp. 697–706.
> <https://aclanthology.org/2021.eacl-main.58>
