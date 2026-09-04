"""Prepare ParlaSpeech-HR v1.0 (Croatian) into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: parlaspeech_hr
Languages: hr
Source: https://www.clarin.si/repository/xmlui/handle/11356/1494

Format notes, from the source page:

- 116 G of FLAC in three parts, publicly available with no login.
- Transcripts are JSONL with word-level alignments; the segment is the
  utterance. 1816 h total, 309 speakers.
- The split ships, but the repository publishes dev and test as **segment
  counts** (500 over 5 speakers, 513 over 6) rather than hours, so this preparer
  should measure and record the realised hours instead of copying a figure.
- Parliamentary speech. The same reader shape covers ParlaSpeech-RS if Serbian
  is ever added, though its licence tag differs between CLARIN and the HF
  mirror and would need resolving first.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("parlaspeech_hr",)
LANGUAGES = ("hr",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "ParlaSpeech-HR: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
