"""Prepare TalTech Estonian Speech Dataset 1.0 into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: taltech_estonian
Languages: et
Source: https://cs.taltech.ee/staff/tanel.alumae/data/est-pub-asr-data/

Format notes, from the source page:

- ~150 G tar, no registration. 1334 h train / 21 h dev / 23 h test, so the
  split ships and is used as shipped.
- Transcripts are STM and VTT, converted by the publisher from Transcriber TRS.
  Long-form spontaneous speech, so segmentation matters: the STM segment
  boundaries are the utterances.
- **The audio format is not stated on the source page.** Confirm it before
  ingest rather than assuming; the entry in configs/corpora.yaml lists it as
  unverified for that reason.
- Transcription is non-professional and the publisher documents known errors.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("taltech_estonian",)
LANGUAGES = ("et",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "TalTech Estonian: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
