"""Prepare Samrómur 21.05 (Icelandic) into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: slr112_samromur
Languages: is
Source: https://www.openslr.org/112/

Format notes, from the source page:

- One archive, `samromur_21.05.tgz` (7.0 G). FLAC, 16 kHz, 16-bit linear PCM,
  mono. 145 h, 100,000 utterances, 8,392 speakers.
- **The split ships and the page states it has no speaker overlap**, so this
  preparer fills `split` from the corpus metadata and nothing is derived. It is
  the cleanest split of any corpus here; do not replace it with a derivation.
- The metadata file carries the read prompt plus per-utterance and per-speaker
  fields; the prompt is the transcript.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("slr112_samromur",)
LANGUAGES = ("is",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "Samrómur: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
