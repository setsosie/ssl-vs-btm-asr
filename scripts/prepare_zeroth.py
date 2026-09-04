"""Prepare Zeroth-Korean into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: slr40_zeroth_korean
Languages: ko
Source: https://www.openslr.org/40/

Format notes, from the source page:

- One archive, `zeroth_korean.tar.gz` (10 G). Audio format and sample rate are
  not stated on the page; the corpus is Kaldi-shaped, so expect FLAC plus
  `text`/`utt2spk`-style metadata. Confirm before writing the reader.
- 51.6 h train over 105 speakers, 1.2 h test over 10 speakers, speaker-disjoint.
- **The shipped test is under the 2 h evaluation bar.** So this preparer writes
  an empty `split` for every row and lets the shared speaker-disjoint derivation
  repartition all 115 speakers, which puts dev and test both over the bar. The
  published 51.6/1.2 h figures then no longer describe the splits in use, and
  any write-up has to say the split was re-derived.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("slr40_zeroth_korean",)
LANGUAGES = ("ko",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "Zeroth-Korean: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
