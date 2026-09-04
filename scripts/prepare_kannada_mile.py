"""Prepare IISc-MILE Kannada into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: slr126_kannada
Languages: kn
Source: https://www.openslr.org/126/

Format notes, from the source page:

- Two archives: `mile_kannada_train.tar.gz` (22 G) and
  `mile_kannada_test.tar.gz` (6.1 G). 16 kHz 16-bit mono PCM WAV, ~350 h, 915
  speakers.
- Each archive holds `audio_files/` and `trans_files/`; the transcript is one
  UTF-8 `.txt` per `.wav`, matched by stem.
- Train and test ship; **there is no dev split**. The loader refuses a manifest
  that is part shipped and part derived, so this preparer clears the column for
  every row and lets the shared speaker-disjoint derivation produce all three.
  The published train/test division is therefore not the one a run uses, and any
  write-up has to say the split was re-derived.
- Speaker ids are unconfirmed. If the file stems do not carry a speaker the
  derivation degrades to utterance level and the run records that, which is not
  a speaker-independent result.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("slr126_kannada",)
LANGUAGES = ("kn",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "IISc-MILE Kannada: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
