"""Prepare Kazakh Speech Corpus (KSC) into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: slr102_ksc_kazakh
Languages: kk
Source: https://www.openslr.org/102/

Format notes, from the source page:

- One archive, `ISSAI_KSC_335RS_v1.1_flac.tar.gz` (19 G). FLAC; sample rate not
  stated on the page.
- ~332 h over 153,000+ utterances. The page states neither the transcript
  convention nor whether a split ships — both are the first thing to establish.
- Speaker count is not published. If the metadata carries no speaker, say so in
  the manifest by leaving `speaker` empty; the derivation then falls back to
  utterance level rather than inventing disjointness.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("slr102_ksc_kazakh",)
LANGUAGES = ("kk",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "KSC Kazakh: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
