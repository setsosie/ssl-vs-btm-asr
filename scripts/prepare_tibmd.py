"""Prepare TIBMD@MUC (Tibetan) into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: slr124_tibetan
Languages: bo
Source: https://www.openslr.org/124/

Format notes, from the source page:

- One archive, `Tibetan_speech_data.tgz` (29 G), 16 kHz 16-bit audio.
- **The transcript format is not documented on the source page.** Inspect the
  tarball before committing to a reader; that inspection is the first task here.
- Three dialects (Ü-Tsang 52.53 h, Kham 5.87 h, Amdo 25.93 h), 87 named
  speakers, 10,390 unique sentences, 84.33 h total.
- No split ships. Speaker ids are unconfirmed: if they are recoverable, leave
  `split` empty and let the shared speaker-disjoint derivation run; if they are
  not, the split degrades to utterance level and the run must say so.
- Tibetan is written without word boundaries, so the preset entry carries
  `word_boundary: false` and character error rate is the primary metric.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("slr124_tibetan",)
LANGUAGES = ("bo",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "TIBMD@MUC: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
