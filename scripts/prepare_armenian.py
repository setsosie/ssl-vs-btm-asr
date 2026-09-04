"""Prepare Armenian Speech Crowdsourcing Data into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus id: slr160_armenian
Languages: hy
Source: https://www.openslr.org/160/

Format notes, from the source page:

- One archive, `armenian_speech_crowdsourcing_data.tar.gz` (5.6 G), holding a
  `pitched/` audio directory and `pitched.jsonl`.
- `pitched.jsonl` is one JSON object per line carrying the transcript and
  metadata. WAV; sample rate is not stated, so resample from whatever the files
  report rather than assuming 16 kHz.
- 70 h, crowdsourced read speech (Toloka, Yerevan City Magazine text). No split
  ships. **Whether the JSONL carries a speaker field is unconfirmed** — check it
  first; without one the split is utterance-level and not speaker-independent.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("slr160_armenian",)
LANGUAGES = ("hy",)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "Armenian crowdsourcing data: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
