"""Prepare the NCHLT South African corpora into the manifest layer.

**Not implemented.** This is the contract and the format notes; the body is on a
sibling branch. See `docs/data.md` for what a preparer owes the loader and
`configs/corpora.yaml` for this corpus's licence, download and hours record.

Corpus ids: nchlt_zulu, nchlt_xhosa, nchlt_sepedi, nchlt_xitsonga, nchlt_tshivenda
Languages: zu, xh, nso, ts, ve
Source: https://repo.sadilar.org/

Format notes, from the source page:

- One zip per language, served by SADiLaR's DSpace 7 REST API. The resources
  page mentions accepting terms of use, but the bitstream endpoint answers an
  unauthenticated client (verified: HTTP 206, `application/zip`, PK magic).
- Resolve a handle to a download URL through
  `/server/api/core/items/{uuid}/bundles` -> the `ORIGINAL` bundle ->
  `/server/api/core/bundles/{uuid}/bitstreams` -> `_links.content.href`.
- `dc.rights.license` and `dc.description` come from the same API, so **assert
  the licence and the corpus size at fetch time** rather than trusting
  configs/corpora.yaml.
- 16 kHz WAV with per-utterance XML transcripts. Read prompts throughout, so the
  domain is narrow, and the published split's test side is an 8-speaker suite.
- The three-way split ships and is speaker-disjoint; use it as shipped.
"""

from __future__ import annotations

import sys

CORPUS_IDS = ("nchlt_zulu", "nchlt_xhosa", "nchlt_sepedi", "nchlt_xitsonga", "nchlt_tshivenda")
LANGUAGES = ("zu", "xh", "nso", "ts", "ve")


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "NCHLT: preparer not written yet. It must fetch the archives listed in "
        "configs/corpora.yaml, then write $CORPORA_ROOT/<corpus>/<lang>/manifest.tsv "
        "with columns utt_id, path, text, speaker, split (see docs/data.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
