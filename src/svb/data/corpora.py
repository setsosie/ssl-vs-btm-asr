"""The corpora that are not Common Voice, and what is known about each.

``configs/corpora.yaml`` is the licence and provenance record for every corpus
the presets draw on besides Common Voice and the held-out OpenSLR four. It is
read here into typed specs so the preparers, the language selection and the
tests all see the same fields, rather than three YAML readers drifting apart.

Nothing in this module downloads or reads audio. It answers "what is this
corpus, under what licence, in what shape" — the questions a reader of the paper
would ask and a preparer needs answered before it fetches anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

#: Languages held out for the transfer experiment (``configs/scales/heldout.yaml``)
#: plus Odia, the pending fifth. None may appear as training data under any
#: corpus: several corpora here come from the same crowdsourced family as the
#: held-out four, so this is a live hazard rather than a formality.
HELD_OUT_LANGUAGES = frozenset({"ml", "mr", "te", "gu", "or"})

#: What a corpus ships by way of a train/dev/test partition.
#:
#: ``full``      all three splits, and the preparer writes them into the manifest
#: ``partial``   some splits ship; the rest are derived (Kannada has no dev,
#:               Zeroth's test is 1.2 h and under the 2 h bar)
#: ``none``      nothing ships; the whole partition is derived at load time
SPLIT_STATES = ("full", "partial", "none")

#: Where a speaker id comes from, which decides whether a derived split can be
#: speaker-disjoint at all.
#:
#: ``column``         an explicit speaker column or field in the corpus metadata
#: ``file_id``        recoverable from the utterance id, as in the OpenSLR four
#: ``path``           recoverable from the directory the audio sits in
#: ``per-recording``  present, but scoped to one recording rather than to a
#:                    person: the same speaker in two recordings gets two ids, so
#:                    a split cannot be claimed speaker-disjoint across them
#: ``unconfirmed``    the source page does not say; the preparer must check and
#:                    the split degrades to utterance level if it is not there
#: ``absent``         known not to be recoverable
SPEAKER_ID_STATES = ("column", "file_id", "path", "per-recording", "unconfirmed", "absent")


@dataclass(frozen=True)
class CorpusSpec:
    """One downloadable corpus resource.

    Usually one language: the corpora that cover several publish one resource
    per language anyway (each NCHLT language is its own DSpace item with its own
    licence field), so ``languages`` is a list without often holding more than
    one.
    """

    id: str
    name: str
    languages: list[str]
    source_page: str
    licence_name: str  # verbatim, as the source page words it
    licence_url: str
    downloads: list[dict[str, Any]]
    audio_format: str
    transcripts: str
    speaker_ids: str
    ships_split: str
    preparer: str  # repo-relative path
    citation: str
    #: None where the source page does not state it. Recorded as null rather
    #: than guessed: an unstated sample rate is a thing to check at ingest, not
    #: a thing to assume is 16 kHz.
    sample_rate_hz: int | None = None
    total_hours: float | None = None
    train_hours: float | None = None
    dev_hours: float | None = None
    test_hours: float | None = None
    speakers: str = ""
    notes: str = ""
    #: ``file name -> {algorithm, value}`` for the corpora that publish one.
    #: Three do — NCHLT per bitstream from the DSpace API, ParlaSpeech per file
    #: from its METS record, and Zeroth at ``resources/40/checksum.md5``. The
    #: fetcher compares against these rather than digesting for later comparison,
    #: which is all it can do for the rest.
    checksums: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Fields whose value the source page does not publish, so a reader can tell
    #: "not stated" from "not yet filled in".
    unverified: list[str] = field(default_factory=list)

    @property
    def trainable_hours(self) -> float | None:
        """Hours available to train on, on the same footing as the Common Voice
        figure: the training split where one is published, else the total less
        whatever dev and test are known to take."""
        if self.train_hours is not None:
            return self.train_hours
        if self.total_hours is None:
            return None
        return max(0.0, self.total_hours - (self.dev_hours or 0.0) - (self.test_hours or 0.0))


def _spec_from_entry(entry: dict[str, Any]) -> CorpusSpec:
    known = {f.name for f in fields(CorpusSpec)}
    unknown = sorted(set(entry) - known)
    if unknown:
        raise ValueError(f"corpus {entry.get('id', '?')!r} has unknown keys {unknown}")
    try:
        return CorpusSpec(**entry)
    except TypeError as exc:  # a required key is missing
        raise ValueError(f"corpus {entry.get('id', '?')!r}: {exc}") from exc


def load_corpora(configs_dir: Path) -> list[CorpusSpec]:
    """Every corpus in ``<configs_dir>/corpora.yaml``."""
    path = Path(configs_dir) / "corpora.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing corpus registry: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [_spec_from_entry(entry) for entry in raw.get("corpora", [])]


def corpus_by_language(corpora: list[CorpusSpec]) -> dict[str, CorpusSpec]:
    """``language code -> corpus``, for the selection and the presets."""
    return {code: corpus for corpus in corpora for code in corpus.languages}
