"""What the normalizer did to one language's transcripts, as a run artifact.

A normalization policy is a set of claims about the corpus — that Common Voice
carries almost no digits, that the training vocabulary covers the test set, that
a language declared to use word boundaries actually writes them. This module
measures those claims per language and per split so they land in
``text_stats.json`` beside the results, instead of being taken on trust.

Two of the numbers here change how a result must be read. ``unk_rate`` is an
irreducible floor under a language's error rate: characters the vocabulary
lacks cannot be produced no matter how good the model is.
``n_utts_empty_after_norm`` counts utterances that cannot be a CTC target or a
scoreable reference, which is a silent change to the test set unless reported.
"""

from __future__ import annotations

import statistics
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .normalize import (
    DEFAULT_POLICY,
    NormalizerPolicy,
    count_arabic_marks,
    has_digits,
    normalize_text,
)

if TYPE_CHECKING:  # avoids a cycle: the vocab is built through this package
    from ..model.ctc_vocab import CtcVocab

# Below this, whitespace tokenization is not measuring words, so WER over
# ``str.split`` would score whole sentences as single tokens.
_UNSPACED_MEDIAN_TOKENS = 1.0


@dataclass
class TextStats:
    """Per-(language, split) effect of the normalization policy."""

    n_utts: int = 0
    n_utts_empty_after_norm: int = 0
    n_utts_with_digits: int = 0
    n_chars_raw: int = 0
    n_chars_normalized: int = 0
    # Disjoint by construction: a character has exactly one Unicode category,
    # and tatweel (Lm) is neither punctuation, symbol nor mark.
    removed_by_category: dict[str, int] = field(default_factory=dict)
    case_changed_chars: int = 0
    median_tokens_per_utt: float = 0.0
    unk_chars: int = 0
    unk_rate: float = 0.0
    # Filled in by the vocabulary build, which is the only place that knows
    # which characters the frequency floor evicted.
    evicted_by_floor: dict[str, int] = field(default_factory=dict)

    @property
    def looks_unspaced(self) -> bool:
        """Whether the transcripts read as a script written without spaces.

        Checked against each language's declared ``word_boundary`` so a preset
        that mislabels a language is caught by the data rather than by a reader.
        """
        return self.n_utts > 0 and self.median_tokens_per_utt <= _UNSPACED_MEDIAN_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_text_stats(
    texts: Iterable[str],
    policy: NormalizerPolicy = DEFAULT_POLICY,
    vocab: CtcVocab | None = None,
) -> TextStats:
    """Measure one split's transcripts under ``policy``.

    Args:
        texts: Raw corpus transcripts, before normalization.
        policy: The policy the run is using.
        vocab: The vocabulary targets will be encoded against. Supplying it adds
            the unknown-character rate; without it that rate is reported as zero
            rather than guessed.

    Returns:
        A :class:`TextStats`. ``evicted_by_floor`` is left empty for the caller
        to fill from the vocabulary build.
    """
    stats = TextStats(removed_by_category={"P": 0, "S": 0, "Cf": 0, "Cc": 0, "arabic_marks": 0})
    token_counts: list[int] = []

    for raw in texts:
        stats.n_utts += 1
        stats.n_chars_raw += len(raw)
        if has_digits(raw):
            stats.n_utts_with_digits += 1

        # Count removals against the Unicode-normalized form, which is what the
        # policy's later steps actually see.
        formed = unicodedata.normalize(policy.form, raw)
        for ch in formed:
            category = unicodedata.category(ch)
            group = category[0]
            if group == "P":
                stats.removed_by_category["P"] += 1
            elif group == "S":
                stats.removed_by_category["S"] += 1
            elif category in ("Cf", "Cc"):
                stats.removed_by_category[category] += 1
            if ch.casefold() != ch:
                stats.case_changed_chars += 1
        stats.removed_by_category["arabic_marks"] += count_arabic_marks(formed)

        normalized = normalize_text(raw, policy)
        stats.n_chars_normalized += len(normalized)
        if not normalized:
            stats.n_utts_empty_after_norm += 1
            continue

        token_counts.append(len(normalized.split()))
        if vocab is not None:
            stats.unk_chars += sum(1 for ch in normalized if ch not in vocab.char_to_id)

    if token_counts:
        stats.median_tokens_per_utt = float(statistics.median(token_counts))
    if stats.n_chars_normalized:
        stats.unk_rate = stats.unk_chars / stats.n_chars_normalized
    return stats
