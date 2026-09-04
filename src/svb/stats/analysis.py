"""Honest-number machinery: across-seed aggregation, bootstrap CIs, and paired
permutation tests.

Two distinct variance sources are reported, never conflated:

  - **seed variance** — std of the per-seed WER, i.e. training stochasticity.
  - **bootstrap CI** — utterance-level resampling of one run's test set, i.e.
    test-set sampling uncertainty.

Paired permutation compares two systems on the *same* utterances.

Tokenization is a parameter, not a constant. Whitespace tokenization of a
script written without spaces gives one token per sentence, so a word-level
interval for Japanese would be an interval on a number that can only be 0 or
100 per utterance. Characters are tokenized including spaces, matching
``jiwer.cer``, so a bootstrap's point estimate equals the reported metric
exactly.

Both statistics work from per-utterance ``(edits, reference length)`` computed
once. Corpus error rate is a ratio of sums, so a resample is two array sums
rather than a re-alignment — which is what makes ten thousand draws over a real
test set finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Tokenization = Literal["word", "char"]

# Resampling draws an (chunk, n_utterances) index matrix. The chunk is sized to
# keep that under roughly 32 MB regardless of test-set size.
_MAX_INDEX_CELLS = 4_000_000


@dataclass
class SeedAgg:
    mean: float
    std: float
    n_seeds: int
    per_seed: list[float]


def aggregate_seeds(per_seed_wer: list[float]) -> SeedAgg:
    """Mean ± std across seeds. ``ddof=1`` (sample std); std is nan for n<2."""
    arr = np.asarray(per_seed_wer, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return SeedAgg(mean=float(arr.mean()), std=std, n_seeds=arr.size, per_seed=list(arr))


def _tokens(text: str, tokenize: Tokenization) -> list[str]:
    """Word or character tokens. Characters include spaces, as ``jiwer.cer`` does."""
    return text.split() if tokenize == "word" else list(text)


def _edit_counts(
    refs: list[str], hyps: list[str], tokenize: Tokenization
) -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance (edit distance, reference length), computed once.

    Levenshtein distance over token lists is exactly substitutions + deletions +
    insertions, so this is the numerator of the corpus error rate without a
    round-trip back through a string interface.
    """
    from rapidfuzz.distance import Levenshtein

    edits = np.empty(len(refs), dtype=float)
    lengths = np.empty(len(refs), dtype=float)
    for i, (r, h) in enumerate(zip(refs, hyps, strict=True)):
        ref_tokens = _tokens(r, tokenize)
        edits[i] = Levenshtein.distance(ref_tokens, _tokens(h, tokenize))
        lengths[i] = len(ref_tokens)
    return edits, lengths


def corpus_error_rate(refs: list[str], hyps: list[str], tokenize: Tokenization = "word") -> float:
    """Corpus WER (``word``) or CER (``char``) as a percentage.

    Equal to ``100 * jiwer.wer`` and ``100 * jiwer.cer`` respectively, so the
    statistics layer and the evaluator cannot report different numbers for the
    same predictions.
    """
    edits, lengths = _edit_counts(refs, hyps, tokenize)
    return 100.0 * float(edits.sum()) / max(1.0, float(lengths.sum()))


@dataclass
class CIResult:
    point: float
    lo: float
    hi: float


def _chunk_size(n_utts: int, remaining: int) -> int:
    return max(1, min(remaining, _MAX_INDEX_CELLS // max(1, n_utts)))


def bootstrap_ci(
    refs: list[str],
    hyps: list[str],
    n: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
    tokenize: Tokenization = "word",
) -> CIResult:
    """Utterance-level bootstrap percentile CI for the corpus error rate (%)."""
    edits, lengths = _edit_counts(refs, hyps, tokenize)
    point = 100.0 * float(edits.sum()) / max(1.0, float(lengths.sum()))
    m = len(refs)
    if m == 0:
        return CIResult(point=point, lo=point, hi=point)

    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=float)
    done = 0
    while done < n:
        size = _chunk_size(m, n - done)
        idx = rng.integers(0, m, size=(size, m))
        drawn_edits = edits[idx].sum(axis=1)
        drawn_lengths = np.maximum(1.0, lengths[idx].sum(axis=1))
        samples[done : done + size] = 100.0 * drawn_edits / drawn_lengths
        done += size

    lo = float(np.percentile(samples, 100 * (1 - ci) / 2))
    hi = float(np.percentile(samples, 100 * (1 + ci) / 2))
    return CIResult(point=point, lo=lo, hi=hi)


def paired_permutation(
    refs: list[str],
    hyps_a: list[str],
    hyps_b: list[str],
    n: int = 10_000,
    seed: int = 42,
    tokenize: Tokenization = "word",
) -> float:
    """One-sided paired permutation p-value that system A has the lower error rate.

    The statistic is the summed per-utterance difference in edit counts, with
    signs flipped per utterance under the null. That is exactly a permutation
    test on the difference in *corpus* error rate: both systems share the
    reference-length denominator, and that denominator is invariant under the
    sign flips, so the ratio's ordering is the sum's ordering.
    """
    edits_a, _ = _edit_counts(refs, hyps_a, tokenize)
    edits_b, _ = _edit_counts(refs, hyps_b, tokenize)
    diff = edits_a - edits_b  # negative => A better
    observed = float(diff.sum())
    m = diff.shape[0]
    if m == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    while done < n:
        size = _chunk_size(m, n - done)
        signs = rng.integers(0, 2, size=(size, m)) * 2.0 - 1.0
        count += int(((signs * diff).sum(axis=1) <= observed).sum())
        done += size
    return (count + 1) / (n + 1)
