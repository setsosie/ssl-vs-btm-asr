"""Honest-number machinery: across-seed aggregation, bootstrap CIs, and paired
permutation tests.

Two distinct variance sources are reported, never conflated:

  - **seed variance** — std of the per-seed WER, i.e. training stochasticity.
  - **bootstrap CI** — utterance-level resampling of one run's test set, i.e.
    test-set sampling uncertainty.

Paired permutation compares two systems on the *same* utterances.

A claim that one *arm* beats another is a claim about both at once: it has to
survive retraining and it has to survive a different draw of test utterances.
``multibootstrap`` is the statistic for that claim. It resamples the seeds of
each arm and the utterances of each language in the same draw, so its interval
carries both sources, which neither of the two above does alone.

Tokenization is a parameter, not a constant. Whitespace tokenization of a
script written without spaces gives one token per sentence, so a word-level
interval for Japanese would be an interval on a number that can only be 0 or
100 per utterance. Both tokenizations reproduce jiwer's default transforms
step for step, so a bootstrap's point estimate equals the reported metric
exactly — including on text that ends in a space, which ``whisper-basic``
leaves behind and jiwer strips before counting characters.

Both statistics work from per-utterance ``(edits, reference length)`` computed
once. Corpus error rate is a ratio of sums, so a resample is two array sums
rather than a re-alignment — which is what makes ten thousand draws over a real
test set finish.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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


_RUNS_OF_WHITESPACE = re.compile(r"\s\s+")


def _tokens(text: str, tokenize: Tokenization) -> list[str]:
    """Exactly the tokens ``jiwer.wer`` and ``jiwer.cer`` count, by construction.

    Their default transforms: words are what is left after collapsing runs of
    whitespace to one space, stripping the ends and splitting on the plain
    space character; characters are the stripped string, spaces included. A
    plain ``str.split()`` would also split on a non-breaking space that jiwer
    keeps inside a word, and ``list(text)`` would count a trailing space that
    jiwer strips — and every ``whisper-basic`` reference ends in one.
    """
    if tokenize == "word":
        return [w for w in _RUNS_OF_WHITESPACE.sub(" ", text).strip().split(" ") if w]
    return list(text.strip())


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


def edit_counts(
    refs: list[str], hyps: list[str], tokenize: Tokenization = "word"
) -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance ``(edit distance, reference length)`` as two arrays.

    What every statistic here is computed from. Public because a comparison
    across many runs wants to align each run once and resample the counts.
    """
    return _edit_counts(refs, hyps, tokenize)


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


@dataclass(frozen=True)
class LanguageEdits:
    """One language's per-utterance edit counts, for every run of two arms.

    Row ``i`` of ``edits_a`` is run ``i`` of arm A, and must be the same run in
    every language handed to :func:`multibootstrap` together: a seed is one
    training run covering all of its languages, so resampling seeds means
    resampling whole runs.
    """

    lengths: np.ndarray  # (n_utts,) reference lengths, shared by every run
    edits_a: np.ndarray  # (n_runs_a, n_utts)
    edits_b: np.ndarray  # (n_runs_b, n_utts)


@dataclass(frozen=True)
class Contrast:
    """Arm A minus arm B, in percentage points. Negative means A is better."""

    delta: float
    lo: float
    hi: float
    #: Two-sided: twice the smaller share of bootstrap draws on either side of
    #: zero. A reading of the interval, not an independent test.
    p_value: float


def _mean_rates(edits: np.ndarray, lengths: np.ndarray) -> float:
    """Mean over runs of the corpus error rate (%)."""
    return float((100.0 * edits.sum(axis=1) / max(1.0, float(lengths.sum()))).mean())


def _contrast(point: float, draws: np.ndarray, ci: float) -> Contrast:
    n = draws.size
    below = (int((draws <= 0).sum()) + 1) / (n + 1)
    above = (int((draws >= 0).sum()) + 1) / (n + 1)
    return Contrast(
        delta=point,
        lo=float(np.percentile(draws, 100 * (1 - ci) / 2)),
        hi=float(np.percentile(draws, 100 * (1 + ci) / 2)),
        p_value=min(1.0, 2.0 * min(below, above)),
    )


def multibootstrap(
    languages: Sequence[LanguageEdits],
    n: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[list[Contrast], Contrast]:
    """Arm A minus arm B per language, and macro-averaged, with joint intervals.

    Each draw resamples, with replacement, the runs of arm A, the runs of arm B,
    and the test utterances of every language, then recomputes each arm's mean
    error rate. This is the Multi-Bootstrap of Sellam et al. (2022, "The
    MultiBERTs") in its unpaired-seeds, paired-examples form: the two arms
    share a test set, so one utterance draw serves both, but seed 3 of one arm
    has nothing to do with seed 3 of the other, so runs are drawn separately.

    Within a draw the run sample is shared across languages and the utterance
    sample is not. A run is one model evaluated on every language, so its
    languages move together; the test sets of two languages are different
    utterances and do not.

    With few runs the interval is approximate and tends to be too narrow —
    resampling five runs cannot represent more spread than those five show. It
    is still the honest version of the comparison, because the alternatives
    each leave one of the two sources of variation out entirely.

    Returns:
        One contrast per language, in the order given, and the contrast of the
        unweighted macro-average across them.

    Raises:
        ValueError: If no language is given, or the languages do not all carry
            the same number of runs for each arm.
    """
    if not languages:
        raise ValueError("no languages to compare")
    runs_a = {lang.edits_a.shape[0] for lang in languages}
    runs_b = {lang.edits_b.shape[0] for lang in languages}
    if len(runs_a) != 1 or len(runs_b) != 1:
        raise ValueError(
            "every language must carry the same runs of each arm; resampling seeds means "
            "resampling whole runs, and a run missing from one language cannot be drawn"
        )
    k_a, k_b = runs_a.pop(), runs_b.pop()
    empty = [i for i, lang in enumerate(languages) if lang.lengths.shape[0] == 0]
    if empty:
        raise ValueError(
            f"language(s) at position(s) {empty} have no utterances; a sidecar with no "
            "pairs is not a scored language"
        )

    points = [
        _mean_rates(lang.edits_a, lang.lengths) - _mean_rates(lang.edits_b, lang.lengths)
        for lang in languages
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty((len(languages), n), dtype=float)
    widest = max(lang.lengths.shape[0] for lang in languages)
    done = 0
    while done < n:
        size = _chunk_size(widest, n - done)
        pick_a = rng.integers(0, k_a, size=(size, k_a))
        pick_b = rng.integers(0, k_b, size=(size, k_b))
        for row, lang in enumerate(languages):
            m = lang.lengths.shape[0]
            # One utterance draw per language, used for both arms and every
            # run. Drawn as indices and counted, which is the same multinomial
            # as ``rng.multinomial`` draws one category at a time; the count
            # matrix then turns every run's resampled total into one matrix
            # product instead of a gather per run.
            idx = rng.integers(0, m, size=(size, m))
            # One bincount over every draw at once, each draw offset into its
            # own range of bins, rather than one call per draw.
            offsets = (np.arange(size) * m)[:, None]
            flat = np.bincount((idx + offsets).ravel(), minlength=size * m)
            counts = flat.reshape(size, m).astype(float)
            denominator = np.maximum(1.0, counts @ lang.lengths)[:, None]
            rates_a = 100.0 * (counts @ lang.edits_a.T) / denominator  # (size, k_a)
            rates_b = 100.0 * (counts @ lang.edits_b.T) / denominator
            mean_a = np.take_along_axis(rates_a, pick_a, axis=1).mean(axis=1)
            mean_b = np.take_along_axis(rates_b, pick_b, axis=1).mean(axis=1)
            draws[row, done : done + size] = mean_a - mean_b
        done += size

    per_language = [_contrast(points[row], draws[row], ci) for row in range(len(languages))]
    macro = _contrast(float(np.mean(points)), draws.mean(axis=0), ci)
    return per_language, macro
