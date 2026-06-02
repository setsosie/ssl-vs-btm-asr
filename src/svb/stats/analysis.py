"""Honest-number machinery: across-seed aggregation, bootstrap CIs, and paired
permutation tests.

Two distinct variance sources are reported, never conflated:

  - **seed variance** — std of the per-seed WER, i.e. training stochasticity.
  - **bootstrap CI** — utterance-level resampling of one run's test set, i.e.
    test-set sampling uncertainty.

Paired permutation compares two systems on the *same* utterances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _corpus_wer(ref_words: list[list[str]], hyp_words: list[list[str]]) -> float:
    """Corpus WER from per-utterance (edits, ref_len); uses jiwer per utterance."""
    import jiwer

    total_edits = 0
    total_len = 0
    for r, h in zip(ref_words, hyp_words, strict=True):
        m = jiwer.process_words(" ".join(r), " ".join(h))
        total_edits += m.substitutions + m.deletions + m.insertions
        total_len += len(r)
    return 100.0 * total_edits / max(1, total_len)


@dataclass
class CIResult:
    point: float
    lo: float
    hi: float


def bootstrap_ci(
    refs: list[str],
    hyps: list[str],
    n: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> CIResult:
    """Utterance-level bootstrap percentile CI for corpus WER (%)."""
    rw = [r.split() for r in refs]
    hw = [h.split() for h in hyps]
    point = _corpus_wer(rw, hw)
    rng = np.random.default_rng(seed)
    m = len(rw)
    samples = np.empty(n, dtype=float)
    idx_all = np.arange(m)
    for b in range(n):
        idx = rng.choice(idx_all, size=m, replace=True)
        samples[b] = _corpus_wer([rw[i] for i in idx], [hw[i] for i in idx])
    lo = float(np.percentile(samples, 100 * (1 - ci) / 2))
    hi = float(np.percentile(samples, 100 * (1 + ci) / 2))
    return CIResult(point=point, lo=lo, hi=hi)


def paired_permutation(
    refs: list[str],
    hyps_a: list[str],
    hyps_b: list[str],
    n: int = 10_000,
    seed: int = 42,
) -> float:
    """One-sided paired permutation p-value that system A has lower WER than B.

    Test statistic is the difference in per-utterance edit counts; signs are
    flipped per utterance under the null.
    """
    import jiwer

    rw = [r.split() for r in refs]

    def edits(hyps: list[str]) -> np.ndarray:
        out = np.empty(len(rw), dtype=float)
        for i, (r, h) in enumerate(zip(rw, hyps, strict=True)):
            m = jiwer.process_words(" ".join(r), h)
            out[i] = m.substitutions + m.deletions + m.insertions
        return out

    diff = edits(hyps_a) - edits(hyps_b)  # negative => A better
    observed = diff.sum()
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        flip = rng.choice([1.0, -1.0], size=diff.shape[0])
        if (diff * flip).sum() <= observed:
            count += 1
    return (count + 1) / (n + 1)
