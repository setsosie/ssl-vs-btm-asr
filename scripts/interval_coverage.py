"""How often the `svb compare` interval covers a true difference of zero.

The contrast interval resamples seeds and test utterances together. Resampling
five seeds cannot show more spread than those five do, so at the seed counts
this study can afford the interval is narrower than its nominal level. The
docs and the table caption quote how much narrower; this script is where that
number comes from, so it can be regenerated rather than trusted.

Two arms are simulated with **no true difference**. Each run's error rate is
the base rate perturbed by a per-run effect (the seed), shared across languages
as a real run's is, and each utterance's edit count is a binomial draw around
it. Every replicate calls the real ``multibootstrap`` and records whether its
95% interval contains zero. Coverage is the fraction that do; ``1 - coverage``
is the false-positive rate of ``p < 0.05``.

    uv run python scripts/interval_coverage.py                 # the table in the docs
    uv run python scripts/interval_coverage.py --reps 50       # quicker, noisier

The regimes are stylised — one base rate, one sentence length, no speaker
clustering — and are meant to bracket the study, not to model it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from svb.stats.analysis import LanguageEdits, multibootstrap


@dataclass(frozen=True)
class Regime:
    label: str
    seeds: int
    languages: int
    utterances: int
    #: Relative standard deviation of a run's error rate around the base rate.
    seed_sd: float


REGIMES = (
    Regime("1 language, utterance noise dominant", 5, 1, 2000, 0.03),
    Regime("1 language, seed noise dominant", 5, 1, 300, 0.15),
    Regime("4-language macro, seed noise dominant", 3, 4, 300, 0.15),
    Regime("4-language macro, seed noise dominant", 5, 4, 300, 0.15),
    Regime("4-language macro, seed noise dominant", 10, 4, 300, 0.15),
    Regime("4-language macro, utterance noise dominant", 5, 4, 2000, 0.03),
)

BASE_RATE = 0.25
WORDS_PER_UTTERANCE = 10


def _arm(rng: np.random.Generator, regime: Regime, lengths: list[np.ndarray]) -> list[np.ndarray]:
    """Edit counts for one arm: ``regime.seeds`` runs over every language."""
    # One draw per run, applied to all of its languages: a seed is one model.
    rates = BASE_RATE * (1.0 + regime.seed_sd * rng.standard_normal(regime.seeds))
    rates = np.clip(rates, 0.01, 0.99)
    return [
        np.stack([rng.binomial(length.astype(int), rate).astype(float) for rate in rates])
        for length in lengths
    ]


def coverage(regime: Regime, reps: int, draws: int, seed: int) -> tuple[float, float]:
    """(per-language coverage, macro coverage) at true delta zero."""
    rng = np.random.default_rng(seed)
    language_hits = 0
    macro_hits = 0
    for rep in range(reps):
        lengths = [
            np.full(regime.utterances, float(WORDS_PER_UTTERANCE)) for _ in range(regime.languages)
        ]
        edits_a = _arm(rng, regime, lengths)
        edits_b = _arm(rng, regime, lengths)
        per_language, macro = multibootstrap(
            [
                LanguageEdits(lengths=length, edits_a=a, edits_b=b)
                for length, a, b in zip(lengths, edits_a, edits_b, strict=True)
            ],
            n=draws,
            seed=rep,
        )
        language_hits += sum(row.lo <= 0.0 <= row.hi for row in per_language)
        macro_hits += macro.lo <= 0.0 <= macro.hi
    return language_hits / (reps * regime.languages), macro_hits / reps


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--reps", type=int, default=300, help="replicates per regime")
    parser.add_argument("--draws", type=int, default=1000, help="bootstrap draws per replicate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print("| regime | seeds/arm | utts/lang | seed SD | per-language | macro |")
    print("|---|---:|---:|---:|---:|---:|")
    for regime in REGIMES:
        per_language, macro = coverage(regime, args.reps, args.draws, args.seed)
        print(
            f"| {regime.label} | {regime.seeds} | {regime.utterances} | "
            f"{regime.seed_sd:.0%} | {per_language:.2f} | {macro:.2f} |"
        )
        sys.stdout.flush()
    print()
    print(
        f"Coverage of the nominal 95% interval at a true difference of zero, "
        f"{args.reps} replicates of {args.draws} draws each."
    )


if __name__ == "__main__":
    main()
