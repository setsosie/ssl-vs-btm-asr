"""Across-seed aggregation for one ``(arm, scale)``.

Reports word **and** character error rate for every language, plus whichever of
the two is that language's primary metric. Reporting only one of them was the
old behaviour and it misreported every unspaced language: whitespace
tokenization of a Japanese transcript yields one token per sentence, so its word
error rate is not a measurement.

Which metric is primary is read from the run's own ``text_stats.json`` rather
than from today's presets. The choice is a property of the run that produced the
number, and a preset edited afterwards must not retroactively change what an
existing result means.

The macro-average across languages carries an explicit kind label. Averaging one
language's WER with another's CER produces a number that is neither, and the
only safe version of that number is one that says so wherever it appears.

The macro is aggregated the same way a single language is: computed per seed
first, then averaged across seeds. Its ``±`` is therefore the same quantity as
every per-language row's — run-to-run spread. Taking the standard deviation of
the per-language means instead would put the gap *between* languages after a
``±`` and invite a reader to take it for training noise; two languages thirty
points apart, each moving two points between seeds, would report a spread of
twenty-one where the truth is one and a half. That dispersion is still worth
having, so it is kept under its own name and never after a ``±``.

A language missing from some seed is left out of the macro and named. A mean
whose membership changes between seeds is not comparable seed to seed, and the
change would otherwise be invisible.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..stats.analysis import SeedAgg, aggregate_seeds
from .analyze import language_policies
from .tables import fmt_mean_std, fmt_value, render_table

SECTIONS = ("in_distribution", "transfer")
_METRIC_LABEL = {"wer": "WER", "cer": "CER"}


@dataclass(frozen=True)
class RunRecord:
    """One seed's results plus the metric choice that run recorded."""

    seed: int
    path: Path
    results: dict[str, Any]
    word_boundary: dict[str, bool]
    #: Which normalization policy each language ran under. Empty for a run
    #: written before that was recorded.
    policies: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LanguageAgg:
    code: str
    section: str
    primary_kind: str  # "wer" or "cer"
    wer: SeedAgg
    cer: SeedAgg
    #: This language's primary metric per seed, keyed by seed. The macro needs
    #: the values seed by seed, which a mean and a standard deviation cannot
    #: give back.
    primary_by_seed: dict[int, float] = field(default_factory=dict)

    @property
    def primary(self) -> SeedAgg:
        return self.wer if self.primary_kind == "wer" else self.cer


@dataclass(frozen=True)
class MacroAgg:
    """Macro-average of per-language primaries, with what it is made of.

    ``mean`` and ``std`` are computed over the *per-seed* macro values held in
    ``per_seed``, so ``std`` is run-to-run spread — the same quantity as every
    per-language row's. ``spread_across_languages`` is the dispersion between
    the languages themselves; it is a different thing and is never printed as
    an error bar.
    """

    mean: float
    std: float
    n_languages: int
    kind: str  # "wer", "cer", "mixed", or "empty"
    composition: dict[str, int]
    #: The macro value for each seed, in seed order.
    per_seed: list[float] = field(default_factory=list)
    #: Sample standard deviation of the per-language means. Between-language
    #: dispersion, not uncertainty about the macro.
    spread_across_languages: float = float("nan")
    #: Languages left out because they are absent from at least one seed.
    excluded_languages: list[str] = field(default_factory=list)

    @property
    def n_seeds(self) -> int:
        return len(self.per_seed)

    @property
    def is_mixed(self) -> bool:
        return self.kind == "mixed"

    @property
    def label(self) -> str:
        """How this number must be named. Never just "primary" when mixed."""
        if self.kind in _METRIC_LABEL:
            return _METRIC_LABEL[self.kind]
        if self.kind == "mixed":
            parts = ", ".join(
                f"{_METRIC_LABEL[k]}×{self.composition[k]}"
                for k in ("wer", "cer")
                if self.composition.get(k)
            )
            return f"mixed ({parts})"
        return "—"


@dataclass(frozen=True)
class ScaleAggregate:
    arm: str
    scale: str
    seeds: list[int]
    languages: list[LanguageAgg]
    macro: dict[str, MacroAgg]
    #: The policy each language ran under, for the table's per-language column
    #: and for the macro's caption.
    policies: dict[str, str] = field(default_factory=dict)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


def _seed_of(path: Path) -> int:
    return int(path.name.removeprefix("seed"))


def load_runs(root: Path, arm: str, scale: str) -> list[RunRecord]:
    """Every completed seed of one ``(arm, scale)``, ordered numerically.

    A ``seed*`` directory with no ``results.json`` is a run that did not finish;
    it is skipped rather than counted as a seed with no languages.

    Raises:
        FileNotFoundError: When nothing under ``root`` matches, so an empty
            table cannot be mistaken for a set of runs that all scored zero.
    """
    base = root / arm / scale
    candidates = sorted(
        (p for p in base.glob("seed*") if (p / "results.json").exists()),
        # Numeric: "seed10" sorts before "seed2" as a string.
        key=_seed_of,
    )
    if not candidates:
        raise FileNotFoundError(f"no runs with a results.json under {base}")
    return [
        RunRecord(
            seed=_seed_of(p),
            path=p,
            results=json.loads((p / "results.json").read_text(encoding="utf-8")),
            word_boundary=_word_boundary(p),
            policies=language_policies(p),
        )
        for p in candidates
    ]


def _word_boundary(run_dir: Path) -> dict[str, bool]:
    """Per-language ``word_boundary`` as recorded by the run itself.

    Absent for runs written before ``text_stats.json`` existed; those languages
    fall back to WER, which is what those runs reported at the time.
    """
    path = run_dir / "text_stats.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        code: bool(entry.get("word_boundary", True))
        for code, entry in payload.get("languages", {}).items()
    }


def aggregate_runs(runs: list[RunRecord]) -> ScaleAggregate:
    """Mean ± std of WER, CER and the primary metric, per language and section."""
    collected: dict[tuple[str, str], dict[str, dict[int, float]]] = {}
    kinds: dict[tuple[str, str], str] = {}

    for run in runs:
        for section in SECTIONS:
            for code, entry in run.results.get(section, {}).items():
                key = (section, code)
                bucket = collected.setdefault(key, {"wer": {}, "cer": {}})
                for metric in ("wer", "cer"):
                    value = entry.get(metric)
                    if value is not None:
                        bucket[metric][run.seed] = float(value)
                # First run to mention the language fixes its metric; later
                # seeds of the same experiment agree by construction.
                kinds.setdefault(key, "wer" if run.word_boundary.get(code, True) else "cer")

    seeds = [run.seed for run in runs]
    languages = [
        LanguageAgg(
            code=code,
            section=section,
            primary_kind=kinds[(section, code)],
            wer=aggregate_seeds(_in_seed_order(values["wer"], seeds)),
            cer=aggregate_seeds(_in_seed_order(values["cer"], seeds)),
            primary_by_seed=dict(values[kinds[(section, code)]]),
        )
        for (section, code), values in sorted(
            collected.items(), key=lambda kv: (kv[0][0], kv[0][1])
        )
    ]

    macro = {
        section: _macro([row for row in languages if row.section == section], seeds)
        for section in SECTIONS
    }
    return ScaleAggregate(
        arm=runs[0].results.get("arm", ""),
        scale=runs[0].results.get("scale", ""),
        seeds=seeds,
        languages=languages,
        macro=macro,
        # From the first seed: every seed of one arm and scale ran the same
        # preset, and a seed that disagreed would already have been rejected by
        # the reference check before any table was drawn.
        policies=dict(runs[0].policies),
    )


def _in_seed_order(by_seed: dict[int, float], seeds: list[int]) -> list[float]:
    """The values a language has, ordered by seed rather than by read order."""
    return [by_seed[seed] for seed in seeds if seed in by_seed]


def _macro(rows: list[LanguageAgg], seeds: list[int]) -> MacroAgg:
    """Macro-average of the per-language primaries, computed seed by seed.

    Unweighted by utterance count on purpose: the question is how a system does
    across languages, and a corpus-weighted mean is dominated by whichever
    language happened to ship the most audio.

    Averaging within a seed and only then across seeds is what makes the
    reported ``±`` run-to-run spread rather than the gap between the languages.
    Languages absent from any seed are excluded and named: a macro whose
    membership changes between seeds is not comparable seed to seed.
    """
    complete = [row for row in rows if all(seed in row.primary_by_seed for seed in seeds)]
    excluded = sorted(row.code for row in rows if row not in complete)
    if not complete:
        return MacroAgg(
            mean=float("nan"),
            std=float("nan"),
            n_languages=0,
            kind="empty",
            composition={},
            excluded_languages=excluded,
        )

    composition: dict[str, int] = {}
    for row in complete:
        composition[row.primary_kind] = composition.get(row.primary_kind, 0) + 1
    kind = next(iter(composition)) if len(composition) == 1 else "mixed"

    per_seed = [
        sum(row.primary_by_seed[seed] for row in complete) / len(complete) for seed in seeds
    ]
    across_seeds = aggregate_seeds(per_seed)
    across_languages = aggregate_seeds([row.primary.mean for row in complete])
    return MacroAgg(
        mean=across_seeds.mean,
        std=across_seeds.std,
        n_languages=len(complete),
        kind=kind,
        composition=composition,
        per_seed=per_seed,
        spread_across_languages=across_languages.std,
        excluded_languages=excluded,
    )


def to_json(agg: ScaleAggregate) -> dict[str, Any]:
    """Machine-readable form. ``is_mixed`` is explicit so a consumer never has
    to parse the human-facing label to discover it."""
    return {
        "arm": agg.arm,
        "scale": agg.scale,
        "seeds": agg.seeds,
        "languages": [
            {
                "code": row.code,
                "section": row.section,
                "primary_kind": row.primary_kind,
                "wer": {"mean": row.wer.mean, "std": row.wer.std, "n_seeds": row.wer.n_seeds},
                "cer": {"mean": row.cer.mean, "std": row.cer.std, "n_seeds": row.cer.n_seeds},
            }
            for row in agg.languages
        ],
        "macro": {
            section: {
                "mean": macro.mean,
                "std": macro.std,
                # Named, so a consumer can never mistake this for the spread
                # between the languages — which is the next field along.
                "std_is": "across_seeds",
                "per_seed": macro.per_seed,
                "n_seeds": macro.n_seeds,
                "spread_across_languages": macro.spread_across_languages,
                "excluded_languages": macro.excluded_languages,
                "n_languages": macro.n_languages,
                "kind": macro.kind,
                "is_mixed": macro.is_mixed,
                "composition": macro.composition,
                "label": macro.label,
            }
            for section, macro in agg.macro.items()
        },
    }


def to_markdown(agg: ScaleAggregate) -> str:
    """The table a write-up embeds, with the mixed-metric warning inline."""
    lines = [
        f"# {agg.arm} — scale {agg.scale}",
        "",
        f"{agg.n_seeds} seed(s): {', '.join(str(s) for s in agg.seeds)}. "
        "Mean ± sample standard deviation across seeds, in percent.",
        "",
    ]
    for section in SECTIONS:
        rows = [row for row in agg.languages if row.section == section]
        if not rows:
            continue
        macro = agg.macro[section]
        lines += [f"## {section.replace('_', ' ')}", ""]
        lines.append(
            render_table(
                ["language", "WER", "CER", "primary", "metric", "policy", "seeds"],
                [
                    [
                        row.code,
                        fmt_mean_std(row.wer.mean, row.wer.std),
                        fmt_mean_std(row.cer.mean, row.cer.std),
                        fmt_mean_std(row.primary.mean, row.primary.std),
                        _METRIC_LABEL[row.primary_kind],
                        agg.policies.get(row.code, "—"),
                        str(row.wer.n_seeds),
                    ]
                    for row in rows
                ],
                aligns=["left", "right", "right", "right", "left", "left", "right"],
            )
        )
        lines += [
            "",
            f"**Macro-average of primaries ({macro.label}):** "
            f"{fmt_mean_std(macro.mean, macro.std)} over {macro.n_languages} language(s), "
            f"unweighted by utterance count, across {macro.n_seeds} seed(s). The spread is "
            "across seeds, computed by averaging the languages within each seed first.",
        ]
        if macro.n_languages > 1:
            lines += [
                "",
                "Spread *across languages* (how far apart the languages are, not an error "
                f"bar on the number above): {fmt_value(macro.spread_across_languages)}.",
            ]
        if macro.excluded_languages:
            lines += [
                "",
                "Left out of the macro because they are absent from at least one seed, so "
                "including them would change what the average is an average of: "
                f"{', '.join(macro.excluded_languages)}.",
            ]
        caption = macro_policy_caption(agg.policies, [row.code for row in rows])
        if caption:
            lines += [
                "",
                f"The languages in this average were normalized under different rules: "
                f"{caption}. Averaging across them is how a multilingual system is "
                "summarised at all, but the result is not one quantity, and it is not "
                "comparable with a number produced under a single policy.",
            ]
        if macro.is_mixed:
            lines += [
                "",
                "> This macro-average pools metrics of two kinds and is therefore neither a "
                "word nor a character error rate. Quote it only with the label above, or "
                "quote the per-language column instead.",
            ]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def macro_policy_caption(policies: dict[str, str], languages: Iterable[str]) -> str:
    """A note for a macro-average whose languages were normalized differently.

    Averaging error rates computed under different text rules is a real thing to
    do — it is the only way to say anything about a multilingual system at all —
    but the number is not one quantity, and a reader who does not know that will
    compare it against one that is.
    """
    used = sorted({policies[code] for code in languages if code in policies})
    if len(used) <= 1:
        return ""
    return f"policies differ by language ({', '.join(used)})"
