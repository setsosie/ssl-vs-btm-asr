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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stats.analysis import SeedAgg, aggregate_seeds
from .tables import fmt_mean_std, render_table

SECTIONS = ("in_distribution", "transfer")
_METRIC_LABEL = {"wer": "WER", "cer": "CER"}


@dataclass(frozen=True)
class RunRecord:
    """One seed's results plus the metric choice that run recorded."""

    seed: int
    path: Path
    results: dict[str, Any]
    word_boundary: dict[str, bool]


@dataclass(frozen=True)
class LanguageAgg:
    code: str
    section: str
    primary_kind: str  # "wer" or "cer"
    wer: SeedAgg
    cer: SeedAgg

    @property
    def primary(self) -> SeedAgg:
        return self.wer if self.primary_kind == "wer" else self.cer


@dataclass(frozen=True)
class MacroAgg:
    """Macro-average of per-language primaries, with what it is made of."""

    mean: float
    std: float
    n_languages: int
    kind: str  # "wer", "cer", "mixed", or "empty"
    composition: dict[str, int]

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
    collected: dict[tuple[str, str], dict[str, list[float]]] = {}
    kinds: dict[tuple[str, str], str] = {}

    for run in runs:
        for section in SECTIONS:
            for code, entry in run.results.get(section, {}).items():
                key = (section, code)
                bucket = collected.setdefault(key, {"wer": [], "cer": []})
                for metric in ("wer", "cer"):
                    value = entry.get(metric)
                    if value is not None:
                        bucket[metric].append(float(value))
                # First run to mention the language fixes its metric; later
                # seeds of the same experiment agree by construction.
                kinds.setdefault(key, "wer" if run.word_boundary.get(code, True) else "cer")

    languages = [
        LanguageAgg(
            code=code,
            section=section,
            primary_kind=kinds[(section, code)],
            wer=aggregate_seeds(values["wer"]),
            cer=aggregate_seeds(values["cer"]),
        )
        for (section, code), values in sorted(
            collected.items(), key=lambda kv: (kv[0][0], kv[0][1])
        )
    ]

    macro = {
        section: _macro([row for row in languages if row.section == section])
        for section in SECTIONS
    }
    return ScaleAggregate(
        arm=runs[0].results.get("arm", ""),
        scale=runs[0].results.get("scale", ""),
        seeds=[run.seed for run in runs],
        languages=languages,
        macro=macro,
    )


def _macro(rows: list[LanguageAgg]) -> MacroAgg:
    """Macro-average of per-language primary means, labelled by what it mixes.

    Unweighted by utterance count on purpose: the question is how a system does
    across languages, and a corpus-weighted mean is dominated by whichever
    language happened to ship the most audio.
    """
    if not rows:
        return MacroAgg(
            mean=float("nan"), std=float("nan"), n_languages=0, kind="empty", composition={}
        )
    composition: dict[str, int] = {}
    for row in rows:
        composition[row.primary_kind] = composition.get(row.primary_kind, 0) + 1
    kind = next(iter(composition)) if len(composition) == 1 else "mixed"
    agg = aggregate_seeds([row.primary.mean for row in rows])
    return MacroAgg(
        mean=agg.mean, std=agg.std, n_languages=len(rows), kind=kind, composition=composition
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
                ["language", "WER", "CER", "primary", "metric", "seeds"],
                [
                    [
                        row.code,
                        fmt_mean_std(row.wer.mean, row.wer.std),
                        fmt_mean_std(row.cer.mean, row.cer.std),
                        fmt_mean_std(row.primary.mean, row.primary.std),
                        _METRIC_LABEL[row.primary_kind],
                        str(row.wer.n_seeds),
                    ]
                    for row in rows
                ],
                aligns=["left", "right", "right", "right", "left", "right"],
            )
        )
        lines += [
            "",
            f"**Macro-average of primaries ({macro.label}):** "
            f"{fmt_mean_std(macro.mean, macro.std)} over {macro.n_languages} language(s).",
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
