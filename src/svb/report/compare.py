"""The arms side by side: the table the study exists to produce.

``svb aggregate`` describes one arm and ``svb analyze`` one run. Neither answers
the question in the repository's name, which is a comparison *between* arms, and
a reader assembling that comparison by hand from three aggregate tables has no
interval to put on a difference.

This module writes ``<scale>_arms.{md,json}``: each arm's mean ± standard
deviation across seeds for every language, and for every pair of arms the
difference with an interval that accounts for both retraining and the test set
(:func:`svb.stats.analysis.multibootstrap`). Per-language numbers use that
language's primary metric, as recorded by the runs themselves.

What it refuses, it refuses for the reason ``svb analyze`` does. The arms must
have been scored on identical references: a difference between two error rates
computed against different strings is partly a difference in scoring. And a
language is only contrasted where every run of both arms evaluated it, because
a mean over seeds whose membership differs between the arms is not one
comparison.

The intervals are per contrast and unadjusted, and at the seed counts a study
can afford they are narrower than their nominal level: resampling five runs
cannot show more spread than those five do. ``scripts/interval_coverage.py``
measures it — at a true difference of zero, about 87–94% of nominal-95%
intervals cover zero with five seeds per arm, about 84–87% with three — and the
rendered table says so. Which contrast is the pre-specified one is the
protocol's job (``docs/small-study.md``), not this module's; every other row is
a breakdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ..stats.analysis import LanguageEdits, Tokenization, edit_counts, multibootstrap
from .aggregate import SECTIONS, RunRecord, ScaleAggregate, aggregate_runs, load_runs
from .analyze import Sidecar, load_sidecars
from .tables import MISSING, fmt_mean_std, fmt_value, render_table

_METRIC_LABEL = {"wer": "WER", "cer": "CER"}
_TOKENIZE: dict[str, Tokenization] = {"wer": "word", "cer": "char"}


@dataclass(frozen=True)
class ArmRuns:
    """Every finished seed of one arm, and its across-seed aggregate."""

    arm: str
    runs: list[RunRecord]
    aggregate: ScaleAggregate


@dataclass(frozen=True)
class ContrastRow:
    code: str  # a language code, or "macro"
    metric: str  # "wer", "cer", or the macro's label
    delta: float
    lo: float
    hi: float
    p_value: float
    n_utts: int | None  # None for the macro row

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


@dataclass(frozen=True)
class ArmContrast:
    """Arm ``a`` minus arm ``b`` over one section."""

    a: str
    b: str
    section: str
    rows: list[ContrastRow]
    macro: ContrastRow | None
    #: Languages that some run of either arm did not evaluate.
    excluded: list[str]


def load_arms(root: Path, scale: str, arms: list[str]) -> list[ArmRuns]:
    """The arms that have at least one finished run, in the order asked for.

    An arm with nothing on disk is left out instead of raising: the table is
    useful while a study is still running, and it says which arms it found.

    Raises:
        FileNotFoundError: When no arm has a finished run.
    """
    found: list[ArmRuns] = []
    for arm in arms:
        try:
            runs = load_runs(root, arm, scale)
        except FileNotFoundError:
            continue
        found.append(ArmRuns(arm=arm, runs=runs, aggregate=aggregate_runs(runs)))
    if not found:
        raise FileNotFoundError(
            f"no finished runs at scale {scale} under {root} for any of: {', '.join(arms)}"
        )
    return found


def _require_same_policies(arms: list[ArmRuns]) -> None:
    """Every arm must have normalized each shared language under the same rules."""
    first = arms[0]
    for other in arms[1:]:
        shared = set(first.aggregate.policies) & set(other.aggregate.policies)
        differing = sorted(
            code
            for code in shared
            if first.aggregate.policies[code] != other.aggregate.policies[code]
        )
        if differing:
            detail = "; ".join(
                f"{c}: {first.aggregate.policies[c]} vs {other.aggregate.policies[c]}"
                for c in differing
            )
            raise ValueError(
                f"{first.arm} and {other.arm} normalized these languages differently, so "
                f"their error rates are not comparable — {detail}"
            )


class _Edits:
    """Per-utterance edit counts for each run, aligned once and reused.

    Three pairs of arms share the same runs, and aligning a sixteen-thousand
    utterance test set is the expensive step; the resampling after it is array
    arithmetic.
    """

    def __init__(self) -> None:
        self._sidecars: dict[Path, dict[tuple[str, str], Sidecar]] = {}
        self._counts: dict[tuple[Path, str, str], tuple[np.ndarray, np.ndarray]] = {}

    def sidecars(self, run: Path) -> dict[tuple[str, str], Sidecar]:
        if run not in self._sidecars:
            self._sidecars[run] = {(s.section, s.code): s for s in load_sidecars(run)}
        return self._sidecars[run]

    def counts(self, run: Path, section: str, code: str) -> tuple[np.ndarray, np.ndarray]:
        key = (run, section, code)
        if key not in self._counts:
            sidecar = self.sidecars(run)[(section, code)]
            metric = "wer" if sidecar.is_primary_word else "cer"
            self._counts[key] = edit_counts(sidecar.refs, sidecar.hyps, _TOKENIZE[metric])
        return self._counts[key]


def _language_edits(
    edits: _Edits, a: ArmRuns, b: ArmRuns, section: str, code: str
) -> tuple[LanguageEdits, str]:
    """Stack one language's edit counts over every run of both arms."""
    runs = [run.path for run in (*a.runs, *b.runs)]
    reference = edits.sidecars(runs[0])[(section, code)]
    for run in runs[1:]:
        other = edits.sidecars(run)[(section, code)]
        if other.refs != reference.refs:
            raise ValueError(
                f"{code}: {run} and {runs[0]} were scored on different references, so their "
                "error rates cannot be contrasted. Every run in a comparison must share a "
                "test set and a normalization policy."
            )
        if other.is_primary_word != reference.is_primary_word:
            # Each run's counts are taken on its own primary metric; two runs
            # that disagree would put a word count beside a character count.
            raise ValueError(
                f"{code}: {run} and {runs[0]} record different primary metrics for this "
                "language, so their error rates are not the same quantity"
            )
    _, lengths = edits.counts(runs[0], section, code)
    return (
        LanguageEdits(
            lengths=lengths,
            edits_a=np.stack([edits.counts(run.path, section, code)[0] for run in a.runs]),
            edits_b=np.stack([edits.counts(run.path, section, code)[0] for run in b.runs]),
        ),
        "wer" if reference.is_primary_word else "cer",
    )


def contrast_arms(
    a: ArmRuns,
    b: ArmRuns,
    section: str,
    edits: _Edits | None = None,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> ArmContrast:
    """Arm ``a`` minus arm ``b`` for every language both evaluated in every run."""
    edits = edits or _Edits()
    runs = [run.path for run in (*a.runs, *b.runs)]
    per_run = [{code for sec, code in edits.sidecars(run) if sec == section} for run in runs]
    everywhere = sorted(set.intersection(*per_run))
    excluded = sorted(set.union(*per_run) - set(everywhere))
    if not everywhere:
        return ArmContrast(
            a=a.arm, b=b.arm, section=section, rows=[], macro=None, excluded=excluded
        )

    stacked = [_language_edits(edits, a, b, section, code) for code in everywhere]
    per_language, macro = multibootstrap(
        [language for language, _ in stacked], n=n_resamples, seed=seed
    )
    rows = [
        ContrastRow(
            code=code,
            metric=metric,
            delta=result.delta,
            lo=result.lo,
            hi=result.hi,
            p_value=result.p_value,
            n_utts=int(language.lengths.shape[0]),
        )
        for code, (language, metric), result in zip(everywhere, stacked, per_language, strict=True)
    ]
    kinds = sorted({row.metric for row in rows})
    macro_row = ContrastRow(
        code="macro",
        metric=_METRIC_LABEL[kinds[0]] if len(kinds) == 1 else "mixed",
        delta=macro.delta,
        lo=macro.lo,
        hi=macro.hi,
        p_value=macro.p_value,
        n_utts=None,
    )
    return ArmContrast(
        a=a.arm, b=b.arm, section=section, rows=rows, macro=macro_row, excluded=excluded
    )


def compare_arms(
    arms: list[ArmRuns], n_resamples: int = 10_000, seed: int = 42
) -> list[ArmContrast]:
    """Every pair of arms, in every section, sharing one alignment of each run."""
    _require_same_policies(arms)
    edits = _Edits()
    return [
        contrast_arms(a, b, section, edits, n_resamples=n_resamples, seed=seed)
        for a, b in combinations(arms, 2)
        for section in SECTIONS
    ]


def _cell(arm: ArmRuns, section: str, code: str) -> str:
    for row in arm.aggregate.languages:
        if (row.section, row.code) == (section, code):
            return fmt_mean_std(row.primary.mean, row.primary.std)
    return MISSING


def _metric_of(arms: list[ArmRuns], section: str, code: str) -> str:
    for arm in arms:
        for row in arm.aggregate.languages:
            if (row.section, row.code) == (section, code):
                return _METRIC_LABEL[row.primary_kind]
    return MISSING


def _fmt_p(p_value: float, n_resamples: int) -> str:
    """The smallest p the draws can produce is a floor, not a measurement."""
    floor = 2.0 / (n_resamples + 1)
    return f"<{floor:.2g}" if p_value <= floor else f"{p_value:.4f}"


def _contrast_cells(row: ContrastRow, n_resamples: int) -> list[str]:
    return [
        row.code,
        _METRIC_LABEL.get(row.metric, row.metric),
        fmt_value(row.delta),
        f"[{fmt_value(row.lo)}, {fmt_value(row.hi)}]",
        _fmt_p(row.p_value, n_resamples),
        MISSING if row.n_utts is None else str(row.n_utts),
    ]


def to_markdown(scale: str, arms: list[ArmRuns], contrasts: list[ArmContrast], n: int) -> str:
    lines = [
        f"# scale {scale} — arms compared",
        "",
        "Mean ± sample standard deviation across seeds, in percent, on each language's "
        "primary metric.",
        "",
        *(
            f"- **{arm.arm}**: {arm.aggregate.n_seeds} seed(s) — "
            f"{', '.join(str(s) for s in arm.aggregate.seeds)}"
            for arm in arms
        ),
        "",
    ]
    for section in SECTIONS:
        codes = sorted(
            {row.code for arm in arms for row in arm.aggregate.languages if row.section == section}
        )
        if not codes:
            continue
        lines += [f"## {section.replace('_', ' ')}", ""]
        lines.append(
            render_table(
                ["language", "metric", *(arm.arm for arm in arms)],
                [
                    *(
                        [code, _metric_of(arms, section, code)]
                        + [_cell(arm, section, code) for arm in arms]
                        for code in codes
                    ),
                    [
                        "macro",
                        MISSING,
                        *(
                            fmt_mean_std(
                                arm.aggregate.macro[section].mean, arm.aggregate.macro[section].std
                            )
                            for arm in arms
                        ),
                    ],
                ],
                aligns=["left", "left", *("right" for _ in arms)],
            )
        )
        lines += [
            "",
            "Macro rows average each arm's languages within a seed and then across seeds: "
            + "; ".join(
                f"{arm.arm} {arm.aggregate.macro[section].label} over "
                f"{arm.aggregate.macro[section].n_languages} language(s)"
                for arm in arms
            )
            + ".",
            "",
        ]
        if section == "in_distribution":
            lines += [
                "In-distribution, arm A and the BTM arms are not matched on training "
                "budget — a language sees more epochs of its own data under arm A — so a "
                "difference here is not the effect of the pipeline alone. The transfer "
                "section is the matched comparison.",
                "",
            ]
        for contrast in (c for c in contrasts if c.section == section):
            lines += [f"### {contrast.a} − {contrast.b}", ""]
            if contrast.macro is None:
                lines += [
                    "No language was evaluated by every run of both arms — some run is "
                    f"missing a prediction sidecar for: {', '.join(contrast.excluded)}.",
                    "",
                ]
                continue
            lines.append(
                render_table(
                    ["language", "metric", "Δ", "95% CI", "p", "n utts"],
                    [_contrast_cells(row, n) for row in (*contrast.rows, contrast.macro)],
                    aligns=["left", "left", "right", "right", "right", "right"],
                )
            )
            lines.append("")
            if contrast.excluded:
                lines += [
                    "Not contrasted, because some run of one of the two arms did not "
                    f"evaluate them: {', '.join(contrast.excluded)}. The macro Δ is over "
                    f"the {len(contrast.rows)} language(s) above and is therefore not the "
                    "difference of the two arm macros in the first table.",
                    "",
                ]
    lines += [
        "## How to read the contrasts",
        "",
        f"Δ is the first arm's mean error rate minus the second's, in percentage points; "
        f"negative means the first arm is better. Intervals are 95% percentile intervals "
        f"from {n} draws that resample each arm's seeds and each language's test utterances "
        "together, so they carry both run-to-run and test-set variation. p is two-sided, "
        "read off the same draws, and is not the one-sided paired permutation p that "
        "`svb analyze` reports for two runs at one seed. Nothing is adjusted for the "
        "number of contrasts; the study protocol says which single contrast is the "
        "pre-specified one, and every other row is a breakdown.",
        "",
        "With few seeds the intervals are narrower than their nominal level, because "
        "resampling a handful of runs cannot show more spread than those runs do. In "
        "simulation at a true difference of zero (scripts/interval_coverage.py), five seeds "
        "per arm cover zero about 87–94% of the time and three seeds about 84–87%, so read "
        "p < 0.05 as roughly one-in-ten evidence, not one-in-twenty. An arm with one seed "
        "contributes no run-to-run variation at all. Test utterances are resampled as "
        "independent, though a corpus holds many clips per speaker and per sentence, which "
        "also makes the intervals somewhat too narrow.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def to_json(
    scale: str, arms: list[ArmRuns], contrasts: list[ArmContrast], n: int, seed: int
) -> dict[str, Any]:
    from .aggregate import to_json as aggregate_json

    return {
        "scale": scale,
        "n_resamples": n,
        "seed": seed,
        "interval": "multibootstrap: seeds and utterances resampled together, 95% percentile",
        "arms": {arm.arm: aggregate_json(arm.aggregate) for arm in arms},
        "contrasts": [
            {
                "a": contrast.a,
                "b": contrast.b,
                "section": contrast.section,
                "delta_is": "a minus b, percentage points; negative means a is better",
                "excluded_languages": contrast.excluded,
                "rows": [row.__dict__ for row in contrast.rows],
                "macro": contrast.macro.__dict__ if contrast.macro else None,
            }
            for contrast in contrasts
        ],
    }


def write_tables(
    root: Path,
    scale: str,
    arms: list[str],
    out_dir: Path,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> list[Path]:
    """Write ``<scale>_arms.md`` and ``.json``; return both paths."""
    loaded = load_arms(root, scale, arms)
    contrasts = compare_arms(loaded, n_resamples=n_resamples, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{scale}_arms.md"
    json_path = out_dir / f"{scale}_arms.json"
    md_path.write_text(to_markdown(scale, loaded, contrasts, n_resamples), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            to_json(scale, loaded, contrasts, n_resamples, seed), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return [md_path, json_path]
