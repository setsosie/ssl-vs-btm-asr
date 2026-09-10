"""Utterance-level intervals and paired tests, recomputed from the sidecars.

Seed spread and test-set sampling uncertainty are different quantities and this
module reports the second. ``svb aggregate`` answers "how much does retraining
move this number"; here the question is "how much of this number is the test set
we happened to draw", which needs the per-utterance pairs rather than the corpus
figure. Both come out of files already on disk, so neither costs GPU time.

One table per metric rather than one table of primaries. A file of WERs is a
file of comparable numbers; a file that silently switches to CER for Japanese is
not, and a reader who sorts it gets nonsense. Each row still marks whether that
metric is the language's primary, so the information is not lost.

Paired permutation requires the two systems to have been scored on the same
utterances, and that is checked rather than assumed. Two runs under different
normalization policies produce different reference strings from the same corpus,
and pairing those would compare two systems on two test sets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stats.analysis import Tokenization, bootstrap_ci, corpus_error_rate, paired_permutation
from .tables import fmt_value, render_table

#: Which tokenization each metric is defined over. A property of the metric, not
#: of the language: CER is character-level for English too.
METRIC_TOKENIZE: dict[str, Tokenization] = {"wer": "word", "cer": "char"}
METRICS = ("wer", "cer")
_METRIC_LABEL = {"wer": "WER", "cer": "CER"}


@dataclass(frozen=True)
class Sidecar:
    """One language's per-utterance references and hypotheses from one run."""

    code: str
    section: str
    path: Path
    refs: list[str]
    hyps: list[str]
    is_primary_word: bool  # whether WER is this language's primary metric


@dataclass(frozen=True)
class Interval:
    code: str
    section: str
    metric: str
    is_primary: bool
    point: float
    lo: float
    hi: float
    n: int


@dataclass(frozen=True)
class Comparison:
    code: str
    section: str
    metric: str
    is_primary: bool
    point_a: float
    point_b: float
    delta: float  # a - b; negative means A has the lower error rate
    p_value: float
    n: int


def _word_boundary(run_dir: Path) -> dict[str, bool]:
    path = run_dir / "text_stats.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        code: bool(entry.get("word_boundary", True))
        for code, entry in payload.get("languages", {}).items()
    }


def _read_pairs(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs", [])
    return [str(r) for r, _ in pairs], [str(h) for _, h in pairs]


def load_sidecars(run_dir: Path) -> list[Sidecar]:
    """Every prediction sidecar a run wrote, in-distribution and transfer.

    The two live at different depths because they are written by different code
    paths: in-distribution evaluation writes ``predictions/<code>.json`` beside
    the results, while each held-out language gets its own directory.
    """
    boundary = _word_boundary(run_dir)
    found: list[Sidecar] = []

    for path in sorted((run_dir / "predictions").glob("*.json")):
        refs, hyps = _read_pairs(path)
        found.append(
            Sidecar(
                code=path.stem,
                section="in_distribution",
                path=path,
                refs=refs,
                hyps=hyps,
                is_primary_word=boundary.get(path.stem, True),
            )
        )

    for path in sorted((run_dir / "transfer").glob("*/predictions.json")):
        code = path.parent.name
        refs, hyps = _read_pairs(path)
        found.append(
            Sidecar(
                code=code,
                section="transfer",
                path=path,
                refs=refs,
                hyps=hyps,
                is_primary_word=boundary.get(code, True),
            )
        )
    return found


def _require_sidecars(run_dir: Path) -> list[Sidecar]:
    found = load_sidecars(run_dir)
    if not found:
        raise FileNotFoundError(
            f"no prediction sidecars under {run_dir}; expected predictions/*.json or "
            "transfer/*/predictions.json from a completed `svb run`"
        )
    return found


def _is_primary(sidecar: Sidecar, metric: str) -> bool:
    return sidecar.is_primary_word == (metric == "wer")


def intervals_for_run(
    run_dir: Path, metric: str, n_resamples: int = 10_000, seed: int = 42, ci: float = 0.95
) -> list[Interval]:
    """Per-language bootstrap percentile interval for one run and one metric."""
    tokenize = METRIC_TOKENIZE[metric]
    rows = []
    for sidecar in _require_sidecars(run_dir):
        result = bootstrap_ci(
            sidecar.refs, sidecar.hyps, n=n_resamples, ci=ci, seed=seed, tokenize=tokenize
        )
        rows.append(
            Interval(
                code=sidecar.code,
                section=sidecar.section,
                metric=metric,
                is_primary=_is_primary(sidecar, metric),
                point=result.point,
                lo=result.lo,
                hi=result.hi,
                n=len(sidecar.refs),
            )
        )
    return rows


def compare_runs(
    run_a: Path, run_b: Path, metric: str, n_resamples: int = 10_000, seed: int = 42
) -> list[Comparison]:
    """Paired permutation test per language, for languages both runs scored.

    Raises:
        ValueError: If a language's references differ between the two runs. The
            test is only meaningful on a shared test set, and non-identical
            references mean the two systems were not scored on one.
    """
    tokenize = METRIC_TOKENIZE[metric]
    by_code_b = {(s.section, s.code): s for s in _require_sidecars(run_b)}
    rows = []

    for a in _require_sidecars(run_a):
        b = by_code_b.get((a.section, a.code))
        if b is None:
            continue
        if a.refs != b.refs:
            raise ValueError(
                f"{a.code}: the two runs were scored on different references, so this is "
                f"not a paired comparison. {_first_difference(a.refs, b.refs)}. Compare "
                f"runs that share a normalization policy.\n  {a.path}\n  {b.path}"
            )
        point_a = corpus_error_rate(a.refs, a.hyps, tokenize)
        point_b = corpus_error_rate(b.refs, b.hyps, tokenize)
        rows.append(
            Comparison(
                code=a.code,
                section=a.section,
                metric=metric,
                is_primary=_is_primary(a, metric),
                point_a=point_a,
                point_b=point_b,
                delta=point_a - point_b,
                p_value=paired_permutation(
                    a.refs, a.hyps, b.hyps, n=n_resamples, seed=seed, tokenize=tokenize
                ),
                n=len(a.refs),
            )
        )
    return rows


def _first_difference(a: list[str], b: list[str]) -> str:
    """Where two reference lists diverge, in a form a reader can act on.

    Reporting only the counts describes the rarer case. Two runs under different
    normalization policies produce the *same number* of references from one
    corpus and different strings, so the counts match and the message says
    nothing about what is wrong.
    """
    if len(a) != len(b):
        return f"{len(a)} vs {len(b)} utterances"
    for index, (left, right) in enumerate(zip(a, b, strict=True)):
        if left != right:
            return f"same {len(a)} utterances, but reference {index} is {left!r} vs {right!r}"
    return "the lists differ but no element does"  # pragma: no cover - unreachable


def _label(run_dir: Path) -> str:
    """``arm/scale/seedN`` — how a run is named in a table."""
    return "/".join(run_dir.parts[-3:])


def render_metric_tables(
    scale: str,
    runs: list[Path],
    comparisons: list[tuple[Path, Path]],
    out_dir: Path,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> list[Path]:
    """Write ``<scale>_<metric>.md`` and ``.json`` for each metric.

    Returns every path written, so a caller can report exactly what it produced
    rather than describing what it intended to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for metric in METRICS:
        payload: dict[str, Any] = {
            "scale": scale,
            "metric": metric,
            "tokenization": METRIC_TOKENIZE[metric],
            "n_resamples": n_resamples,
            "seed": seed,
            "runs": {},
            "comparisons": [],
        }
        body = [
            f"# scale {scale} — {_METRIC_LABEL[metric]}",
            "",
            f"Utterance-level bootstrap percentile intervals, {n_resamples} resamples, "
            f"{METRIC_TOKENIZE[metric]}-level tokenization. This is test-set sampling "
            "uncertainty for a single run, not spread across seeds — `svb aggregate` "
            "reports that.",
            "",
        ]

        for run in runs:
            rows = intervals_for_run(run, metric=metric, n_resamples=n_resamples, seed=seed)
            payload["runs"][_label(run)] = [row.__dict__ for row in rows]
            body += [f"## {_label(run)}", ""]
            body.append(
                render_table(
                    ["language", "section", _METRIC_LABEL[metric], "95% CI", "primary", "n"],
                    [
                        [
                            row.code,
                            row.section.replace("_", " "),
                            fmt_value(row.point),
                            f"[{fmt_value(row.lo)}, {fmt_value(row.hi)}]",
                            "yes" if row.is_primary else "no",
                            str(row.n),
                        ]
                        for row in rows
                    ],
                    aligns=["left", "left", "right", "right", "left", "right"],
                )
            )
            body.append("")

        for run_a, run_b in comparisons:
            rows_c = compare_runs(run_a, run_b, metric=metric, n_resamples=n_resamples, seed=seed)
            payload["comparisons"].append(
                {"a": _label(run_a), "b": _label(run_b), "rows": [r.__dict__ for r in rows_c]}
            )
            body += [f"## {_label(run_a)} vs {_label(run_b)}", ""]
            body.append(
                render_table(
                    ["language", "section", "A", "B", "Δ (A−B)", "p", "primary", "n"],
                    [
                        [
                            row.code,
                            row.section.replace("_", " "),
                            fmt_value(row.point_a),
                            fmt_value(row.point_b),
                            fmt_value(row.delta),
                            f"{row.p_value:.4f}",
                            "yes" if row.is_primary else "no",
                            str(row.n),
                        ]
                        for row in rows_c
                    ],
                    aligns=["left", "left", "right", "right", "right", "right", "left", "right"],
                )
            )
            body += [
                "",
                "One-sided paired permutation on per-utterance edit counts; a small p "
                "supports A having the lower error rate. Negative Δ means A is better.",
                "",
            ]

        md_path = out_dir / f"{scale}_{metric}.md"
        json_path = out_dir / f"{scale}_{metric}.json"
        md_path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += [md_path, json_path]

    return written
