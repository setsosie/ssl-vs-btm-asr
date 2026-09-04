"""Markdown rendering shared by the reporting commands.

Small on purpose. The value is that every generated table formats a missing
number, a single-seed standard deviation and a stray pipe the same way, so two
tables in one write-up cannot disagree about what a blank cell means.
"""

from __future__ import annotations

from collections.abc import Sequence

#: What an absent number prints as. Not "0", which is a measurement, and not
#: "nan", which reads like a computation that went wrong rather than one that
#: was never possible.
MISSING = "—"

_ALIGN_RULE = {"left": "---", "right": "---:", "center": ":---:"}


def fmt_value(value: float | None, places: int = 2) -> str:
    """A number, or the missing marker when there is nothing to report."""
    if value is None:
        return MISSING
    return f"{value:.{places}f}"


def fmt_mean_std(mean: float, std: float, places: int = 2) -> str:
    """``mean ± std``, with the missing marker when std is undefined.

    ``aggregate_seeds`` returns nan for a single seed, which is correct and
    unprintable: a lone run has a mean but no spread, and "± nan" invites a
    reader to think the spread was computed and came out broken.
    """
    spread = MISSING if std != std else f"{std:.{places}f}"
    return f"{mean:.{places}f} ± {spread}"


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Sequence[str] | None = None,
) -> str:
    """Render a GitHub-flavoured Markdown table.

    Args:
        headers: Column headings.
        rows: Cells, already formatted as strings.
        aligns: Per-column "left", "right" or "center". Defaults to all-left.

    Raises:
        ValueError: If a row's width does not match the header's. A short row
            shifts every later cell one column left, which in a results table
            silently reassigns numbers to the wrong metric.
    """
    width = len(headers)
    if aligns is None:
        aligns = ["left"] * width
    if len(aligns) != width:
        raise ValueError(f"aligns has {len(aligns)} entries for {width} columns")

    lines = [_row(headers), "| " + " | ".join(_ALIGN_RULE[a] for a in aligns) + " |"]
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"row {index} has {len(row)} cells for {width} columns: {list(row)}")
        lines.append(_row(row))
    return "\n".join(lines)


def _row(cells: Sequence[str]) -> str:
    # A pipe inside a cell would otherwise open a column that the separator row
    # does not have, and the whole table stops rendering.
    return "| " + " | ".join(str(c).replace("|", r"\|") for c in cells) + " |"
