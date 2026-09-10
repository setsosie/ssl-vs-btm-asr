"""Markdown rendering for generated tables."""

from __future__ import annotations

import re

import pytest

from svb.report.tables import fmt_mean_std, fmt_value, render_table


def test_render_table_has_header_rule_and_cells():
    out = render_table(["lang", "wer"], [["en", "12.3"], ["ja", "4.5"]])
    lines = out.splitlines()
    assert lines[0] == "| lang | wer |"
    assert set(lines[1]) <= set("|-: ")
    assert lines[2] == "| en | 12.3 |"
    assert lines[3] == "| ja | 4.5 |"


def test_render_table_right_aligns_declared_numeric_columns():
    out = render_table(["lang", "wer"], [["en", "12.3"]], aligns=["left", "right"])
    rule = out.splitlines()[1]
    assert rule == "| --- | ---: |"


def test_render_table_escapes_pipes_so_a_cell_cannot_forge_a_column():
    out = render_table(["a"], [["x|y"]])
    assert r"x\|y" in out
    # One header column means two *delimiting* pipes, however many the cell
    # held. The escaped one is still a pipe character, so count only unescaped.
    row = out.splitlines()[2]
    assert len(re.findall(r"(?<!\\)\|", row)) == 2


def test_render_table_rejects_a_row_of_the_wrong_width():
    # A short row silently shifts every later cell into the wrong column, which
    # is worse in a results table than a crash.
    with pytest.raises(ValueError, match="2 columns"):
        render_table(["a", "b"], [["only-one"]])


def test_fmt_mean_std_shows_a_dash_for_a_single_seed():
    # std is nan for n<2; printing "nan" reads like a computation that failed.
    assert fmt_mean_std(5.0, float("nan")) == "5.00 ± —"
    assert fmt_mean_std(5.0, 0.25) == "5.00 ± 0.25"


def test_fmt_value_marks_a_missing_number_rather_than_printing_zero():
    assert fmt_value(None) == "—"
    assert fmt_value(1.5) == "1.50"
