"""`svb aggregate` end to end: what it prints, and what `--json` emits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.cli import build_parser
from tests.test_report_aggregate import metrics, write_run


def run_cli(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for seed, bump in ((0, 0.0), (1, 2.0)):
        write_run(
            tmp_path,
            "A_ssl",
            "3",
            seed,
            in_dist={"en": metrics(10.0 + bump, 4.0 + bump), "ja": metrics(90.0, 20.0 + bump)},
            transfer={"telugu": metrics(30.0 + bump, 8.0 + bump)},
            word_boundary={"en": True, "ja": False, "telugu": True},
        )
    return tmp_path


def test_table_shows_both_metrics_and_the_primary_marker(root: Path, capsys) -> None:
    run_cli(["aggregate", "--arm", "A_ssl", "--scale", "3", "--results-root", str(root)])
    out = capsys.readouterr().out

    assert "WER" in out and "CER" in out
    # Japanese is scored by CER, English by WER, and the table says which.
    assert "en" in out and "ja" in out
    assert "11.00" in out  # en WER mean over the two seeds
    assert "21.00" in out  # ja CER mean


def test_mixed_macro_is_never_printed_without_its_label(root: Path, capsys) -> None:
    run_cli(["aggregate", "--arm", "A_ssl", "--scale", "3", "--results-root", str(root)])
    out = capsys.readouterr().out

    assert "mixed" in out
    assert "WER×1" in out and "CER×1" in out


def test_json_flag_emits_parseable_output_only(root: Path, capsys) -> None:
    run_cli(["aggregate", "--arm", "A_ssl", "--scale", "3", "--results-root", str(root), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["arm"] == "A_ssl"
    assert payload["seeds"] == [0, 1]
    assert payload["macro"]["in_distribution"]["is_mixed"] is True
    codes = {row["code"] for row in payload["languages"]}
    assert codes == {"en", "ja", "telugu"}


def test_missing_runs_report_the_path_rather_than_an_empty_table(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no runs"):
        run_cli(["aggregate", "--arm", "A_ssl", "--scale", "3", "--results-root", str(tmp_path)])
