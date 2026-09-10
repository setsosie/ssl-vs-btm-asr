"""`svb analyze`: which runs it picks up and what it writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.cli import build_parser
from tests.test_report_aggregate import metrics, write_run
from tests.test_report_analyze import write_sidecar


def run_cli(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


def make_run(root: Path, arm: str, seed: int, hyp: str) -> Path:
    run = write_run(
        root,
        arm,
        "3",
        seed,
        in_dist={"en": metrics(25.0, 10.0)},
        word_boundary={"en": True},
    )
    write_sidecar(run, "en", [("a b c d", hyp)] * 12, transfer=False)
    return run


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_run(tmp_path, "A_ssl", 0, "a b c x")
    make_run(tmp_path, "A_ssl", 1, "a b c x")
    make_run(tmp_path, "B_btm_ssl", 0, "a b q r")
    return tmp_path


def test_writes_a_markdown_and_json_table_per_metric(root: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "out"
    run_cli(
        [
            "analyze",
            "--arm",
            "A_ssl",
            "--scale",
            "3",
            "--results-root",
            str(root),
            "--tables-dir",
            str(out),
            "--resamples",
            "200",
        ],
    )

    assert sorted(p.name for p in out.iterdir()) == [
        "3_cer.json",
        "3_cer.md",
        "3_wer.json",
        "3_wer.md",
    ]
    # The command names what it produced rather than describing an intention.
    assert "3_wer.md" in capsys.readouterr().out


def test_all_seeds_are_analyzed_when_none_are_named(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_cli(
        [
            "analyze",
            "--arm",
            "A_ssl",
            "--scale",
            "3",
            "--results-root",
            str(root),
            "--tables-dir",
            str(out),
            "--resamples",
            "200",
        ],
    )

    payload = json.loads((out / "3_wer.json").read_text(encoding="utf-8"))
    assert set(payload["runs"]) == {"A_ssl/3/seed0", "A_ssl/3/seed1"}


def test_a_named_seed_narrows_the_analysis(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_cli(
        [
            "analyze",
            "--arm",
            "A_ssl",
            "--scale",
            "3",
            "--seed",
            "1",
            "--results-root",
            str(root),
            "--tables-dir",
            str(out),
            "--resamples",
            "200",
        ],
    )

    payload = json.loads((out / "3_wer.json").read_text(encoding="utf-8"))
    assert set(payload["runs"]) == {"A_ssl/3/seed1"}


def test_compare_to_pairs_the_same_seed_of_two_arms(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_cli(
        [
            "analyze",
            "--arm",
            "A_ssl",
            "--scale",
            "3",
            "--compare-to",
            "B_btm_ssl",
            "--results-root",
            str(root),
            "--tables-dir",
            str(out),
            "--resamples",
            "500",
        ],
    )

    payload = json.loads((out / "3_wer.json").read_text(encoding="utf-8"))
    # Only seed 0 exists for both arms, so only seed 0 can be paired.
    assert [c["a"] for c in payload["comparisons"]] == ["A_ssl/3/seed0"]
    assert payload["comparisons"][0]["b"] == "B_btm_ssl/3/seed0"
    row = payload["comparisons"][0]["rows"][0]
    assert row["delta"] < 0  # A has the lower error rate
    assert row["p_value"] < 0.05


def test_missing_runs_are_reported_rather_than_producing_an_empty_table(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no runs"):
        run_cli(
            [
                "analyze",
                "--arm",
                "A_ssl",
                "--scale",
                "3",
                "--results-root",
                str(tmp_path),
                "--tables-dir",
                str(tmp_path / "out"),
            ],
        )
