"""CLI: where results land, and which arm/scale names are accepted."""

from __future__ import annotations

from pathlib import Path

import pytest

from svb.cli import _run_dir, build_parser, results_root


def test_results_root_defaults_to_a_relative_results_dir(monkeypatch) -> None:
    monkeypatch.delenv("SVB_RESULTS_ROOT", raising=False)
    assert results_root(None) == Path("results")


def test_results_root_reads_the_environment(monkeypatch, tmp_path) -> None:
    """A cluster job writes to scratch, not into the checkout it launched from."""
    monkeypatch.setenv("SVB_RESULTS_ROOT", str(tmp_path))
    assert results_root(None) == tmp_path


def test_an_explicit_flag_beats_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SVB_RESULTS_ROOT", str(tmp_path / "env"))
    assert results_root(str(tmp_path / "flag")) == tmp_path / "flag"


def test_run_dir_layout() -> None:
    assert _run_dir(Path("/r"), "A_ssl", "16", 3) == Path("/r/A_ssl/16/seed3")


def test_aggregate_rejects_an_arm_that_cannot_exist() -> None:
    """Unvalidated, `aggregate` printed an empty table for a typo.

    A zero-seed table looks like a result that has not been produced yet, not
    like a misspelled argument, so the mistake survives to the next reader.
    """
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["aggregate", "--arm", "A_sssl", "--scale", "3"])


def test_aggregate_rejects_a_scale_that_cannot_exist() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["aggregate", "--arm", "A_ssl", "--scale", "7"])


def test_aggregate_accepts_the_same_names_as_run() -> None:
    parser = build_parser()

    args = parser.parse_args(["aggregate", "--arm", "B_btm_ssl", "--scale", "64"])

    assert args.arm == "B_btm_ssl"
    assert args.scale == "64"
