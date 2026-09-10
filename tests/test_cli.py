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


def test_run_manifest_records_both_resolved_language_sets() -> None:
    """The held-out set is a fact of the results file, not an inference from it.

    Reading it off the keys of `transfer` cannot distinguish "four languages
    were configured" from "five were configured and one crashed".
    """
    from svb.cli import run_manifest
    from svb.config import load_config
    from svb.data.registry import LangSpec

    specs = [LangSpec(code="hi", source="commonvoice", hf_config="hi")]
    heldout = [
        LangSpec(
            code="telugu", source="openslr", slr=66, archives=("a.zip",), index_files=("i.tsv",)
        )
    ]

    manifest = run_manifest(load_config("A_ssl", "3", 0), specs, heldout)

    assert manifest["languages"] == ["hi"]
    assert manifest["heldout_langs"] == ["telugu"]
    assert manifest["in_distribution"] == {}
    assert (manifest["arm"], manifest["scale"], manifest["seed"]) == ("A_ssl", "3", 0)


def test_merge_head_can_be_turned_off_from_the_command_line() -> None:
    from svb.cli import build_parser

    parser = build_parser()
    base = ["run", "--arm", "B_btm_ssl", "--scale", "3", "--seed", "0"]

    # None, not True, when unset: an unpassed flag must not outrank a
    # `merge_head: false` in the YAML with the parser's own default.
    assert parser.parse_args(base).merge_head is None
    assert parser.parse_args([*base, "--no-merge-head"]).merge_head is False
    assert parser.parse_args([*base, "--merge-head"]).merge_head is True
