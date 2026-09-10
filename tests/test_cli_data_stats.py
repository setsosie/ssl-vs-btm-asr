"""`svb data-stats`: which languages it sweeps and what it writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.cli import build_parser, specs_for_scope


def run_cli(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


def test_heldout_scope_is_the_four_openslr_languages() -> None:
    specs = specs_for_scope("heldout")

    assert [s.code for s in specs] == ["malayalam", "marathi", "telugu", "gujarati"]
    assert {s.source for s in specs} == {"openslr"}


def test_all_scope_skips_an_unpopulated_preset_without_failing(capsys, monkeypatch) -> None:
    # An unpopulated preset raises, which is right for a run and wrong for a
    # report of what is on disk: that has to name the gap and keep going. Every
    # preset in the tree is populated now, so the gap is simulated.
    def unpopulated(scale, configs_dir=None):
        raise ValueError(f"preset {scale} is not populated")

    monkeypatch.setattr("svb.data.registry.get_preset", unpopulated)

    specs = specs_for_scope("all")
    out = capsys.readouterr().out

    assert [s.code for s in specs if s.source == "openslr"]
    assert "64" in out
    assert "not populated" in out


def test_all_scope_lists_each_language_once() -> None:
    # en appears in both the 3 and 16 presets; sweeping it twice would double
    # its hours in the total.
    codes = [s.code for s in specs_for_scope("all")]

    assert len(codes) == len(set(codes))


def test_writes_markdown_and_json_for_the_heldout_set(
    tmp_path: Path, slr_root: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))
    out = tmp_path / "tables"

    run_cli(["data-stats", "--scope", "heldout", "--tables-dir", str(out)])

    assert (out / "data_durations.md").exists()
    payload = json.loads((out / "data_durations.json").read_text(encoding="utf-8"))
    by_code = {row["code"]: row for row in payload["languages"]}
    # Only Malayalam is staged in the fixture root; the other three report the
    # reason they could not be read rather than a zero.
    assert by_code["malayalam"]["error"] is None
    assert by_code["malayalam"]["total_hours"] > 0
    assert by_code["marathi"]["error"] is not None
    assert "data_durations.md" in capsys.readouterr().out


def test_threshold_flag_reaches_the_report(tmp_path: Path, slr_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))
    out = tmp_path / "tables"

    run_cli(
        [
            "data-stats",
            "--scope",
            "heldout",
            "--tables-dir",
            str(out),
            "--min-train-hours",
            "50",
        ]
    )

    payload = json.loads((out / "data_durations.json").read_text(encoding="utf-8"))
    assert payload["min_train_hours"] == 50.0
    by_code = {row["code"]: row for row in payload["languages"]}
    # The fixture holds a few seconds of silence, so it is well under 50 hours.
    assert by_code["malayalam"]["below_threshold"] is True


def test_scope_rejects_an_unknown_name() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data-stats", "--scope", "nope"])
