"""Utterance-level intervals and paired tests, read back from the sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.report.analyze import (
    METRIC_TOKENIZE,
    compare_runs,
    intervals_for_run,
    load_sidecars,
    render_metric_tables,
)
from tests.test_report_aggregate import metrics, write_run


def write_sidecar(run: Path, code: str, pairs: list[tuple[str, str]], *, transfer: bool) -> Path:
    path = (
        run / "transfer" / code / "predictions.json"
        if transfer
        else run / "predictions" / f"{code}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"wer": 0.0, "cer": 0.0, "n": len(pairs), "pairs": [list(p) for p in pairs]}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def run_with_sidecars(tmp_path: Path) -> Path:
    run = write_run(
        tmp_path,
        "A_ssl",
        "3",
        0,
        in_dist={"en": metrics(25.0, 10.0), "ja": metrics(90.0, 20.0)},
        transfer={"telugu": metrics(30.0, 8.0)},
        word_boundary={"en": True, "ja": False, "telugu": True},
    )
    write_sidecar(run, "en", [("a b c d", "a b c x")] * 8, transfer=False)
    write_sidecar(run, "ja", [("あいうえ", "あいうお")] * 8, transfer=False)
    write_sidecar(run, "telugu", [("x y z", "x y z")] * 8, transfer=True)
    return tmp_path


def test_load_sidecars_finds_both_in_distribution_and_transfer(run_with_sidecars: Path):
    found = load_sidecars(run_with_sidecars / "A_ssl" / "3" / "seed0")
    by_code = {s.code: s for s in found}

    assert set(by_code) == {"en", "ja", "telugu"}
    assert by_code["en"].section == "in_distribution"
    # Transfer sidecars live one level deeper, under the language's own dir.
    assert by_code["telugu"].section == "transfer"
    assert by_code["en"].refs == ["a b c d"] * 8


def test_metric_tokenization_is_fixed_by_the_metric_not_the_language():
    assert METRIC_TOKENIZE == {"wer": "word", "cer": "char"}


def test_interval_brackets_the_point_estimate(run_with_sidecars: Path):
    rows = intervals_for_run(
        run_with_sidecars / "A_ssl" / "3" / "seed0", metric="wer", n_resamples=200
    )
    en = next(r for r in rows if r.code == "en")

    # One substitution in every four-word utterance.
    assert en.point == pytest.approx(25.0)
    assert en.lo <= en.point <= en.hi
    assert en.n == 8


def test_interval_marks_which_languages_the_metric_is_primary_for(run_with_sidecars: Path):
    rows = {
        r.code: r
        for r in intervals_for_run(
            run_with_sidecars / "A_ssl" / "3" / "seed0", metric="wer", n_resamples=200
        )
    }
    # Japanese has no word boundaries, so WER is reported but is not its primary.
    assert rows["en"].is_primary is True
    assert rows["ja"].is_primary is False


def test_character_metric_uses_character_tokenization(run_with_sidecars: Path):
    rows = {
        r.code: r
        for r in intervals_for_run(
            run_with_sidecars / "A_ssl" / "3" / "seed0", metric="cer", n_resamples=200
        )
    }
    # One of four characters wrong.
    assert rows["ja"].point == pytest.approx(25.0)
    assert rows["ja"].is_primary is True


def test_compare_runs_pairs_systems_on_the_same_utterances(tmp_path: Path):
    for arm, hyp in (("A_ssl", "a b c x"), ("B_btm_ssl", "a b q r")):
        run = write_run(
            tmp_path,
            arm,
            "3",
            0,
            in_dist={"en": metrics(0.0, 0.0)},
            word_boundary={"en": True},
        )
        write_sidecar(run, "en", [("a b c d", hyp)] * 30, transfer=False)

    rows = compare_runs(
        tmp_path / "A_ssl" / "3" / "seed0",
        tmp_path / "B_btm_ssl" / "3" / "seed0",
        metric="wer",
        n_resamples=500,
    )
    row = rows[0]

    assert row.code == "en"
    assert row.point_a == pytest.approx(25.0)  # one error in four
    assert row.point_b == pytest.approx(50.0)  # two errors in four
    assert row.delta == pytest.approx(-25.0)  # A better
    assert row.p_value < 0.05


def test_compare_refuses_systems_scored_on_different_references(tmp_path: Path):
    # A paired test on non-identical references is not a paired test. This can
    # happen for real: two runs under different normalization policies produce
    # different reference strings from the same corpus.
    for arm, ref in (("A_ssl", "a b c d"), ("B_btm_ssl", "a b c e")):
        run = write_run(
            tmp_path, arm, "3", 0, in_dist={"en": metrics(0.0, 0.0)}, word_boundary={"en": True}
        )
        write_sidecar(run, "en", [(ref, "a b c d")] * 4, transfer=False)

    with pytest.raises(ValueError, match="different references"):
        compare_runs(
            tmp_path / "A_ssl" / "3" / "seed0",
            tmp_path / "B_btm_ssl" / "3" / "seed0",
            metric="wer",
        )


def test_compare_skips_a_language_only_one_system_has(tmp_path: Path):
    run_a = write_run(
        tmp_path,
        "A_ssl",
        "3",
        0,
        in_dist={"en": metrics(0.0, 0.0), "de": metrics(0.0, 0.0)},
        word_boundary={"en": True, "de": True},
    )
    write_sidecar(run_a, "en", [("a b", "a b")] * 4, transfer=False)
    write_sidecar(run_a, "de", [("c d", "c d")] * 4, transfer=False)
    run_b = write_run(
        tmp_path, "B_btm_ssl", "3", 0, in_dist={"en": metrics(0.0, 0.0)}, word_boundary={"en": True}
    )
    write_sidecar(run_b, "en", [("a b", "a b")] * 4, transfer=False)

    rows = compare_runs(run_a, run_b, metric="wer", n_resamples=100)

    assert [r.code for r in rows] == ["en"]


def test_render_writes_one_markdown_and_one_json_per_metric(
    run_with_sidecars: Path, tmp_path: Path
):
    out = tmp_path / "tables"
    written = render_metric_tables(
        scale="3",
        runs=[run_with_sidecars / "A_ssl" / "3" / "seed0"],
        comparisons=[],
        out_dir=out,
        n_resamples=200,
    )

    assert (out / "3_wer.md").exists()
    assert (out / "3_wer.json").exists()
    assert (out / "3_cer.md").exists()
    assert set(written) == {
        out / "3_wer.md",
        out / "3_wer.json",
        out / "3_cer.md",
        out / "3_cer.json",
    }
    body = (out / "3_wer.md").read_text(encoding="utf-8")
    assert "95%" in body and "en" in body
    payload = json.loads((out / "3_wer.json").read_text(encoding="utf-8"))
    assert payload["metric"] == "wer"
    assert payload["scale"] == "3"


def test_a_run_with_no_sidecars_is_named_rather_than_silently_empty(tmp_path: Path):
    run = write_run(
        tmp_path, "A_ssl", "3", 0, in_dist={"en": metrics(1.0, 1.0)}, word_boundary={"en": True}
    )
    with pytest.raises(FileNotFoundError, match="no prediction sidecars"):
        intervals_for_run(run, metric="wer")
