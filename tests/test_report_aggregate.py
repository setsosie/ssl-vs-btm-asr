"""Across-seed aggregation: both metrics, the per-language primary, and the
labelling that stops two different metrics being averaged into one number."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from svb.report.aggregate import aggregate_runs, load_runs, to_json, to_markdown


def write_run(
    root: Path,
    arm: str,
    scale: str,
    seed: int,
    in_dist: dict[str, dict[str, float]],
    transfer: dict[str, dict[str, float]] | None = None,
    word_boundary: dict[str, bool] | None = None,
    policies: dict[str, str] | None = None,
) -> Path:
    run = root / arm / scale / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    results = {
        "arm": arm,
        "scale": scale,
        "seed": seed,
        "languages": sorted(in_dist),
        "heldout_langs": sorted(transfer or {}),
        "in_distribution": in_dist,
        "transfer": transfer or {},
    }
    (run / "results.json").write_text(json.dumps(results), encoding="utf-8")
    if word_boundary is not None:
        stats = {
            "languages": {
                code: {
                    "word_boundary": wb,
                    "heldout": code in (transfer or {}),
                    # A real run always records this; the reports refuse to
                    # compare two runs that cannot show they used the same rules.
                    "policy": (policies or {}).get(code, "whisper-basic"),
                    "splits": {},
                }
                for code, wb in word_boundary.items()
            }
        }
        (run / "text_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return run


def metrics(wer: float, cer: float, n: int = 100) -> dict[str, float]:
    return {"wer": wer, "cer": cer, "n": n, "n_empty_refs": 0}


@pytest.fixture
def two_seed_root(tmp_path: Path) -> Path:
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


def test_reports_both_metrics_not_just_wer(two_seed_root: Path):
    agg = aggregate_runs(load_runs(two_seed_root, "A_ssl", "3"))
    en = next(row for row in agg.languages if row.code == "en")
    assert en.wer.mean == pytest.approx(11.0)
    assert en.wer.std == pytest.approx(math.sqrt(2.0))
    assert en.cer.mean == pytest.approx(5.0)


def test_primary_follows_word_boundary_per_language(two_seed_root: Path):
    agg = aggregate_runs(load_runs(two_seed_root, "A_ssl", "3"))
    by_code = {row.code: row for row in agg.languages}
    # Spaced script -> WER is primary; unspaced -> CER.
    assert by_code["en"].primary_kind == "wer"
    assert by_code["en"].primary.mean == pytest.approx(by_code["en"].wer.mean)
    assert by_code["ja"].primary_kind == "cer"
    assert by_code["ja"].primary.mean == pytest.approx(by_code["ja"].cer.mean)


def test_word_boundary_comes_from_the_run_not_todays_config(tmp_path: Path):
    # The preset may have been edited since the run. The metric a number was
    # chosen under is a property of that run, so it is read back from the
    # artifact the run wrote.
    write_run(
        tmp_path,
        "A_ssl",
        "3",
        0,
        in_dist={"en": metrics(10.0, 4.0)},
        word_boundary={"en": False},
    )
    agg = aggregate_runs(load_runs(tmp_path, "A_ssl", "3"))
    assert agg.languages[0].primary_kind == "cer"


def test_macro_over_one_kind_is_labelled_with_that_kind(tmp_path: Path):
    write_run(
        tmp_path,
        "A_ssl",
        "3",
        0,
        in_dist={"en": metrics(10.0, 4.0), "de": metrics(20.0, 6.0)},
        word_boundary={"en": True, "de": True},
    )
    agg = aggregate_runs(load_runs(tmp_path, "A_ssl", "3"))
    macro = agg.macro["in_distribution"]
    assert macro.kind == "wer"
    assert macro.mean == pytest.approx(15.0)
    assert macro.label == "WER"


def test_macro_over_mixed_kinds_names_the_mixture(two_seed_root: Path):
    # en is scored by WER and ja by CER. Their average is not a WER and not a
    # CER, so the label has to say so wherever the number appears.
    agg = aggregate_runs(load_runs(two_seed_root, "A_ssl", "3"))
    macro = agg.macro["in_distribution"]
    assert macro.kind == "mixed"
    assert macro.composition == {"wer": 1, "cer": 1}
    assert "mixed" in macro.label
    assert "WER×1" in macro.label and "CER×1" in macro.label


def test_a_mixed_macro_carries_its_warning_into_every_rendering(two_seed_root: Path):
    agg = aggregate_runs(load_runs(two_seed_root, "A_ssl", "3"))
    text = to_markdown(agg)
    assert "mixed" in text
    payload = to_json(agg)
    macro = payload["macro"]["in_distribution"]
    assert macro["kind"] == "mixed"
    assert macro["composition"] == {"wer": 1, "cer": 1}
    # The flag a downstream consumer would branch on, so it never has to parse
    # the human-facing label to find out.
    assert payload["macro"]["in_distribution"]["is_mixed"] is True


def test_transfer_is_aggregated_separately_from_in_distribution(two_seed_root: Path):
    agg = aggregate_runs(load_runs(two_seed_root, "A_ssl", "3"))
    sections = {row.section for row in agg.languages}
    assert sections == {"in_distribution", "transfer"}
    assert agg.macro["transfer"].mean == pytest.approx(31.0)
    # Held-out is one spaced language, so its macro is a clean WER.
    assert agg.macro["transfer"].kind == "wer"


def test_single_seed_std_is_nan_and_renders_as_a_dash(tmp_path: Path):
    write_run(
        tmp_path, "A_ssl", "3", 0, in_dist={"en": metrics(10.0, 4.0)}, word_boundary={"en": True}
    )
    agg = aggregate_runs(load_runs(tmp_path, "A_ssl", "3"))
    assert math.isnan(agg.languages[0].wer.std)
    assert "± —" in to_markdown(agg)


def test_a_language_missing_from_one_seed_reports_the_seeds_it_has(tmp_path: Path):
    write_run(
        tmp_path,
        "A_ssl",
        "3",
        0,
        in_dist={"en": metrics(10.0, 4.0), "de": metrics(20.0, 6.0)},
        word_boundary={"en": True, "de": True},
    )
    write_run(
        tmp_path, "A_ssl", "3", 1, in_dist={"en": metrics(12.0, 5.0)}, word_boundary={"en": True}
    )
    agg = aggregate_runs(load_runs(tmp_path, "A_ssl", "3"))
    by_code = {row.code: row for row in agg.languages}
    assert by_code["en"].wer.n_seeds == 2
    # Reported, not dropped and not silently averaged over a smaller n as if it
    # were the same sample as its neighbours.
    assert by_code["de"].wer.n_seeds == 1
    assert agg.n_seeds == 2


def test_load_runs_ignores_a_seed_directory_with_no_results(tmp_path: Path):
    write_run(
        tmp_path, "A_ssl", "3", 0, in_dist={"en": metrics(10.0, 4.0)}, word_boundary={"en": True}
    )
    (tmp_path / "A_ssl" / "3" / "seed9").mkdir(parents=True)
    runs = load_runs(tmp_path, "A_ssl", "3")
    assert [r.seed for r in runs] == [0]


def test_load_runs_on_an_empty_tree_says_so(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no runs"):
        load_runs(tmp_path, "A_ssl", "3")


def test_seeds_are_ordered_numerically_not_lexically(tmp_path: Path):
    # seed10 sorts before seed2 as a string, which would misreport the run list.
    for seed in (2, 10):
        write_run(
            tmp_path,
            "A_ssl",
            "3",
            seed,
            in_dist={"en": metrics(10.0, 4.0)},
            word_boundary={"en": True},
        )
    assert [r.seed for r in load_runs(tmp_path, "A_ssl", "3")] == [2, 10]


def test_the_macro_error_bar_is_across_seeds_not_across_languages(tmp_path: Path):
    """The headline number's ``±`` must be the quantity the caption promises.

    Two languages 30 points apart, each moving 2 points between two seeds. The
    macro-average moves 2 points between seeds, so its spread across seeds is
    sqrt(2) — the same as every per-language row. Aggregating across languages
    instead would report the 30-point gap between the languages as if it were
    run-to-run noise.
    """
    for seed, bump in ((0, 0.0), (1, 2.0)):
        write_run(
            tmp_path,
            "A_ssl",
            "3",
            seed,
            in_dist={"en": metrics(10.0 + bump, 4.0), "de": metrics(40.0 + bump, 9.0)},
            word_boundary={"en": True, "de": True},
        )

    agg = aggregate_runs(load_runs(tmp_path, "A_ssl", "3"))
    macro = agg.macro["in_distribution"]

    assert macro.mean == pytest.approx(26.0)
    assert macro.std == pytest.approx(math.sqrt(2.0))
    assert macro.per_seed == pytest.approx([25.0, 27.0])
    assert macro.n_seeds == 2
    assert macro.n_languages == 2
    # Between-language dispersion is a different quantity. It is kept, but
    # under its own name and never after a "±".
    assert macro.spread_across_languages == pytest.approx(21.213203435596427)


def test_the_macro_says_how_many_seeds_and_languages_it_covers(tmp_path: Path):
    write_run(
        tmp_path,
        "A_ssl",
        "3",
        0,
        in_dist={"en": metrics(10.0, 4.0), "de": metrics(40.0, 9.0)},
        word_boundary={"en": True, "de": True},
    )
    agg = aggregate_runs(load_runs(tmp_path, "A_ssl", "3"))
    text = to_markdown(agg)

    assert "2 language(s)" in text
    assert "1 seed(s)" in text
    # A lone seed has a mean but no spread, and "± nan" would read like a
    # computation that broke rather than one that was never possible.
    assert "± —" in text


def test_a_language_missing_from_one_seed_is_excluded_from_the_macro(tmp_path: Path):
    """A macro whose membership changes between seeds is not comparable seed to
    seed, so the incomplete language is dropped and named rather than quietly
    shifting the mean."""
    write_run(
        tmp_path,
        "A_ssl",
        "3",
        0,
        in_dist={"en": metrics(10.0, 4.0), "de": metrics(40.0, 9.0)},
        word_boundary={"en": True, "de": True},
    )
    write_run(
        tmp_path, "A_ssl", "3", 1, in_dist={"en": metrics(12.0, 5.0)}, word_boundary={"en": True}
    )

    agg = aggregate_runs(load_runs(tmp_path, "A_ssl", "3"))
    macro = agg.macro["in_distribution"]

    assert macro.n_languages == 1
    assert macro.excluded_languages == ["de"]
    assert macro.mean == pytest.approx(11.0)
    assert "de" in to_markdown(agg)


def test_the_json_says_which_quantity_the_macro_std_is(tmp_path: Path):
    write_run(
        tmp_path, "A_ssl", "3", 0, in_dist={"en": metrics(10.0, 4.0)}, word_boundary={"en": True}
    )
    macro = to_json(aggregate_runs(load_runs(tmp_path, "A_ssl", "3")))["macro"]["in_distribution"]

    assert macro["std_is"] == "across_seeds"
    assert "spread_across_languages" in macro
    assert macro["per_seed"] == [10.0]
