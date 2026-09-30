"""`svb compare`: the arms side by side, and what a difference between them is worth."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from svb.cli import build_parser
from svb.report.compare import compare_arms, load_arms, write_tables
from svb.stats.analysis import LanguageEdits, multibootstrap
from tests.test_report_aggregate import metrics, write_run
from tests.test_report_analyze import write_sidecar

REFS = ["a b c d"] * 40
#: Hypotheses with 0, 1 and 2 word errors against the reference above.
HYP = {0: "a b c d", 1: "a b c x", 2: "a b x y"}


def make_run(
    root: Path, arm: str, seed: int, errors: int, transfer_errors: int | None = None
) -> Path:
    """One run whose English WER is ``errors``/4 and whose Telugu is likewise."""
    transfer_errors = errors if transfer_errors is None else transfer_errors
    run = write_run(
        root,
        arm,
        "3",
        seed,
        in_dist={"en": metrics(25.0 * errors, 10.0 * errors)},
        transfer={"telugu": metrics(25.0 * transfer_errors, 10.0 * transfer_errors)},
        word_boundary={"en": True, "telugu": True},
    )
    write_sidecar(run, "en", [(r, HYP[errors]) for r in REFS], transfer=False)
    write_sidecar(run, "telugu", [(r, HYP[transfer_errors]) for r in REFS], transfer=True)
    return run


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for seed in (11, 22, 33):
        make_run(tmp_path, "A_ssl", seed, errors=2)
        make_run(tmp_path, "B_btm_ssl", seed, errors=1)
    return tmp_path


def test_a_consistent_difference_is_reported_with_an_interval_that_excludes_zero(
    root: Path,
) -> None:
    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl", "C_btm_scratch"])
    contrasts = compare_arms(arms, n_resamples=500)

    assert [arm.arm for arm in arms] == ["A_ssl", "B_btm_ssl"]  # C has no runs yet
    assert {(c.a, c.b, c.section) for c in contrasts} == {
        ("A_ssl", "B_btm_ssl", "in_distribution"),
        ("A_ssl", "B_btm_ssl", "transfer"),
    }
    transfer = next(c for c in contrasts if c.section == "transfer")
    (telugu,) = transfer.rows
    # Arm A makes two errors in four words and arm B one: 50% against 25%.
    assert telugu.delta == pytest.approx(25.0)
    assert telugu.excludes_zero
    assert transfer.macro is not None and transfer.macro.delta == pytest.approx(25.0)


def test_seed_to_seed_variation_widens_the_interval(tmp_path: Path) -> None:
    """The reason to resample seeds at all: a difference that one seed of three
    reverses is not the same finding as one every seed agrees on."""
    steady, noisy = tmp_path / "steady", tmp_path / "noisy"
    for seed in (11, 22, 33):
        make_run(steady, "A_ssl", seed, errors=2)
        make_run(steady, "B_btm_ssl", seed, errors=1)
        make_run(noisy, "A_ssl", seed, errors=2)
    for seed, errors in ((11, 0), (22, 1), (33, 2)):
        make_run(noisy, "B_btm_ssl", seed, errors=errors)

    def telugu(root: Path):
        arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
        contrast = next(c for c in compare_arms(arms, n_resamples=500) if c.section == "transfer")
        return contrast.rows[0]

    assert telugu(steady).delta == pytest.approx(telugu(noisy).delta)
    assert (telugu(noisy).hi - telugu(noisy).lo) > (telugu(steady).hi - telugu(steady).lo)
    assert not telugu(noisy).excludes_zero


def test_arms_scored_on_different_references_are_not_contrasted(root: Path) -> None:
    run = root / "B_btm_ssl" / "3" / "seed22"
    write_sidecar(run, "telugu", [("a b c e", "a b c e")] * 40, transfer=True)

    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
    with pytest.raises(ValueError, match="different references"):
        compare_arms(arms, n_resamples=50)


def test_arms_under_different_normalization_are_refused_before_any_alignment(
    tmp_path: Path,
) -> None:
    for arm, policy in (("A_ssl", "whisper-basic"), ("B_btm_ssl", "latin-marks")):
        write_run(
            tmp_path,
            arm,
            "3",
            11,
            in_dist={"en": metrics(1.0, 1.0)},
            word_boundary={"en": True},
            policies={"en": policy},
        )

    arms = load_arms(tmp_path, "3", ["A_ssl", "B_btm_ssl"])
    with pytest.raises(ValueError, match="normalized these languages differently"):
        compare_arms(arms, n_resamples=50)


def test_a_language_one_run_did_not_evaluate_is_left_out_and_named(root: Path) -> None:
    (root / "B_btm_ssl" / "3" / "seed33" / "transfer" / "telugu" / "predictions.json").unlink()

    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
    transfer = next(c for c in compare_arms(arms, n_resamples=50) if c.section == "transfer")

    assert transfer.rows == []
    assert transfer.excluded == ["telugu"]


def test_a_language_reported_but_scored_nowhere_is_named_not_dropped(root: Path) -> None:
    """The arm macros include it and the contrast cannot, and the table has to
    say so rather than print two macros that are not over the same set."""
    for arm in ("A_ssl", "B_btm_ssl"):
        for seed in (11, 22, 33):
            (root / arm / "3" / f"seed{seed}" / "transfer" / "telugu" / "predictions.json").unlink()

    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
    transfer = next(c for c in compare_arms(arms, n_resamples=50) if c.section == "transfer")

    assert transfer.rows == []
    assert transfer.excluded == ["telugu"]


def test_a_sidecar_for_a_language_the_run_did_not_report_is_not_contrasted(root: Path) -> None:
    """A --restart after a language was dropped from a preset leaves its old
    sidecar behind; the run's results.json is what says what was scored."""
    for arm in ("A_ssl", "B_btm_ssl"):
        for seed in (11, 22, 33):
            write_sidecar(
                root / arm / "3" / f"seed{seed}", "stale", [("a b", "a b")] * 4, transfer=False
            )

    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
    in_dist = next(c for c in compare_arms(arms, n_resamples=50) if c.section == "in_distribution")

    assert [row.code for row in in_dist.rows] == ["en"]
    assert "stale" not in in_dist.excluded


def test_an_arm_that_records_no_policies_is_refused(root: Path) -> None:
    for seed in (11, 22, 33):
        (root / "B_btm_ssl" / "3" / f"seed{seed}" / "text_stats.json").unlink()

    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
    with pytest.raises(ValueError, match="record no per-language policies"):
        compare_arms(arms, n_resamples=50)


def test_every_pair_of_arms_is_checked_for_policy_agreement(tmp_path: Path) -> None:
    """Checking each arm against the first would let B and C disagree unseen:
    here A shares no language with either, and only B and C differ."""
    for arm, code, policy in (
        ("A_ssl", "de", "whisper-basic"),
        ("B_btm_ssl", "en", "whisper-basic"),
        ("C_btm_scratch", "en", "latin-marks"),
    ):
        write_run(
            tmp_path,
            arm,
            "3",
            11,
            in_dist={code: metrics(1.0, 1.0)},
            word_boundary={code: True},
            policies={code: policy},
        )

    arms = load_arms(tmp_path, "3", ["A_ssl", "B_btm_ssl", "C_btm_scratch"])
    with pytest.raises(ValueError, match="B_btm_ssl and C_btm_scratch normalized"):
        compare_arms(arms, n_resamples=50)


def test_arms_that_disagree_on_a_primary_metric_say_so_in_the_table(
    root: Path, tmp_path: Path
) -> None:
    """One arm scored Telugu by characters. Its cell is a CER beside a WER, and
    the row must not carry one arm's label for both."""
    for seed in (11, 22, 33):
        run = root / "B_btm_ssl" / "3" / f"seed{seed}"
        stats = json.loads((run / "text_stats.json").read_text(encoding="utf-8"))
        stats["languages"]["telugu"]["word_boundary"] = False
        (run / "text_stats.json").write_text(json.dumps(stats), encoding="utf-8")
        (run / "transfer" / "telugu" / "predictions.json").unlink()

    out = tmp_path / "tables"
    write_tables(root, "3", ["A_ssl", "B_btm_ssl"], out, n_resamples=50)

    body = (out / "3_arms.md").read_text(encoding="utf-8")
    assert "| telugu | CER/WER (arms disagree) |" in body
    assert "different primary metrics for telugu" in body


def test_no_finished_runs_is_an_error_not_an_empty_table(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no finished runs"):
        load_arms(tmp_path, "3", ["A_ssl"])


def test_the_tables_name_every_arm_and_every_contrast(root: Path, tmp_path: Path) -> None:
    make_run(root, "C_btm_scratch", 11, errors=2)
    out = tmp_path / "tables"

    written = write_tables(root, "3", ["A_ssl", "B_btm_ssl", "C_btm_scratch"], out, n_resamples=100)

    assert {path.name for path in written} == {"3_arms.md", "3_arms.json"}
    body = (out / "3_arms.md").read_text(encoding="utf-8")
    for heading in ("A_ssl − B_btm_ssl", "A_ssl − C_btm_scratch", "B_btm_ssl − C_btm_scratch"):
        assert f"### {heading}" in body
    assert "not matched on training budget" in body
    payload = json.loads((out / "3_arms.json").read_text(encoding="utf-8"))
    assert set(payload["arms"]) == {"A_ssl", "B_btm_ssl", "C_btm_scratch"}
    assert len(payload["contrasts"]) == 6  # three pairs, two sections


def test_the_command_writes_the_tables(root: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "tables"
    argv = ["compare", "--scale", "3", "--results-root", str(root)]
    argv += ["--tables-dir", str(out), "--resamples", "100"]
    args = build_parser().parse_args(argv)
    args.func(args)

    assert (out / "3_arms.md").is_file()
    assert "3_arms.md" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The statistic itself
# --------------------------------------------------------------------------- #


def _language(rates_a: list[float], rates_b: list[float], n_utts: int = 200) -> LanguageEdits:
    """Runs whose per-utterance error count is constant, so only seeds vary."""
    lengths = np.full(n_utts, 10.0)
    return LanguageEdits(
        lengths=lengths,
        edits_a=np.array([np.full(n_utts, 10.0 * r) for r in rates_a]),
        edits_b=np.array([np.full(n_utts, 10.0 * r) for r in rates_b]),
    )


def test_the_point_estimate_is_the_difference_of_the_arm_means() -> None:
    per_language, macro = multibootstrap([_language([0.30, 0.40], [0.20, 0.20])], n=200)

    assert per_language[0].delta == pytest.approx(15.0)
    assert macro.delta == pytest.approx(15.0)


def test_identical_arms_give_an_interval_around_zero() -> None:
    rng = np.random.default_rng(0)
    edits = rng.binomial(10, 0.2, size=(4, 300)).astype(float)
    language = LanguageEdits(lengths=np.full(300, 10.0), edits_a=edits, edits_b=edits.copy())

    (contrast,), _ = multibootstrap([language], n=500)

    assert contrast.delta == pytest.approx(0.0)
    assert contrast.lo <= 0.0 <= contrast.hi
    assert contrast.p_value > 0.5


def test_the_macro_is_the_unweighted_mean_over_languages() -> None:
    """A language with ten times the utterances does not count ten times."""
    big = _language([0.30], [0.10], n_utts=1000)
    small = _language([0.10], [0.30], n_utts=100)

    _, macro = multibootstrap([big, small], n=200)

    assert macro.delta == pytest.approx(0.0)


def _noisy(rng: np.random.Generator, n_utts: int = 400, rate: float = 0.2) -> np.ndarray:
    """One run whose per-utterance edit counts vary, so utterances matter."""
    return rng.binomial(10, rate, size=(1, n_utts)).astype(float)


# The four structural claims of the statistic, each pinned by a case that only
# holds if the claim does. Together they fail under: no utterance resampling,
# a run pick drawn per language, a separate utterance draw per arm, and a macro
# taken from one language's draws.


def test_utterances_are_resampled() -> None:
    """One run per arm, so seeds contribute nothing: the interval's width is
    the utterance draw's, and without one it would be a point."""
    rng = np.random.default_rng(1)
    language = LanguageEdits(np.full(400, 10.0), _noisy(rng), _noisy(rng, rate=0.3))

    (contrast,), _ = multibootstrap([language], n=300)

    assert contrast.hi - contrast.lo > 0.5


def test_one_utterance_draw_serves_both_arms() -> None:
    """Two identical arms differ by exactly nothing in every draw — which is
    only so if the same utterances are drawn for both."""
    rng = np.random.default_rng(2)
    edits = _noisy(rng)
    language = LanguageEdits(np.full(400, 10.0), edits, edits.copy())

    (contrast,), macro = multibootstrap([language], n=300)

    assert (contrast.lo, contrast.hi) == (0.0, 0.0)
    assert (macro.lo, macro.hi) == (0.0, 0.0)


def test_runs_are_drawn_once_for_every_language() -> None:
    """Two languages with the same per-run rates and no utterance noise give
    the same draws only if each draw picks the same runs for both."""
    first = _language([0.30, 0.50, 0.40], [0.20, 0.25, 0.45], n_utts=100)
    second = _language([0.30, 0.50, 0.40], [0.20, 0.25, 0.45], n_utts=100)

    (one, two), _ = multibootstrap([first, second], n=300)

    assert (one.lo, one.hi) == (two.lo, two.hi)


def test_utterance_draws_differ_between_languages() -> None:
    """Two languages holding the same noisy edits are still different test
    sets, and their draws must not move together."""
    rng = np.random.default_rng(3)
    edits = _noisy(rng)
    zeros = np.zeros_like(edits)
    twins = [LanguageEdits(np.full(400, 10.0), edits, zeros) for _ in range(2)]

    (one, two), _ = multibootstrap(twins, n=300)

    assert (one.lo, one.hi) != (two.lo, two.hi)


def test_the_macro_draws_are_the_mean_of_the_languages_draws() -> None:
    """Two languages whose contrasts cancel exactly, with nothing to resample,
    give a macro of exactly zero — not the first language's interval."""
    up = _language([0.30], [0.10], n_utts=100)
    down = _language([0.10], [0.30], n_utts=100)

    (first, _), macro = multibootstrap([up, down], n=200)

    assert (first.lo, first.hi) == pytest.approx((20.0, 20.0))
    assert (macro.lo, macro.hi) == pytest.approx((0.0, 0.0))


def test_a_language_with_no_utterances_is_refused() -> None:
    empty = LanguageEdits(np.zeros(0), np.zeros((1, 0)), np.zeros((1, 0)))

    with pytest.raises(ValueError, match="no utterances"):
        multibootstrap([empty], n=10)


def test_languages_must_carry_the_same_runs() -> None:
    with pytest.raises(ValueError, match="same runs"):
        multibootstrap([_language([0.1, 0.2], [0.1]), _language([0.1], [0.1])], n=10)


def test_the_draws_are_reproducible() -> None:
    language = _language([0.30, 0.50, 0.40], [0.20, 0.25, 0.45])

    assert multibootstrap([language], n=300, seed=5) == multibootstrap([language], n=300, seed=5)


def test_the_coverage_script_runs_the_real_statistic(interval_coverage: ModuleType) -> None:
    """The figures quoted in the docs come from this script; it has to keep running."""
    regime = interval_coverage.Regime("tiny", seeds=3, languages=2, utterances=40, seed_sd=0.1)

    per_language, macro = interval_coverage.coverage(regime, reps=3, draws=30, seed=0)

    assert 0.0 <= per_language <= 1.0
    assert 0.0 <= macro <= 1.0


def test_a_single_run_without_recorded_policies_is_enough_to_refuse(root: Path) -> None:
    """The aggregate carries the first seed's policies; the check must look at
    every seed, or one run written before policies were recorded slips by."""
    (root / "B_btm_ssl" / "3" / "seed33" / "text_stats.json").unlink()

    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
    with pytest.raises(ValueError, match=r"B_btm_ssl seed\(s\) \[33\] record no"):
        compare_arms(arms, n_resamples=50)


def test_an_empty_sidecar_is_named_by_language(root: Path) -> None:
    for arm in ("A_ssl", "B_btm_ssl"):
        for seed in (11, 22, 33):
            write_sidecar(root / arm / "3" / f"seed{seed}", "telugu", [], transfer=True)

    arms = load_arms(root, "3", ["A_ssl", "B_btm_ssl"])
    with pytest.raises(ValueError, match="telugu: a prediction sidecar with no pairs"):
        compare_arms(arms, n_resamples=50)
