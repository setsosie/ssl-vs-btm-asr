"""Statistics: seed aggregation, bootstrap CI, paired permutation."""

import math

import pytest

from svb.stats.analysis import aggregate_seeds, bootstrap_ci, paired_permutation


def test_aggregate_seeds():
    agg = aggregate_seeds([10.0, 12.0, 14.0])
    assert agg.mean == 12.0
    assert agg.n_seeds == 3
    assert abs(agg.std - 2.0) < 1e-9  # sample std of {10,12,14}


def test_aggregate_single_seed_std_nan():
    agg = aggregate_seeds([5.0])
    assert agg.mean == 5.0
    assert math.isnan(agg.std)


def test_bootstrap_ci_brackets_point():
    refs = ["the cat sat", "a dog ran", "hello world", "good morning"]
    hyps = ["the cat sat", "a dog run", "hello word", "good morning"]
    ci = bootstrap_ci(refs, hyps, n=200, seed=0)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.lo >= 0.0 and ci.hi <= 100.0


def test_paired_permutation_identical_systems_not_significant():
    refs = ["the cat sat", "a dog ran", "hello world"]
    p = paired_permutation(refs, hyps_a=refs, hyps_b=refs, n=200, seed=0)
    assert p > 0.05  # no difference -> not significant


def test_paired_permutation_clear_winner():
    refs = ["the cat sat on the mat"] * 8
    good = list(refs)
    bad = ["completely wrong text here now"] * 8
    p = paired_permutation(refs, hyps_a=good, hyps_b=bad, n=500, seed=0)
    assert p < 0.05  # A clearly better -> significant


def test_corpus_error_rate_matches_jiwer_word_for_word():
    """The statistics layer and the evaluator must not disagree on a number."""
    import jiwer

    from svb.stats.analysis import corpus_error_rate

    refs = ["the quick brown fox", "hello world", "a b c d e"]
    hyps = ["the quick brown box", "hello", "a x c d"]

    # approx, not equality: the two differ in the last bit because they scale by
    # 100 at different points in the division, not because they disagree.
    assert corpus_error_rate(refs, hyps) == pytest.approx(100.0 * jiwer.wer(refs, hyps))
    assert corpus_error_rate(refs, hyps, tokenize="char") == pytest.approx(
        100.0 * jiwer.cer(refs, hyps)
    )


def test_word_tokenization_is_meaningless_for_a_script_without_spaces():
    """The reason the tokenization is a parameter rather than a constant.

    Whitespace tokenization of Japanese yields one token per sentence, so word
    error rate can only ever be 0 or 100 per utterance.
    """
    from svb.stats.analysis import corpus_error_rate

    refs = ["コーヒーを飲む", "今日はいい天気"]
    hyps = ["コーヒーを飲んだ", "今日はいい天気"]

    assert corpus_error_rate(refs, hyps) == 50.0
    assert corpus_error_rate(refs, hyps, tokenize="char") < 20.0


def test_bootstrap_point_estimate_is_the_reported_error_rate():
    from svb.stats.analysis import corpus_error_rate

    refs = ["the cat sat", "a dog ran", "hello world"]
    hyps = ["the cat sat", "a dog run", "hello word"]

    ci = bootstrap_ci(refs, hyps, n=200, seed=0)
    assert ci.point == corpus_error_rate(refs, hyps)


def test_bootstrap_can_resample_characters():
    refs = ["コーヒーを飲む", "今日はいい天気", "犬が走る"]
    hyps = ["コーヒーを飲んだ", "今日はいい天気", "猫が走る"]

    ci = bootstrap_ci(refs, hyps, n=200, seed=0, tokenize="char")

    assert ci.lo <= ci.point <= ci.hi
    assert ci.point < 50.0  # word tokenization would say 66.7


def test_bootstrap_is_reproducible_from_the_seed():
    refs = ["the cat sat", "a dog ran", "hello world", "good morning"]
    hyps = ["the cat sat", "a dog run", "hello word", "good morning"]

    first = bootstrap_ci(refs, hyps, n=300, seed=7)
    second = bootstrap_ci(refs, hyps, n=300, seed=7)

    assert (first.lo, first.point, first.hi) == (second.lo, second.point, second.hi)


def test_paired_permutation_can_compare_on_characters():
    refs = ["コーヒーを飲む"] * 8
    good = list(refs)
    bad = ["まったく違う文章"] * 8

    p = paired_permutation(refs, hyps_a=good, hyps_b=bad, n=500, seed=0, tokenize="char")

    assert p < 0.05


def test_paired_permutation_is_reproducible_from_the_seed():
    refs = ["the cat sat on the mat"] * 6
    bad = ["the cat sat on a mat"] * 6

    assert paired_permutation(refs, refs, bad, n=300, seed=1) == paired_permutation(
        refs, refs, bad, n=300, seed=1
    )


def test_an_empty_evaluation_does_not_divide_by_zero():
    from svb.stats.analysis import corpus_error_rate

    assert corpus_error_rate([], []) == 0.0
