"""Statistics: seed aggregation, bootstrap CI, paired permutation."""

import math

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
    assert 0.0 <= ci.lo and ci.hi <= 100.0


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
