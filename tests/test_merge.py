"""Merge strategies: identities and shape/sign behavior on tiny tensors."""

import torch

from svb.merge.strategy import average, dare_ties, ties


def _sd(**kw):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in kw.items()}


def test_average_of_identical_is_identity():
    a = _sd(w=[1.0, 2.0, 3.0])
    out = average([a, {k: v.clone() for k, v in a.items()}])
    assert torch.allclose(out["w"], a["w"])


def test_average_is_mean():
    a = _sd(w=[0.0, 0.0])
    b = _sd(w=[2.0, 4.0])
    out = average([a, b])
    assert torch.allclose(out["w"], torch.tensor([1.0, 2.0]))


def test_average_equals_task_arithmetic_identity():
    # AVERAGE == base + mean(task vectors) at lambda = 1/N. This is the algebraic
    # identity behind "AVERAGE == TASK_ARITHMETIC == DARE_LINEAR" in the paper.
    base = _sd(w=[1.0, 1.0])
    e1 = _sd(w=[2.0, 0.0])
    e2 = _sd(w=[0.0, 4.0])
    avg = average([e1, e2])["w"]
    tv_mean = ((e1["w"] - base["w"]) + (e2["w"] - base["w"])) / 2
    assert torch.allclose(avg, base["w"] + tv_mean)


def test_ties_preserves_non_float_and_shapes():
    base = _sd(w=[0.0, 0.0, 0.0, 0.0])
    e1 = _sd(w=[1.0, 1.0, 0.0, 0.0])
    e2 = _sd(w=[1.0, -1.0, 1.0, 0.0])
    out = ties([e1, e2], base=base, density=1.0)
    assert out["w"].shape == base["w"].shape


def test_ties_sign_election_zeros_conflicts():
    # Element 1: experts disagree (+1 vs -1) -> elected sign breaks the tie by
    # the sum (0 -> sign 0) so the conflicting entry collapses toward zero.
    base = _sd(w=[0.0, 0.0])
    e1 = _sd(w=[1.0, 1.0])
    e2 = _sd(w=[1.0, -1.0])
    out = ties([e1, e2], base=base, density=1.0)["w"]
    assert out[0] == 1.0  # agreeing dimension survives
    assert out[1] == 0.0  # conflicting dimension elected to zero


def test_dare_ties_runs_and_is_seeded():
    base = _sd(w=[0.0] * 8)
    e1 = _sd(w=[1.0] * 8)
    e2 = _sd(w=[2.0] * 8)
    o1 = dare_ties([e1, e2], base=base, density=1.0, drop_p=0.5, seed=0)["w"]
    o2 = dare_ties([e1, e2], base=base, density=1.0, drop_p=0.5, seed=0)["w"]
    assert torch.allclose(o1, o2)  # deterministic given seed
