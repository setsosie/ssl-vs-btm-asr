"""Merge strategies: identities, sign election, trimming, seeding, head scope."""

import pytest
import torch

from svb.merge.strategy import _trim, average, dare_ties, ties


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


def test_non_float_entries_pass_through_untouched():
    """Integer buffers are not parameters and must survive the merge as-is.

    The previous test of this name contained no non-float tensor at all, so the
    passthrough path had never been exercised.
    """
    base = _sd(w=[0.0, 0.0])
    steps = torch.tensor([7, 7], dtype=torch.int64)
    e1 = {**_sd(w=[1.0, 1.0]), "steps": steps.clone()}
    e2 = {**_sd(w=[1.0, -1.0]), "steps": steps.clone()}

    out = ties([e1, e2], base=base, density=1.0)

    assert out["steps"].dtype == torch.int64
    assert torch.equal(out["steps"], steps)


def test_disagreeing_non_float_entries_are_refused():
    """Silently taking experts[0]'s copy would hide a real mismatch."""
    base = _sd(w=[0.0])
    e1 = {**_sd(w=[1.0]), "steps": torch.tensor([1], dtype=torch.int64)}
    e2 = {**_sd(w=[1.0]), "steps": torch.tensor([2], dtype=torch.int64)}

    with pytest.raises(ValueError, match="steps"):
        ties([e1, e2], base=base, density=1.0)


def test_ties_sign_election_zeros_conflicts():
    # Element 1: experts disagree (+1 vs -1) -> elected sign breaks the tie by
    # the sum (0 -> sign 0) so the conflicting entry collapses toward zero.
    base = _sd(w=[0.0, 0.0])
    e1 = _sd(w=[1.0, 1.0])
    e2 = _sd(w=[1.0, -1.0])
    out = ties([e1, e2], base=base, density=1.0)["w"]
    assert out[0] == 1.0  # agreeing dimension survives
    assert out[1] == 0.0  # conflicting dimension elected to zero


def test_trim_keeps_exactly_the_density_fraction_under_ties():
    """Equal magnitudes must not defeat the density budget.

    A threshold test of `>= kthvalue` keeps every entry whenever the cut value
    repeats — which it does for a task vector of uniform magnitude, and for one
    padded with the exact zeros of an untouched parameter. TIES would then trim
    nothing while reporting a density of 0.3.
    """
    tv = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])

    trimmed = _trim(tv, density=0.3)

    assert int((trimmed != 0).sum()) == 3


def test_trim_keeps_the_largest_entries():
    tv = torch.tensor([[0.1, 5.0, 0.2, 4.0]])
    trimmed = _trim(tv, density=0.5)
    assert torch.equal(trimmed, torch.tensor([[0.0, 5.0, 0.0, 4.0]]))


def test_dare_ties_is_deterministic_given_a_seed():
    base = _sd(w=[0.0] * 8)
    e1 = _sd(w=[1.0] * 8)
    e2 = _sd(w=[2.0] * 8)
    o1 = dare_ties([e1, e2], base=base, density=1.0, drop_p=0.5, seed=0)["w"]
    o2 = dare_ties([e1, e2], base=base, density=1.0, drop_p=0.5, seed=0)["w"]
    assert torch.allclose(o1, o2)


def test_dare_ties_actually_varies_with_the_seed():
    """DARE's drop mask is its only source of stochasticity.

    With the seed pinned at its default the mask was byte-identical across all
    five run seeds, so the reported std for the DARE arm measured everything
    except the thing DARE does.
    """
    base = _sd(w=[0.0] * 64)
    e1 = _sd(w=[1.0] * 64)
    e2 = _sd(w=[2.0] * 64)

    o0 = dare_ties([e1, e2], base=base, density=1.0, drop_p=0.5, seed=0)["w"]
    o1 = dare_ties([e1, e2], base=base, density=1.0, drop_p=0.5, seed=1)["w"]

    assert not torch.allclose(o0, o1)


def test_merge_head_false_takes_the_head_from_the_base():
    """Encoder-only merging is a protocol choice the paper has to be able to make."""
    base = {**_sd(w=[0.0, 0.0]), **_sd(**{"ctc_proj.weight": [9.0, 9.0]})}
    e1 = {**_sd(w=[1.0, 1.0]), **_sd(**{"ctc_proj.weight": [1.0, 1.0]})}
    e2 = {**_sd(w=[3.0, 3.0]), **_sd(**{"ctc_proj.weight": [3.0, 3.0]})}

    merged = average([e1, e2], base=base, merge_head=False)
    both = average([e1, e2], base=base, merge_head=True)

    assert torch.allclose(merged["w"], torch.tensor([2.0, 2.0]))  # encoder merged
    assert torch.allclose(merged["ctc_proj.weight"], base["ctc_proj.weight"])
    assert torch.allclose(both["ctc_proj.weight"], torch.tensor([2.0, 2.0]))


def test_merge_head_false_needs_a_base():
    e1 = {**_sd(w=[1.0]), **_sd(**{"ctc_norm.weight": [1.0]})}
    with pytest.raises(ValueError, match="base"):
        average([e1, e1], base=None, merge_head=False)


def test_merge_experts_threads_the_run_seed(tmp_path):
    """The pipeline must hand DARE the run's seed, not the function default."""
    from svb.btm.pipeline import merge_experts

    base_path = tmp_path / "base.pt"
    torch.save({"w": torch.zeros(64)}, base_path)
    experts = {}
    for name, value in (("a", 1.0), ("b", 2.0)):
        path = tmp_path / f"{name}.pt"
        torch.save({"w": torch.full((64,), value)}, path)
        experts[name] = path

    merged = []
    for seed in (0, 1):
        out = merge_experts(
            experts,
            "dare_ties",
            tmp_path / f"seed{seed}",
            base_ckpt=base_path,
            seed=seed,
        )
        merged.append(torch.load(out, weights_only=True)["w"])

    assert not torch.allclose(merged[0], merged[1])
