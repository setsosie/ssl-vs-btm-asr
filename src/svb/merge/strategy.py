"""Model-merging strategies for BTM experts.

Operates on parameter ``state_dict``s. Task-vector methods (TIES, DARE-TIES)
need the pre-branch base (the phase-0 checkpoint): the task vector for an expert
is ``expert - base``.

The paper's finding #2 is that AVERAGE is the only safe strategy at scale, and
that the sign-election penalty in TIES/DARE-TIES tracks linguistic typological
distance. These implementations are the standard ones (Yadav et al. 2023; Yu et
al. 2024) so that finding is not an artifact of a bespoke variant.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

StateDict = dict[str, torch.Tensor]


def _mergeable_keys(experts: list[StateDict]) -> list[str]:
    """Keys present in all experts with a floating-point dtype."""
    common = set(experts[0])
    for e in experts[1:]:
        common &= set(e)
    return [k for k in experts[0] if k in common and experts[0][k].is_floating_point()]


def average(experts: list[StateDict], **_: object) -> StateDict:
    """Elementwise mean of expert parameters (== model soup)."""
    keys = _mergeable_keys(experts)
    out: StateDict = {k: experts[0][k].clone() for k in experts[0]}
    for k in keys:
        stacked = torch.stack([e[k].float() for e in experts], dim=0)
        out[k] = stacked.mean(dim=0).to(experts[0][k].dtype)
    return out


def _task_vectors(experts: list[StateDict], base: StateDict, keys: list[str]) -> dict[str, torch.Tensor]:
    """Stack per-expert task vectors (expert - base) as (n_experts, ...)."""
    return {k: torch.stack([e[k].float() - base[k].float() for e in experts], dim=0) for k in keys}


def _trim(tv: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top-``density`` fraction of entries by magnitude per expert."""
    if density >= 1.0:
        return tv
    flat = tv.reshape(tv.shape[0], -1)
    k = max(1, int(flat.shape[1] * density))
    thresh = flat.abs().kthvalue(flat.shape[1] - k + 1, dim=1, keepdim=True).values
    mask = flat.abs() >= thresh
    return (flat * mask).reshape(tv.shape)


def _ties_combine(tv: torch.Tensor) -> torch.Tensor:
    """Elect a sign per parameter, then average agreeing experts.

    ``tv`` is (n_experts, ...). Returns the merged task vector (...).
    """
    sign = torch.sign(tv.sum(dim=0))  # elected sign per element
    agree = (torch.sign(tv) == sign) & (tv != 0)
    summed = (tv * agree).sum(dim=0)
    count = agree.sum(dim=0).clamp(min=1)
    return summed / count


def ties(
    experts: list[StateDict], base: StateDict | None, density: float = 0.2, **_: object
) -> StateDict:
    """TIES-merging: trim → elect sign → disjoint-average → add base."""
    if base is None:
        raise ValueError("TIES requires a base (phase-0) state_dict")
    keys = _mergeable_keys(experts)
    tvs = _task_vectors(experts, base, keys)
    out: StateDict = {k: experts[0][k].clone() for k in experts[0]}
    for k in keys:
        merged = _ties_combine(_trim(tvs[k], density))
        out[k] = (base[k].float() + merged).to(experts[0][k].dtype)
    return out


def dare_ties(
    experts: list[StateDict],
    base: StateDict | None,
    density: float = 0.2,
    drop_p: float = 0.5,
    seed: int = 0,
    **_: object,
) -> StateDict:
    """DARE drop-and-rescale on task vectors, then TIES."""
    if base is None:
        raise ValueError("DARE-TIES requires a base (phase-0) state_dict")
    keys = _mergeable_keys(experts)
    tvs = _task_vectors(experts, base, keys)
    gen = torch.Generator().manual_seed(seed)
    out: StateDict = {k: experts[0][k].clone() for k in experts[0]}
    for k in keys:
        tv = tvs[k]
        mask = (torch.rand(tv.shape, generator=gen) >= drop_p).float()
        tv = tv * mask / (1.0 - drop_p)
        merged = _ties_combine(_trim(tv, density))
        out[k] = (base[k].float() + merged).to(experts[0][k].dtype)
    return out


MERGE_STRATEGIES: dict[str, Callable[..., StateDict]] = {
    "average": average,
    "ties": ties,
    "dare_ties": dare_ties,
}
