"""Model-merging strategies for BTM experts.

Operates on parameter ``state_dict``s. Task-vector methods (TIES, DARE-TIES)
need the pre-branch base (the phase-0 checkpoint): the task vector for an expert
is ``expert - base``.

The paper's finding #2 is that AVERAGE is the only safe strategy at scale, and
that the sign-election penalty in TIES/DARE-TIES tracks linguistic typological
distance. These implementations are the standard ones (Yadav et al. 2023; Yu et
al. 2024) so that finding is not an artifact of a bespoke variant.

**What gets merged.** By default every floating-point tensor is merged,
*including the CTC head* (``ctc_proj.*`` and ``ctc_norm.*``) — not just the
encoder. That is a research decision, not an implementation detail: the experts
share one vocabulary, so their heads are commensurable, and merging them keeps
the merged model usable without a further head-fitting step. Pass
``merge_head=False`` to merge the encoder only and take the head from the base
(phase-0) checkpoint, which is the natural ablation for asking how much of the
merging penalty lives in the head.

Non-floating-point entries (integer buffers, counters) cannot be averaged. They
are carried through from the first expert, and the merge refuses to run if the
experts disagree about one, since that would mean silently discarding a real
difference.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

StateDict = dict[str, torch.Tensor]

# The CTC head, as named by ``XeusCTC``.
HEAD_PREFIXES = ("ctc_proj.", "ctc_norm.")


def _is_head_key(key: str) -> bool:
    return key.startswith(HEAD_PREFIXES)


def _mergeable_keys(experts: list[StateDict], merge_head: bool = True) -> list[str]:
    """Keys present in all experts with a floating-point dtype."""
    common = set(experts[0])
    for e in experts[1:]:
        common &= set(e)
    keys = [k for k in experts[0] if k in common and experts[0][k].is_floating_point()]
    if not merge_head:
        keys = [k for k in keys if not _is_head_key(k)]
    return keys


def _scaffold(experts: list[StateDict], base: StateDict | None, merge_head: bool) -> StateDict:
    """The output dict before any merged values are written into it.

    Starts as a copy of the first expert, which supplies the entries no
    strategy touches. Non-float entries are checked for agreement first: taking
    experts[0]'s copy of a value the experts disagree about would discard a
    real difference without a word.
    """
    for key, value in experts[0].items():
        if value.is_floating_point():
            continue
        for other in experts[1:]:
            if key in other and not torch.equal(value, other[key]):
                raise ValueError(
                    f"experts disagree on the non-float entry {key!r}, which the "
                    "merge would silently take from the first expert"
                )

    out: StateDict = {k: v.clone() for k, v in experts[0].items()}
    if not merge_head:
        if base is None:
            raise ValueError("merge_head=False needs a base (phase-0) state_dict for the head")
        for key in out:
            if not _is_head_key(key):
                continue
            if key not in base:
                # Falling back to experts[0] here would leave the head half from
                # phase 0 and half from one arbitrary expert, which is neither
                # protocol and would not show up in any artifact.
                raise ValueError(
                    f"merge_head=False takes the head from the base, but the base has no "
                    f"{key!r}; it is not the phase-0 checkpoint these experts branched from"
                )
            out[key] = base[key].clone()
    return out


def average(
    experts: list[StateDict],
    base: StateDict | None = None,
    merge_head: bool = True,
    **_: object,
) -> StateDict:
    """Elementwise mean of expert parameters (== model soup)."""
    keys = _mergeable_keys(experts, merge_head)
    out = _scaffold(experts, base, merge_head)
    for k in keys:
        stacked = torch.stack([e[k].float() for e in experts], dim=0)
        out[k] = stacked.mean(dim=0).to(experts[0][k].dtype)
    return out


def _task_vector(experts: list[StateDict], base: StateDict, key: str) -> torch.Tensor:
    """Stack one key's per-expert task vectors (expert - base) as (n_experts, ...).

    One key at a time, deliberately. Building the whole dict up front costs
    ``n_experts`` float32 copies of the entire model on top of the experts
    themselves: for a 577M-parameter encoder that is about 2.3 GB per expert,
    so the 64-language tier this study targets would need roughly 150 GB for
    the task vectors alone and would never reach the merge. Streaming holds one
    key's stack instead, which is bounded by the largest single parameter.
    """
    return torch.stack([e[key].float() - base[key].float() for e in experts], dim=0)


def _trim(tv: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top-``density`` fraction of entries by magnitude per expert.

    Selection is by ``topk`` rather than by comparing against the k-th largest
    magnitude. A threshold comparison keeps every tied entry, so a task vector
    of uniform magnitude — or one padded with the exact zeros of a parameter
    the expert never moved — would be kept in full while reporting that it had
    been trimmed to ``density``.

    ``topk``'s tie-break among equal magnitudes is unspecified and may differ
    between the CPU and CUDA kernels. That is not a live risk only because
    merging is CPU-only by construction (``merge_experts`` refuses any other
    device); moving it to a GPU would quietly change which entries survive.
    """
    if density >= 1.0:
        return tv
    flat = tv.reshape(tv.shape[0], -1)
    k = max(1, int(flat.shape[1] * density))
    keep = flat.abs().topk(k, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, keep, True)
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
    experts: list[StateDict],
    base: StateDict | None,
    density: float = 0.2,
    merge_head: bool = True,
    **_: object,
) -> StateDict:
    """TIES-merging: trim → elect sign → disjoint-average → add base."""
    if base is None:
        raise ValueError("TIES requires a base (phase-0) state_dict")
    keys = _mergeable_keys(experts, merge_head)
    out = _scaffold(experts, base, merge_head)
    for k in keys:
        merged = _ties_combine(_trim(_task_vector(experts, base, k), density))
        out[k] = (base[k].float() + merged).to(experts[0][k].dtype)
    return out


def dare_ties(
    experts: list[StateDict],
    base: StateDict | None,
    density: float = 0.2,
    drop_p: float = 0.5,
    seed: int = 0,
    merge_head: bool = True,
    **_: object,
) -> StateDict:
    """DARE drop-and-rescale on task vectors, then TIES.

    The drop mask is the method's only stochastic element, so ``seed`` must be
    the run's seed: pinned at a constant, every seed of a multi-seed study
    would share one mask and the reported spread would omit the variance DARE
    itself contributes.
    """
    if base is None:
        raise ValueError("DARE-TIES requires a base (phase-0) state_dict")
    keys = _mergeable_keys(experts, merge_head)
    gen = torch.Generator().manual_seed(seed)
    out = _scaffold(experts, base, merge_head)
    # The generator is consumed in `keys` order, so the mask a key receives is
    # a function of its position. Reordering this loop would silently reassign
    # every mask and change the merge without changing the seed.
    for k in keys:
        tv = _task_vector(experts, base, k)
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
