"""Deriving a train/validation/test split for a corpus that ships none.

Eleven of the corpora this repository can draw on publish no partition, so one
is derived. It is derived from a stable SHA-1 of the grouping key, never from a
library's shuffle, whose output is only reproducible for a pinned version of
that library — which is not something a results file records.

The split is **speaker-disjoint** wherever a speaker is recoverable: test
speakers are then never seen in training, which is what an evaluation number
should measure. Where no speaker is recoverable the derivation degrades to
utterance level and says so, and that carries an obvious caveat — the same
speaker appears in train and test, so the result is not speaker-independent.
Which policy was used is returned, never assumed, and every run records it.

Groups are assigned to whichever split has the largest remaining shortfall
rather than bucketed by hash, which is far less lumpy at these sizes. It is not
exact: whole speakers are indivisible, so the realised proportions drift from
80/10/10, and the fewer and more uneven a corpus's speakers the further they
drift. Nine evenly-sized speakers land near 78/11/11; nine uneven ones — one
dominant speaker and eight small — land near 94/3/3. Read the realised sizes
rather than assuming the nominal fractions.

The shortfall rule alone can leave a split empty, which one dominant speaker is
enough to do, so a second pass moves the smallest group out of whichever split
holds the most until none is empty. That pass, not luck, is why a nine-speaker
corpus still has a test split.

This module knows nothing about corpora. It takes items and the key to group
them by, so the OpenSLR held-out four and the manifest layer share one
implementation instead of two that drift.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

SPLITS = ("train", "validation", "test")
_FRACTIONS = (0.8, 0.1, 0.1)
_MIN_GROUPS = len(SPLITS)

#: How the partition was derived. ``speaker`` is what a result should be able to
#: claim; ``utterance`` is the fallback and is not speaker-independent.
SPLIT_POLICIES = ("speaker", "utterance")


def _assign_groups(ordered: list[str], sizes: dict[str, int]) -> dict[str, list[str]]:
    """Give each group to the split with the largest remaining shortfall."""
    total = sum(sizes[key] for key in ordered)
    targets = {name: frac * total for name, frac in zip(SPLITS, _FRACTIONS, strict=True)}
    counts = dict.fromkeys(SPLITS, 0)
    assigned: dict[str, list[str]] = {name: [] for name in SPLITS}

    for key in ordered:
        # Ties break towards the earlier split, so the choice never depends on
        # dict ordering.
        name = max(SPLITS, key=lambda n: (targets[n] - counts[n], -SPLITS.index(n)))
        assigned[name].append(key)
        counts[name] += sizes[key]

    _fill_empty_splits(assigned, sizes)
    return assigned


def _fill_empty_splits(assigned: dict[str, list[str]], sizes: dict[str, int]) -> None:
    """Make sure no split is empty when there are groups enough to go round.

    One dominant group can otherwise swallow the whole target for train and leave
    validation or test with nothing. Donate the smallest group of the split that
    holds the most, which costs the donor the least mass.
    """
    if sum(len(keys) for keys in assigned.values()) < _MIN_GROUPS:
        return
    for name in SPLITS:
        if assigned[name]:
            continue
        donor = max(SPLITS, key=lambda n: (len(assigned[n]), -SPLITS.index(n)))
        if len(assigned[donor]) < 2:
            return
        key = min(assigned[donor], key=lambda k: (sizes[k], k))
        assigned[donor].remove(key)
        assigned[name].append(key)


def derive_splits[T](
    items: Sequence[T],
    speaker_of: Callable[[T], str | None],
    identity_of: Callable[[T], str],
) -> tuple[dict[str, list[T]], str, str]:
    """Partition ``items`` 80/10/10, speaker-disjoint where that is possible.

    Args:
        items: The rows to partition.
        speaker_of: The speaker key for a row, or None when it has none. If
            *any* row returns None, or fewer than three distinct speakers turn
            up, the whole corpus falls back to utterance level — one row with a
            missing speaker is enough, because a partition that is disjoint for
            most rows is not disjoint.
        identity_of: A stable id for a row, used as the utterance-level grouping
            key and to order and digest the result.

    Returns:
        The partition, the policy used (``"speaker"`` or ``"utterance"``), and
        the SHA-1 of the resulting test id list, which callers record so a run's
        provenance pins the exact set it scored on.
    """
    speakers = [speaker_of(item) for item in items]
    resolved = [s for s in speakers if s is not None]
    if len(resolved) == len(items) and len(set(resolved)) >= _MIN_GROUPS:
        policy, keys = "speaker", resolved
    else:
        policy, keys = "utterance", [identity_of(item) for item in items]

    groups: dict[str, list[T]] = {}
    for key, item in zip(keys, items, strict=True):
        groups.setdefault(key, []).append(item)

    ordered = sorted(groups, key=lambda k: hashlib.sha1(k.encode("utf-8")).hexdigest())
    assigned = _assign_groups(ordered, {k: len(v) for k, v in groups.items()})

    # Sorting by identity makes the result independent of the input row order.
    parts = {
        name: sorted((row for key in assigned[name] for row in groups[key]), key=identity_of)
        for name in SPLITS
    }
    listing = "\n".join(identity_of(row) for row in parts["test"])
    return parts, policy, hashlib.sha1(listing.encode("utf-8")).hexdigest()
