"""An hours cap on the training and validation splits.

Without one the smallest preset cannot be run at all. Common Voice 25 holds
about 2,700 trainable hours of English, and arm A fine-tunes each language for
fifty epochs; validation is scored every epoch over another 24. A study that is
meant to finish has to say how much audio each language contributes, and say it
in hours, because that is the unit a low-resource claim is made in.

The subset is chosen by hash, not by a random generator:

* **It is the same for every arm and every seed.** The run seed does not enter
  the hash, so two arms at one seed — and one arm at two seeds — train on the
  same utterances. Seed spread then measures initialisation and data order, and
  an arm comparison is not confounded with which hours each arm happened to get.
* **It is nested.** Rows are taken in hash order until the next one would cross
  the cap, so the 5-hour subset is a prefix of the 10-hour subset. A later run
  at a larger cap adds data to the smaller one rather than replacing it.
* **It does not depend on row order or on any library's shuffle**, for the same
  reason the held-out split is derived from a hash: the order rows sit in on
  disk and the output of ``random.shuffle`` are not stable things to build a
  published number on.

Durations come from metadata, never from decoding: Common Voice's own
``clip_durations.tsv`` where the release ships one, audio headers otherwise. A
row whose duration cannot be read, or reads as zero, is left out of a capped
split and counted — audio that cannot be measured cannot be budgeted, and a clip
of no length always fits a budget while contributing nothing but a bad batch.

The test split is never capped. A number scored on part of a test set is a
number about that part.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapSelection:
    """The rows an hours cap kept for one split, and what it left behind."""

    max_hours: float
    #: Positions in the uncapped split, ascending, so a capped dataset keeps the
    #: relative order of the split it was cut from.
    indices: tuple[int, ...]
    n_available: int
    seconds_available: float
    seconds_selected: float
    #: Rows left out because no positive duration could be read for them.
    n_unknown_duration: int
    #: SHA-1 over the selected row keys in selection order. Two runs that report
    #: the same digest trained on the same utterances.
    subset_sha1: str

    @property
    def n_selected(self) -> int:
        return len(self.indices)

    @property
    def binding(self) -> bool:
        """Whether the cap removed anything that could have been used."""
        return self.n_selected < self.n_available - self.n_unknown_duration

    def to_record(self) -> dict[str, Any]:
        return {
            "max_hours": self.max_hours,
            "binding": self.binding,
            "n_available": self.n_available,
            "n_selected": self.n_selected,
            "hours_available": self.seconds_available / 3600.0,
            "hours_selected": self.seconds_selected / 3600.0,
            "n_unknown_duration": self.n_unknown_duration,
            "subset_sha1": self.subset_sha1,
        }


def _rank(salt: str, key: str) -> str:
    return hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()


def select_within_hours(
    rows: Sequence[tuple[str, float | None]], max_hours: float, salt: str
) -> CapSelection:
    """Take rows in hash order until the next one would cross ``max_hours``.

    Args:
        rows: ``(key, seconds)`` per row, in the split's own order. ``seconds``
            is None where no duration could be read.
        max_hours: The cap. Must be positive.
        salt: Mixed into the hash — the language code — so two languages whose
            clips happen to share names are not ranked identically.

    Stops at the first row that does not fit instead of skipping it and packing
    smaller ones behind it. Packing would fill the budget more tightly and
    break nesting: the rows a smaller cap skipped past would differ from the
    ones a larger cap took.

    Raises:
        ValueError: If the cap is not positive, two rows share a key, or nothing
            fits under the cap.
    """
    if max_hours <= 0:
        raise ValueError(f"an hours cap must be positive, got {max_hours}")
    keys = [key for key, _ in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"{salt}: row keys are not unique, so a hash order over them is not an "
            "order over the rows"
        )

    budget = max_hours * 3600.0
    known = [(i, key, sec) for i, (key, sec) in enumerate(rows) if sec is not None and sec > 0]
    chosen: list[tuple[int, str]] = []
    total = 0.0
    for index, key, seconds in sorted(known, key=lambda row: _rank(salt, row[1])):
        if total + seconds > budget:
            break
        chosen.append((index, key))
        total += seconds
    if not chosen:
        raise ValueError(
            f"{salt}: nothing fits under a cap of {max_hours} h "
            f"({len(known)} of {len(rows)} rows have a usable duration)"
        )

    digest = hashlib.sha1("\n".join(key for _, key in chosen).encode("utf-8")).hexdigest()
    return CapSelection(
        max_hours=max_hours,
        indices=tuple(sorted(index for index, _ in chosen)),
        n_available=len(rows),
        seconds_available=sum(sec for _, _, sec in known),
        seconds_selected=total,
        n_unknown_duration=len(rows) - len(known),
        subset_sha1=digest,
    )
