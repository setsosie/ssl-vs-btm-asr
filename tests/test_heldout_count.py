"""The held-out set is stated in prose in several places. It has one definition.

``configs/scales/heldout.yaml`` is the definition and ``get_heldout`` reads it.
A count written into a README, a citation file or a protocol is a second copy,
and a second copy drifts: the citation abstract said five for as long as the
set has been four. This fails when any write-up-facing file states a count the
configuration does not have, so the drift is caught here and not in a paper.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from svb.data.registry import get_heldout

_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
# "four held-out", "4 held out", "the four held-out corpora" — a count directly
# in front of the phrase. Counts elsewhere in a sentence are not caught, which
# is the price of not flagging every number near the words.
_COUNT = re.compile(rf"\b({'|'.join(_WORDS)}|\d+)\s+held[- ]out\b", re.IGNORECASE)


def _surfaces(root: Path) -> list[Path]:
    return [root / "README.md", root / "CITATION.cff", *sorted((root / "docs").glob("*.md"))]


def test_no_write_up_states_a_heldout_count_the_config_does_not_have(
    pytestconfig: pytest.Config,
) -> None:
    n = len(get_heldout())
    allowed = {_WORDS[n], str(n)}

    wrong = [
        f"{path.name}: {match.group(0)!r}"
        for path in _surfaces(Path(pytestconfig.rootpath))
        for match in _COUNT.finditer(path.read_text(encoding="utf-8"))
        if match.group(1).lower() not in allowed
    ]

    assert not wrong, f"the held-out set has {n} languages, but: {wrong}"


def test_the_check_would_catch_the_count_it_was_written_for() -> None:
    assert _COUNT.search("and five held-out Indic transfer languages")
    assert _COUNT.search("on 7 held out languages")
    assert not _COUNT.search("the held-out set is small")
