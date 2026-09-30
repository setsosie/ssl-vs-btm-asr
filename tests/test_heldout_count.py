"""The held-out set is stated in prose in several places. It has one definition.

``configs/scales/heldout.yaml`` is the definition and ``get_heldout`` reads it.
A count written into a README, a citation file or a protocol is a second copy,
and a second copy drifts: the citation abstract said five for as long as the
set has been four. This fails when any write-up-facing file states a count the
configuration does not have, so the drift is caught here and not in a paper.

Two lists in code also name the set by Common Voice locale, because the
selection of training languages has to keep clear of it and the preset carries
no locale. They are hand-maintained copies, and the last test pins them to the
preset so that a language added to it without them fails the suite.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import pytest

from svb.data.corpora import HELD_OUT_LANGUAGES
from svb.data.registry import get_heldout

_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_NUMBER = rf"(?<![\w/.-])({'|'.join(_WORDS)}|\d+)(?![\w/.-])"
# A count within three words on either side of "held-out": "four held-out
# languages", "the held-out set is four languages", "the held-out four",
# "four OpenSLR held-out sets". Words are separated by spaces or a line break
# only, so a table cell or an "80/10/10" three cells away is not a count. A
# count further away in the sentence is not caught, which is the price of not
# flagging every number near the words.
_NEAR = r"(?:[ \n]+[\w*'’-]+){0,3}[ \n]+"
_COUNT = re.compile(rf"{_NUMBER}{_NEAR}held[- ]out\b|\bheld[- ]out{_NEAR}{_NUMBER}", re.IGNORECASE)
# "one held-out language" is how a docstring speaks of each of them, not a size.
_DISTRIBUTIVE = {"one"}


def _surfaces(root: Path) -> list[Path]:
    """Every file a reader of the write-up sees. ``docs/planning/`` is a local,
    untracked scratch area and is left out on purpose."""
    docs = [p for p in sorted((root / "docs").rglob("*.md")) if "planning" not in p.parts]
    return [root / "README.md", root / "CITATION.cff", *docs]


def _counts(text: str) -> list[tuple[str, str]]:
    """``(count, phrase)`` for every count the pattern finds near "held-out"."""
    found = []
    for match in _COUNT.finditer(text):
        count = (match.group(1) or match.group(2)).lower()
        if count not in _DISTRIBUTIVE:
            found.append((count, match.group(0)))
    return found


def test_no_write_up_states_a_heldout_count_the_config_does_not_have(
    pytestconfig: pytest.Config,
) -> None:
    root = Path(pytestconfig.rootpath)
    n = len(get_heldout(configs_dir=root / "configs"))
    assert n < len(_WORDS), f"{n} held-out languages: extend _WORDS before extending the set"
    allowed = {_WORDS[n], str(n)}

    wrong = [
        f"{path.relative_to(root)}: {phrase!r}"
        for path in _surfaces(root)
        for count, phrase in _counts(path.read_text(encoding="utf-8"))
        if count not in allowed
    ]

    assert not wrong, f"the held-out set has {n} languages, but: {wrong}"


def test_the_check_reaches_the_file_that_was_wrong(pytestconfig: pytest.Config) -> None:
    root = Path(pytestconfig.rootpath)
    surfaces = _surfaces(root)

    assert root / "CITATION.cff" in surfaces
    assert (
        root / "docs" / "small-study.md" in surfaces
        or not (root / "docs" / "small-study.md").exists()
    )
    assert any(p.parent.name == "corpora" for p in surfaces)
    assert not any("planning" in p.parts for p in surfaces)


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("and five held-out Indic transfer languages", "five"),
        ("on 7 held out languages", "7"),
        ("**The held-out set is four languages.**", "four"),
        ("the four OpenSLR held-out sets", "four"),
        ("the held-out four", "four"),
    ],
)
def test_the_check_would_catch_the_phrasings_the_docs_use(text: str, count: str) -> None:
    assert [c for c, _ in _counts(text)] == [count]


@pytest.mark.parametrize(
    "text",
    [
        "the held-out set is small",
        "adapt to one held-out language and evaluate",
        "twenty-four held-out utterances",  # a compound numeral is not "four"
    ],
)
def test_the_check_leaves_alone_what_is_not_a_set_size(text: str) -> None:
    assert _counts(text) == []


def test_the_locale_lists_in_code_name_the_same_languages_as_the_preset(
    pytestconfig: pytest.Config, select_languages: ModuleType
) -> None:
    """Not one definition, but pinned to it. The preset carries no Common Voice
    locale, so the training-language selection keeps its own locale→name map and
    the corpus registry its own locale set; a language added to the preset
    without them would be selectable for training. The registry's set may also
    carry Odia, documented there as the pending fifth."""
    codes = {spec.code for spec in get_heldout(configs_dir=Path(pytestconfig.rootpath) / "configs")}

    assert set(select_languages.HELD_OUT_LOCALES.values()) == codes
    assert set(select_languages.HELD_OUT_LOCALES) <= HELD_OUT_LANGUAGES
    assert HELD_OUT_LANGUAGES - set(select_languages.HELD_OUT_LOCALES) <= {"or"}
