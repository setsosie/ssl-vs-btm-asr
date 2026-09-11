"""Resolving a language to the policy its script family needs.

Every shipped preset names its policy explicitly, so this table is the fallback
for a language added without one. It is keyed on script rather than on language
because the reason a policy exists is a property of the writing system.
"""

from __future__ import annotations

import pytest

from svb.text.registry import (
    LANGUAGE_SCRIPTS,
    NO_SPACE_POLICIES,
    SCRIPT_POLICIES,
    policy_for_language,
    script_for_language,
)


def test_an_explicit_name_wins_over_the_script_default() -> None:
    """Which is how Turkish and Arabic get policies their script alone would not."""
    assert policy_for_language("tr", "turkic-tr").version == "turkic-tr"
    assert policy_for_language("de", "whisper-basic").version == "whisper-basic"


def test_the_script_default_covers_a_language_with_no_explicit_name() -> None:
    assert policy_for_language("hi").version == "indic-vistaar"
    assert policy_for_language("ja").version == "ja-cer"
    assert policy_for_language("ru").version == "whisper-basic"


def test_a_script_with_no_exception_takes_the_default() -> None:
    """Latin, Cyrillic and Georgian have no entry at all: Whisper's normalizer
    is what they wanted, and it is now the default rather than a table row."""
    assert "Latn" not in SCRIPT_POLICIES
    assert "Cyrl" not in SCRIPT_POLICIES
    assert policy_for_language("sw", "latin-marks").version == "latin-marks"
    assert policy_for_language("lg").version == "whisper-basic"


def test_the_arabic_script_has_no_entry_of_its_own(monkeypatch) -> None:
    """Its two conventions fold letters in opposite directions, so there is no
    script-wide answer and each language names the one it follows. One absent
    from the language table takes the default, which normalizes nothing away:
    the default strips its vocalization along with all of category M."""
    assert "Arab" not in SCRIPT_POLICIES
    monkeypatch.setitem(LANGUAGE_SCRIPTS, "zq", "Arab")

    assert policy_for_language("zq").version == "whisper-basic"
    assert policy_for_language("zq", "arabic-ouaal").version == "arabic-ouaal"


def test_the_arabic_script_languages_each_name_their_own_policy() -> None:
    """They do not share one: Arabic and Perso-Arabic disagree by design."""
    assert policy_for_language("ar").version == "arabic-ouaal"
    assert policy_for_language("fa").version == "perso-arabic"
    assert policy_for_language("ug").version == "uyghur-ug"


def test_an_unknown_language_takes_the_default_and_says_so() -> None:
    """Loudly. A default nobody sees is how a script gets scored under rules
    written for another one."""
    with pytest.warns(UserWarning, match="no script recorded"):
        assert policy_for_language("xx").version == "whisper-basic"


def test_an_unknown_policy_name_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown normalization policy"):
        policy_for_language("de", "whisper")


IN_SCOPE = {
    "en": "Latn",
    "de": "Latn",
    "fr": "Latn",
    "es": "Latn",
    "it": "Latn",
    "nl": "Latn",
    "pl": "Latn",
    "fi": "Latn",
    "ca": "Latn",
    "eo": "Latn",
    "eu": "Latn",
    "hu": "Latn",
    "gl": "Latn",
    "tr": "Latn",
    "sw": "Latn",
    "rw": "Latn",
    "lg": "Latn",
    "kab": "Latn",
    "uz": "Latn",
    "ru": "Cyrl",
    "uk": "Cyrl",
    "be": "Cyrl",
    "ab": "Cyrl",
    "ba": "Cyrl",
    "mhr": "Cyrl",
    "ka": "Geor",
    "hi": "Deva",
    "mr": "Deva",
    "ta": "Taml",
    "ml": "Mlym",
    "te": "Telu",
    "gu": "Gujr",
    "ar": "Arab",
    "ps": "Arab",
    "ug": "Arab",
    "ja": "Jpan",
}


@pytest.mark.parametrize(("code", "script"), sorted(IN_SCOPE.items()))
def test_every_in_scope_language_has_a_recorded_script(code: str, script: str) -> None:
    assert script_for_language(code) == script
    assert LANGUAGE_SCRIPTS[code] == script


def test_every_script_with_an_entry_is_an_exception_to_the_default() -> None:
    """The table holds only exceptions, so no entry may name the default."""
    assert "whisper-basic" not in set(SCRIPT_POLICIES.values())
    undefaulted = set(LANGUAGE_SCRIPTS.values()) - set(SCRIPT_POLICIES)
    assert "Latn" in undefaulted and "Arab" in undefaulted


def test_every_shipped_preset_states_its_policy(pytestconfig) -> None:
    """Explicit beside the language, not inferred three modules away."""
    from pathlib import Path

    from svb.data.registry import get_heldout, get_preset

    configs = Path(pytestconfig.rootpath) / "configs"
    specs = list(get_heldout(configs_dir=configs))
    for scale in ("3", "16", "64"):
        specs += get_preset(scale, configs_dir=configs)

    unstated = sorted({s.code for s in specs if s.normalizer is None})
    assert unstated == []


def test_the_preset_assignments_are_the_intended_ones(pytestconfig) -> None:
    from pathlib import Path

    from svb.data.registry import get_heldout, get_preset

    configs = Path(pytestconfig.rootpath) / "configs"
    specs = list(get_heldout(configs_dir=configs))
    for scale in ("3", "16", "64"):
        specs += get_preset(scale, configs_dir=configs)
    assigned = {s.code: s.normalizer for s in specs}

    assert assigned["tr"] == "turkic-tr"
    assert assigned["ar"] == "arabic-ouaal"
    assert assigned["ps"] == "perso-arabic"
    assert assigned["ug"] == "uyghur-ug"
    assert assigned["ja"] == "ja-cer"
    assert assigned["hi"] == assigned["ta"] == "indic-vistaar"
    assert {assigned[c] for c in ("sw", "rw")} == {"latin-marks"}
    assert {assigned[c] for c in ("lg", "kab", "uz")} == {"whisper-basic"}
    assert {assigned[c] for c in ("en", "de", "ru", "ka", "hu")} == {"whisper-basic"}
    assert {assigned[c] for c in ("malayalam", "marathi", "telugu")} == {"indic-vistaar"}


def test_a_policy_named_for_characters_goes_to_a_language_written_without_spaces(
    pytestconfig,
) -> None:
    """A policy name and a word_boundary have to agree, since one sets the
    metric and the other names the family that expects it.

    The large tier has five languages written without spaces — Japanese, plus
    Chinese, Cantonese and Thai from the wider Common Voice statistic and
    Tibetan from the corpora — so the check is that the two declarations agree,
    not that only Japanese is scored on characters.
    """
    from pathlib import Path

    from svb.data.registry import get_preset

    configs = Path(pytestconfig.rootpath) / "configs"
    for scale in ("3", "16", "64"):
        specs = get_preset(scale, configs_dir=configs)
        assert any(not s.word_boundary for s in specs)
        for spec in specs:
            # Extends to every policy whose script is written without word
            # separators, so a regenerated preset that adds Thai or Chinese
            # cannot quietly leave word error rate as their primary metric.
            if spec.normalizer in NO_SPACE_POLICIES:
                assert not spec.word_boundary, spec.code


def test_an_unknown_normalizer_in_a_preset_is_refused() -> None:
    from svb.data.registry import LangSpec

    with pytest.raises(ValueError, match="unknown normalizer"):
        LangSpec(code="en", source="commonvoice", hf_config="en", normalizer="nope")
