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


def test_the_latin_default_is_the_mark_preserving_one() -> None:
    """Failing safe. A European language added without a line gets a policy that
    differs from whisper-basic only in ways Latin text barely notices; a Yoruba
    or Vietnamese one added without a line keeps its tone marks."""
    assert policy_for_language("sw").version == "latin-marks"
    assert SCRIPT_POLICIES["Latn"] == "latin-marks"


def test_the_arabic_script_has_no_default_and_says_why(monkeypatch) -> None:
    """Its two conventions fold letters in opposite directions, so there is no
    safe guess: Arabic folds the Persian letters onto the Arabic ones and
    Perso-Arabic folds them back.

    Every Arabic-script language currently in use names its policy in the
    language table, so this exercises the mechanism on one that does not.
    """
    from svb.text.registry import SCRIPT_POLICIES

    assert "Arab" not in SCRIPT_POLICIES
    monkeypatch.setitem(LANGUAGE_SCRIPTS, "zz", "Arab")

    with pytest.raises(ValueError, match="no default normalization policy"):
        policy_for_language("zz")
    assert policy_for_language("zz", "arabic-ouaal").version == "arabic-ouaal"


def test_the_arabic_script_languages_each_name_their_own_policy() -> None:
    """They do not share one: Arabic and Perso-Arabic disagree by design."""
    assert policy_for_language("ar").version == "arabic-ouaal"
    assert policy_for_language("fa").version == "perso-arabic"
    assert policy_for_language("ug").version == "uyghur-ug"


def test_an_unknown_language_is_refused_rather_than_guessed() -> None:
    with pytest.raises(KeyError, match="no script recorded"):
        policy_for_language("xx")


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


def test_every_recorded_script_either_has_a_default_or_is_documented() -> None:
    """A script in the table with no policy must be one the docs explain."""
    undefaulted = set(LANGUAGE_SCRIPTS.values()) - set(SCRIPT_POLICIES)
    assert undefaulted == {"Arab"}


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
    assert {assigned[c] for c in ("sw", "rw", "lg", "kab", "uz")} == {"latin-marks"}
    assert {assigned[c] for c in ("en", "de", "ru", "ka", "hu")} == {"whisper-basic"}
    assert {assigned[c] for c in ("malayalam", "marathi", "telugu")} == {"indic-vistaar"}


def test_japanese_is_the_only_preset_language_scored_on_characters(pytestconfig) -> None:
    """Its policy name and its word_boundary have to agree, since one sets the
    metric and the other names the family that expects it."""
    from pathlib import Path

    from svb.data.registry import get_preset

    configs = Path(pytestconfig.rootpath) / "configs"
    for scale in ("3", "16", "64"):
        specs = get_preset(scale, configs_dir=configs)
        unspaced = {s.code for s in specs if not s.word_boundary}
        assert unspaced <= {"ja"}
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
