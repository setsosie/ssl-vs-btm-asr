"""The per-script pipeline: its rules, its ordering, and what it must not move.

The two existing pipelines are frozen. `svb-norm-1` is what the comparability
paragraph describes and `whisper-basic` is a reproduction of someone else's
published algorithm, so neither may change; a policy records and hashes only the
fields its own pipeline reads, which is what lets a third pipeline arrive with
ten new fields and move neither hash.
"""

from __future__ import annotations

import pytest

from svb.text.normalize import NormalizerPolicy, get_policy, normalize_text


def test_the_two_frozen_hashes_did_not_move() -> None:
    assert NormalizerPolicy().policy_hash() == "205fefc26d0e"
    assert get_policy("whisper-basic").policy_hash() == "3870eb0765ba"


def test_a_script_policy_records_only_script_rules() -> None:
    record = get_policy("whisper-marks").to_dict()

    assert record["pipeline"] == "script"
    assert record["marks"] == "keep"
    for absent in ("strip_marks", "diacritics", "strip_invisibles", "malayalam_chillu"):
        assert absent not in record


def test_a_rule_from_another_pipeline_is_refused() -> None:
    with pytest.raises(ValueError, match=r"script.*does not read"):
        NormalizerPolicy(pipeline="script", version="x", diacritics="drop")
    with pytest.raises(ValueError, match=r"svb.*does not read"):
        NormalizerPolicy(marks="space")


def test_whisper_marks_is_whisper_with_the_marks_kept() -> None:
    """The one-character `MSP` to `SP` fix the Open ASR Leaderboard already made.

    Whisper reduces an abugida to bare consonants; the same pipeline with the
    mark rule off leaves the word intact.
    """
    hindi = "यह हिंदी है"

    assert normalize_text(hindi, get_policy("whisper-basic")) == "यह ह द ह "
    assert normalize_text(hindi, get_policy("whisper-marks")) == hindi


def test_whisper_marks_protects_the_intra_word_apostrophe() -> None:
    assert normalize_text("don't", get_policy("whisper-basic")) == "don t"
    assert normalize_text("don't", get_policy("whisper-marks")) == "don't"


def test_whisper_marks_deletes_zero_width_characters() -> None:
    """Category Cf is not in Whisper's MSP set, so upstream leaves them in."""
    assert normalize_text("a\u200bb", get_policy("whisper-basic")) == "a\u200bb"
    assert normalize_text("a\u200bb", get_policy("whisper-marks")) == "ab"


def test_the_script_pipeline_strips_its_output() -> None:
    """Unlike Whisper, which never strips."""
    assert normalize_text("Hello!", get_policy("whisper-basic")) == "hello "
    assert normalize_text("Hello!", get_policy("whisper-marks")) == "hello"


def test_zero_width_can_become_a_space_instead_of_vanishing() -> None:
    """U+200B is the word separator in several scripts, not noise."""
    spacing = NormalizerPolicy(pipeline="script", version="t", zero_width="space")
    assert normalize_text("a\u200bb", spacing) == "a b"


def test_zero_width_can_keep_the_non_joiner_alone() -> None:
    """ZWNJ is orthographic in Perso-Arabic; ZWSP and the rest are not."""
    keep = NormalizerPolicy(pipeline="script", version="t", zero_width="keep_zwnj")
    assert normalize_text("a‌b", keep) == "a‌b"
    assert normalize_text("a\u200bb", keep) == "ab"


def test_punctuation_can_be_deleted_rather_than_spaced() -> None:
    """Vistaar deletes; Whisper spaces. The difference shows on a danda."""
    deleting = NormalizerPolicy(pipeline="script", version="t", punct_action="delete")
    spacing = NormalizerPolicy(pipeline="script", version="t", punct_action="space")

    assert normalize_text("a,b", deleting) == "ab"
    assert normalize_text("a,b", spacing) == "a b"


def test_a_turkish_locale_premap_beats_the_combining_dot() -> None:
    """`İ`.lower() is i + U+0307, which a mark rule then turns into a space."""
    tr = NormalizerPolicy(pipeline="script", version="t", locale_case="tr")

    assert normalize_text("İstanbul", get_policy("whisper-basic")) == "i stanbul"
    assert normalize_text("İstanbul", tr) == "istanbul"
    assert normalize_text("Irmak", tr) == "ırmak"


def test_marks_can_be_scoped_to_one_script() -> None:
    """Arabic vocalization is optional in writing; Devanagari matras are not."""
    arabic = NormalizerPolicy(pipeline="script", version="t", marks="arabic")

    assert normalize_text("مَرْحَبًا", arabic) == "مرحبا"
    assert normalize_text("हिंदी", arabic) == "हिंदी"


def test_the_malayalam_chillu_map_runs_before_zero_width_deletion() -> None:
    """The atomic chillu is reached *through* a joiner.

    Deleting zero-width characters first strands a bare virama, which is a real
    ordering bug in the reference Indic normalizer this policy is drawn from.
    """
    policy = NormalizerPolicy(pipeline="script", version="t", script_map="malayalam")
    zwj_form = "മലയാളത്തിന്‍"

    assert normalize_text(zwj_form, policy) == "മലയാളത്തിൻ"
    without_map = NormalizerPolicy(pipeline="script", version="t")
    assert normalize_text(zwj_form, without_map) == "മലയാളത്തിന്"


@pytest.mark.parametrize(
    "text",
    ["Hello, World!", "यह हिंदी है", "مَرْحَبًا", "İstanbul", "a‌b", "…!?"],
)
def test_the_script_pipeline_is_idempotent(text: str) -> None:
    for name in ("whisper-marks",):
        policy = get_policy(name)
        once = normalize_text(text, policy)
        assert normalize_text(once, policy) == once
