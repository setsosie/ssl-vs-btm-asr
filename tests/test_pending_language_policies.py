"""Policies for the locales the validated-minus-eval statistic brings in.

Fourteen Common Voice locales clear the training-hours line under that count and
none of them could be placed before: three are written in scripts the table had
no entry for, four in scripts it had no policy for, three in the Arabic script
where there is deliberately no default, and four are European languages the
Latin default would have sent to the wrong policy.

The Thai and Han expectations come from the design document's case list and were
cross-checked by running these policies; all ten agreed.
"""

from __future__ import annotations

import unicodedata

import pytest

from svb.text.normalize import get_policy, normalize_text
from svb.text.registry import (
    LANGUAGE_SCRIPTS,
    NO_SPACE_POLICIES,
    policy_for_language,
    script_for_language,
)

ZWSP = "\u200b"

PENDING = {
    "fa": ("Arab", "perso-arabic"),
    "kbd": ("Cyrl", "whisper-basic"),
    "lv": ("Latn", "whisper-basic"),
    "zh-CN": ("Hans", "han-mer"),
    "yue": ("Hant", "han-mer"),
    "pt": ("Latn", "whisper-basic"),
    "th": ("Thai", "thai-cer"),
    "ckb": ("Arab", "perso-arabic"),
    "cy": ("Latn", "whisper-basic"),
    "ur": ("Arab", "perso-arabic"),
    "kmr": ("Latn", "whisper-basic"),
    "fy-NL": ("Latn", "whisper-basic"),
    "ady": ("Cyrl", "whisper-basic"),
    "cs": ("Latn", "whisper-basic"),
}


@pytest.mark.parametrize(("code", "expected"), sorted(PENDING.items()))
def test_every_pending_locale_resolves(code: str, expected: tuple[str, str]) -> None:
    script, policy = expected

    assert script_for_language(code) == script
    assert LANGUAGE_SCRIPTS[code] == script
    assert policy_for_language(code).version == policy


def test_the_latin_locales_take_the_default_and_the_tables_stay_quiet() -> None:
    """Whisper's normalizer is the default, so a Latin language needs no entry
    unless the default is wrong for it — and for none of these it is."""
    from svb.text.registry import LANGUAGE_POLICIES, SCRIPT_POLICIES

    assert "Latn" not in SCRIPT_POLICIES
    for code in ("lv", "pt", "cy", "cs", "fy-NL", "kmr"):
        assert code not in LANGUAGE_POLICIES
        assert policy_for_language(code).version == "whisper-basic"


def test_whisper_basic_costs_the_new_european_locales_nothing() -> None:
    """Their diacritics are precomposed, so its mark rule finds nothing.

    This is the check that makes assigning them Whisper's normalizer safe rather
    than merely conventional.
    """
    samples = {
        "cy": "Dŵr a thân yn Nghymru",
        "lv": "Latviešu valoda ir skaista",
        "pt": "Não é fácil coração",
        "cs": "Příliš žluťoučký kůň",
        "fy-NL": "Frysk is myn memmetaal",
        "kbd": "Адыгэбзэ",
        "ady": "Адыгабзэ",
    }
    for code, text in samples.items():
        normalized = unicodedata.normalize("NFKC", text)
        marks = [c for c in normalized if unicodedata.category(c)[0] == "M"]
        assert marks == [], code
        assert normalize_text(text, policy_for_language(code)).strip() == text.lower()


# (id, policy, input, expected) — the design document's case list, confirmed by
# running these policies.
CASES = [
    ("66-th-sara-am-stays-one-code-point", "thai-cer", "ทำงาน", "ทำงาน"),
    ("67-th-tone-marks-survive", "thai-cer", "สวัสดีครับ น้ำ", "สวัสดีครับ น้ำ"),
    ("68-th-zwsp-becomes-a-space", "thai-cer", f"ภาษา{ZWSP}ไทย", "ภาษา ไทย"),
    ("69-th-punctuation", "thai-cer", "เขาบอกว่า “ดี” ๆ!", "เขาบอกว่า ดี ๆ"),
    ("70-th-paiyannoi-is-a-letter", "thai-cer", "ฯลฯ และ ๆ", "ฯลฯ และ ๆ"),
    ("76-zh-fullwidth-punctuation", "han-mer", "你好，世界！", "你好 世界"),
    ("77-zh-latin-run-kept", "han-mer", "我用 Python 写代码。", "我用 python 写代码"),
    ("78-zh-traditional-preserved", "han-mer", "繁體字與简体字", "繁體字與简体字"),
    ("79-yue-cantonese-traditional", "han-mer", "你哋食咗飯未呀？", "你哋食咗飯未呀"),
    ("80-zh-fullwidth-digits", "han-mer", "２０２６年", "2026年"),
]


@pytest.mark.parametrize(
    ("policy", "text", "expected"),
    [(c[1], c[2], c[3]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_policy_table(policy: str, text: str, expected: str) -> None:
    assert normalize_text(text, get_policy(policy)) == expected


@pytest.mark.parametrize(
    ("policy", "text"), [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES]
)
def test_every_case_is_idempotent(policy: str, text: str) -> None:
    once = normalize_text(text, get_policy(policy))
    assert normalize_text(once, get_policy(policy)) == once


def test_thai_uses_nfc_because_nfkc_splits_a_letter() -> None:
    """SARA AM U+0E33 is category Lo, and NFKC turns it into a combining mark
    plus a vowel — lengthening every word that contains it by one code point,
    which moves the character error rate denominator, and manufacturing an Mn
    for a mark rule to find."""
    assert unicodedata.category("ำ") == "Lo"
    assert len(unicodedata.normalize("NFC", "ทำงาน")) == 5
    assert len(unicodedata.normalize("NFKC", "ทำงาน")) == 6

    assert get_policy("thai-cer").form == "NFC"
    assert normalize_text("ทำงาน", get_policy("thai-cer")) == "ทำงาน"


def test_thai_spaces_the_zero_width_separator_rather_than_deleting_it() -> None:
    """U+200B is where Thai puts its word boundary; deleting it fuses words."""
    assert get_policy("thai-cer").zero_width == "space"
    assert normalize_text(f"ภาษา{ZWSP}ไทย", get_policy("thai-cer")) == "ภาษา ไทย"
    assert normalize_text(f"ภาษา{ZWSP}ไทย", get_policy("whisper-marks")) == "ภาษาไทย"


def test_han_keeps_traditional_and_folds_only_width() -> None:
    """No Chinese benchmark converts between the character sets, and Unicode
    normalization does not either."""
    assert unicodedata.normalize("NFKC", "體") == "體"
    assert normalize_text("繁體字", get_policy("han-mer")) == "繁體字"
    assert normalize_text("２０２６", get_policy("han-mer")) == "2026"


def test_the_no_space_policies_are_the_ones_scored_on_characters() -> None:
    # Membership, not equality: the set grows as scripts without word
    # separators arrive, and what has to hold is that these three are in it.
    assert {"ja-cer", "thai-cer", "han-mer"} <= NO_SPACE_POLICIES
    for name in NO_SPACE_POLICIES:
        get_policy(name)  # each exists


def test_no_existing_policy_hash_moved() -> None:
    """Adding families must not re-label results already produced."""
    frozen = {
        "svb-norm-1": "205fefc26d0e",
        "whisper-basic": "3870eb0765ba",
        "whisper-basic-nodiacritics": "be6ae6dbd8dc",
        "whisper-marks": "4edf68941a4d",
        "cyrillic-yo": "0cf3c74a7ac6",
        "turkic-tr": "9e1473d7cf88",
        "latin-marks": "fada7b86ad40",
        "indic-vistaar": "c06db1f88465",
        "arabic-ouaal": "ff67c557c4e9",
        "perso-arabic": "37de70b2f78b",
        "uyghur-ug": "014787ce54cb",
        "ja-cer": "d5ab6fdc697d",
    }
    for name, digest in frozen.items():
        assert get_policy(name).policy_hash() == digest, name


def test_the_two_new_policies_have_their_own_hashes() -> None:
    hashes = {n: get_policy(n).policy_hash() for n in ("thai-cer", "han-mer", "whisper-marks")}

    assert len(set(hashes.values())) == 3
