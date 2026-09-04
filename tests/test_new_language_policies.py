"""Policies for the eighteen languages the non-Common-Voice corpora bring in.

Three need a policy that did not exist: Armenian writes a punctuation mark
*inside* the word, Korean must never see the compatibility form, and Tibetan is
an abugida written without spaces whose corpus states no evaluation convention
at all.
"""

from __future__ import annotations

import unicodedata

import pytest

from svb.text.normalize import get_policy, normalize_text
from svb.text.registry import LANGUAGE_SCRIPTS, NO_SPACE_POLICIES, policy_for_language

NEW = {
    "bo": ("Tibt", "tibetan-syllable"),
    "jv": ("Latn", "latin-marks"),
    "su": ("Latn", "latin-marks"),
    "si": ("Sinh", "indic-vistaar"),
    "ne": ("Deva", "indic-vistaar"),
    "bn": ("Beng", "indic-vistaar"),
    "kn": ("Knda", "indic-vistaar"),
    "ko": ("Kore", "ko-kspon"),
    "hy": ("Armn", "armenian-hy"),
    "kk": ("Cyrl", "whisper-basic"),
    "et": ("Latn", "whisper-basic"),
    "is": ("Latn", "whisper-basic"),
    "hr": ("Latn", "whisper-basic"),
    "zu": ("Latn", "latin-marks"),
    "xh": ("Latn", "latin-marks"),
    "nso": ("Latn", "latin-marks"),
    "ts": ("Latn", "latin-marks"),
    "ve": ("Latn", "latin-marks"),
}


@pytest.mark.parametrize(("code", "expected"), sorted(NEW.items()))
def test_every_new_language_resolves(code: str, expected: tuple[str, str]) -> None:
    script, policy = expected

    assert LANGUAGE_SCRIPTS[code] == script
    assert policy_for_language(code).version == policy


def test_armenian_keeps_the_word_its_question_mark_sits_inside() -> None:
    """The Armenian question mark is written on the stressed syllable, inside
    the word, so a rule that turns punctuation into a space splits it in two."""
    assert unicodedata.category("՞") == "Po"

    assert normalize_text("Ի՞նչ կա։", get_policy("whisper-marks")) == "ի նչ կա"
    assert normalize_text("Ի՞նչ կա։", get_policy("armenian-hy")) == "ինչ կա"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ի՞նչ կա։", "ինչ կա"),
        ("Բարև ձեզ", "բարեւ ձեզ"),
        ("Այո՛, ճիշտ է։", "այո ճիշտ է"),
        ("և՛ այս և՛ այն", "եւ այս եւ այն"),
        ("Հայերեն լեզու", "հայերեն լեզու"),
    ],
)
def test_armenian_table(text: str, expected: str) -> None:
    assert normalize_text(text, get_policy("armenian-hy")) == expected


def test_the_armenian_ligature_is_a_letter_that_the_form_splits_in_two() -> None:
    """Applied identically to both sides, so it costs nothing — but it is why
    the expected output reads two letters where the input read one."""
    assert unicodedata.category("և") == "Ll"
    assert unicodedata.normalize("NFKC", "և") == "եւ"


def test_korean_never_sees_the_compatibility_form() -> None:
    """NFKC composes two compatibility jamo into one syllable, which moves the
    character error rate denominator for a language scored on characters."""
    assert len(unicodedata.normalize("NFC", "ㄱㅡ 그")) == 4
    assert len(unicodedata.normalize("NFKC", "ㄱㅡ 그")) == 3

    assert get_policy("ko-kspon").form == "NFC"
    assert normalize_text("ㄱㅡ 그", get_policy("ko-kspon")) == "ㄱㅡ 그"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("안녕하세요, 반갑습니다.", "안녕하세요 반갑습니다"),
        ("AI 모델을 학습합니다.", "ai 모델을 학습합니다"),
        ("네? 아니요!", "네 아니요"),
        ("한국어 인식", "한국어 인식"),
    ],
)
def test_korean_table(text: str, expected: str) -> None:
    assert normalize_text(text, get_policy("ko-kspon")) == expected


def test_korean_is_written_with_spaces_so_word_error_rate_is_primary() -> None:
    """Unlike the other character-scored languages. Korean spacing is flexible,
    which inflates its word error rate, but the words are there to count."""
    assert "ko-kspon" not in NO_SPACE_POLICIES
    assert len(normalize_text("한국어 인식", get_policy("ko-kspon")).split()) == 2


def test_tibetan_keeps_its_marks_where_whisper_would_delete_them() -> None:
    """Tibetan stacks vowel signs and subjoined consonants as category Mn, so
    the mark rule reduces a word to its root letters."""
    text = "བོད་སྐད་ཡིན།"

    assert normalize_text(text, get_policy("whisper-basic")) == "བ ད ས ད ཡ ན "
    assert normalize_text(text, get_policy("tibetan-syllable")) == "བོད སྐད ཡིན"


def test_the_tibetan_syllable_mark_becomes_a_space() -> None:
    """The tsheg is where Tibetan separates syllables, and it is punctuation, so
    spacing it costs no characters and leaves the boundary visible."""
    assert unicodedata.category("་") == "Po"

    out = normalize_text("བོད་སྐད", get_policy("tibetan-syllable"))
    assert out == "བོད སྐད"
    assert len(out) == len("བོད་སྐད")


def test_tibetan_is_scored_on_characters() -> None:
    assert "tibetan-syllable" in NO_SPACE_POLICIES


def test_kazakh_cyrillic_is_safe_under_whisper() -> None:
    """Which is what makes assigning it Whisper's normalizer safe; no Kazakh
    normalization convention could be sourced."""
    text = "Қазақ тілі әдемі"
    assert [
        c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c)[0] == "M"
    ] == []
    assert normalize_text(text, get_policy("whisper-basic")).strip() == text.lower()


def test_kazakh_does_not_get_the_russian_yo_fold() -> None:
    """Folding ё to е is a Russian convention from Russian corpora. Importing it
    into Kazakh would be applying another language's rule."""
    assert policy_for_language("kk").version != "cyrillic-yo"


def test_no_existing_policy_hash_moved() -> None:
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
        "thai-cer": "ba69b63847b2",
        "han-mer": "eec15f0541a3",
    }
    for name, digest in frozen.items():
        assert get_policy(name).policy_hash() == digest, name


def test_the_three_new_policies_have_distinct_hashes() -> None:
    """Two of them carry the same rules as an existing policy and differ only in
    name and in the metric their languages take, which is why the version is
    part of what is hashed."""
    names = ("armenian-hy", "ko-kspon", "tibetan-syllable")
    hashes = {n: get_policy(n).policy_hash() for n in names}

    assert len(set(hashes.values())) == 3
    assert get_policy("ko-kspon").policy_hash() != get_policy("latin-marks").policy_hash()


@pytest.mark.parametrize(
    ("policy", "text"),
    [
        ("armenian-hy", "Ի՞նչ կա։"),
        ("ko-kspon", "안녕하세요, 반갑습니다."),
        ("tibetan-syllable", "བོད་སྐད་ཡིན།"),
    ],
)
def test_the_new_policies_are_idempotent(policy: str, text: str) -> None:
    once = normalize_text(text, get_policy(policy))
    assert normalize_text(once, get_policy(policy)) == once
