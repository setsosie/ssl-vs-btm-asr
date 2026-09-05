"""Whisper's normalizer is the default; everything else is a documented exception.

A language uses something other than `whisper-basic` only where that normalizer
is demonstrably destructive or word-breaking for its orthography. This file is
the evidence: the check that decides it, run per language, so a future language
cannot be moved onto the default without passing the same test.
"""

from __future__ import annotations

import unicodedata

import pytest

from svb.text.normalize import get_policy, normalize_text
from svb.text.registry import (
    LANGUAGE_POLICIES,
    SCRIPT_POLICIES,
    policy_for_language,
)

WHISPER = get_policy("whisper-basic")


def _damage(letters: str, phrase: str) -> tuple[int, list[str], int, int]:
    """What Whisper's normalizer costs an orthography.

    Returns the residual combining marks its letters leave under NFKC, any
    letter its mark/symbol/punctuation rule would eat, and the word count of a
    real phrase before and after. A language is safe on the default when the
    marks are zero, no letter is eaten, and the word count is unchanged.
    """
    residual = [
        c
        for letter in letters
        for c in unicodedata.normalize("NFKC", letter)
        if unicodedata.category(c)[0] == "M"
    ]
    eaten = [
        letter
        for letter in letters
        if any(unicodedata.category(c)[0] in "SP" for c in unicodedata.normalize("NFKC", letter))
    ]
    phrase_marks = [
        c for c in unicodedata.normalize("NFKC", phrase) if unicodedata.category(c)[0] == "M"
    ]
    return (
        len(residual) + len(phrase_marks),
        eaten,
        len(phrase.split()),
        len(normalize_text(phrase, WHISPER).split()),
    )


# (code, the orthography's non-ASCII letters, a phrase in the language)
SAFE_ON_THE_DEFAULT = [
    ("lg", "ŋ", "ŋŋaana eŋŋoma"),
    ("kab", "ɛɣḥṭẓčǧṛ", "aɣerbaz iḥeddaden ṭṭaxi"),
    ("uz", "ʻ", "oʻzbek gʻalaba"),
    ("kmr", "çêîşû", "kurdî ye şêrîn"),
    ("jv", "éèêåḍṭ", "sugeng énjang"),
    ("su", "é", "kumaha damang"),
    ("zu", "", "sawubona umhlaba"),
    ("xh", "", "molo mhlaba"),
    ("nso", "šêô", "dumela lefase"),
    ("ts", "", "avuxeni misava"),
    ("ve", "ḓḽṋṱṅ", "ndaa shango ḽoṱhe"),
    ("ha", "ƴƙɓɗ", "ƴan ƙasar ɓera"),
    ("vi", "ếệđươ", "Tiếng Việt rất đẹp"),
    ("id", "", "halo dunia"),
    ("ms", "", "helo dunia"),
]


@pytest.mark.parametrize(
    ("code", "letters", "phrase"), SAFE_ON_THE_DEFAULT, ids=[s[0] for s in SAFE_ON_THE_DEFAULT]
)
def test_the_default_is_harmless_for_these_orthographies(
    code: str, letters: str, phrase: str
) -> None:
    """The gate for putting a language on the default. Nothing may skip it."""
    marks, eaten, words_in, words_out = _damage(letters, phrase)

    assert marks == 0, f"{code}: {marks} combining mark(s) would be replaced with a space"
    assert eaten == [], f"{code}: {eaten} would be replaced with a space"
    assert words_in == words_out, f"{code}: {words_in} words became {words_out}"
    assert policy_for_language(code).version == "whisper-basic"


# (code, policy, the concrete failure)
LATIN_EXCEPTIONS = [
    ("sw", "latin-marks", "ng'ombe na ng'ambo", 3),
    ("rw", "latin-marks", "amashyirahamwe y'u Rwanda", 3),
]


@pytest.mark.parametrize(
    ("code", "policy", "phrase", "words"),
    LATIN_EXCEPTIONS,
    ids=[e[0] for e in LATIN_EXCEPTIONS],
)
def test_an_intra_word_apostrophe_is_word_breaking_under_the_default(
    code: str, policy: str, phrase: str, words: int
) -> None:
    """The apostrophe is punctuation, so the default turns one word into two.

    Swahili writes its velar nasal `ng'` and Kinyarwanda its elision `y'u` with
    one, inside the word.
    """
    assert len(normalize_text(phrase, WHISPER).split()) > words
    assert len(normalize_text(phrase, get_policy(policy)).split()) == words
    assert policy_for_language(code).version == policy


@pytest.mark.parametrize(("code", "phrase"), [("yo", "Ọjọ́ ìbílẹ̀"), ("ig", "Ọ́ bụ̀ ihe")])
def test_a_tone_mark_with_no_precomposed_form_survives_only_off_the_default(
    code: str, phrase: str
) -> None:
    """Yoruba and Igbo stack a tone accent on a dot-below vowel, and that
    combination has no single code point, so the mark rule deletes the tone."""
    marks, _, _, _ = _damage("", phrase)
    assert marks > 0

    assert normalize_text(phrase, get_policy("latin-marks")) == phrase.lower()
    assert normalize_text(phrase, WHISPER).strip() != phrase.lower()
    assert policy_for_language(code).version == "latin-marks"


EXCEPTIONS = {
    "hi": "indic-vistaar",
    "bn": "indic-vistaar",
    "kn": "indic-vistaar",
    "ne": "indic-vistaar",
    "si": "indic-vistaar",
    "ta": "indic-vistaar",
    "th": "thai-cer",
    "zh-CN": "han-mer",
    "yue": "han-mer",
    "ja": "ja-cer",
    "ko": "ko-kspon",
    "hy": "armenian-hy",
    "bo": "tibetan-syllable",
    "tr": "turkic-tr",
    "ar": "arabic-ouaal",
    "fa": "perso-arabic",
    "ur": "perso-arabic",
    "ps": "perso-arabic",
    "ckb": "perso-arabic",
    "ug": "uyghur-ug",
    "sw": "latin-marks",
    "rw": "latin-marks",
}


@pytest.mark.parametrize(("code", "policy"), sorted(EXCEPTIONS.items()))
def test_every_exception_still_resolves(code: str, policy: str) -> None:
    assert policy_for_language(code).version == policy


def test_a_language_the_tables_do_not_mention_takes_the_default() -> None:
    """With a warning: a default that nobody sees is how a script gets scored
    under rules written for another one."""
    with pytest.warns(UserWarning, match="zz.*whisper-basic"):
        assert policy_for_language("zz").version == "whisper-basic"


def test_a_language_with_a_recorded_script_and_no_exception_takes_the_default() -> None:
    """European Latin, Cyrillic and Georgian are no longer table entries at all,
    because the default is what they wanted."""
    for code in ("en", "de", "ru", "ka", "kk", "pt", "cs", "et"):
        assert code not in LANGUAGE_POLICIES
        assert policy_for_language(code).version == "whisper-basic"


def test_the_tables_hold_only_exceptions() -> None:
    """Neither table may name a language or script that takes the default: an
    entry that agrees with the default is a rule nobody can see the point of."""
    assert "whisper-basic" not in set(LANGUAGE_POLICIES.values())
    assert "whisper-basic" not in set(SCRIPT_POLICIES.values())
    assert "Latn" not in SCRIPT_POLICIES


def test_the_unassigned_policies_are_still_reachable() -> None:
    """Kept available by name even where no language uses them."""
    for name in ("latin-marks", "cyrillic-yo", "whisper-marks", "svb-norm-1"):
        assert get_policy(name).version == name


def test_no_policy_hash_moved() -> None:
    frozen = {
        "svb-norm-1": "205fefc26d0e",
        "whisper-basic": "3870eb0765ba",
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
        "armenian-hy": "4044356f052f",
        "ko-kspon": "f47fec6c0d16",
        "tibetan-syllable": "e116c770ae0c",
    }
    for name, digest in frozen.items():
        assert get_policy(name).policy_hash() == digest, name
