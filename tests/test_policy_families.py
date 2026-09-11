"""The per-script policy table.

Numbered against the design document's case list; the expected strings were
produced by running these policies and cross-checked against that document's own
values. Forty of the forty-four agreed. The four that did not are all
`whisper-basic`, and all differ only by a leading or trailing space, because
upstream's `BasicTextNormalizer` collapses whitespace runs but never strips —
verified by running the upstream code, which contains no `.strip()`. The
document's prototype stripped; this table follows upstream.
"""

from __future__ import annotations

import unicodedata

import pytest

from svb.text.normalize import POLICIES, get_policy, normalize_text

FAMILY_POLICIES = (
    "whisper-basic",
    "whisper-marks",
    "cyrillic-yo",
    "turkic-tr",
    "latin-marks",
    "indic-vistaar",
    "arabic-ouaal",
    "perso-arabic",
    "uyghur-ug",
    "ja-cer",
)

# (id, policy, input, expected)
CASES: list[tuple[str, str, str, str]] = [
    (
        "01-whisper-basic-en-case-and-punctuation",
        "whisper-basic",
        "Hello, World! It's Bob's.",
        "hello world it s bob s ",
    ),
    (
        "02-whisper-basic-de-eszett-survives-lower()",
        "whisper-basic",
        "Die Straße war groß.",
        "die straße war groß ",
    ),
    (
        "03-whisper-basic-fr-curly-apostrophe-splits",
        "whisper-basic",
        "L’été de l'amour",
        "l été de l amour",
    ),
    (
        "04-whisper-basic-es-inverted-marks",
        "whisper-basic",
        "¿Cómo estás? ¡Bien!",
        " cómo estás bien ",
    ),
    (
        "05-whisper-basic-pl-precomposed-diacritics",
        "whisper-basic",
        "Łódź — Zażółć gęślą jaźń!",
        "łódź zażółć gęślą jaźń ",
    ),
    ("06-whisper-basic-nl-digraph", "whisper-basic", "IJsland is mooi", "ijsland is mooi"),
    (
        "07-whisper-basic-it-word-final-apostrophe",
        "whisper-basic",
        "Un po' d'acqua",
        "un po d acqua",
    ),
    ("08-cyrillic-yo-ru-yo-folded-to-e", "cyrillic-yo", "Ещё её приём, мир!", "еще ее прием мир"),
    ("09-cyrillic-yo-uk-apostrophe-protected", "cyrillic-yo", "П'ять — це число", "п'ять це число"),
    ("10-cyrillic-yo-be-Belarusian", "cyrillic-yo", "Прывітанне, свет!", "прывітанне свет"),
    (
        "14-turkic-tr-tr-dotted-I-locale-premap",
        "turkic-tr",
        "İstanbul'da yaşıyorum.",
        "istanbul'da yaşıyorum",
    ),
    ("15-turkic-tr-tr-dotless-I-preserved", "turkic-tr", "IRMAK ve ırmak", "ırmak ve ırmak"),
    ("16-turkic-tr-tr-Isik-keeps-dotless", "turkic-tr", "Işık ve ışık", "ışık ve ışık"),
    ("17-turkic-tr-az-Azerbaijani-schwa", "turkic-tr", "Əlifba və İldə", "əlifba və ildə"),
    (
        "29-latin-marks-vi-tone-marks-precomposed",
        "latin-marks",
        "Tiếng Việt rất đẹp, chào bạn!",
        "tiếng việt rất đẹp chào bạn",
    ),
    (
        "30-latin-marks-yo-combining-tone-and-dot",
        "latin-marks",
        "Ọjọ́ ìbílẹ̀ ni mo ń sọ.",
        "ọjọ́ ìbílẹ̀ ni mo ń sọ",
    ),
    ("31-latin-marks-yo-reordered-marks-unify", "latin-marks", "Ọjọ́ vs Ọjọ́", "ọjọ́ vs ọjọ́"),
    ("32-latin-marks-ig-dot-below-plus-tone", "latin-marks", "Ọ́ bụ̀ ihe.", "ọ́ bụ̀ ihe"),
    (
        "33-latin-marks-ha-hooked-letters",
        "latin-marks",
        "Ƴan ƙasar Hausa, ɓera.",
        "ƴan ƙasar hausa ɓera",
    ),
    (
        "34-latin-marks-sw-ng-apostrophe-protected",
        "latin-marks",
        "Ng'ombe na ng'ambo",
        "ng'ombe na ng'ambo",
    ),
    (
        "35-latin-marks-rw-apostrophe-protected",
        "latin-marks",
        "Amashyirahamwe y'u Rwanda",
        "amashyirahamwe y'u rwanda",
    ),
    ("37-indic-vistaar-hi-danda-deleted", "indic-vistaar", "यह हिंदी है।", "यह हिंदी है"),
    (
        "38-indic-vistaar-hi-matras-survive",
        "indic-vistaar",
        "मेरे पास ३ किताबें हैं।",
        "मेरे पास ३ किताबें हैं",
    ),
    ("39-indic-vistaar-mr-eyelash-ra-joiner", "indic-vistaar", "सूर्\u200dय आणि सूर्य", "सूर्य आणि सूर्य"),
    (
        "40-indic-vistaar-ml-joiner-chillu-unified",
        "indic-vistaar",
        "മലയാളത്തിന്\u200d മലയാളത്തിൻ",
        "മലയാളത്തിൻ മലയാളത്തിൻ",
    ),
    ("41-indic-vistaar-te-Telugu-conjunct", "indic-vistaar", "ఇది తెలుగు భాష.", "ఇది తెలుగు భాష"),
    ("42-indic-vistaar-gu-Gujarati", "indic-vistaar", "આ ગુજરાતી છે.", "આ ગુજરાતી છે"),
    ("43-indic-vistaar-or-Odia-nukta-decomposes", "indic-vistaar", "ଡ଼ି ଓଡ଼ିଆ।", "ଡ଼ି ଓଡ଼ିଆ"),
    ("44-indic-vistaar-ta-Tamil", "indic-vistaar", "இது தமிழ் மொழி.", "இது தமிழ் மொழி"),
    ("48-arabic-ouaal-ar-tashkeel-stripped", "arabic-ouaal", "مَرْحَبًا بِٱلْعَالَم", "مرحبا بالعالم"),
    ("49-arabic-ouaal-ar-tatweel-stripped", "arabic-ouaal", "مـــرحبا", "مرحبا"),
    (
        "50-arabic-ouaal-ar-alef-and-hamza-unified",
        "arabic-ouaal",
        "أحمد إبراهيم آمن ٱلله",
        "احمد ابراهيم امن الله",
    ),
    (
        "51-arabic-ouaal-ar-eastern-digits-to-ascii",
        "arabic-ouaal",
        "عندي ٣ كتب و ١٢ قلم",
        "عندي 3 كتب و 12 قلم",
    ),
    ("52-arabic-ouaal-ar-presentation-form", "arabic-ouaal", "ﻻ إله", "لا اله"),
    (
        "53-arabic-ouaal-ar-punctuation",
        "arabic-ouaal",
        "مرحبا بالعالم، كيف حالك؟",
        "مرحبا بالعالم كيف حالك",
    ),
    (
        "54-perso-arabic-fa-non-joiner-kept",
        "perso-arabic",
        "می\u200cروم به خانه.",
        "می\u200cروم به خانه",
    ),
    (
        "55-perso-arabic-fa-arabic-yeh-to-farsi-yeh",
        "perso-arabic",
        "علي ياسين كتاب",
        "علی یاسین کتاب",
    ),
    ("56-perso-arabic-fa-eastern-digits", "perso-arabic", "۱۲۳ کتاب", "123 کتاب"),
    ("57-perso-arabic-ur-Urdu-yeh-unified", "perso-arabic", "یہ اردو زبان ہے۔", "یہ اردو زبان ہے"),
    ("58-perso-arabic-ur-diacritics-stripped", "perso-arabic", "اُردُو زَبان", "اردو زبان"),
    ("59-uyghur-ug-ug-maksura-is-a-letter", "uyghur-ug", "ئۇيغۇر تىلى", "ئۇیغۇر تىلى"),
    ("60-uyghur-ug-ug-kaf-folded,-i-preserved", "uyghur-ug", "كىتاب ياخشى", "کىتاب یاخشى"),
    (
        "81-ja-cer-ja-fullwidth-punctuation",
        "ja-cer",
        "コーヒーを飲む。「はい」",
        "コーヒーを飲む はい",
    ),
    ("82-ja-cer-ja-halfwidth-katakana", "ja-cer", "ｺｰﾋｰ を飲む", "コーヒー を飲む"),
    ("83-ja-cer-ja-prolonged-mark-kept", "ja-cer", "ラーメンとビール", "ラーメンとビール"),
    ("84-ja-cer-ja-ideographic-space", "ja-cer", "今日は\u3000いい天気", "今日は いい天気"),
]


@pytest.mark.parametrize(
    ("policy", "text", "expected"),
    [(c[1], c[2], c[3]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_policy_table(policy: str, text: str, expected: str) -> None:
    assert normalize_text(text, get_policy(policy)) == expected


@pytest.mark.parametrize(
    ("policy", "text"),
    [(c[1], c[2]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_every_case_is_idempotent(policy: str, text: str) -> None:
    once = normalize_text(text, get_policy(policy))
    assert normalize_text(once, get_policy(policy)) == once


MARK_BEARING = [
    ("hi", "हिंदी में"),
    ("ml", "മലയാളം ഭാഷ"),
    ("te", "తెలుగు భాష"),
    ("gu", "ગુજરાતી ભાષા"),
    ("ta", "தமிழ் மொழி"),
    ("yo", "Ọjọ́ ìbílẹ̀"),
]


@pytest.mark.parametrize(("lang", "text"), MARK_BEARING, ids=[m[0] for m in MARK_BEARING])
def test_only_whisper_basic_destroys_combining_marks(lang: str, text: str) -> None:
    """The one defect the whole family table exists to avoid.

    Whisper replaces Unicode category M with a space, which deletes the vowel
    signs an abugida is written with. Every other policy leaves marks that it
    does not explicitly target alone.
    """
    marks = [c for c in unicodedata.normalize("NFC", text) if unicodedata.category(c)[0] == "M"]
    assert marks, "the fixture must contain combining marks"

    destroyed = normalize_text(text, get_policy("whisper-basic"))
    assert not [c for c in destroyed if unicodedata.category(c)[0] == "M"]

    for name in ("whisper-marks", "latin-marks", "indic-vistaar", "ja-cer"):
        kept = normalize_text(text, get_policy(name))
        assert [c for c in kept if unicodedata.category(c)[0] == "M"] == marks, name


def test_the_non_joiner_is_kept_for_perso_arabic_and_dropped_elsewhere() -> None:
    """It marks a morpheme boundary: deleting it turns one Persian word into two."""
    word = "می\u200cروم"

    assert "\u200c" in normalize_text(word, get_policy("perso-arabic"))
    assert "\u200c" not in normalize_text(word, get_policy("whisper-marks"))
    assert len(normalize_text(word, get_policy("perso-arabic")).split()) == 1


def test_uyghur_keeps_the_letter_persian_would_fold_away() -> None:
    """U+0649 writes /i/ in Uyghur; folding it onto yeh merges two letters."""
    text = "تىلى"

    assert "\u0649" in normalize_text(text, get_policy("uyghur-ug"))
    assert "\u0649" not in normalize_text(text, get_policy("perso-arabic"))


def test_arabic_and_perso_letter_maps_run_in_opposite_directions() -> None:
    """Arabic folds Persian letters onto Arabic ones; Perso-Arabic folds back."""
    assert normalize_text("كتاب", get_policy("arabic-ouaal")) == "كتاب"
    assert normalize_text("كتاب", get_policy("perso-arabic")) == "کتاب"


def test_every_family_policy_has_its_own_hash() -> None:
    """Two families sharing a hash would let their results be pooled."""
    hashes = {name: get_policy(name).policy_hash() for name in FAMILY_POLICIES}

    assert len(set(hashes.values())) == len(FAMILY_POLICIES)
    assert all(name in POLICIES for name in FAMILY_POLICIES)


def test_the_two_frozen_policies_are_still_reachable_by_name() -> None:
    """svb-norm-1 is what the comparability paragraph describes."""
    assert get_policy("svb-norm-1").policy_hash() == "205fefc26d0e"
    assert get_policy("whisper-basic").policy_hash() == "3870eb0765ba"
