"""The ``whisper-basic`` preset, against openai/whisper's own output.

Every expected string in ``CASES`` was produced by *running* the upstream
normalizer (``whisper/normalizers/basic.py``, ``BasicTextNormalizer``), not by
reading it. The preset's claim is exact reproduction, so a table of hand-written
approximations would be worse than no table.

The Indic and Arabic rows are the reason the default policy is not this one.
Whisper replaces every character in Unicode category M with a space, which
reduces an abugida to its bare consonants and shatters vocalized Arabic into
single letters. See ``docs/normalization.md``.
"""

from __future__ import annotations

import pytest

from svb.text.normalize import (
    ADDITIONAL_DIACRITICS,
    POLICIES,
    NormalizerPolicy,
    get_policy,
    normalize_text,
    policy_name,
)

# (id, input, whisper-basic, whisper-basic-nodiacritics)
CASES: list[tuple[str, str, str, str]] = [
    ("en-case-and-punctuation", "Hello, World!", "hello world ", "hello world "),
    ("en-apostrophe-split", "I don't know", "i don t know", "i don t know"),
    ("en-bracket-and-paren", "[noise] hello (laughs) there", " hello there", " hello there"),
    ("en-angle-span", "<unk> spoken words", " spoken words", " spoken words"),
    ("en-nested-parens-are-non-greedy", "a (b (c) d) e", "a d e", "a d e"),
    ("de-eszett", "Die Straße war groß", "die straße war groß", "die strasse war gross"),
    ("fr-curly-apostrophe", "L’été de l'amour", "l été de l amour", "l ete de l amour"),
    (
        "pl-diacritics",
        "Łódź — Zażółć gęślą jaźń!",
        "łódź zażółć gęślą jaźń ",
        "lodz zazolc gesla jazn ",
    ),
    ("tr-dotted-i", "İstanbul'da yaşıyorum.", "i stanbul da yaşıyorum ", "istanbul da yasıyorum "),
    ("ru-cyrillic", "Привет, мир!", "привет мир ", "привет мир "),
    ("hi-devanagari-destroyed", "हिंदी में", "ह द म ", "ह द म"),
    ("hi-danda", "यह हिंदी है।", "यह ह द ह ", "यह ह द ह "),
    ("ml-malayalam-destroyed", "ഇത് മലയാളം ആണ്.", "ഇത മലയ ള ആണ ", "ഇത മലയ ള ആണ "),
    ("te-telugu-destroyed", "ఇది తెలుగు భాష.", "ఇద త ల గ భ ష ", "ఇద తల గ భష "),
    ("ar-tashkeel", "مَرْحَبًا بِٱلْعَالَم", "م ر ح ب ا ب ٱل ع ال م", "مرحبا بٱلعالم"),
    (
        "ar-punctuation",
        "مرحبا بالعالم، كيف حالك؟",
        "مرحبا بالعالم كيف حالك ",
        "مرحبا بالعالم كيف حالك ",
    ),
    (
        "ja-fullwidth-punctuation",
        "コーヒーを飲む。「はい」",
        "コーヒーを飲む はい ",
        "コーヒーを飲む はい ",
    ),
    ("ja-halfwidth-katakana", "ｺｰﾋｰ を飲む", "コーヒー を飲む", "コーヒー を飲む"),
    ("es-inverted", "¿Cómo estás? ¡Bien!", " cómo estás bien ", " como estas bien "),
    ("symbols-and-currency", "It costs €5 + 10% more", "it costs 5 10 more", "it costs 5 10 more"),
    ("cafe-accent", "café naïve", "café naïve", "cafe naive"),
    ("empty-after-normalization", "…!? 。", " ", " "),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [(c[1], c[2]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_whisper_basic_matches_upstream(text: str, expected: str) -> None:
    assert normalize_text(text, get_policy("whisper-basic")) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [(c[1], c[3]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_whisper_basic_nodiacritics_matches_upstream(text: str, expected: str) -> None:
    assert normalize_text(text, get_policy("whisper-basic-nodiacritics")) == expected


def test_the_diacritics_table_is_upstreams() -> None:
    """Copied verbatim from openai/whisper; a divergence here is a wrong answer."""
    assert ADDITIONAL_DIACRITICS == UPSTREAM_ADDITIONAL_DIACRITICS


def test_whisper_output_is_not_stripped() -> None:
    """Upstream collapses whitespace runs but never strips the ends.

    Faithful rather than tidied: a leading space is one character of CER.
    """
    assert normalize_text("Hello!", get_policy("whisper-basic")) == "hello "
    assert normalize_text("[noise] hi", get_policy("whisper-basic")) == " hi"
    assert normalize_text("Hello!") == "hello"


def test_the_registry_names_every_preset_and_refuses_the_rest() -> None:
    assert set(POLICIES) == {"svb-norm-1", "whisper-basic", "whisper-basic-nodiacritics"}
    assert get_policy("svb-norm-1") == NormalizerPolicy()
    with pytest.raises(KeyError, match="unknown normalization policy"):
        get_policy("whisper")


def test_every_preset_carries_its_own_version_and_hash() -> None:
    """The version is what tells two pipelines apart inside a policy record,
    since each records only the fields its own pipeline reads."""
    versions = {name: p.version for name, p in POLICIES.items()}
    hashes = {name: p.policy_hash() for name, p in POLICIES.items()}

    assert len(set(versions.values())) == len(POLICIES)
    assert len(set(hashes.values())) == len(POLICIES)
    for name, policy in POLICIES.items():
        assert policy_name(policy) == name


def test_a_hand_rolled_policy_has_no_preset_name() -> None:
    assert policy_name(NormalizerPolicy(case="lower")) is None


def test_a_whisper_policy_records_only_whisper_rules() -> None:
    record = get_policy("whisper-basic").to_dict()

    assert record["pipeline"] == "whisper"
    assert record["strip_marks"] is True
    assert record["drop_bracketed_spans"] is True
    assert record["strip_whitespace"] is False
    # Rules that belong to the other pipeline are absent, not reported as on.
    for absent in ("turkish_dotted_i", "malayalam_chillu", "strip_arabic_marks"):
        assert absent not in record


DEVIATIONS = [
    (
        "marks are replaced with a space",
        "\u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902",
    ),
    ("the intra-word apostrophe is not protected", "I don't know"),
    ("bracketed and parenthesized spans are deleted", "[noise] hello (laughs)"),
]


@pytest.mark.parametrize(("reason", "text"), DEVIATIONS, ids=[d[0] for d in DEVIATIONS])
def test_the_two_policies_differ_where_documented(reason: str, text: str) -> None:
    assert normalize_text(text) != normalize_text(text, get_policy("whisper-basic"))


AGREEMENTS = [
    "the cat sat on the mat",
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",
    "hyv\u00e4\u00e4 p\u00e4iv\u00e4\u00e4",
]


@pytest.mark.parametrize("text", AGREEMENTS)
def test_the_two_policies_agree_on_text_that_exercises_no_deviation(text: str) -> None:
    """Latin and Cyrillic prose with no marks, apostrophes or brackets.

    Whisper's trailing space is the only remaining difference, which is why the
    comparison strips: it is not one of the deviations under test.
    """
    assert normalize_text(text) == normalize_text(text, get_policy("whisper-basic")).strip()


UPSTREAM_ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}
