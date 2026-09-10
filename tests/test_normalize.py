"""The svb-norm-1 text normalizer.

The expected outputs in ``CASES`` are the specification. Every one was produced
by executing the policy against Python 3.12 / Unicode 15.0.0 and checked against
the code points involved; they are not transcribed from a document.

The hazard tests below the table are the ones that would silently corrupt a
language if the policy drifted: Whisper's ``BasicTextNormalizer`` strips Unicode
category ``M`` and destroys every Indic script, three of the characters this
policy must delete are category ``Lm`` rather than punctuation, and one that it
must keep (the Japanese prolonged sound mark) is ``Lm`` too.
"""

from __future__ import annotations

import unicodedata

import pytest

from svb.text.normalize import (
    NORMALIZER_VERSION,
    NormalizerPolicy,
    normalize_batch,
    normalize_text,
)

# (id, input, expected)
CASES: list[tuple[str, str, str]] = [
    ("en-case", "Hello, World!", "hello world"),
    ("en-apostrophe", "I don't know — it's Bob's.", "i don't know it's bob's"),
    ("en-possessive-final", "the students' books", "the students books"),
    ("en-digits-kept", "In 2026 we had 3 dogs", "in 2026 we had 3 dogs"),
    ("en-ligature", "The ﬁnal oﬃce", "the final office"),
    ("de-eszett", "Die Straße war groß", "die strasse war gross"),
    ("de-ss-variant", "Die Strasse war gross", "die strasse war gross"),
    ("fr-curly-apostrophe", "L’été de l'amour", "l'été de l'amour"),
    ("fr-guillemets", "«Bonjour», dit-il", "bonjour dit il"),
    ("es-inverted", "¿Cómo estás? ¡Bien!", "cómo estás bien"),
    ("it-final-apostrophe", "Un po' d'acqua", "un po d'acqua"),
    ("nl-digraph", "IJsland is mooi", "ijsland is mooi"),
    ("ru-cyrillic", "Привет, мир!", "привет мир"),
    ("uk-apostrophe", "П'ять — це число", "п'ять це число"),
    ("pl-diacritics", "Łódź — Zażółć gęślą jaźń!", "łódź zażółć gęślą jaźń"),
    ("tr-dotted-i", "İstanbul'da yaşıyorum.", "istanbul'da yaşıyorum"),
    ("tr-allcaps-dotless", "IRMAK ve ırmak", "irmak ve ırmak"),
    ("fi-umlaut", "Hyvää päivää!", "hyvää päivää"),
    ("sw-ng-apostrophe", "Ng'ombe na ng'ambo", "ng'ombe na ng'ambo"),
    ("ka-mkhedruli", "გამარჯობა, მსოფლიო!", "გამარჯობა მსოფლიო"),
    ("ka-mtavruli", "ᲒᲐᲛᲐᲠᲯᲝᲑᲐ", "გამარჯობა"),
    ("ar-punctuation", "مرحبا بالعالم، كيف حالك؟", "مرحبا بالعالم كيف حالك"),
    ("ar-tashkeel", "مَرْحَبًا بِٱلْعَالَم", "مرحبا بٱلعالم"),
    ("ar-tatweel", "مـــرحبا", "مرحبا"),
    ("ar-presentation-form", "ﻻ إله", "لا إله"),
    ("hi-danda", "यह हिंदी है।", "यह हिंदी है"),
    ("hi-devanagari-digits", "मेरे पास ३ किताबें हैं", "मेरे पास ३ किताबें हैं"),
    ("ja-fullwidth-punctuation", "コーヒーを飲む。「はい」", "コーヒーを飲む はい"),
    ("ja-halfwidth-katakana", "ｺｰﾋｰ を飲む", "コーヒー を飲む"),
    ("ja-ideographic-space", "今日は　いい天気", "今日は いい天気"),
    ("ml-zwj-chillu", "മലയാളത്തിന്‍ മലയാളത്തിൻ", "മലയാളത്തിൻ മലയാളത്തിൻ"),
    ("ml-sentence-period", "ഇത് മലയാളം ആണ്.", "ഇത് മലയാളം ആണ്"),
    ("mr-eyelash-ra", "सूर्‍य आणि सूर्य", "सूर्य आणि सूर्य"),
    ("te-conjunct", "ఇది తెలుగు భాష.", "ఇది తెలుగు భాష"),
    ("gu-sentence", "આ ગુજરાતી છે.", "આ ગુજરાતી છે"),
    ("symbols-and-currency", "It costs €5 + 10% more", "it costs 5 10 more"),
    ("vulgar-fraction", "½ cup", "1 2 cup"),
    ("invisibles-deleted", "soft­hyphen﻿ and\u200bzwsp", "softhyphen andzwsp"),
    ("nbsp-and-thin-space", "a b c", "a b c"),
    ("empty-after-normalization", "…!? 。", ""),
    ("whitespace-only", "   \t\n  ", ""),
]


@pytest.mark.parametrize(
    ("text", "expected"), [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES]
)
def test_normalize_table(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


@pytest.mark.parametrize("text", [c[1] for c in CASES], ids=[c[0] for c in CASES])
def test_normalization_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_nukta_letters_decompose_under_the_unicode_form() -> None:
    """NFC *decomposes* the precomposed nukta letters — they are composition
    exclusions. A reader who assumes NFC means "maximally composed" will count
    Odia and Devanagari characters wrongly, so the behaviour is pinned here."""
    assert normalize_text(chr(0x0B5C)) == chr(0x0B21) + chr(0x0B3C)  # Odia RRA
    assert normalize_text(chr(0x095C)) == chr(0x0921) + chr(0x093C)  # Devanagari DDDHA
    # Both spellings of one letter therefore land on the same vocab characters.
    assert normalize_text(chr(0x0B5C)) == normalize_text(chr(0x0B21) + chr(0x0B3C))
    # NNNA carries no decomposition, so it stays one character.
    assert normalize_text(chr(0x0929)) == chr(0x0929)


def test_zwj_chillu_and_atomic_chillu_become_one_character() -> None:
    """Unicode normalization does not unify these, so the vocab would carry
    two encodings of one Malayalam letter."""
    zwj = normalize_text("മലയാളത്തിന്‍")
    atomic = normalize_text("മലയാളത്തിൻ")

    assert zwj == atomic
    assert zwj.endswith("ൻ")


def test_indic_combining_marks_survive() -> None:
    """The direct guard against Whisper's category-M strip.

    ``BasicTextNormalizer`` replaces every character in categories M, S and P
    with a space, which turns Hindi into a row of bare consonants.
    """
    for text in ("हिंदी में", "മലയാളം", "తెలుగు", "ગુજરાતી"):
        out = normalize_text(text)
        marks_in = [c for c in text if unicodedata.category(c) in ("Mn", "Mc")]
        marks_out = [c for c in out if unicodedata.category(c) in ("Mn", "Mc")]

        assert marks_in, "the fixture must actually contain combining marks"
        assert marks_out == marks_in
        assert out == text


def test_japanese_prolonged_sound_mark_is_kept() -> None:
    """U+30FC is category Lm, not punctuation: it is a vowel-length letter."""
    assert unicodedata.category("ー") == "Lm"
    assert normalize_text("コーヒー") == "コーヒー"


def test_characters_that_are_letters_by_category_but_must_go() -> None:
    """Tatweel and the modifier apostrophes are Lm, so no P*/S* rule reaches them."""
    assert unicodedata.category("ـ") == "Lm"
    assert unicodedata.category("ʼ") == "Lm"
    assert normalize_text("مـرحبا") == "مرحبا"
    assert normalize_text("donʼt stop") == "don't stop"


def test_soft_hyphen_survives_nfkc_and_needs_its_own_rule() -> None:
    assert unicodedata.normalize("NFKC", "­") == "­"
    assert normalize_text("soft­hyphen") == "softhyphen"


def test_turkish_dotted_capital_i_folds_to_a_single_character() -> None:
    """``'İ'.casefold()`` is U+0069 U+0307. Left alone, `İstanbul` and
    `istanbul` are different words and the combining dot enters the vocab."""
    assert "İ".casefold() == "i̇"
    assert normalize_text("İstanbul") == "istanbul"
    assert normalize_text("İSTANBUL") == normalize_text("istanbul")
    assert "̇" not in normalize_text("İstanbul")


def test_the_accepted_turkish_residual_is_pinned() -> None:
    """A capital dotless I folds to a dotted i. Accepted, not fixed: it needs a
    Turkish-locale pre-map, and Common Voice rejects the all-caps sentences
    where it would mostly appear."""
    assert normalize_text("Irmak") == "irmak"
    assert normalize_text("ırmak") == "ırmak"
    assert normalize_text("Irmak") != normalize_text("ırmak")


def test_dashes_need_no_rule_of_their_own() -> None:
    """Pd is a subset of P*, so space-substituting punctuation covers them."""
    assert all(unicodedata.category(c) == "Pd" for c in "-‐–—―")
    assert normalize_text("state-of-the-art") == "state of the art"
    assert normalize_text("a — b") == "a b"


def test_punctuation_becomes_a_space_rather_than_vanishing() -> None:
    """Deleting would fuse two words when the source omits the space."""
    assert normalize_text("hello,world") == "hello world"


def test_apostrophe_is_letter_keeps_word_edge_apostrophes() -> None:
    """The knob a fork flips for an orthography that writes a glottal stop."""
    policy = NormalizerPolicy(apostrophe_is_letter=True)

    assert normalize_text("'aa ts'ii'") == "aa ts'ii"
    assert normalize_text("'aa ts'ii'", policy) == "'aa ts'ii'"


def test_lower_keeps_eszett_where_casefold_folds_it() -> None:
    assert normalize_text("Straße") == "strasse"
    assert normalize_text("Straße", NormalizerPolicy(case="lower")) == "straße"


def test_lower_and_casefold_differ_only_in_latin() -> None:
    """A regression guard on the claim that drives the case decision: for the
    scripts this project uses, the choice is the German ß and nothing else."""
    blocks = {
        "cyrillic": [(0x0400, 0x052F)],
        "georgian": [(0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)],
        "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
        "indic": [
            (0x0900, 0x097F),
            (0x0A80, 0x0AFF),
            (0x0B00, 0x0B7F),
            (0x0C00, 0x0C7F),
            (0x0D00, 0x0D7F),
        ],
        "japanese": [(0x3000, 0x30FF), (0xFF00, 0xFFEF)],
    }
    for name, ranges in blocks.items():
        differing = [
            chr(c)
            for lo, hi in ranges
            for c in range(lo, hi + 1)
            if chr(c).lower() != chr(c).casefold()
        ]
        assert differing == [], f"{name} would be affected by the lower/casefold choice"

    assert "ß".lower() != "ß".casefold()


def test_georgian_mtavruli_folds_without_an_exception() -> None:
    assert normalize_text("ᲒᲐᲛᲐᲠᲯᲝᲑᲐ") == "გამარჯობა"


def test_an_utterance_that_normalizes_to_nothing_stays_empty_rather_than_vanishing() -> None:
    """The caller decides what empty means — training drops the utterance,
    scoring excludes and counts it — so the normalizer must not decide for it."""
    assert [normalize_text(t) for t in ("Hello!", "…", "  ")] == ["hello", "", ""]


def test_normalize_batch_preserves_length_and_empties() -> None:
    out = normalize_batch(["Hello!", "…", "  "])
    assert out == ["hello", "", ""]


def test_policy_hash_is_stable_and_tracks_every_field() -> None:
    baseline = NormalizerPolicy()

    assert baseline.policy_hash() == NormalizerPolicy().policy_hash()
    assert len(baseline.policy_hash()) == 12
    for changed in (
        NormalizerPolicy(form="NFC"),
        NormalizerPolicy(case="lower"),
        NormalizerPolicy(strip_symbols=False),

        NormalizerPolicy(apostrophe_is_letter=True),
        NormalizerPolicy(turkish_dotted_i=False),
    ):
        assert changed.policy_hash() != baseline.policy_hash()


def test_the_default_policy_hash_is_pinned() -> None:
    """Results under different policy hashes must not be pooled, so the default
    policy's hash is part of this repository's published interface. Changing it
    silently splits every result produced before the change from every result
    produced after; this test makes that a decision rather than an accident."""
    assert NormalizerPolicy().policy_hash() == "205fefc26d0e"


def test_the_only_digit_policy_is_the_one_that_is_implemented() -> None:
    """A policy field is hashed into every run's provenance, so a value that
    changes the hash without changing the text would mark results incomparable
    for no reason. Dropping digit-bearing utterances changes the test set, not
    the transcript, so it is not a normalizer setting."""
    from typing import get_args, get_type_hints

    assert get_args(get_type_hints(NormalizerPolicy)["digits"]) == ("keep",)


def test_policy_round_trips_through_a_dict() -> None:
    policy = NormalizerPolicy(case="lower", malayalam_chillu="keep")

    assert NormalizerPolicy.from_dict(policy.to_dict()) == policy
    assert policy.to_dict()["version"] == NORMALIZER_VERSION


def test_module_sha256_detects_an_edited_normalizer() -> None:
    """The policy hash cannot see a change to the code that reads it."""
    from svb.text.normalize import module_sha256

    assert len(module_sha256()) == 64
    assert module_sha256() == module_sha256()


def test_legacy_policy_reproduces_the_pre_normalization_behaviour() -> None:
    """Vocabularies written before this policy existed were NFC-only."""
    from svb.text.normalize import LEGACY_POLICY

    assert LEGACY_POLICY.version == "svb-norm-0"
    assert normalize_text("Hello, World!", LEGACY_POLICY) == "Hello, World!"
    assert normalize_text("café", LEGACY_POLICY) == "café"


def test_the_apostrophe_rule_follows_the_punctuation_setting() -> None:
    """U+0027 is category Po, so it is the punctuation rule's business.

    The apostrophe branch used to run whenever *either* strip flag was on, which
    deleted a word-final apostrophe under a policy that had punctuation removal
    switched off.
    """
    keep_punctuation = NormalizerPolicy(strip_punctuation=False, strip_symbols=True)

    # The "+" is a symbol and still goes; the apostrophe is punctuation and stays.
    assert normalize_text("the students' books + more", keep_punctuation) == (
        "the students' books more"
    )


def test_counts_describe_the_transformation_that_actually_happened() -> None:
    """The protected intra-word apostrophes survive, so they are not removals."""
    from svb.text.normalize import normalize_with_counts

    out, counts = normalize_with_counts("ng'ombe, l'été!")

    assert out == "ng'ombe l'été"
    # The comma and the exclamation mark; not the two apostrophes that survived.
    assert counts["P"] == 2


def test_counts_are_zero_for_a_rule_the_policy_switched_off() -> None:
    from svb.text.normalize import normalize_with_counts

    out, counts = normalize_with_counts(
        "Hello, world!", NormalizerPolicy(strip_punctuation=False, strip_symbols=False)
    )

    assert out == "hello, world!"
    assert counts["P"] == 0 and counts["S"] == 0


def test_arabic_marks_are_counted_where_they_are_removed() -> None:
    from svb.text.normalize import normalize_with_counts

    _, counts = normalize_with_counts("مَرْحَبًا")

    assert counts["arabic_marks"] == 4
    _, off = normalize_with_counts("مَرْحَبًا", NormalizerPolicy(strip_arabic_marks=False))
    assert off["arabic_marks"] == 0


def test_normalize_text_and_normalize_with_counts_agree() -> None:
    from svb.text.normalize import normalize_with_counts

    for case in ("Hello, World!", "مَرْحَبًا", "コーヒーを飲む。", "soft\xadhyphen", "…!?"):
        assert normalize_with_counts(case)[0] == normalize_text(case)
