"""Which normalization policy a language gets, and why it is keyed on script.

A policy exists because of a writing system, not because of a language: the
reason Devanagari needs its combining marks kept is the same reason Telugu does,
and it has nothing to do with Hindi or Telugu in particular. So the default
table is script to policy, and the language table records only which script each
language is written in.

Every shipped preset names its policy explicitly anyway, in
``configs/scales/*.yaml``, so the choice is visible next to the language rather
than inferred three modules away. This table is the fallback for a language
added without that line, and the fallbacks are chosen to fail safe.

Scripts are ISO 15924 codes, matching ``docs/languages.md``. They are recorded
here rather than derived, because deriving a script from a language code means
guessing, and a wrong guess silently destroys a corpus.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

from .normalize import NormalizerPolicy, get_policy

# Language code to ISO 15924 script. Common Voice codes, plus the OpenSLR
# held-out set. A language absent here has to be added before it can be used,
# which is deliberate: the alternative is guessing.
LANGUAGE_SCRIPTS: dict[str, str] = {
    # European Latin
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
    "pt": "Latn",
    "cs": "Latn",
    "sk": "Latn",
    "ro": "Latn",
    "sv-SE": "Latn",
    "da": "Latn",
    "et": "Latn",
    "is": "Latn",
    "hr": "Latn",
    "lt": "Latn",
    "lv": "Latn",
    "cy": "Latn",
    "fy-NL": "Latn",
    # Latin, outside Europe or with live combining marks
    "tr": "Latn",
    "az": "Latn",
    "uz": "Latn",
    "sw": "Latn",
    "rw": "Latn",
    "lg": "Latn",
    "kab": "Latn",
    "kmr": "Latn",
    "vi": "Latn",
    "yo": "Latn",
    "ig": "Latn",
    "ha": "Latn",
    "id": "Latn",
    "ms": "Latn",
    "jv": "Latn",
    "su": "Latn",
    "zu": "Latn",
    "xh": "Latn",
    "nso": "Latn",
    "ts": "Latn",
    "ve": "Latn",
    # Cyrillic
    "ru": "Cyrl",
    "uk": "Cyrl",
    "be": "Cyrl",
    "ab": "Cyrl",
    "ba": "Cyrl",
    "mhr": "Cyrl",
    "bg": "Cyrl",
    "sr": "Cyrl",
    "mn": "Cyrl",
    "tt": "Cyrl",
    "kk": "Cyrl",
    "ky": "Cyrl",
    "kbd": "Cyrl",
    "ady": "Cyrl",
    # Other European scripts
    "ka": "Geor",
    "el": "Grek",
    "hy-AM": "Armn",
    "hy": "Armn",
    # Indic abugidas
    "hi": "Deva",
    "mr": "Deva",
    "ne-NP": "Deva",
    "ne": "Deva",
    "bn": "Beng",
    "pa-IN": "Guru",
    "gu": "Gujr",
    "or": "Orya",
    "ta": "Taml",
    "te": "Telu",
    "kn": "Knda",
    "ml": "Mlym",
    "si": "Sinh",
    # Arabic script
    "ar": "Arab",
    "fa": "Arab",
    "ur": "Arab",
    "ps": "Arab",
    "ug": "Arab",
    "ckb": "Arab",
    # East Asian and mainland South-East Asian
    "ja": "Jpan",
    "zh-CN": "Hans",
    "zh-TW": "Hant",
    "yue": "Hant",
    "th": "Thai",
    "bo": "Tibt",
    "ko": "Kore",
    # The held-out corpora name their languages in full rather than by code.
    "malayalam": "Mlym",
    "marathi": "Deva",
    "telugu": "Telu",
    "gujarati": "Gujr",
    "odia": "Orya",
}

#: What every language gets unless an exception below says otherwise.
DEFAULT_POLICY_NAME = "whisper-basic"

# The scripts whose reference systems disagree with the default, and why. A
# script absent here takes the default; an entry that agreed with the default
# would be a rule nobody could see the point of, so there are none.
#
# Latin, Cyrillic and Georgian are deliberately absent: Whisper's normalizer is
# what those want, and it is now the default rather than a table entry.
SCRIPT_POLICIES: dict[str, str] = {
    # Abugidas. Whisper replaces Unicode category M with a space, and an Indic
    # vowel sign is a combining mark, so the default deletes the vowels and
    # leaves the consonant skeleton.
    "Deva": "indic-vistaar",
    "Beng": "indic-vistaar",
    "Guru": "indic-vistaar",
    "Gujr": "indic-vistaar",
    "Orya": "indic-vistaar",
    "Taml": "indic-vistaar",
    "Telu": "indic-vistaar",
    "Knda": "indic-vistaar",
    "Mlym": "indic-vistaar",
    "Sinh": "indic-vistaar",
    # Same defect, plus a word separator the default would delete rather than
    # space, and a letter the compatibility form splits in two.
    "Thai": "thai-cer",
    # Same defect: Tibetan stacks vowel signs and subjoined consonants as marks.
    "Tibt": "tibetan-syllable",
    # The compatibility form composes two jamo into one syllable, which moves
    # the character count for a language scored on characters.
    "Kore": "ko-kspon",
    # The question and emphasis marks are written inside the word, so the
    # default's punctuation rule splits every question in two.
    "Armn": "armenian-hy",
    # No text defect; a different metric and a name for it.
    "Hans": "han-mer",
    "Hant": "han-mer",
    "Jpan": "ja-cer",
}

# The languages whose policy differs from their script's answer, and why.
#
# The Arabic script has no entry of its own: its two conventions fold letters in
# opposite directions, so there is no script-wide answer, and each language names
# the one it follows. An Arabic-script language absent from this table takes the
# default rather than being refused, which normalizes nothing away — the default
# strips its vocalization along with everything else in category M.
LANGUAGE_POLICIES: dict[str, str] = {
    # Lowercasing the dotted capital I yields a combining dot that the default's
    # mark rule turns into a space, splitting every sentence-initial I-word.
    "tr": "turkic-tr",
    "az": "turkic-tr",
    # Arabic script: the conventions disagree, so each language names its own.
    "ar": "arabic-ouaal",
    **dict.fromkeys(["fa", "ur", "ckb", "ps"], "perso-arabic"),
    "ug": "uyghur-ug",
    # Latin orthographies the default is demonstrably wrong for. Swahili writes
    # its velar nasal `ng'` and Kinyarwanda its elision `y'u` with an apostrophe
    # *inside* the word, and an apostrophe is punctuation, so the default turns
    # one word into two. Yoruba and Igbo stack a tone accent on a dot-below
    # vowel, a combination with no single code point, so the mark rule deletes
    # the tone. Every other Latin orthography in the presets is verified
    # harmless on the default; see tests/test_whisper_default.py.
    **dict.fromkeys(["sw", "rw", "yo", "ig"], "latin-marks"),
}

# Policies whose languages are written without word separators, so character
# error rate is primary and a preset carrying one must set ``word_boundary``
# false. Checked by the test suite, so a regenerated preset cannot quietly
# disagree.
NO_SPACE_POLICIES = frozenset({"ja-cer", "thai-cer", "han-mer", "tibetan-syllable"})


def script_for_language(code: str) -> str:
    """The ISO 15924 script a language is written in.

    Raises:
        KeyError: When the language is not in the table. Deliberate: guessing a
            script from a language code is how a corpus gets silently destroyed.
    """
    try:
        return LANGUAGE_SCRIPTS[code]
    except KeyError:
        raise KeyError(
            f"no script recorded for language {code!r}; add it to LANGUAGE_SCRIPTS "
            "with its ISO 15924 script before using it"
        ) from None


def policy_for_language(code: str, explicit: str | None = None) -> NormalizerPolicy:
    """The policy a language is normalized under.

    Whisper's normalizer is the default. A language moves off it only where that
    normalizer is demonstrably destructive or word-breaking for its orthography,
    and every such case is an entry in one of the two tables above with its
    reason. Falling through to the default warns, because a default nobody sees
    is how a script ends up scored under rules written for another one.

    Args:
        code: The language code.
        explicit: The ``normalizer:`` named in the preset, which wins. Every
            shipped preset names one, including where it equals the default, so
            the choice is auditable from the preset alone.

    Raises:
        KeyError: The named policy does not exist.
    """
    if explicit is not None:
        return get_policy(explicit)
    if code in LANGUAGE_POLICIES:
        return get_policy(LANGUAGE_POLICIES[code])
    try:
        script = script_for_language(code)
    except KeyError:
        warnings.warn(
            f"no script recorded for {code!r}, so it falls back to the default "
            f"normalization policy {DEFAULT_POLICY_NAME!r}; add it to "
            "LANGUAGE_SCRIPTS, and to the exception tables if the default is "
            "wrong for its orthography",
            UserWarning,
            stacklevel=2,
        )
        return get_policy(DEFAULT_POLICY_NAME)
    return get_policy(SCRIPT_POLICIES.get(script, DEFAULT_POLICY_NAME))


def policies_for_specs(
    specs: Iterable[Any], override: str | None = None
) -> dict[str, NormalizerPolicy]:
    """The policy map a vocabulary is built with, one entry per language.

    Args:
        specs: Language specs, each carrying its ``code`` and its optional
            ``normalizer`` name.
        override: A policy name that replaces every language's own. This is the
            switch for scoring an entire run the way one published system
            scores, at the cost of whatever that system's rules do to the
            scripts it was not designed for.
    """
    return {
        spec.code: policy_for_language(spec.code, override or spec.normalizer) for spec in specs
    }
