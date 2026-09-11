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
    # Indic abugidas
    "hi": "Deva",
    "mr": "Deva",
    "ne-NP": "Deva",
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
    # The held-out corpora name their languages in full rather than by code.
    "malayalam": "Mlym",
    "marathi": "Deva",
    "telugu": "Telu",
    "gujarati": "Gujr",
    "odia": "Orya",
}

# Script to its default policy.
#
# ``Latn`` defaults to the mark-preserving policy rather than to
# ``whisper-basic``, which fails safe in both directions: a European language
# added without an explicit line gets a policy differing from whisper-basic only
# in ways Latin text barely notices, while a Yoruba or Vietnamese one keeps its
# tone marks instead of having them replaced with spaces.
#
# ``Arab`` is deliberately absent. Its two conventions fold letters in *opposite*
# directions — the Arabic one folds the Persian letters onto the Arabic ones, the
# Perso-Arabic one folds them back — so there is no guess that is not wrong for
# half the family. A language written in it must name its policy.
SCRIPT_POLICIES: dict[str, str] = {
    "Latn": "latin-marks",
    "Thai": "thai-cer",
    "Hans": "han-mer",
    "Hant": "han-mer",
    "Cyrl": "whisper-basic",
    "Geor": "whisper-basic",
    "Grek": "whisper-basic",
    "Armn": "whisper-basic",
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
    "Jpan": "ja-cer",
}


# Languages whose policy is not their script's default.
#
# The Latin script cannot decide this on its own: European Latin uses Whisper's
# normalizer as published, while Latin orthographies with live combining marks
# or an in-word apostrophe need the mark-preserving variant, and Turkish needs a
# locale case pre-map because lowercasing its dotted capital I yields a
# combining dot that the mark rule then turns into a space. The Arabic script
# cannot decide it either, for the reason given above.
#
# A language here overrides its script default; a preset naming a policy on the
# language's own line overrides both, and every shipped preset does.
LANGUAGE_POLICIES: dict[str, str] = {
    # European Latin and Cyrillic, and Georgian: Whisper's normalizer as
    # published. Verified safe for each — their diacritics are precomposed, so
    # the mark rule finds nothing to remove.
    **dict.fromkeys(
        [
            "en",
            "de",
            "fr",
            "es",
            "it",
            "nl",
            "pl",
            "fi",
            "ca",
            "eo",
            "eu",
            "hu",
            "gl",
            "pt",
            "cs",
            "sk",
            "ro",
            "sv-SE",
            "da",
            "et",
            "lt",
            "lv",
            "cy",
            "fy-NL",
            "ru",
            "uk",
            "be",
            "ab",
            "ba",
            "mhr",
            "bg",
            "sr",
            "mn",
            "tt",
            "kk",
            "ky",
            "kbd",
            "ady",
            "ka",
            "el",
            "hy-AM",
        ],
        "whisper-basic",
    ),
    "tr": "turkic-tr",
    "az": "turkic-tr",
    # Latin outside Europe, and Kurmanji, which the coordinator groups here.
    **dict.fromkeys(
        ["uz", "sw", "rw", "lg", "kab", "kmr", "vi", "yo", "ig", "ha", "id", "ms"], "latin-marks"
    ),
    "ar": "arabic-ouaal",
    **dict.fromkeys(["fa", "ur", "ckb", "ps"], "perso-arabic"),
    "ug": "uyghur-ug",
}

# Policies whose languages are written without word separators, so character
# error rate is primary and a preset carrying one must set ``word_boundary``
# false. Checked by the test suite, so a regenerated preset cannot quietly
# disagree.
NO_SPACE_POLICIES = frozenset({"ja-cer", "thai-cer", "han-mer"})


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

    Args:
        code: The language code.
        explicit: The ``normalizer:`` named in the preset, which wins. This is
            how Turkish and Arabic get policies their script default would not
            give them.

    Raises:
        KeyError: The language has no recorded script, or the named policy does
            not exist.
        ValueError: The language's script has no default and none was named.
    """
    if explicit is not None:
        return get_policy(explicit)
    if code in LANGUAGE_POLICIES:
        return get_policy(LANGUAGE_POLICIES[code])
    script = script_for_language(code)
    try:
        return get_policy(SCRIPT_POLICIES[script])
    except KeyError:
        raise ValueError(
            f"{code!r} is written in {script!r}, which has no default normalization "
            "policy because its conventions disagree with each other; name one in "
            "the preset's `normalizer:` field"
        ) from None


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
