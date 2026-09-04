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
    # Latin, outside Europe or with live combining marks
    "tr": "Latn",
    "az": "Latn",
    "uz": "Latn",
    "sw": "Latn",
    "rw": "Latn",
    "lg": "Latn",
    "kab": "Latn",
    "vi": "Latn",
    "yo": "Latn",
    "ig": "Latn",
    "ha": "Latn",
    "cy": "Latn",
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
    # East Asian
    "ja": "Jpan",
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
    script = script_for_language(code)
    try:
        return get_policy(SCRIPT_POLICIES[script])
    except KeyError:
        raise ValueError(
            f"{code!r} is written in {script!r}, which has no default normalization "
            "policy because its conventions disagree with each other; name one in "
            "the preset's `normalizer:` field"
        ) from None
