"""One versioned text normalizer, applied at exactly three places.

Every transcript this project reads passes through :func:`normalize_text` when
the vocabulary is built, when a training target is encoded, and when a reference
or hypothesis is scored. A single policy object is threaded through all three
and hashed into each run's provenance, so training and evaluation cannot drift
apart on text policy.

The order of operations is fixed and each step earns its place:

1. **Unicode form (NFKC).** Fullwidth and halfwidth Japanese, Arabic
   presentation forms and Latin ligatures are acoustically identical but encode
   differently; under NFC each would take a vocabulary slot the model can never
   learn to disambiguate. Note that NFC and NFKC both *decompose* the
   precomposed Indic nukta letters, which are Unicode composition exclusions:
   Odia U+0B5C becomes U+0B21 U+0B3C. Character counts, and therefore CER
   denominators, are over the decomposed form.
2. **Malayalam chillu letters.** Consonant + virama + ZWJ and the atomic chillu
   are the same letter, and Unicode normalization does *not* unify them. Without
   this step one letter has two encodings in the vocabulary.
3. **Invisible characters.** Zero-width joiner and non-joiner, soft hyphen, the
   byte-order mark, bidi controls. A CTC model cannot align an invisible
   character to audio, so it would learn to emit them from context alone. Note
   that NFKC does not remove the soft hyphen.
4. **Arabic vocalization.** Tashkeel is optional in writing, so it appears in
   some sentences and not others; an inconsistently written target is
   unlearnable. Scoped by code range rather than by language, because it is a
   property of the script. Tatweel U+0640 is pure typographic elongation and is
   category ``Lm``, so removing it takes an explicit rule.
5. **Case folding**, then the Unicode form again, since folding can produce
   decomposed sequences.
6. **The Turkish dotted capital I.** ``'İ'.casefold()`` is *two* code points,
   U+0069 U+0307. Left alone, ``İstanbul`` and ``istanbul`` are different words
   and a combining dot above enters the vocabulary. The one case-motivated
   carve-out, applied globally. A capital dotless ``I`` still folds to a dotted
   ``i``; fixing that needs a Turkish-locale mapping, and Common Voice rejects
   the all-caps sentences where it would mostly appear.
7. **Apostrophes**, unified to U+0027 first because Common Voice's French and
   Italian text uses U+2019 far more than the ASCII form, and because U+02BC is
   category ``Lm`` and would otherwise survive as a stray letter.
8. **Punctuation and symbols**, replaced with a *space* rather than deleted, so
   that ``hello,world`` becomes two words. Hyphens and dashes are category
   ``Pd``, so no separate dash rule is needed. An apostrophe between two word
   characters survives, which keeps French ``l'été``, Ukrainian ``п'ять`` and
   Swahili ``ng'ombe`` as single words.
9. **Whitespace**, collapsed and stripped.

Combining marks are otherwise **kept**. This is a deliberate departure from
Whisper's ``BasicTextNormalizer``, which replaces every character in categories
M, S and P with a space and so destroys every Indic script: ``हिंदी में``
becomes ``ह  द  म  ``. We also protect the intra-word apostrophe, where that
normalizer turns ``don't`` into ``don t``, and we do not delete bracketed spans.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

NORMALIZER_VERSION = "svb-norm-1"

APOSTROPHE = "'"

# Apostrophe-like code points folded onto U+0027. U+02BC and U+02B9 are category
# Lm: they are letters as far as Unicode is concerned, so the punctuation pass
# never reaches them. A fork adding an orthography where those are letters in
# earnest (Hawaiian, Uzbek) must remove them from this map.
_APOSTROPHE_FORMS = "’‘‛ʼʹ՚"

# Arabic-script combining marks: harakat, tanwin, sukun, shadda, superscript
# alef, and the Quranic annotation marks.
_ARABIC_MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"

# Malayalam chillu letters, which Unicode 5.1 gave atomic code points. Text
# predating that spells them consonant + virama + ZWJ, and no normalization
# form unifies the two.
_MALAYALAM_CHILLU = {
    "ണ്‍": "ൺ",  # chillu NN
    "ന്‍": "ൻ",  # chillu N
    "ര്‍": "ർ",  # chillu RR
    "ല്‍": "ൽ",  # chillu L
    "ള്‍": "ൾ",  # chillu LL
    "ക്‍": "ൿ",  # chillu K
}

# Categories deleted outright rather than spaced: they sit inside words.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc", "Cs", "Co", "Cn"})

_DOTTED_I = "i̇"

_WHITESPACE = re.compile(r"\s+")

_DIGIT_CATEGORIES = frozenset({"Nd", "Nl", "No"})


@dataclass(frozen=True)
class NormalizerPolicy:
    """Every knob that can change a transcript, hashed into run provenance.

    Defaults are ``svb-norm-1``. Changing any field changes the reported
    numbers, so results produced under different :meth:`policy_hash` values are
    not comparable and must not be pooled.
    """

    version: str = NORMALIZER_VERSION
    form: Literal["NFC", "NFKC"] = "NFKC"
    case: Literal["casefold", "lower", "none"] = "casefold"
    malayalam_chillu: Literal["atomic", "keep"] = "atomic"
    strip_invisibles: bool = True
    strip_arabic_marks: bool = True
    turkish_dotted_i: bool = True
    unify_apostrophes: bool = True
    strip_punctuation: bool = True
    strip_symbols: bool = True
    # Set True for an orthography that writes a glottal stop as an apostrophe,
    # where a word-initial or word-final one is a letter rather than a quote.
    apostrophe_is_letter: bool = False
    # Digits are kept as characters: they are not spelled out, and utterances
    # containing them are not dropped. The field exists to make that choice
    # explicit in the dumped config and the policy hash rather than leave it
    # implicit in the absence of a rule.
    #
    # Deliberately a single value. Dropping digit-bearing utterances is the
    # obvious ablation, but it changes the test set rather than the text, so it
    # belongs to whatever assembles a split — not here. An accepted second value
    # that `normalize_text` never read would have changed the policy hash, and
    # so marked results incomparable, while changing not one character.
    digits: Literal["keep"] = "keep"

    def policy_hash(self) -> str:
        """Short digest over every field, for the resolved config and env.json."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizerPolicy:
        return cls(**data)


DEFAULT_POLICY = NormalizerPolicy()

# What a vocabulary written before this module existed was built with: NFC on
# the training targets and nothing else. Kept so an old vocab.json still loads
# and still describes itself honestly.
LEGACY_POLICY = NormalizerPolicy(
    version="svb-norm-0",
    form="NFC",
    case="none",
    malayalam_chillu="keep",
    strip_invisibles=False,
    strip_arabic_marks=False,
    turkish_dotted_i=False,
    unify_apostrophes=False,
    strip_punctuation=False,
    strip_symbols=False,
)


def _is_word_char(ch: str) -> bool:
    """The ``\\w`` predicate, spelled out so the apostrophe rule is inspectable."""
    return ch.isalnum() or ch == "_"


def _replace_punctuation_and_symbols(text: str, policy: NormalizerPolicy) -> str:
    """Replace category P* and S* with a space, keeping the intra-word apostrophe.

    A space rather than a deletion: sources that omit the space after a comma
    would otherwise have two words fused into one.
    """
    out: list[str] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        if ch == APOSTROPHE:
            neighboured = (
                i > 0 and i < last and _is_word_char(text[i - 1]) and _is_word_char(text[i + 1])
            )
            out.append(APOSTROPHE if policy.apostrophe_is_letter or neighboured else " ")
            continue
        group = unicodedata.category(ch)[0]
        drop = (group == "P" and policy.strip_punctuation) or (
            group == "S" and policy.strip_symbols
        )
        out.append(" " if drop else ch)
    return "".join(out)


def normalize_text(text: str, policy: NormalizerPolicy = DEFAULT_POLICY) -> str:
    """Apply the normalization policy. Idempotent.

    Args:
        text: A raw corpus transcript.
        policy: The policy to apply; defaults to ``svb-norm-1``.

    Returns:
        The normalized transcript, which may be empty when the input held only
        punctuation. Callers decide what an empty result means — training drops
        the utterance, scoring excludes and counts it.
    """
    if not text:
        return ""

    text = unicodedata.normalize(policy.form, text)

    if policy.malayalam_chillu == "atomic":
        for sequence, atomic in _MALAYALAM_CHILLU.items():
            text = text.replace(sequence, atomic)

    if policy.strip_invisibles:
        text = "".join(c for c in text if unicodedata.category(c) not in _INVISIBLE_CATEGORIES)

    if policy.strip_arabic_marks:
        text = _ARABIC_MARKS.sub("", text).replace(_TATWEEL, "")

    if policy.case == "casefold":
        text = unicodedata.normalize(policy.form, text.casefold())
    elif policy.case == "lower":
        text = unicodedata.normalize(policy.form, text.lower())

    if policy.turkish_dotted_i:
        text = text.replace(_DOTTED_I, "i")

    if policy.unify_apostrophes:
        for form in _APOSTROPHE_FORMS:
            text = text.replace(form, APOSTROPHE)

    if policy.strip_punctuation or policy.strip_symbols:
        text = _replace_punctuation_and_symbols(text, policy)

    return _WHITESPACE.sub(" ", text).strip()


def normalize_batch(texts: Iterable[str], policy: NormalizerPolicy = DEFAULT_POLICY) -> list[str]:
    """:func:`normalize_text` over an iterable, preserving length.

    Results that normalize to empty are kept as empty strings rather than
    dropped, so positions still line up with whatever the caller read them from.
    """
    return [normalize_text(t, policy) for t in texts]


def has_digits(text: str) -> bool:
    """Whether a transcript carries any numeric character, in any script.

    Common Voice's sentence collector rejects sentences containing digits, but
    its default rule matches ASCII only, so Devanagari and Arabic-Indic digits
    pass; OpenSLR applies no such rule at all. Reported as a data-quality
    statistic rather than acted on.
    """
    return any(unicodedata.category(c) in _DIGIT_CATEGORIES for c in text)


def count_arabic_marks(text: str) -> int:
    """How many characters the Arabic-script rule would remove.

    Vocalization and tatweel, counted for the run's text statistics. Neither is
    reachable by a punctuation or symbol rule: the marks are category ``Mn`` and
    tatweel is ``Lm``.
    """
    return len(_ARABIC_MARKS.findall(text)) + text.count(_TATWEEL)


def module_sha256() -> str:
    """Digest of this source file, recorded alongside the policy hash.

    The policy hash describes the settings; it cannot see a change to the code
    that reads them.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
