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
becomes ``ह द म``. We also protect the intra-word apostrophe, where that
normalizer turns ``don't`` into ``don t``, and we do not delete bracketed spans.

Whisper's normalizer is nevertheless available, as the ``whisper-basic`` preset,
reproduced exactly rather than approximated — comparing against published
Whisper numbers means scoring the way Whisper scored. It runs its own step
order, because it is not a reordering of the one above. Which preset a run used
is recorded in its config; the choice, and when each is the right one, is in
``docs/normalization.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

NORMALIZER_VERSION = "svb-norm-1"

Pipeline = Literal["svb", "whisper", "script"]

# The fields each pipeline reads. A policy records and hashes exactly these:
# a field its pipeline never consults did not change a character, so recording
# it would imply a rule that ran and hashing it would mark results incomparable
# for a setting that had no effect. Constructing a policy with an unread field
# off its default is refused rather than silently ignored, which is what makes
# leaving it out of the record sound.
_HASHED_FIELDS: dict[Pipeline, frozenset[str]] = {
    "svb": frozenset(
        {
            "version",
            "form",
            "case",
            "malayalam_chillu",
            "strip_invisibles",
            "strip_arabic_marks",
            "turkish_dotted_i",
            "unify_apostrophes",
            "strip_punctuation",
            "strip_symbols",
            "apostrophe_is_letter",
            "digits",
        }
    ),
    "whisper": frozenset(
        {
            "version",
            "pipeline",
            "form",
            "case",
            "drop_bracketed_spans",
            "strip_marks",
            "strip_punctuation",
            "strip_symbols",
            "diacritics",
            "strip_whitespace",
        }
    ),
    # The per-script pipeline of docs/normalization.md. A third pipeline rather
    # than more switches on the first two, because both of those are frozen:
    # `svb-norm-1` is what this project's comparability paragraph describes, and
    # `whisper-basic` reproduces someone else's published algorithm. Neither
    # hash may move, and a policy hashing only its own pipeline's fields is what
    # lets ten new fields arrive without moving either.
    "script": frozenset(
        {
            "version",
            "pipeline",
            "form",
            "case",
            "bracket_spans",
            "zero_width",
            "marks",
            "script_map",
            "locale_case",
            "cyrillic_yo",
            "delete_quotes",
            "fold_alef_maksura",
            "unify_apostrophes",
            "apostrophe_is_letter",
            "turkish_dotted_i",
            "punct_action",
            "strip_symbols",
            "arabic_digits",
        }
    ),
}

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

# --------------------------------------------------------------------------- #
# Whisper's normalizer, transcribed from openai/whisper
# --------------------------------------------------------------------------- #
#
# Source: whisper/normalizers/basic.py, sha256
# 4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38, and
# Radford et al. 2023 appendix C. ``BasicTextNormalizer.__call__`` is, verbatim:
#
#     s = s.lower()
#     s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
#     s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
#     s = self.clean(s).lower()
#     ...
#     s = re.sub(r"\s+", " ", s)
#
# where ``clean`` is ``remove_symbols``:
#
#     "".join(" " if unicodedata.category(c)[0] in "MSP" else c
#             for c in unicodedata.normalize("NFKC", s))
#
# or, with remove_diacritics=True, ``remove_symbols_and_diacritics``, which runs
# over ``unicodedata.normalize("NFKD", s)`` and, per character, prefers
# ADDITIONAL_DIACRITICS, then deletes category Mn, then spaces category MSP.
#
# Three details decide whether a reimplementation is faithful. The lowercasing
# and the span deletion happen BEFORE the Unicode form, so a fullwidth-
# parenthesized span survives. The lowercasing happens again after cleaning,
# which is what makes the uppercase entries of the diacritics table work. And
# there is no strip, so the output routinely carries a trailing space.

# Non-ASCII letters that NFKD does not separate. Copied verbatim from upstream;
# a divergence here is a wrong answer, not a style choice.
ADDITIONAL_DIACRITICS = {
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

_BRACKETED_SPAN = re.compile(r"[<\[][^>\]]*[>\]]")
_PARENTHESIZED_SPAN = re.compile(r"\(([^)]+?)\)")

# --------------------------------------------------------------------------- #
# Script-family tables
# --------------------------------------------------------------------------- #

# Hebrew niqqud and cantillation. Deleted rather than spaced, following
# ivrit.ai's normalizer, which strips them *before* handing the text to
# Whisper's, so they never reach the rule that would turn them into spaces.
_HEBREW_MARKS = re.compile("[֑-ׇ]")

# Zero-width and format characters, by role. ZWNJ is the one that is
# orthographic somewhere (Perso-Arabic morpheme boundaries), so it is named
# separately from the rest.
_ZWNJ = "\u200c"
_ZERO_WIDTH = (
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u00ad",  # soft hyphen
    "\ufeff",  # byte order mark
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u061c",  # arabic letter mark
)

# Eastern Arabic-Indic (U+0660) and Extended (U+06F0) digits to ASCII. Both the
# Arabic leaderboard and the Persian toolkits specify this.
_ARABIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)} | {
    chr(0x06F0 + i): str(i) for i in range(10)
}

# Modern Standard Arabic unifications. Hamza and madda carriers fold onto plain
# alef, the Persian/Urdu letters fold onto their Arabic counterparts, and the
# bare hamza and the tatweel elongation are deleted. Alef wasla is included on
# the leaderboard's stated policy ("normalizing characters with Hamzas and
# Maddas") even though its published code covers only the other three.
_ARABIC_MSA_MAP = {
    "آ": "ا",  # alef with madda
    "أ": "ا",  # alef with hamza above
    "إ": "ا",  # alef with hamza below
    "ٱ": "ا",  # alef wasla
    "ؤ": "و",  # waw with hamza
    "ئ": "ي",  # yeh with hamza
    "پ": "ب",  # peh
    "ڤ": "ف",  # veh
    "ء": "",  # bare hamza
    "ـ": "",  # tatweel
}

# Perso-Arabic folds the other way: the Arabic forms fold onto the Persian
# letters. Alef maksura is listed separately because Uyghur writes /i/ with it.
_PERSO_MAP = {
    "ي": "ی",  # arabic yeh -> farsi yeh
    "ك": "ک",  # arabic kaf -> keheh
    "ة": "ه",  # teh marbuta -> heh
    "ـ": "",  # tatweel
}
_ALEF_MAKSURA = "ى"

_CYRILLIC_YO = {"ё": "е", "Ё": "Е"}

# Armenian U+055B-U+055F. Deleted rather than spaced, because the question and
# emphasis marks are written *inside* the word on the stressed syllable, so
# turning them into a space splits one word into two.
_ARMENIAN_MARKS = re.compile("[\u055b-\u055f]")


@dataclass(frozen=True)
class NormalizerPolicy:
    """Every knob that can change a transcript, hashed into run provenance.

    Defaults are ``svb-norm-1``. Changing any field changes the reported
    numbers, so results produced under different :meth:`policy_hash` values are
    not comparable and must not be pooled.
    """

    version: str = NORMALIZER_VERSION
    # Which order the rules run in. The two pipelines are not reorderings of one
    # another: Whisper lowercases and deletes bracketed spans *before* the
    # Unicode form, so a fullwidth-parenthesized span survives it, and a policy
    # that applied the form first would delete that span while claiming to be
    # Whisper's normalizer.
    pipeline: Pipeline = "svb"
    form: Literal["NFC", "NFKC", "NFKD"] = "NFKC"
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

    # --- read by the whisper pipeline only ---------------------------------
    # Whisper deletes <...>, [...] and (...) spans before anything else, a
    # transcription convention Common Voice and OpenSLR do not use; on corpora
    # that do not use it, it deletes real words.
    drop_bracketed_spans: bool = False
    # Replace Unicode category M with a space. Whisper does this to every mark,
    # which is what reduces an Indic script to its bare consonants.
    strip_marks: bool = False
    # "drop" is Whisper's remove_diacritics path: NFKD, delete category Mn, and
    # map the letters NFKD does not separate through ADDITIONAL_DIACRITICS.
    diacritics: Literal["keep", "drop"] = "keep"
    # Whisper collapses runs of whitespace but never strips the ends, so its
    # output routinely carries a trailing space. Kept faithful rather than
    # tidied: a leading space is one character of CER.
    strip_whitespace: bool = True

    # --- read by the script pipeline only ----------------------------------
    # Whisper deletes bracketed spans; the script pipeline does so only where a
    # family's reference system does, and after the Unicode form rather than
    # before it.
    bracket_spans: Literal["keep", "delete"] = "keep"
    # Zero-width characters are a word separator in Khmer, Myanmar, Thai and Lao,
    # and orthographic in Perso-Arabic, where deleting the non-joiner silently
    # turns one word into two. One global answer is wrong for one of them.
    zero_width: Literal["delete", "space", "keep", "keep_zwnj"] = "delete"
    # Category M handling. "space" is Whisper's rule, which the Open ASR
    # Leaderboard forked away from because it deletes Indic vowel signs; the
    # scoped values remove vocalization that its own script writes optionally.
    marks: Literal["keep", "space", "arabic", "hebrew"] = "keep"
    # The named letter tables: Malayalam chillu joiners, and the Arabic and
    # Perso-Arabic unifications, which run in opposite directions.
    script_map: Literal["none", "malayalam", "arabic_msa", "perso", "armenian"] = "none"
    # Turkish "İ to i, I to ı" before folding. Without it a mark rule turns the
    # combining dot lowercasing produces into a space, splitting the word.
    locale_case: Literal["none", "tr"] = "none"
    # The Russian convention: ё is written optionally, so it folds to е.
    cyrillic_yo: bool = False
    # Hebrew deletes ASCII quotes outright, so an acronym stays one token.
    delete_quotes: bool = False
    # False for Uyghur, where U+0649 is the letter i rather than a variant of yeh.
    fold_alef_maksura: bool = True
    # Vistaar deletes punctuation where Whisper spaces it.
    punct_action: Literal["space", "delete", "keep"] = "space"
    # Eastern Arabic-Indic digits to ASCII, which the Arabic and Persian
    # conventions both specify.
    arabic_digits: Literal["keep", "to_ascii"] = "keep"

    def __post_init__(self) -> None:
        """Refuse a setting this pipeline cannot act on.

        Accepting one silently would be a knob that changes nothing, and would
        also make it unsound to leave the field out of :meth:`to_dict`.
        """
        read = _HASHED_FIELDS[self.pipeline]
        for spec in fields(self):
            if spec.name in read:
                continue
            if getattr(self, spec.name) != spec.default:
                raise ValueError(
                    f"the {self.pipeline!r} pipeline does not read {spec.name!r}; "
                    f"leave it at {spec.default!r} or choose a pipeline that runs it"
                )

    def policy_hash(self) -> str:
        """Short digest over the rules this pipeline runs."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """The rules this pipeline actually runs, and nothing else."""
        read = _HASHED_FIELDS[self.pipeline]
        return {spec.name: getattr(self, spec.name) for spec in fields(self) if spec.name in read}

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


WHISPER_BASIC_POLICY = NormalizerPolicy(
    version="whisper-basic",
    pipeline="whisper",
    form="NFKC",
    case="lower",
    drop_bracketed_spans=True,
    strip_marks=True,
    strip_punctuation=True,
    strip_symbols=True,
    diacritics="keep",
    strip_whitespace=False,
)

WHISPER_BASIC_NODIACRITICS_POLICY = NormalizerPolicy(
    version="whisper-basic-nodiacritics",
    pipeline="whisper",
    form="NFKD",
    case="lower",
    drop_bracketed_spans=True,
    strip_marks=True,
    strip_punctuation=True,
    strip_symbols=True,
    diacritics="drop",
    strip_whitespace=False,
)

WHISPER_MARKS_POLICY = NormalizerPolicy(
    version="whisper-marks",
    pipeline="script",
    form="NFKC",
    case="lower",
    bracket_spans="delete",
    zero_width="delete",
    marks="keep",
    punct_action="space",
    strip_symbols=True,
    unify_apostrophes=True,
)


# --- the script families -------------------------------------------------- #
#
# Each is whisper-marks plus the rules its family's reference system specifies.
# Sources are named in docs/normalization.md; the deviations are labelled there
# too, along with the two policies whose base could not be verified at all.

CYRILLIC_YO_POLICY = dataclasses.replace(
    WHISPER_MARKS_POLICY, version="cyrillic-yo", cyrillic_yo=True
)

TURKIC_TR_POLICY = dataclasses.replace(
    WHISPER_MARKS_POLICY, version="turkic-tr", locale_case="tr", turkish_dotted_i=True
)

# NFC rather than NFKC: the only thing NFKC adds for Latin is fullwidth folding,
# which these corpora do not need, while NFC's canonical reordering is what makes
# the two keystroke orders of Yoruba e-dot-below-acute compare equal.
LATIN_MARKS_POLICY = dataclasses.replace(WHISPER_MARKS_POLICY, version="latin-marks", form="NFC")

# Vistaar deletes punctuation rather than spacing it, applies no Unicode form,
# and does not lowercase. We add NFC and lowercasing, both labelled deviations:
# without a form the composed and decomposed nukta spellings are two vocabulary
# entries, and lowercasing costs nothing for a caseless script while correcting
# the Latin contamination these corpora carry.
INDIC_VISTAAR_POLICY = NormalizerPolicy(
    version="indic-vistaar",
    pipeline="script",
    form="NFC",
    case="lower",
    bracket_spans="keep",
    zero_width="delete",
    marks="keep",
    script_map="malayalam",
    punct_action="delete",
    strip_symbols=True,
    unify_apostrophes=True,
)

# Marks are removed by code range, letters unified onto plain alef, and Eastern
# digits mapped to ASCII, per the Open Universal Arabic ASR Leaderboard.
ARABIC_OUAAL_POLICY = NormalizerPolicy(
    version="arabic-ouaal",
    pipeline="script",
    form="NFKC",
    case="lower",
    zero_width="delete",
    marks="arabic",
    script_map="arabic_msa",
    punct_action="space",
    strip_symbols=True,
    unify_apostrophes=True,
    arabic_digits="to_ascii",
)

# The letter table runs the other way: the Arabic forms fold onto the Persian
# letters. The non-joiner is kept, because both Persian toolkits not only
# preserve it but insert it — it marks a morpheme boundary, and deleting it
# would silently turn one word into two.
PERSO_ARABIC_POLICY = dataclasses.replace(
    ARABIC_OUAAL_POLICY,
    version="perso-arabic",
    script_map="perso",
    zero_width="keep_zwnj",
)

# Uyghur writes /i/ with alef maksura and /j/ with yeh, so the Persian fold of
# the first onto the second would merge two distinct letters.
UYGHUR_UG_POLICY = dataclasses.replace(
    PERSO_ARABIC_POLICY, version="uyghur-ug", fold_alef_maksura=False
)

# Japanese needs no mark or letter rule; what it needs is the character error
# rate, which is the language's setting rather than the policy's.
JA_CER_POLICY = dataclasses.replace(WHISPER_MARKS_POLICY, version="ja-cer")


# Thai puts its word separator in an invisible character rather than a space, so
# the zero-width rule maps to a space instead of deleting. NFC, never NFKC: Thai
# SARA AM U+0E33 is a letter (category Lo) that NFKC splits into a combining
# mark plus a vowel, lengthening every word containing it by one code point and
# manufacturing an Mn for a mark rule to find. Verified: "ทำงาน" is 5 code
# points under NFC and 6 under NFKC.
#
# Thai has two live conventions and they disagree. Whisper's appendix C measures
# character error rate; Thonburian Whisper, the strongest published Thai system,
# reports word error rate after `deepcut` segmentation. Character error rate is
# primary here because putting a segmenter in the metric makes the number depend
# on a third-party model version. Thonburian's own cleaner agrees with this
# policy on the rule that matters: it replaces U+200B with a space.
#
# Not adopted: PyThaiNLP's tone-mark reordering and SARA AM composition. Those
# are corpus-cleaning rules, and applying them to a hypothesis would silently
# repair model errors.
THAI_CER_POLICY = dataclasses.replace(
    WHISPER_MARKS_POLICY, version="thai-cer", form="NFC", zero_width="space"
)

# Han script. The rules are whisper-marks unchanged — NFKC folds the fullwidth
# punctuation and digits Chinese text carries, and there are no combining marks
# to protect — so this policy differs from its base only in name and in the
# metric its languages are scored on, which lives on the language rather than
# here.
#
# Traditional is preserved, not folded to Simplified. NFKC does not touch it
# (verified: U+9AD4 is stable), no Chinese benchmark converts, and MDCC keeps
# Traditional for Cantonese. WenetSpeech-Yue does fold with OpenCC, so Cantonese
# is genuinely contested; see docs/normalization.md.
HAN_MER_POLICY = dataclasses.replace(WHISPER_MARKS_POLICY, version="han-mer")


# Armenian writes its question and emphasis marks inside the word, on the
# stressed syllable, so the punctuation pass would split every question in two.
# Deleting them first is our fix: no Armenian ASR evaluation convention was
# found. NFKC's rewrite of the ligature U+0587 into two letters is left in
# place, applied identically to both sides of every score.
ARMENIAN_HY_POLICY = dataclasses.replace(
    WHISPER_MARKS_POLICY, version="armenian-hy", script_map="armenian"
)

# NFC, never NFKC: the compatibility jamo compose into syllables under the
# compatibility form, so two code points silently become one and the character
# count moves. KsponSpeech reports character and word error rate, and word error
# rate is primary here because Korean is written with spaces — unlike the other
# character-scored languages. Korean spacing is flexible, so that word error rate
# is inflated by spacing choices that are not recognition errors; KsponSpeech
# answers this with a space-normalized variant that rewrites the hypothesis
# toward the reference, which is a convention this repository would have to
# defend separately and does not adopt.
KO_KSPON_POLICY = dataclasses.replace(WHISPER_MARKS_POLICY, version="ko-kspon", form="NFC")

# Tibetan stacks vowel signs and subjoined consonants as combining marks, so
# Whisper's mark rule reduces a word to its root letters. The tsheg U+0F0B, which
# separates syllables, is punctuation and therefore becomes a space: that costs
# no characters, since it is one code point either way, and leaves the syllable
# boundary visible to anything that wants to count syllables later.
#
# UNVERIFIED, like `uyghur-ug`. The corpus this policy exists for states no
# evaluation metric and no transcription convention, and no Tibetan ASR
# normalizer was found. These rules are reasoned from the orthography.
TIBETAN_SYLLABLE_POLICY = dataclasses.replace(
    WHISPER_MARKS_POLICY, version="tibetan-syllable", form="NFC"
)


# The policies a config may name. Each records only the rules its own pipeline
# runs, so the version string is what distinguishes two records that would
# otherwise share a shape — hence one version per preset, checked below.
POLICIES: dict[str, NormalizerPolicy] = {
    "svb-norm-1": DEFAULT_POLICY,
    "whisper-basic": WHISPER_BASIC_POLICY,
    "whisper-basic-nodiacritics": WHISPER_BASIC_NODIACRITICS_POLICY,
    "whisper-marks": WHISPER_MARKS_POLICY,
    "cyrillic-yo": CYRILLIC_YO_POLICY,
    "turkic-tr": TURKIC_TR_POLICY,
    "latin-marks": LATIN_MARKS_POLICY,
    "indic-vistaar": INDIC_VISTAAR_POLICY,
    "arabic-ouaal": ARABIC_OUAAL_POLICY,
    "perso-arabic": PERSO_ARABIC_POLICY,
    "uyghur-ug": UYGHUR_UG_POLICY,
    "ja-cer": JA_CER_POLICY,
    "thai-cer": THAI_CER_POLICY,
    "han-mer": HAN_MER_POLICY,
    "armenian-hy": ARMENIAN_HY_POLICY,
    "ko-kspon": KO_KSPON_POLICY,
    "tibetan-syllable": TIBETAN_SYLLABLE_POLICY,
}

assert len({p.version for p in POLICIES.values()}) == len(POLICIES), (
    "two presets share a version, so their records could not be told apart"
)


def get_policy(name: str) -> NormalizerPolicy:
    """Look up a named policy.

    Args:
        name: A key of :data:`POLICIES`.

    Raises:
        KeyError: When no preset carries that name.
    """
    try:
        return POLICIES[name]
    except KeyError:
        known = ", ".join(sorted(POLICIES))
        raise KeyError(f"unknown normalization policy {name!r}; known presets: {known}") from None


def policy_name(policy: NormalizerPolicy) -> str | None:
    """The preset name for a policy, or ``None`` when it is hand-rolled.

    Recorded beside the resolved fields so a config says which published policy
    it is, without that name being the thing the run trusts.
    """
    for name, preset in POLICIES.items():
        if preset == policy:
            return name
    return None


def _is_word_char(ch: str) -> bool:
    """The ``\\w`` predicate, spelled out so the apostrophe rule is inspectable."""
    return ch.isalnum() or ch == "_"


def _replace_punctuation_and_symbols(
    text: str, policy: NormalizerPolicy, counts: dict[str, int]
) -> str:
    """Replace category P* and S* with a space, keeping the intra-word apostrophe.

    A space rather than a deletion: sources that omit the space after a comma
    would otherwise have two words fused into one.
    Tallies what it actually replaced into ``counts``. A character it decided to
    keep — a protected apostrophe, or anything at all under a policy with the
    rule switched off — is not a removal and is not counted.
    """
    out: list[str] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        if ch == APOSTROPHE:
            # U+0027 is category Po, so it is the punctuation rule's business
            # and follows that rule's setting.
            neighboured = (
                i > 0 and i < last and _is_word_char(text[i - 1]) and _is_word_char(text[i + 1])
            )
            keep = policy.apostrophe_is_letter or neighboured or not policy.strip_punctuation
            if keep:
                out.append(APOSTROPHE)
            else:
                counts["P"] += 1
                out.append(" ")
            continue
        group = unicodedata.category(ch)[0]
        drop = (group == "P" and policy.strip_punctuation) or (
            group == "S" and policy.strip_symbols
        )
        if drop:
            counts[group] += 1
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _apply_case(text: str, case: str) -> str:
    if case == "casefold":
        return text.casefold()
    return text.lower() if case == "lower" else text


def _whisper_normalize(text: str, policy: NormalizerPolicy, counts: dict[str, int]) -> str:
    """Whisper's ``BasicTextNormalizer``, in its own order.

    Deliberately not routed through the svb steps. Its lowercasing and span
    deletion run before the Unicode form, so folding it into a pipeline that
    normalizes first would delete fullwidth-parenthesized spans that upstream
    keeps — a normalizer that is not Whisper's, under Whisper's name.
    """
    text = _apply_case(text, policy.case)
    if policy.drop_bracketed_spans:
        text = _BRACKETED_SPAN.sub("", text)
        text = _PARENTHESIZED_SPAN.sub("", text)

    out: list[str] = []
    for ch in unicodedata.normalize(policy.form, text):
        category = unicodedata.category(ch)
        if policy.diacritics == "drop":
            if ch in ADDITIONAL_DIACRITICS:
                out.append(ADDITIONAL_DIACRITICS[ch])
                continue
            if category == "Mn":
                counts["M"] += 1
                continue
        group = category[0]
        if (
            (group == "M" and policy.strip_marks)
            or (group == "S" and policy.strip_symbols)
            or (group == "P" and policy.strip_punctuation)
        ):
            counts[group] += 1
            out.append(" ")
        else:
            out.append(ch)

    text = _apply_case("".join(out), policy.case)
    text = _WHITESPACE.sub(" ", text)
    return text.strip() if policy.strip_whitespace else text


def _script_zero_width(text: str, policy: NormalizerPolicy, counts: dict[str, int]) -> str:
    """Delete, space or keep the invisible format characters."""
    if policy.zero_width == "keep":
        return text
    for ch in _ZERO_WIDTH:
        if policy.zero_width == "keep_zwnj" and ch == _ZWNJ:
            continue
        if ch in text:
            counts["Cf"] += text.count(ch)
            text = text.replace(ch, " " if policy.zero_width == "space" else "")
    return text


def _script_marks(text: str, policy: NormalizerPolicy, counts: dict[str, int]) -> str:
    """Remove combining marks, scoped by the policy.

    ``space`` is Whisper's rule and destroys any abugida. The scoped values
    remove only vocalization that its own script writes optionally, which is
    what makes an inconsistently vocalized corpus learnable.
    """
    if policy.marks == "keep":
        return text
    if policy.marks == "space":
        out = []
        for ch in text:
            if unicodedata.category(ch)[0] == "M":
                counts["M"] += 1
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)
    pattern = _ARABIC_MARKS if policy.marks == "arabic" else _HEBREW_MARKS
    counts["M"] += len(pattern.findall(text))
    return pattern.sub("", text)


def _script_letter_map(text: str, policy: NormalizerPolicy) -> str:
    """Apply the family's letter table, if it has one."""
    if policy.script_map == "malayalam":
        for sequence, atomic in _MALAYALAM_CHILLU.items():
            text = text.replace(sequence, atomic)
        return text
    if policy.script_map == "arabic_msa":
        for src, dst in _ARABIC_MSA_MAP.items():
            text = text.replace(src, dst)
        return text
    if policy.script_map == "armenian":
        return _ARMENIAN_MARKS.sub("", text)
    if policy.script_map == "perso":
        for src, dst in _PERSO_MAP.items():
            text = text.replace(src, dst)
        if policy.fold_alef_maksura:
            text = text.replace(_ALEF_MAKSURA, "ی")
        return text
    return text


def _script_punctuation(text: str, policy: NormalizerPolicy, counts: dict[str, int]) -> str:
    """Punctuation and symbols, spaced or deleted, with the apostrophe carve-out."""
    if policy.punct_action == "keep" and not policy.strip_symbols:
        return text
    replacement = " " if policy.punct_action == "space" else ""
    out: list[str] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        if ch == APOSTROPHE:
            neighboured = (
                i > 0 and i < last and _is_word_char(text[i - 1]) and _is_word_char(text[i + 1])
            )
            if policy.apostrophe_is_letter or neighboured or policy.punct_action == "keep":
                out.append(APOSTROPHE)
            else:
                counts["P"] += 1
                out.append(replacement)
            continue
        group = unicodedata.category(ch)[0]
        if group == "P" and policy.punct_action != "keep":
            counts["P"] += 1
            out.append(replacement)
        elif group == "S" and policy.strip_symbols:
            counts["S"] += 1
            out.append(replacement)
        else:
            out.append(ch)
    return "".join(out)


def _script_normalize(text: str, policy: NormalizerPolicy, counts: dict[str, int]) -> str:
    """The per-script pipeline, in the fixed order documented for it.

    Step 3 must precede step 4: the atomic Malayalam chillu is reached *through*
    a zero-width joiner, so deleting those first strands a bare virama. The
    reference Indic normalizer this policy is drawn from has that ordering the
    wrong way round.
    """
    text = unicodedata.normalize(policy.form, text)  # 1
    if policy.bracket_spans == "delete":  # 2
        text = _BRACKETED_SPAN.sub("", text)
        text = _PARENTHESIZED_SPAN.sub("", text)
    if policy.script_map == "malayalam":  # 3
        text = _script_letter_map(text, policy)
    text = _script_zero_width(text, policy, counts)  # 4
    text = _script_marks(text, policy, counts)  # 5
    if policy.locale_case == "tr":  # 6
        text = text.replace("İ", "i").replace("I", "ı")
    if policy.script_map != "malayalam":  # 7
        text = _script_letter_map(text, policy)
    if policy.delete_quotes:  # 8
        text = text.replace('"', "").replace(APOSTROPHE, "")
    text = unicodedata.normalize(policy.form, _apply_case(text, policy.case))  # 9
    if policy.turkish_dotted_i:  # 10
        text = text.replace(_DOTTED_I, "i")
    if policy.cyrillic_yo:  # 11
        for src, dst in _CYRILLIC_YO.items():
            text = text.replace(src, dst)
    if policy.unify_apostrophes:  # 12
        for form in _APOSTROPHE_FORMS:
            text = text.replace(form, APOSTROPHE)
    text = _script_punctuation(text, policy, counts)  # 13
    if policy.arabic_digits == "to_ascii":  # 14
        for src, dst in _ARABIC_DIGITS.items():
            text = text.replace(src, dst)
    return _WHITESPACE.sub(" ", text).strip()  # 15


def empty_removal_counts() -> dict[str, int]:
    """The removal tally's fixed shape: every category any rule can delete.

    Fixed rather than grown on demand, so a zero is reported as a zero instead
    of as an absent key, and every deletable category has somewhere to be
    counted — a removal with nowhere to go is a removal nobody sees.
    """
    return {"M": 0, "P": 0, "S": 0, "arabic_marks": 0} | dict.fromkeys(
        sorted(_INVISIBLE_CATEGORIES), 0
    )


def normalize_with_counts(
    text: str, policy: NormalizerPolicy = DEFAULT_POLICY
) -> tuple[str, dict[str, int]]:
    """Normalize, and report what each rule actually did.

    The single implementation of the policy; :func:`normalize_text` is this
    function with the tally discarded. Counting from a separate scan of the
    input would let the two drift, and the drift is invisible: a scan that
    counts every punctuation character reports the intra-word apostrophes this
    policy goes out of its way to *keep* as though it had removed them.

    Returns:
        The normalized text, and a ``{category: characters removed}`` tally.
        The extra ``case_changed`` entry counts characters the case rule
        altered rather than removed, and is zero when that rule is off.
    """
    counts = empty_removal_counts() | {"case_changed": 0}
    if not text:
        return "", counts

    if policy.case in ("casefold", "lower"):
        # Counted here rather than at the case step, so both pipelines tally it
        # against the same input and neither has to duplicate the rule.
        counts["case_changed"] += sum(1 for ch in text if ch.casefold() != ch)
    if policy.pipeline == "whisper":
        return _whisper_normalize(text, policy, counts), counts
    if policy.pipeline == "script":
        return _script_normalize(text, policy, counts), counts

    text = unicodedata.normalize(policy.form, text)

    if policy.malayalam_chillu == "atomic":
        for sequence, atomic in _MALAYALAM_CHILLU.items():
            text = text.replace(sequence, atomic)

    if policy.strip_invisibles:
        kept: list[str] = []
        for ch in text:
            category = unicodedata.category(ch)
            if category in _INVISIBLE_CATEGORIES:
                counts[category] += 1
            else:
                kept.append(ch)
        text = "".join(kept)

    if policy.strip_arabic_marks:
        counts["arabic_marks"] += count_arabic_marks(text)
        text = _ARABIC_MARKS.sub("", text).replace(_TATWEEL, "")

    if policy.case in ("casefold", "lower"):
        text = unicodedata.normalize(policy.form, _apply_case(text, policy.case))

    if policy.turkish_dotted_i:
        text = text.replace(_DOTTED_I, "i")

    if policy.unify_apostrophes:
        for form in _APOSTROPHE_FORMS:
            text = text.replace(form, APOSTROPHE)

    if policy.strip_punctuation or policy.strip_symbols:
        text = _replace_punctuation_and_symbols(text, policy, counts)

    return _WHITESPACE.sub(" ", text).strip(), counts


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
    return normalize_with_counts(text, policy)[0]


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


def registry_digest() -> str:
    """One digest over every named policy.

    Two runs with the same digest normalized every language the same way,
    whichever languages they happened to use, so text equivalence can be checked
    without walking the per-language table.
    """
    payload = json.dumps(
        {name: policy.to_dict() for name, policy in sorted(POLICIES.items())},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
