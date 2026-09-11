"""Choose the Common Voice languages a scale preset trains on, from the release's own numbers.

The selection rule is: **at least 50 hours of training audio, plus a real dev
and a real test split** — not a sliver of one. A language under that line
contributes more variance than signal to a multilingual mix and teaches the
merge nothing, which is the whole reason the large scale exists.

The training audio is *validated minus the evaluation splits*, which is what a
run reads (:mod:`svb.data.commonvoice_local`), not the official ``train.tsv``.
Corpora Creator builds ``train.tsv`` from a deduplicated frame at roughly one
clip per sentence, so it is about a third of the validated audio and it is what
made most Common Voice languages look too small to train on. Both figures are
in the table; the rule is applied to the first.

Nothing here guesses at corpus size. Mozilla publishes one JSON per release in
`common-voice/cv-dataset`, giving per locale the clip count of each bucket
(`train`, `dev`, `test`, `validated`), the validated hours (`validHrs`) and one
mean clip duration (`avgDurationSecs`). It publishes no per-split duration, so a
split's hours are the product of a count and that mean — an approximation the
release document itself cannot improve on, and the one every number in
`docs/languages.md` rests on.

The trainable figure is an **upper bound**: the release carries no speaker
information, so it cannot subtract the validated clips the loader's speaker
guard removes, which are further recordings by a dev or test speaker. The
measured figure comes from ``svb data-stats`` against a corpus on disk.

Fetch the release document and regenerate the table:

    gh api repos/common-voice/cv-dataset/contents/datasets/scripted-speech/\\
cv-corpus-25.0-2026-03-09.json --jq '.content' | base64 -d > cv25.json
    python scripts/select_languages.py --release cv25.json \\
        --table-out docs/languages.md

Writing over a preset that already exists needs ``--write-preset`` as well, so
the script can be run to see what a new release would change without changing
it. The prose header of a committed preset is written by hand; what this script
regenerates is the language list and the table of evidence behind it. The two
are kept in step by a test that the locales marked included in `docs/languages.md`
are exactly the locales the preset lists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Provenance label recorded in run metadata; not a Hugging Face Hub id. See
# src/svb/data/registry.py.
RELEASE_LABEL = "common_voice_25"

# The 16-language preset, which the paper's design is already committed to. A
# member that misses the rule stays in and is flagged, because dropping it would
# stop the smaller presets from being subsets of the larger one.
COMMITTED_LOCALES: tuple[str, ...] = (
    "en", "de", "fr", "es", "it", "nl", "ru", "uk",
    "pl", "hi", "ar", "ka", "tr", "sw", "ja", "fi",
)  # fmt: skip

# Common Voice locales for the languages held out for transfer
# (configs/scales/heldout.yaml, sourced from OpenSLR). Training on the Common
# Voice half of one of these would make the held-out number measure transfer to
# a language the pipeline had already supervised. Excluded unconditionally.
HELD_OUT_LOCALES: Mapping[str, str] = {
    "ml": "malayalam",
    "mr": "marathi",
    "te": "telugu",
    "gu": "gujarati",
}

#: Languages kept although their trainable hours miss the rule, with the reason.
#: Separate from COMMITTED_LOCALES, which is about the smaller presets: these are
#: chosen for typological coverage and each says what it is short of.
CARRIED_LANGUAGES: Mapping[str, str] = {
    **dict.fromkeys(
        ("zu", "xh", "ts", "ve"),
        "kept for typological coverage; its NCHLT corpus totals about 56 h and the "
        "shortfall is in the shipped train split, not in the corpus",
    ),
    # Zeroth is the only openly-licensed Korean corpus the survey found; the
    # others are all gated. Its shipped test split is 1.2 h, under the
    # evaluation bar, so the whole 52.8 h corpus is repartitioned — which leaves
    # 42.2 h to train on. Dropping it would cost the preset the Koreanic family
    # and the Hangul script outright, so it is kept and the number is stated.
    "ko": (
        "kept because it is the only open Korean corpus, and Koreanic and Hangul are in "
        "the preset only through it; the shipped 1.2 h test is under the evaluation bar, "
        "so all 52.8 h are repartitioned"
    ),
}

# ISO 15924 codes for writing systems that do not put spaces between words.
# Whitespace tokenization of such a transcript yields one token per sentence, so
# word error rate over it is meaningless and character error rate is primary.
# Korean is deliberately absent: Hangul is written with spaces between words, so
# word error rate means the same thing for it as for a Latin-script language.
NO_SPACE_SCRIPTS = frozenset(
    {"Hani", "Hans", "Hant", "Jpan", "Thai", "Laoo", "Khmr", "Mymr", "Tibt"}
)


@dataclass(frozen=True)
class LanguageInfo:
    """What the release document does not carry: a name, a family, a script.

    ``script`` is the ISO 15924 code CLDR resolves the locale to
    (``common/supplemental/likelySubtags.xml``); it decides ``word_boundary``,
    so it is the field worth checking. ``family`` is the conventional
    genealogical label and is descriptive only — it is what makes the
    typological spread of a preset readable, and nothing computes on it.
    """

    name: str
    family: str
    script: str


LANGUAGES: Mapping[str, LanguageInfo] = {
    "ab": LanguageInfo("Abkhaz", "Northwest Caucasian", "Cyrl"),
    "ady": LanguageInfo("Adyghe", "Northwest Caucasian", "Cyrl"),
    "ar": LanguageInfo("Arabic", "Afro-Asiatic (Semitic)", "Arab"),
    "ba": LanguageInfo("Bashkir", "Turkic (Kipchak)", "Cyrl"),
    "be": LanguageInfo("Belarusian", "Indo-European (Slavic)", "Cyrl"),
    "bn": LanguageInfo("Bengali", "Indo-European (Indo-Aryan)", "Beng"),
    "bo": LanguageInfo("Tibetan", "Sino-Tibetan (Bodish)", "Tibt"),
    "ca": LanguageInfo("Catalan", "Indo-European (Romance)", "Latn"),
    "ckb": LanguageInfo("Central Kurdish", "Indo-European (Iranian)", "Arab"),
    "cs": LanguageInfo("Czech", "Indo-European (Slavic)", "Latn"),
    "cy": LanguageInfo("Welsh", "Indo-European (Celtic)", "Latn"),
    "de": LanguageInfo("German", "Indo-European (Germanic)", "Latn"),
    "en": LanguageInfo("English", "Indo-European (Germanic)", "Latn"),
    "eo": LanguageInfo("Esperanto", "Constructed", "Latn"),
    "es": LanguageInfo("Spanish", "Indo-European (Romance)", "Latn"),
    "et": LanguageInfo("Estonian", "Uralic (Finnic)", "Latn"),
    "eu": LanguageInfo("Basque", "Isolate", "Latn"),
    "fa": LanguageInfo("Persian", "Indo-European (Iranian)", "Arab"),
    "fi": LanguageInfo("Finnish", "Uralic (Finnic)", "Latn"),
    "fr": LanguageInfo("French", "Indo-European (Romance)", "Latn"),
    "fy-NL": LanguageInfo("West Frisian", "Indo-European (Germanic)", "Latn"),
    "gl": LanguageInfo("Galician", "Indo-European (Romance)", "Latn"),
    "hi": LanguageInfo("Hindi", "Indo-European (Indo-Aryan)", "Deva"),
    "hr": LanguageInfo("Croatian", "Indo-European (Slavic)", "Latn"),
    "hu": LanguageInfo("Hungarian", "Uralic (Ugric)", "Latn"),
    "hy": LanguageInfo("Armenian", "Indo-European (Armenian)", "Armn"),
    "is": LanguageInfo("Icelandic", "Indo-European (Germanic)", "Latn"),
    "it": LanguageInfo("Italian", "Indo-European (Romance)", "Latn"),
    "ja": LanguageInfo("Japanese", "Japonic", "Jpan"),
    "jv": LanguageInfo("Javanese", "Austronesian (Malayo-Polynesian)", "Latn"),
    "ka": LanguageInfo("Georgian", "Kartvelian", "Geor"),
    "kab": LanguageInfo("Kabyle", "Afro-Asiatic (Berber)", "Latn"),
    "kbd": LanguageInfo("Kabardian", "Northwest Caucasian", "Cyrl"),
    "kk": LanguageInfo("Kazakh", "Turkic (Kipchak)", "Cyrl"),
    # CLDR has no kmr entry; its `ku` (Kurmanji) resolves to ku_Latn_TR.
    "kmr": LanguageInfo("Northern Kurdish", "Indo-European (Iranian)", "Latn"),
    "kn": LanguageInfo("Kannada", "Dravidian", "Knda"),
    "ko": LanguageInfo("Korean", "Koreanic", "Kore"),
    "lg": LanguageInfo("Luganda", "Atlantic-Congo (Bantu)", "Latn"),
    "lv": LanguageInfo("Latvian", "Indo-European (Baltic)", "Latn"),
    # CLDR has no mhr entry; it resolves the Mari macrolanguage chm to chm_Cyrl_RU.
    "mhr": LanguageInfo("Meadow Mari", "Uralic (Mari)", "Cyrl"),
    "ne": LanguageInfo("Nepali", "Indo-European (Indo-Aryan)", "Deva"),
    "nl": LanguageInfo("Dutch", "Indo-European (Germanic)", "Latn"),
    "nso": LanguageInfo("Sepedi", "Atlantic-Congo (Sotho-Tswana)", "Latn"),
    "pl": LanguageInfo("Polish", "Indo-European (Slavic)", "Latn"),
    "ps": LanguageInfo("Pashto", "Indo-European (Iranian)", "Arab"),
    "pt": LanguageInfo("Portuguese", "Indo-European (Romance)", "Latn"),
    "ru": LanguageInfo("Russian", "Indo-European (Slavic)", "Cyrl"),
    "rw": LanguageInfo("Kinyarwanda", "Atlantic-Congo (Bantu)", "Latn"),
    "si": LanguageInfo("Sinhala", "Indo-European (Indo-Aryan)", "Sinh"),
    "su": LanguageInfo("Sundanese", "Austronesian (Malayo-Polynesian)", "Latn"),
    "sw": LanguageInfo("Swahili", "Atlantic-Congo (Bantu)", "Latn"),
    "ta": LanguageInfo("Tamil", "Dravidian", "Taml"),
    "th": LanguageInfo("Thai", "Kra-Dai (Tai)", "Thai"),
    "tr": LanguageInfo("Turkish", "Turkic (Oghuz)", "Latn"),
    "ts": LanguageInfo("Xitsonga", "Atlantic-Congo (Tswa-Ronga)", "Latn"),
    "ug": LanguageInfo("Uyghur", "Turkic (Karluk)", "Arab"),
    "uk": LanguageInfo("Ukrainian", "Indo-European (Slavic)", "Cyrl"),
    "ur": LanguageInfo("Urdu", "Indo-European (Indo-Aryan)", "Arab"),
    "uz": LanguageInfo("Uzbek", "Turkic (Karluk)", "Latn"),
    "ve": LanguageInfo("Tshivenda", "Atlantic-Congo (Venda)", "Latn"),
    "xh": LanguageInfo("isiXhosa", "Atlantic-Congo (Nguni)", "Latn"),
    "yue": LanguageInfo("Cantonese", "Sino-Tibetan (Sinitic)", "Hant"),
    "zh-CN": LanguageInfo("Chinese (Mandarin)", "Sino-Tibetan (Sinitic)", "Hans"),
    "zu": LanguageInfo("isiZulu", "Atlantic-Congo (Nguni)", "Latn"),
}


def language_info(locale: str) -> LanguageInfo:
    """The curated entry for one locale, or a failure naming it.

    Family, script and English name are not in the release document, so they are
    curated here. Relaxing the thresholds selects locales this file has never
    described; that has to stop and name them rather than emit a preset entry
    whose choice of primary metric was a guess.
    """
    try:
        return LANGUAGES[locale]
    except KeyError:
        raise ValueError(
            f"{locale}: no curated language entry. Add its name, family and the "
            f"CLDR-resolved ISO 15924 script code to LANGUAGES in {Path(__file__).name}; "
            "the script code is what decides whether word error rate means anything."
        ) from None


def word_boundary(locale: str) -> bool:
    """Whether the language's writing system separates words with spaces."""
    return language_info(locale).script not in NO_SPACE_SCRIPTS


@dataclass(frozen=True)
class LocaleStats:
    """One locale's split sizes, in hours, as the release document implies them."""

    locale: str
    train_hours: float  # the official train.tsv, kept for comparison
    dev_hours: float
    test_hours: float
    validated_hours: float
    avg_clip_secs: float
    train_clips: int
    #: "commonvoice", or "manifest" for a corpus in configs/corpora.yaml.
    source: str = "commonvoice"
    #: Corpus id, for a manifest language. Empty for Common Voice.
    corpus: str = ""
    #: Set for a manifest language, whose hours come from its corpus record
    #: rather than from a bucket count times a mean clip length.
    stated_trainable_hours: float | None = None

    @property
    def trainable_hours(self) -> float:
        """The pool a run actually trains on: validated, less dev and test.

        An upper bound. The release document has no speaker information, so this
        cannot subtract the validated clips the loader's speaker guard removes —
        further recordings by a dev or test speaker. The measured figure comes
        from ``svb data-stats``, which reads the split files themselves.

        Clamped at zero: ``validHrs`` is the release's own number while dev and
        test are a count times a mean, so the subtraction can go slightly
        negative on a rounding edge, and a negative sorts above real hours.
        """
        if self.stated_trainable_hours is not None:
            return self.stated_trainable_hours
        return max(0.0, self.validated_hours - self.dev_hours - self.test_hours)


@dataclass(frozen=True)
class Thresholds:
    """The rule, in the units the release publishes."""

    min_train_hours: float = 50.0
    min_dev_hours: float = 2.0
    min_test_hours: float = 2.0

    def describe(self) -> str:
        return (
            f"trainable >= {self.min_train_hours} h, dev >= {self.min_dev_hours} h, "
            f"test >= {self.min_test_hours} h"
        )


@dataclass(frozen=True)
class Decision:
    """A locale, the verdict on it, and why."""

    stats: LocaleStats
    included: bool
    reason: str


def load_locales(release: Mapping[str, Any]) -> list[LocaleStats]:
    """Split hours per locale from one release document.

    A bucket a locale does not have counts as zero rather than as an error:
    locales added late in a release cycle carry fewer buckets than the ones that
    have been through a full round of validation.
    """
    stats: list[LocaleStats] = []
    for locale, entry in release["locales"].items():
        buckets = entry.get("buckets") or {}
        avg = float(entry.get("avgDurationSecs") or 0.0)
        stats.append(
            LocaleStats(
                locale=locale,
                train_hours=buckets.get("train", 0) * avg / 3600,
                dev_hours=buckets.get("dev", 0) * avg / 3600,
                test_hours=buckets.get("test", 0) * avg / 3600,
                validated_hours=float(entry.get("validHrs") or 0.0),
                avg_clip_secs=avg,
                train_clips=int(buckets.get("train", 0)),
            )
        )
    return stats


#: What the loader derives for a corpus that ships no usable split. A corpus
#: over the training threshold therefore clears the evaluation thresholds by
#: construction, which is why an unpublished dev or test figure is not a gap.
_DERIVED_EVAL_FRACTION = 0.1


def corpus_locales(corpora: Iterable[Any]) -> list[LocaleStats]:
    """One candidate per language of every corpus in ``configs/corpora.yaml``.

    A corpus that ships a full split is taken at its published train, dev and
    test hours. Anything else is re-derived by the loader, so its trainable
    hours are the total less the tenth each that the derivation gives dev and
    test — the same arithmetic the loader will do, stated here rather than left
    for a reader to work out.
    """
    stats: list[LocaleStats] = []
    for corpus in corpora:
        for code in corpus.languages:
            if corpus.ships_split == "full" and corpus.train_hours is not None:
                trainable = corpus.train_hours
                dev = corpus.dev_hours or 0.0
                test = corpus.test_hours or 0.0
            else:
                total = corpus.total_hours or (corpus.train_hours or 0.0)
                dev = test = total * _DERIVED_EVAL_FRACTION
                trainable = total - dev - test
            stats.append(
                LocaleStats(
                    locale=code,
                    train_hours=corpus.train_hours or 0.0,
                    dev_hours=dev,
                    test_hours=test,
                    validated_hours=corpus.total_hours or 0.0,
                    avg_clip_secs=0.0,
                    train_clips=0,
                    source="manifest",
                    corpus=corpus.id,
                    stated_trainable_hours=trainable,
                )
            )
    return stats


def _shortfall(stats: LocaleStats, thresholds: Thresholds) -> str:
    """Which part of the rule a locale misses, phrased with the number that missed."""
    for label, have, need in (
        ("trainable", stats.trainable_hours, thresholds.min_train_hours),
        ("dev", stats.dev_hours, thresholds.min_dev_hours),
        ("test", stats.test_hours, thresholds.min_test_hours),
    ):
        if have < need:
            return f"{label} {have:.1f} h < {need} h"
    return ""


def _base(locale: str) -> str:
    """The language subtag of a locale, with any region or variant dropped."""
    return locale.split("-", 1)[0]


def _key(stats: LocaleStats) -> tuple[str, str]:
    """A language can be a candidate twice — from Common Voice and from a
    corpus — so a verdict is keyed by both."""
    return (stats.source, stats.locale)


def decide(
    stats: Iterable[LocaleStats],
    thresholds: Thresholds | None = None,
    committed: Sequence[str] = COMMITTED_LOCALES,
    held_out: Mapping[str, str] = HELD_OUT_LOCALES,
    carried: Mapping[str, str] = CARRIED_LANGUAGES,
) -> list[Decision]:
    """Apply the rule to every candidate, most trainable audio first.

    Four things override the arithmetic:

    * A held-out transfer language is excluded however large it is.
    * A language served by a dedicated corpus is read from there rather than
      from Common Voice.
    * A language the smaller presets commit to, or one carried for typological
      coverage, is kept however small and says why.
    * Where a language appears under more than one locale, only the one with the
      most trainable audio survives — two variants are two spellings of the same
      training signal, and keeping both would weight that language twice in a
      mix whose point is breadth across languages.
    """
    thresholds = thresholds or Thresholds()
    conflict = sorted((set(committed) | set(carried)) & set(held_out))
    if conflict:
        raise ValueError(
            f"{', '.join(conflict)}: listed both as trained on and as held out for transfer. "
            "A language cannot be trained on and held out of training at once."
        )

    ordered = sorted(stats, key=lambda s: (-s.trainable_hours, s.locale))
    committed_set = set(committed)

    # A language served by a dedicated corpus is read from there, not from
    # Common Voice. Bengali is the case that matters: 31.5 trainable hours in
    # Common Voice against 229 in SLR53, and it is the corpus that puts the
    # language in the preset at all.
    by_corpus = {s.locale: s.corpus for s in ordered if s.source == "manifest"}

    provisional: dict[tuple[str, str], tuple[bool, str]] = {}
    for entry in ordered:
        key = _key(entry)
        if entry.source == "commonvoice" and entry.locale in by_corpus:
            provisional[key] = (
                False,
                f"read from {by_corpus[entry.locale]} instead, which has more trainable audio",
            )
        elif entry.locale in held_out:
            provisional[key] = (
                False,
                f"held-out transfer language ({held_out[entry.locale]}); never trained on",
            )
        elif not (shortfall := _shortfall(entry, thresholds)):
            provisional[key] = (True, f"meets the rule ({thresholds.describe()})")
        elif entry.locale in carried:
            provisional[key] = (True, f"below the rule ({shortfall}); {carried[entry.locale]}")
        elif entry.locale in committed_set:
            provisional[key] = (
                True,
                f"below the rule ({shortfall}); kept because the smaller presets commit to it",
            )
        else:
            provisional[key] = (False, shortfall)

    kept_per_language: dict[str, str] = {}
    for entry in ordered:  # largest first, so the first one seen is the largest
        if provisional[_key(entry)][0]:
            kept_per_language.setdefault(_base(entry.locale), entry.locale)

    decisions: list[Decision] = []
    for entry in ordered:
        included, reason = provisional[_key(entry)]
        winner = kept_per_language.get(_base(entry.locale))
        if included and winner != entry.locale:
            included, reason = (
                False,
                f"same language as {winner}, which has more trainable audio",
            )
        decisions.append(Decision(stats=entry, included=included, reason=reason))
    return decisions


def render_preset(
    decisions: Iterable[Decision],
    release_label: str = RELEASE_LABEL,
    normalizers: Mapping[str, str] | None = None,
) -> str:
    """The `languages:` block of a scale preset, in the shipped flow style.

    ``normalizers`` carries forward the ``normalizer:`` each language already
    had. A language that has none is written **without** the key rather than
    with a placeholder: a placeholder would have to be a policy name, and an
    unknown one makes the preset unloadable, which would take every test that
    reads a preset down with it. Leaving the key out keeps the preset loadable
    and still fails a run at policy resolution, which is where the gap belongs.
    """
    normalizers = normalizers or {}
    lines = ["languages:"]
    pending: list[str] = []
    for decision in decisions:
        if not decision.included:
            continue
        stats = decision.stats
        code = stats.locale
        if stats.source == "manifest":
            fields = [f"code: {code}", "source: manifest", f"corpus: {stats.corpus}"]
        else:
            fields = [
                f"code: {code}",
                "source: commonvoice",
                f"hf_dataset: {release_label}",
            ]
        fields.append(f"hf_config: {code}")
        if stats.source == "commonvoice":
            fields.append("text_column: sentence")
        if not word_boundary(code):
            fields.append("word_boundary: false")
        if code in normalizers:
            fields.append(f"normalizer: {normalizers[code]}")
        else:
            pending.append(code)
        lines.append("  - { " + ", ".join(fields) + " }")

    if pending:
        lines.insert(
            0,
            "# Awaiting a normalization policy, so they carry no `normalizer:` yet:\n"
            + "\n".join(f"#   {code}" for code in sorted(pending))
            + "\n# A run resolving policies will fail by name on these until they land.",
        )
    return "\n".join(lines) + "\n"


def normalizers_of(path: Path) -> dict[str, str]:
    """``code -> normalizer`` from an existing preset, so regenerating one does
    not silently drop the policy assignments already made."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        entry["code"]: entry["normalizer"]
        for entry in raw.get("languages", [])
        if entry.get("normalizer")
    }


_COLUMNS = (
    "locale", "language", "family", "script", "policy", "corpus", "trainable h",
    "train h", "dev h", "test h", "validated h", "included", "reason",
)  # fmt: skip


def _policy_of(resolve: Any, locale: str) -> str:
    """The policy name, or a marker when the language has no assignment yet.

    A selected locale that cannot resolve is a real gap, so it shows as `?`
    rather than silently reading like a language with no policy needed.
    """
    try:
        return str(resolve(locale).version)
    except (KeyError, ValueError):
        return "?"


def render_table(decisions: Iterable[Decision]) -> str:
    """One markdown row per locale: the numbers, the verdict, and the reason.

    Family, script and normalization policy are shown only for locales that
    were selected. Filling them in for all 290 would be 290 claims nothing in
    this repository checks, so an excluded row says nothing rather than
    something unsourced.

    The policy column is generated rather than written by hand because a
    regeneration that dropped it is how the assignments were lost once already.
    """
    from svb.text.registry import policy_for_language

    rows = ["| " + " | ".join(_COLUMNS) + " |", "|" + "---|" * len(_COLUMNS)]
    for decision in decisions:
        stats = decision.stats
        info = language_info(stats.locale) if decision.included else None
        rows.append(
            "| "
            + " | ".join(
                [
                    stats.locale,
                    info.name if info else "—",
                    info.family if info else "—",
                    info.script if info else "—",
                    _policy_of(policy_for_language, stats.locale) if info else "—",
                    stats.corpus or "common_voice_25",
                    f"{stats.trainable_hours:.1f}",
                    f"{stats.train_hours:.1f}",
                    f"{stats.dev_hours:.1f}",
                    f"{stats.test_hours:.1f}",
                    f"{stats.validated_hours:.1f}",
                    "yes" if decision.included else "no",
                    decision.reason,
                ]
            )
            + " |"
        )
    return "\n".join(rows) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", required=True, type=Path, help="release statistics JSON")
    parser.add_argument(
        "--corpora",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
        help="directory holding corpora.yaml (default: the repo's configs/)",
    )
    parser.add_argument("--preset-out", type=Path, help="write the preset's languages block here")
    parser.add_argument("--table-out", type=Path, help="write the evidence table here")
    parser.add_argument(
        "--write-preset",
        action="store_true",
        help="allow --preset-out to overwrite a preset that already exists",
    )
    parser.add_argument("--min-train-hours", type=float, default=Thresholds.min_train_hours)
    parser.add_argument("--min-dev-hours", type=float, default=Thresholds.min_dev_hours)
    parser.add_argument("--min-test-hours", type=float, default=Thresholds.min_test_hours)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = Thresholds(args.min_train_hours, args.min_dev_hours, args.min_test_hours)

    release = json.loads(args.release.read_text(encoding="utf-8"))
    candidates = load_locales(release)
    if args.corpora is not None:
        from svb.data.corpora import load_corpora

        candidates += corpus_locales(load_corpora(args.corpora))
    decisions = decide(candidates, thresholds)
    included = [d for d in decisions if d.included]

    # A count with no thresholds beside it is not a reproducible selection.
    print(
        f"[svb] {len(included)} of {len(decisions)} locales selected "
        f"({thresholds.describe()}) from {args.release}"
    )
    if args.preset_out:
        if args.preset_out.exists() and not args.write_preset:
            # A committed preset is a decision someone made. Regenerating it
            # from a newer release is a decision someone else has to make, so
            # the default run reports what would change and leaves the file.
            raise SystemExit(
                f"[svb] {args.preset_out} already exists; pass --write-preset to overwrite it. "
                f"The rule selects {len(included)} locales from this release."
            )
        args.preset_out.write_text(
            render_preset(decisions, normalizers=normalizers_of(args.preset_out)),
            encoding="utf-8",
        )
        print(f"[svb] wrote {args.preset_out}")
    if args.table_out:
        args.table_out.write_text(render_table(decisions), encoding="utf-8")
        print(f"[svb] wrote {args.table_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
