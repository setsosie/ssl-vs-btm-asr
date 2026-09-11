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

# ISO 15924 codes for writing systems that do not put spaces between words.
# Whitespace tokenization of such a transcript yields one token per sentence, so
# word error rate over it is meaningless and character error rate is primary.
NO_SPACE_SCRIPTS = frozenset(
    {"Hani", "Hans", "Hant", "Jpan", "Kore", "Thai", "Laoo", "Khmr", "Mymr", "Tibt"}
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
    "ca": LanguageInfo("Catalan", "Indo-European (Romance)", "Latn"),
    "ckb": LanguageInfo("Central Kurdish", "Indo-European (Iranian)", "Arab"),
    "cs": LanguageInfo("Czech", "Indo-European (Slavic)", "Latn"),
    "cy": LanguageInfo("Welsh", "Indo-European (Celtic)", "Latn"),
    "de": LanguageInfo("German", "Indo-European (Germanic)", "Latn"),
    "en": LanguageInfo("English", "Indo-European (Germanic)", "Latn"),
    "eo": LanguageInfo("Esperanto", "Constructed", "Latn"),
    "es": LanguageInfo("Spanish", "Indo-European (Romance)", "Latn"),
    "eu": LanguageInfo("Basque", "Isolate", "Latn"),
    "fa": LanguageInfo("Persian", "Indo-European (Iranian)", "Arab"),
    "fi": LanguageInfo("Finnish", "Uralic (Finnic)", "Latn"),
    "fr": LanguageInfo("French", "Indo-European (Romance)", "Latn"),
    "fy-NL": LanguageInfo("West Frisian", "Indo-European (Germanic)", "Latn"),
    "gl": LanguageInfo("Galician", "Indo-European (Romance)", "Latn"),
    "hi": LanguageInfo("Hindi", "Indo-European (Indo-Aryan)", "Deva"),
    "hu": LanguageInfo("Hungarian", "Uralic (Ugric)", "Latn"),
    "it": LanguageInfo("Italian", "Indo-European (Romance)", "Latn"),
    "ja": LanguageInfo("Japanese", "Japonic", "Jpan"),
    "ka": LanguageInfo("Georgian", "Kartvelian", "Geor"),
    "kab": LanguageInfo("Kabyle", "Afro-Asiatic (Berber)", "Latn"),
    "kbd": LanguageInfo("Kabardian", "Northwest Caucasian", "Cyrl"),
    # CLDR has no kmr entry; its `ku` (Kurmanji) resolves to ku_Latn_TR.
    "kmr": LanguageInfo("Northern Kurdish", "Indo-European (Iranian)", "Latn"),
    "lg": LanguageInfo("Luganda", "Atlantic-Congo (Bantu)", "Latn"),
    "lv": LanguageInfo("Latvian", "Indo-European (Baltic)", "Latn"),
    # CLDR has no mhr entry; it resolves the Mari macrolanguage chm to chm_Cyrl_RU.
    "mhr": LanguageInfo("Meadow Mari", "Uralic (Mari)", "Cyrl"),
    "nl": LanguageInfo("Dutch", "Indo-European (Germanic)", "Latn"),
    "pl": LanguageInfo("Polish", "Indo-European (Slavic)", "Latn"),
    "ps": LanguageInfo("Pashto", "Indo-European (Iranian)", "Arab"),
    "pt": LanguageInfo("Portuguese", "Indo-European (Romance)", "Latn"),
    "ru": LanguageInfo("Russian", "Indo-European (Slavic)", "Cyrl"),
    "rw": LanguageInfo("Kinyarwanda", "Atlantic-Congo (Bantu)", "Latn"),
    "sw": LanguageInfo("Swahili", "Atlantic-Congo (Bantu)", "Latn"),
    "ta": LanguageInfo("Tamil", "Dravidian", "Taml"),
    "th": LanguageInfo("Thai", "Kra-Dai (Tai)", "Thai"),
    "tr": LanguageInfo("Turkish", "Turkic (Oghuz)", "Latn"),
    "ug": LanguageInfo("Uyghur", "Turkic (Karluk)", "Arab"),
    "uk": LanguageInfo("Ukrainian", "Indo-European (Slavic)", "Cyrl"),
    "ur": LanguageInfo("Urdu", "Indo-European (Indo-Aryan)", "Arab"),
    "uz": LanguageInfo("Uzbek", "Turkic (Karluk)", "Latn"),
    "yue": LanguageInfo("Cantonese", "Sino-Tibetan (Sinitic)", "Hant"),
    "zh-CN": LanguageInfo("Chinese (Mandarin)", "Sino-Tibetan (Sinitic)", "Hans"),
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


def decide(
    stats: Iterable[LocaleStats],
    thresholds: Thresholds | None = None,
    committed: Sequence[str] = COMMITTED_LOCALES,
    held_out: Mapping[str, str] = HELD_OUT_LOCALES,
) -> list[Decision]:
    """Apply the rule to every locale, most trainable audio first.

    Three things override the arithmetic:

    * A held-out transfer language is excluded however large it is.
    * A language the smaller presets are already committed to is kept however
      small it is, and carries the shortfall as its reason.
    * Where a language appears under more than one locale, only the one with the
      most training audio survives — two variants are two spellings of the same
      training signal, and keeping both would weight that language twice in a
      mix whose point is breadth across languages.
    """
    thresholds = thresholds or Thresholds()
    conflict = sorted(set(committed) & set(held_out))
    if conflict:
        raise ValueError(
            f"{', '.join(conflict)}: listed both as committed and as held out for transfer. "
            "A language cannot be trained on and held out of training at once."
        )

    ordered = sorted(stats, key=lambda s: (-s.trainable_hours, s.locale))
    committed_set = set(committed)

    # Provisional verdicts, before the one-locale-per-language rule.
    provisional: dict[str, tuple[bool, str]] = {}
    for entry in ordered:
        if entry.locale in held_out:
            provisional[entry.locale] = (
                False,
                f"held-out transfer language ({held_out[entry.locale]}); never trained on",
            )
            continue
        shortfall = _shortfall(entry, thresholds)
        if not shortfall:
            provisional[entry.locale] = (True, f"meets the rule ({thresholds.describe()})")
        elif entry.locale in committed_set:
            provisional[entry.locale] = (
                True,
                f"below the rule ({shortfall}); kept because the smaller presets commit to it",
            )
        else:
            provisional[entry.locale] = (False, shortfall)

    kept_per_language: dict[str, str] = {}
    for entry in ordered:  # largest first, so the first one seen is the largest
        if provisional[entry.locale][0]:
            kept_per_language.setdefault(_base(entry.locale), entry.locale)

    decisions: list[Decision] = []
    for entry in ordered:
        included, reason = provisional[entry.locale]
        winner = kept_per_language.get(_base(entry.locale))
        if included and winner != entry.locale:
            included, reason = (
                False,
                f"same language as {winner}, which has more trainable audio",
            )
        decisions.append(Decision(stats=entry, included=included, reason=reason))
    return decisions


def render_preset(decisions: Iterable[Decision], release_label: str = RELEASE_LABEL) -> str:
    """The `languages:` block of a scale preset, in the shipped flow style."""
    lines = ["languages:"]
    for decision in decisions:
        if not decision.included:
            continue
        code = decision.stats.locale
        fields = [
            f"code: {code}",
            "source: commonvoice",
            f"hf_dataset: {release_label}",
            f"hf_config: {code}",
            "text_column: sentence",
        ]
        if not word_boundary(code):
            fields.append("word_boundary: false")
        lines.append("  - { " + ", ".join(fields) + " }")
    return "\n".join(lines) + "\n"


_COLUMNS = (
    "locale", "language", "family", "script", "trainable h", "train h",
    "dev h", "test h", "validated h", "avg clip s", "included", "reason",
)  # fmt: skip


def render_table(decisions: Iterable[Decision]) -> str:
    """One markdown row per locale: the numbers, the verdict, and the reason.

    Family and script are shown only for locales that were selected. Filling
    them in for all 290 would be 290 claims nothing in this repository checks,
    so an excluded row says nothing rather than something unsourced.
    """
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
                    f"{stats.trainable_hours:.1f}",
                    f"{stats.train_hours:.1f}",
                    f"{stats.dev_hours:.1f}",
                    f"{stats.test_hours:.1f}",
                    f"{stats.validated_hours:.1f}",
                    f"{stats.avg_clip_secs:.3f}",
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
    decisions = decide(load_locales(release), thresholds)
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
        args.preset_out.write_text(render_preset(decisions), encoding="utf-8")
        print(f"[svb] wrote {args.preset_out}")
    if args.table_out:
        args.table_out.write_text(render_table(decisions), encoding="utf-8")
        print(f"[svb] wrote {args.table_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
