"""`scripts/select_languages.py`: the rule that turns release statistics into a preset.

Every test here runs against a synthetic release document rather than Mozilla's
own, so the arithmetic and the rule are pinned without a network fetch and
without committing a third-party 300 kB JSON. What the real release says is
recorded in `docs/languages.md`; what the rule does with it is recorded here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


def _locale(
    train: int, dev: int, test: int, avg: float = 5.0, validated: int | None = None
) -> dict[str, Any]:
    validated = train + dev + test if validated is None else validated
    return {
        "buckets": {"train": train, "dev": dev, "test": test, "validated": validated},
        "avgDurationSecs": avg,
        "validHrs": round(validated * avg / 3600, 2),
    }


def _release(**locales: dict[str, Any]) -> dict[str, Any]:
    return {"locales": locales}


# --------------------------------------------------------------------------- #
# Hours
# --------------------------------------------------------------------------- #


def test_hours_are_clip_counts_times_the_release_mean_clip_duration(
    select_languages: ModuleType,
) -> None:
    """The release publishes counts per split and one mean duration per locale.

    It publishes no per-split duration, so a split's hours can only be the
    product of the two. That approximation is the whole basis of the selection
    and is stated as such in `docs/languages.md`.
    """
    stats = select_languages.load_locales(
        _release(xx=_locale(train=720, dev=72, test=36, avg=10.0))
    )

    (only,) = stats
    assert only.locale == "xx"
    assert only.train_hours == pytest.approx(2.0)  # 720 * 10 / 3600
    assert only.dev_hours == pytest.approx(0.2)
    assert only.test_hours == pytest.approx(0.1)
    assert only.train_clips == 720


def test_a_locale_missing_a_split_counts_as_zero_hours_not_as_an_error(
    select_languages: ModuleType,
) -> None:
    """Locales added late in a release cycle have buckets the older ones do not."""
    stats = select_languages.load_locales(
        {"locales": {"xx": {"buckets": {}, "avgDurationSecs": 5.0}}}
    )

    (only,) = stats
    assert (only.train_hours, only.dev_hours, only.test_hours) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #


def _decide(select_languages: ModuleType, release: dict[str, Any], **kwargs: Any):
    kwargs.setdefault("committed", ())
    return {
        d.stats.locale: d
        for d in select_languages.decide(select_languages.load_locales(release), **kwargs)
    }


def test_a_locale_over_every_threshold_is_included(select_languages: ModuleType) -> None:
    clips = int(60 * 3600 / 5.0)  # 60 h of train audio
    decisions = _decide(select_languages, _release(en=_locale(train=clips, dev=2000, test=2000)))

    assert decisions["en"].included
    assert "50" in decisions["en"].reason


def test_a_locale_short_of_train_audio_is_excluded_and_says_by_how_much(
    select_languages: ModuleType,
) -> None:
    clips = int(40 * 3600 / 5.0)
    decisions = _decide(select_languages, _release(xx=_locale(train=clips, dev=2000, test=2000)))

    assert not decisions["xx"].included
    assert "40.0" in decisions["xx"].reason and "train" in decisions["xx"].reason


def test_training_audio_without_an_evaluation_split_is_not_enough(
    select_languages: ModuleType,
) -> None:
    """The rule is 50 h to train on *plus* a real dev and test split.

    A locale can be large and still have almost nothing held back — the release
    splits are regenerated per release and are not a fixed fraction — so the
    evaluation thresholds are checked separately rather than assumed to follow
    from the training one.
    """
    clips = int(500 * 3600 / 5.0)
    decisions = _decide(select_languages, _release(xx=_locale(train=clips, dev=100, test=2000)))

    assert not decisions["xx"].included
    assert "dev" in decisions["xx"].reason


def test_a_language_the_smaller_presets_committed_to_is_kept_and_flagged(
    select_languages: ModuleType,
) -> None:
    """The 3- and 16-language presets are already part of the paper's design.

    Dropping one of them here would make the smaller presets stop being subsets
    of the larger one, so a committed language stays in and carries the reason
    it would otherwise have been cut.
    """
    decisions = _decide(
        select_languages,
        _release(ja=_locale(train=1000, dev=2000, test=2000)),
        committed=("ja",),
    )

    assert decisions["ja"].included
    assert "below" in decisions["ja"].reason.lower()


def test_a_held_out_transfer_language_is_never_selected(select_languages: ModuleType) -> None:
    """Marathi is in Common Voice as well as in the held-out OpenSLR set.

    Training on the Common Voice half would make the held-out number measure
    transfer to a language the pipeline had already supervised, which is the one
    thing that set exists to avoid.
    """
    clips = int(900 * 3600 / 5.0)
    decisions = _decide(select_languages, _release(mr=_locale(train=clips, dev=9000, test=9000)))

    assert not decisions["mr"].included
    assert "held-out" in decisions["mr"].reason


def test_a_held_out_language_stays_out_even_if_a_caller_lists_it_as_committed(
    select_languages: ModuleType,
) -> None:
    clips = int(900 * 3600 / 5.0)
    with pytest.raises(ValueError, match="mr"):
        _decide(
            select_languages,
            _release(mr=_locale(train=clips, dev=9000, test=9000)),
            committed=("mr",),
        )


def test_only_the_largest_regional_variant_of_a_language_survives(
    select_languages: ModuleType,
) -> None:
    """Two variants of one language are two spellings of the same training signal.

    Keeping both would weight that language twice in a mix whose point is
    breadth across languages, so the larger one stands for it.
    """
    big, small = int(300 * 3600 / 5.0), int(100 * 3600 / 5.0)
    decisions = _decide(
        select_languages,
        _release(
            **{
                "zh-CN": _locale(train=big, dev=9000, test=9000),
                "zh-TW": _locale(train=small, dev=9000, test=9000),
            }
        ),
    )

    assert decisions["zh-CN"].included
    assert not decisions["zh-TW"].included
    assert "zh-CN" in decisions["zh-TW"].reason


def test_a_lone_regional_variant_is_not_a_duplicate_of_anything(
    select_languages: ModuleType,
) -> None:
    clips = int(300 * 3600 / 5.0)
    decisions = _decide(
        select_languages, _release(**{"nan-tw": _locale(train=clips, dev=9000, test=9000)})
    )

    assert decisions["nan-tw"].included


def test_decisions_are_ordered_by_training_hours(select_languages: ModuleType) -> None:
    decisions = select_languages.decide(
        select_languages.load_locales(
            _release(
                small=_locale(train=10, dev=1, test=1),
                big=_locale(train=1000, dev=1, test=1),
                mid=_locale(train=100, dev=1, test=1),
            )
        ),
        committed=(),
    )

    assert [d.stats.locale for d in decisions] == ["big", "mid", "small"]


# --------------------------------------------------------------------------- #
# Writing system
# --------------------------------------------------------------------------- #


def test_word_boundary_follows_the_script_rather_than_a_list_of_codes(
    select_languages: ModuleType,
) -> None:
    """Whether whitespace tokenization means anything is a fact about the script.

    Deriving it from the script the language is written in — the CLDR-resolved
    script recorded beside each language — keeps the two from drifting apart the
    way a hand-maintained list of locale codes would.
    """
    assert select_languages.language_info("ja").script == "Jpan"
    assert select_languages.word_boundary("ja") is False
    assert select_languages.word_boundary("ka") is True
    assert select_languages.word_boundary("ar") is True  # right-to-left, but spaced


def test_a_selected_locale_with_no_curated_entry_fails_by_name(
    select_languages: ModuleType,
) -> None:
    """Family, script and English name are not in the release document.

    They are curated, so lowering the threshold selects locales this file has
    never described. That has to stop the run and name them rather than emit a
    preset entry whose metric choice was guessed.
    """
    with pytest.raises(ValueError, match="xx"):
        select_languages.language_info("xx")


def test_every_committed_language_is_curated(select_languages: ModuleType) -> None:
    for locale in select_languages.COMMITTED_LOCALES:
        assert select_languages.language_info(locale).name


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def test_the_rendered_preset_parses_and_holds_exactly_the_included_locales(
    select_languages: ModuleType,
) -> None:
    clips = int(60 * 3600 / 5.0)
    decisions = select_languages.decide(
        select_languages.load_locales(
            _release(
                en=_locale(train=clips, dev=2000, test=2000),
                ja=_locale(train=10, dev=10, test=10),
                xx=_locale(train=10, dev=10, test=10),
            )
        ),
        committed=("ja",),
    )

    parsed = yaml.safe_load(select_languages.render_preset(decisions))

    assert [entry["code"] for entry in parsed["languages"]] == ["en", "ja"]
    assert all(entry["hf_config"] == entry["code"] for entry in parsed["languages"])
    assert all(entry["hf_dataset"] == "common_voice_25" for entry in parsed["languages"])


def test_the_rendered_preset_declares_word_boundary_only_where_it_is_false(
    select_languages: ModuleType,
) -> None:
    """`word_boundary` defaults to true, so spelling it out on every spaced
    language would bury the one line a reader needs to see."""
    decisions = select_languages.decide(
        select_languages.load_locales(
            _release(
                en=_locale(train=int(60 * 3600 / 5.0), dev=2000, test=2000),
                ja=_locale(train=10, dev=10, test=10),
            )
        ),
        committed=("ja",),
    )

    parsed = yaml.safe_load(select_languages.render_preset(decisions))
    by_code = {entry["code"]: entry for entry in parsed["languages"]}

    assert by_code["ja"]["word_boundary"] is False
    assert "word_boundary" not in by_code["en"]


def test_the_table_gives_every_locale_a_verdict_and_a_reason(
    select_languages: ModuleType,
) -> None:
    decisions = select_languages.decide(
        select_languages.load_locales(
            _release(
                en=_locale(train=int(60 * 3600 / 5.0), dev=2000, test=2000),
                xx=_locale(train=10, dev=10, test=10),
            )
        ),
        committed=(),
    )

    rows = [
        line
        for line in select_languages.render_table(decisions).splitlines()
        if line.startswith("| ")
    ]
    body = rows[1:]  # the column header, then one row per locale

    assert len(body) == 2
    assert all(row.count("|") == len(select_languages._COLUMNS) + 1 for row in body)
    assert "| yes |" in body[0] and "| no |" in body[1]


def test_the_table_leaves_family_and_script_blank_for_locales_it_never_selected(
    select_languages: ModuleType,
) -> None:
    """An excluded locale is a number, not a curated language entry.

    Filling in a family for all 290 of them would be 290 claims nothing in this
    repository checks, so the table says nothing rather than something unsourced.
    """
    decisions = select_languages.decide(
        select_languages.load_locales(_release(xx=_locale(train=10, dev=10, test=10))), committed=()
    )

    (row,) = [
        line
        for line in select_languages.render_table(decisions).splitlines()
        if line.startswith("| xx ")
    ]

    assert "| — | — |" in row


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _empty_registry(tmp_path: Path) -> Path:
    """A corpora registry with nothing in it, so a test of the Common Voice
    half is not also a test of the eighteen corpora."""
    (tmp_path / "corpora.yaml").write_text("corpora: []\n", encoding="utf-8")
    return tmp_path


def test_main_writes_a_preset_and_a_table_from_a_release_document(
    select_languages: ModuleType, tmp_path: Path
) -> None:
    release = tmp_path / "cv.json"
    release.write_text(
        json.dumps(
            _release(
                en=_locale(train=int(60 * 3600 / 5.0), dev=2000, test=2000),
                xx=_locale(train=10, dev=10, test=10),
            )
        ),
        encoding="utf-8",
    )
    preset, table = tmp_path / "64.yaml", tmp_path / "languages.md"

    select_languages.main(
        [
            "--release",
            str(release),
            "--corpora",
            str(_empty_registry(tmp_path)),
            "--preset-out",
            str(preset),
            "--table-out",
            str(table),
        ]
    )

    assert [e["code"] for e in yaml.safe_load(preset.read_text(encoding="utf-8"))["languages"]] == [
        "en"
    ]
    assert "| xx " in table.read_text(encoding="utf-8")


def test_main_reports_the_thresholds_it_applied(
    select_languages: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A count with no thresholds beside it is not a reproducible selection."""
    release = tmp_path / "cv.json"
    release.write_text(
        json.dumps(_release(en=_locale(train=10, dev=10, test=10))), encoding="utf-8"
    )

    select_languages.main(
        [
            "--release",
            str(release),
            "--corpora",
            str(_empty_registry(tmp_path)),
            "--min-train-hours",
            "0.001",
            "--min-dev-hours",
            "0",
            "--min-test-hours",
            "0",
        ]
    )
    out = capsys.readouterr().out

    assert "1 of 1 locales selected" in out
    assert "0.001" in out


# --------------------------------------------------------------------------- #
# The training statistic
# --------------------------------------------------------------------------- #


def test_the_training_statistic_is_validated_minus_the_evaluation_splits(
    select_languages: ModuleType,
) -> None:
    """A run trains on validated minus dev and test, so that is what the rule
    is applied to. `train.tsv` is roughly one clip per sentence and is not the
    training set any more."""
    stats = select_languages.load_locales(
        _release(xx=_locale(train=720, dev=360, test=360, avg=10.0, validated=3600))
    )

    (only,) = stats
    # 3600 clips validated at 10 s = 10 h, minus 1 h of dev and 1 h of test.
    assert only.validated_hours == pytest.approx(10.0)
    assert only.trainable_hours == pytest.approx(8.0)
    assert only.train_hours == pytest.approx(2.0)  # the official split, for comparison


def test_a_locale_whose_validated_pool_is_all_evaluation_has_no_training_hours(
    select_languages: ModuleType,
) -> None:
    """`validHrs` is the release's own figure while dev and test are products of
    a count and a mean, so the subtraction can go negative on a rounding edge.
    Negative training hours would be a nonsense that sorts above real ones."""
    stats = select_languages.load_locales(
        _release(xx=_locale(train=0, dev=1800, test=1800, avg=10.0, validated=3600))
    )

    assert stats[0].trainable_hours == 0.0


def test_a_locale_the_official_split_made_look_small_can_now_qualify(
    select_languages: ModuleType,
) -> None:
    """This is the whole point of the change: the official split is about a
    third of the validated audio, so languages that missed 50 hours on it clear
    the same threshold on the pool a run actually reads."""
    validated = int(120 * 3600 / 5.0)
    decisions = _decide(
        select_languages,
        _release(xx=_locale(train=1000, dev=2000, test=2000, validated=validated)),
    )

    assert decisions["xx"].included


def test_the_shortfall_names_the_trainable_hours_not_the_official_split(
    select_languages: ModuleType,
) -> None:
    validated = int(40 * 3600 / 5.0)
    decisions = _decide(
        select_languages, _release(xx=_locale(train=10, dev=2000, test=2000, validated=validated))
    )

    assert not decisions["xx"].included
    assert "trainable" in decisions["xx"].reason


def test_the_table_reports_both_statistics_side_by_side(select_languages: ModuleType) -> None:
    """A reader comparing these numbers to a published Common Voice table needs
    the official split's hours too, so both columns are there."""
    assert "train h" in select_languages._COLUMNS
    assert "trainable h" in select_languages._COLUMNS


# --------------------------------------------------------------------------- #
# Writing a preset
# --------------------------------------------------------------------------- #


def test_writing_over_an_existing_preset_needs_the_explicit_flag(
    select_languages: ModuleType, tmp_path: Path
) -> None:
    """A committed preset is a decision someone made, and regenerating it from
    a new release is a decision someone else has to make. Refusing by default
    means the script can be run to see what would change without changing it."""
    release = tmp_path / "cv.json"
    release.write_text(
        json.dumps(_release(en=_locale(train=int(60 * 3600 / 5.0), dev=2000, test=2000))),
        encoding="utf-8",
    )
    preset = tmp_path / "64.yaml"
    preset.write_text("languages: []\n", encoding="utf-8")

    corpora = str(_empty_registry(tmp_path))
    with pytest.raises(SystemExit, match="write-preset"):
        select_languages.main(
            ["--release", str(release), "--corpora", corpora, "--preset-out", str(preset)]
        )

    select_languages.main(
        [
            "--release",
            str(release),
            "--corpora",
            corpora,
            "--preset-out",
            str(preset),
            "--write-preset",
        ]
    )

    assert "code: en" in preset.read_text(encoding="utf-8")


def test_writing_a_preset_that_does_not_exist_yet_needs_no_flag(
    select_languages: ModuleType, tmp_path: Path
) -> None:
    release = tmp_path / "cv.json"
    release.write_text(
        json.dumps(_release(en=_locale(train=int(60 * 3600 / 5.0), dev=2000, test=2000))),
        encoding="utf-8",
    )
    preset = tmp_path / "new.yaml"

    select_languages.main(
        [
            "--release",
            str(release),
            "--corpora",
            str(_empty_registry(tmp_path)),
            "--preset-out",
            str(preset),
        ]
    )

    assert preset.exists()


# --------------------------------------------------------------------------- #
# Merging Common Voice with the other corpora
# --------------------------------------------------------------------------- #


def _corpus(select_languages: ModuleType, **kwargs: Any):
    from svb.data.corpora import CorpusSpec

    defaults = {
        "id": "demo_corpus",
        "name": "Demo",
        "languages": ["xx"],
        "source_page": "https://example.org/",
        "licence_name": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "downloads": [{"name": "a.zip", "url": "https://example.org/a.zip"}],
        "audio_format": "wav",
        "transcripts": "tsv",
        "speaker_ids": "column",
        "ships_split": "none",
        "preparer": "scripts/prepare_google_crowdsourced.py",
        "citation": "Demo",
    }
    return CorpusSpec(**{**defaults, **kwargs})


def test_a_corpus_that_derives_its_split_is_priced_at_the_derivation(
    select_languages: ModuleType,
) -> None:
    """A corpus with no shipped split gets a tenth each to dev and test, which
    is exactly what the loader does. Stating it here means the rule is applied
    to the hours a run will train on rather than to the corpus total."""
    (stats,) = select_languages.corpus_locales([_corpus(select_languages, total_hours=100.0)])

    assert stats.trainable_hours == pytest.approx(80.0)
    assert stats.dev_hours == pytest.approx(10.0)
    assert stats.source == "manifest"
    assert stats.corpus == "demo_corpus"


def test_a_corpus_that_ships_a_full_split_is_taken_at_its_published_hours(
    select_languages: ModuleType,
) -> None:
    (stats,) = select_languages.corpus_locales(
        [
            _corpus(
                select_languages,
                ships_split="full",
                train_hours=48.5,
                dev_hours=3.7,
                test_hours=4.0,
                total_hours=56.0,
            )
        ]
    )

    assert stats.trainable_hours == pytest.approx(48.5)
    assert stats.test_hours == pytest.approx(4.0)


def test_a_language_with_a_dedicated_corpus_is_read_from_there_not_common_voice(
    select_languages: ModuleType,
) -> None:
    """Bengali is the case: 31.5 trainable hours in Common Voice against 229 in
    SLR53, and the corpus is what puts the language in the preset at all."""
    cv = select_languages.load_locales(
        _release(bn=_locale(train=1000, dev=2000, test=2000, validated=int(60 * 3600 / 5.0)))
    )
    corpus = select_languages.corpus_locales(
        [_corpus(select_languages, id="slr53_bengali", languages=["bn"], total_hours=229.0)]
    )

    decisions = {(d.stats.source, d.stats.locale): d for d in select_languages.decide(cv + corpus)}

    assert decisions[("manifest", "bn")].included
    assert not decisions[("commonvoice", "bn")].included
    assert "slr53_bengali" in decisions[("commonvoice", "bn")].reason


def test_a_corpus_offering_a_held_out_language_is_still_refused(
    select_languages: ModuleType,
) -> None:
    """The rule that matters most. Five of the corpora come from the same
    crowdsourced programme as the held-out four."""
    corpus = select_languages.corpus_locales(
        [_corpus(select_languages, id="slr66_telugu", languages=["te"], total_hours=900.0)]
    )

    (decision,) = select_languages.decide(corpus)

    assert not decision.included
    assert "held-out" in decision.reason


def test_a_language_carried_for_coverage_says_what_it_is_short_of(
    select_languages: ModuleType,
) -> None:
    """Four of the five NCHLT languages ship a train split just under the bar
    while their corpus totals about 56 hours. Kept, and the reason is the
    reason, not a silent pass."""
    corpus = select_languages.corpus_locales(
        [
            _corpus(
                select_languages,
                id="nchlt_zulu",
                languages=["zu"],
                ships_split="full",
                train_hours=48.5,
                dev_hours=3.7,
                test_hours=4.0,
                total_hours=56.0,
            )
        ]
    )

    (decision,) = select_languages.decide(corpus)

    assert decision.included
    assert "typological coverage" in decision.reason


def test_a_corpus_language_below_the_rule_and_not_carried_is_excluded(
    select_languages: ModuleType,
) -> None:
    corpus = select_languages.corpus_locales(
        [_corpus(select_languages, id="tiny", languages=["fo"], total_hours=10.0)]
    )

    (decision,) = select_languages.decide(corpus)

    assert not decision.included


def test_the_preset_entry_for_a_corpus_language_names_its_corpus(
    select_languages: ModuleType,
) -> None:
    corpus = select_languages.corpus_locales(
        [_corpus(select_languages, id="slr35_javanese", languages=["jv"], total_hours=296.0)]
    )

    (entry,) = yaml.safe_load(select_languages.render_preset(select_languages.decide(corpus)))[
        "languages"
    ]

    assert entry == {
        "code": "jv",
        "source": "manifest",
        "corpus": "slr35_javanese",
        "hf_config": "jv",
    }


def test_a_language_with_no_policy_yet_is_written_without_the_key(
    select_languages: ModuleType,
) -> None:
    """A placeholder would have to be a policy name, and an unknown one makes
    the preset unloadable — which would take down every test that reads a
    preset. Omitting the key keeps it loadable and still fails a run at policy
    resolution, which is where the gap belongs."""
    corpus = select_languages.corpus_locales(
        [_corpus(select_languages, id="slr35_javanese", languages=["jv"], total_hours=296.0)]
    )

    rendered = select_languages.render_preset(select_languages.decide(corpus))

    assert "normalizer" not in yaml.safe_load(rendered)["languages"][0]
    assert "Awaiting a normalization policy" in rendered
    assert "#   jv" in rendered


def test_regenerating_a_preset_carries_its_policy_assignments_forward(
    select_languages: ModuleType, tmp_path: Path
) -> None:
    """Regenerating must not silently drop work done on another branch."""
    existing = tmp_path / "64.yaml"
    existing.write_text(
        "languages:\n  - { code: jv, source: manifest, corpus: slr35_javanese, "
        "hf_config: jv, normalizer: latin-marks }\n",
        encoding="utf-8",
    )
    corpus = select_languages.corpus_locales(
        [_corpus(select_languages, id="slr35_javanese", languages=["jv"], total_hours=296.0)]
    )

    rendered = select_languages.render_preset(
        select_languages.decide(corpus), normalizers=select_languages.normalizers_of(existing)
    )

    assert yaml.safe_load(rendered)["languages"][0]["normalizer"] == "latin-marks"
    assert "Awaiting a normalization policy" not in rendered
