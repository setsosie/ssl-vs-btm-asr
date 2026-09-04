"""LangSpec validates both sources; the shipped presets parse."""

from pathlib import Path

import pytest

from svb.data.registry import LangSpec, get_heldout, get_preset


def _cv() -> LangSpec:
    return LangSpec(code="hi", source="commonvoice", hf_dataset="common_voice_25", hf_config="hi")


def _slr() -> LangSpec:
    return LangSpec(
        code="malayalam",
        source="openslr",
        slr=63,
        archives=("ml_in_female.zip", "ml_in_male.zip"),
        index_files=("line_index_female.tsv", "line_index_male.tsv"),
        license="CC-BY-SA-4.0",
    )


def test_commonvoice_spec_needs_no_openslr_fields():
    spec = _cv()
    assert spec.text_column == "sentence"
    assert spec.slr is None
    assert spec.archives == ()


def test_openslr_spec_needs_no_commonvoice_fields():
    spec = _slr()
    assert spec.hf_config == ""
    assert spec.slr == 63


def test_openslr_sequences_are_coerced_to_tuples_so_the_spec_stays_hashable():
    spec = _slr()
    assert spec.archives == ("ml_in_female.zip", "ml_in_male.zip")
    assert spec.index_files == ("line_index_female.tsv", "line_index_male.tsv")
    assert hash(spec) == hash(_slr())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"code": "x", "source": "kaldi"},
        {"code": "x", "source": "commonvoice"},  # no hf_config
        {"code": "x", "source": "openslr", "archives": ["a.zip"], "index_files": ["i.tsv"]},
        {"code": "x", "source": "openslr", "slr": 63, "index_files": ["i.tsv"]},
        # one index file per archive, or the pairing is ambiguous
        {
            "code": "x",
            "source": "openslr",
            "slr": 63,
            "archives": ["a.zip", "b.zip"],
            "index_files": ["i.tsv"],
        },
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LangSpec(**kwargs)


def test_unknown_yaml_key_names_the_offending_language(tmp_path: Path):
    cfg = tmp_path / "scales"
    cfg.mkdir()
    (cfg / "bad.yaml").write_text("languages:\n  - {code: x, source: openslr, hf_mirror: y}\n")
    with pytest.raises(ValueError, match="hf_mirror"):
        get_preset("bad", configs_dir=tmp_path)


def test_heldout_is_four_openslr_languages(pytestconfig):
    specs = get_heldout(configs_dir=Path(pytestconfig.rootpath) / "configs")
    assert [s.code for s in specs] == ["malayalam", "marathi", "telugu", "gujarati"]
    assert [s.slr for s in specs] == [63, 64, 66, 78]
    assert {s.source for s in specs} == {"openslr"}
    assert {s.license for s in specs} == {"CC-BY-SA-4.0"}
    for s in specs:
        assert len(s.archives) == len(s.index_files) >= 1


def test_heldout_marathi_is_female_only_with_a_single_index(pytestconfig):
    specs = get_heldout(configs_dir=Path(pytestconfig.rootpath) / "configs")
    mr = next(s for s in specs if s.code == "marathi")
    assert mr.archives == ("mr_in_female.zip",)
    assert mr.index_files == ("line_index.tsv",)


SCALES = ["3", "16", "64"]

# Common Voice locales whose writing system does not separate words with spaces.
# Small and hard-coded on purpose: it is the independent check on what the
# presets declare, so deriving it the way the presets were built would make it
# agree with them by construction.
NO_SPACE_LOCALES = frozenset({"ja", "th", "zh-CN", "zh-HK", "zh-TW", "yue", "nan-tw", "bo"})


@pytest.mark.parametrize("scale", ["3", "16"])
def test_the_small_presets_are_common_voice_only(pytestconfig, scale):
    specs = get_preset(scale, configs_dir=Path(pytestconfig.rootpath) / "configs")
    assert specs and {s.source for s in specs} == {"commonvoice"}


def test_the_large_preset_draws_on_both_sources(pytestconfig):
    """Common Voice alone cannot reach 64 languages at 50 training hours, so the
    large tier is a mix and every non-Common-Voice entry names the corpus it is
    read from."""
    specs = get_preset("64", configs_dir=Path(pytestconfig.rootpath) / "configs")

    assert {s.source for s in specs} == {"commonvoice", "manifest"}
    assert all(s.corpus for s in specs if s.source == "manifest")


def test_word_boundary_defaults_to_true_and_is_declarable():
    assert _cv().word_boundary is True
    assert (
        LangSpec(code="ja", source="commonvoice", hf_config="ja", word_boundary=False).word_boundary
        is False
    )


@pytest.mark.parametrize("scale", SCALES)
def test_only_languages_written_without_spaces_are_scored_on_characters(pytestconfig, scale):
    """Which metric is primary for a language is a property of its writing
    system, so it is declared beside the language rather than in a set inside
    the evaluation module that no preset author ever sees.

    The small presets have only Japanese. The large one has five: Chinese,
    Cantonese and Thai came in with the wider Common Voice statistic, and
    Tibetan with the corpora.
    """
    specs = get_preset(scale, configs_dir=Path(pytestconfig.rootpath) / "configs")
    unspaced = {s.hf_config for s in specs if not s.word_boundary}

    assert unspaced <= NO_SPACE_LOCALES
    assert "ja" in unspaced


def test_heldout_indic_languages_are_written_with_spaces(pytestconfig):
    specs = get_heldout(configs_dir=Path(pytestconfig.rootpath) / "configs")

    assert all(s.word_boundary for s in specs)


def test_an_unpopulated_preset_fails_instead_of_running_on_nothing(tmp_path):
    """An empty preset trains on no languages, evaluates nothing, and writes a
    results.json that a reader cannot tell from a completed run. Failing by name
    is the only way that is distinguishable from a run that found no data.

    Every preset in the tree is populated now, so the guard is exercised against
    a written-out empty one rather than against `configs/scales/64.yaml`.
    """
    (tmp_path / "scales").mkdir()
    (tmp_path / "scales" / "64.yaml").write_text("languages: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not populated"):
        get_preset("64", configs_dir=tmp_path)


# --------------------------------------------------------------------------- #
# How the presets relate to each other
# --------------------------------------------------------------------------- #


def _codes(pytestconfig, scale):
    return [
        s.hf_config for s in get_preset(scale, configs_dir=Path(pytestconfig.rootpath) / "configs")
    ]


def test_each_preset_is_a_subset_of_the_next_one_up(pytestconfig):
    """A scale tier is meant to answer "what does adding languages do?".

    That reading only holds if the larger tier contains the smaller one. If the
    16 and the 64 shared only part of their membership, a difference between
    their results would be a difference of *which* languages as much as of how
    many, and the scaling claim would not be about scale.
    """
    three, sixteen, large = (set(_codes(pytestconfig, s)) for s in SCALES)

    assert three <= sixteen, sorted(three - sixteen)
    assert sixteen <= large, sorted(sixteen - large)


@pytest.mark.parametrize("scale", SCALES)
def test_no_preset_lists_a_language_twice(pytestconfig, scale):
    """A duplicate would silently double that language's weight in the mix."""
    codes = _codes(pytestconfig, scale)

    assert len(codes) == len(set(codes)), sorted({c for c in codes if codes.count(c) > 1})


@pytest.mark.parametrize("scale", SCALES)
def test_word_boundary_agrees_with_the_writing_system(pytestconfig, scale):
    """`word_boundary` decides whether word error rate is reported as primary.

    It is declared per language in the preset, where a preset author will see
    it, so it can be declared wrong. This is the check against that.
    """
    for spec in get_preset(scale, configs_dir=Path(pytestconfig.rootpath) / "configs"):
        expected = spec.hf_config not in NO_SPACE_LOCALES
        assert spec.word_boundary is expected, spec.hf_config


def test_the_large_preset_is_sixty_four_languages(pytestconfig):
    """The tier is named for its size and now has it: 46 Common Voice languages
    and 18 from other public corpora. Common Voice alone reaches 44 at the 50
    hour rule, which is why the other corpora exist."""
    codes = _codes(pytestconfig, "64")

    assert len(codes) == 64
    assert len(set(codes)) == 64


def _included_in_evidence(pytestconfig) -> set[str]:
    """The locales `docs/languages.md` marks as selected."""
    text = (Path(pytestconfig.rootpath) / "docs" / "languages.md").read_text(encoding="utf-8")
    return {
        line.split("|")[1].strip()
        for line in text.splitlines()
        if line.startswith("| ") and "| yes |" in line
    }


def test_the_preset_is_exactly_what_the_evidence_table_marks_included(pytestconfig):
    """The table is the published reason each language is in the preset.

    Editing one without the other would leave a preset whose membership no
    longer matches the evidence given for it, which is the failure this catches.
    """
    assert _included_in_evidence(pytestconfig) == set(_codes(pytestconfig, "64"))


def test_a_language_with_no_policy_yet_is_listed_in_the_preset_as_pending(pytestconfig):
    """Thirty-two entries carry no `normalizer:` while their policies land
    elsewhere. The header names them, so a reader of the preset does not have to
    diff it against anything to find out which."""
    configs = Path(pytestconfig.rootpath) / "configs"
    specs = get_preset("64", configs_dir=configs)
    text = (configs / "scales" / "64.yaml").read_text(encoding="utf-8")

    named = {
        line.removeprefix("#   ").strip() for line in text.splitlines() if line.startswith("#   ")
    }
    assert named == {s.hf_config for s in specs if s.normalizer is None}
