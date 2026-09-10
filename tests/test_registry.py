"""LangSpec validates both sources; the shipped presets parse."""

import re
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
NO_SPACE_LOCALES = frozenset({"ja", "th", "zh-CN", "zh-HK", "zh-TW", "yue", "nan-tw"})


@pytest.mark.parametrize("scale", SCALES)
def test_commonvoice_presets_still_parse(pytestconfig, scale):
    specs = get_preset(scale, configs_dir=Path(pytestconfig.rootpath) / "configs")
    assert specs and {s.source for s in specs} == {"commonvoice"}


def test_word_boundary_defaults_to_true_and_is_declarable():
    assert _cv().word_boundary is True
    assert (
        LangSpec(code="ja", source="commonvoice", hf_config="ja", word_boundary=False).word_boundary
        is False
    )


@pytest.mark.parametrize("scale", SCALES)
def test_japanese_is_the_only_preset_language_written_without_spaces(pytestconfig, scale):
    """Which metric is primary for a language is a property of its writing
    system, so it is declared beside the language rather than in a set inside
    the evaluation module that no preset author ever sees."""
    specs = get_preset(scale, configs_dir=Path(pytestconfig.rootpath) / "configs")
    unspaced = [s.code for s in specs if not s.word_boundary]

    assert unspaced == ["ja"]


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


def test_the_large_preset_holds_the_languages_that_qualified_not_sixty_four(pytestconfig):
    """The file is named for the tier the design asked for, not for its size.

    Common Voice 25 has 24 locales with 50 hours of training audio. Eight more
    are in the file because the 16-language preset commits to them. Padding to
    64 would mean adding languages with single-digit training hours, which is
    the opposite of what the threshold is for, so the tier is 32 languages and
    `docs/languages.md` shows every locale that was considered.
    """
    assert len(_codes(pytestconfig, "64")) == 32


def test_the_large_preset_is_exactly_what_the_evidence_table_marks_included(pytestconfig):
    """The table is the published reason each language is in the preset.

    Editing one without the other would leave a preset whose membership no
    longer matches the evidence given for it, which is the failure this catches.
    """
    root = Path(pytestconfig.rootpath)
    table = (root / "docs" / "languages.md").read_text(encoding="utf-8")
    included = {
        line.split("|")[1].strip()
        for line in table.splitlines()
        if line.startswith("| ") and "| yes |" in line
    }

    assert included == set(_codes(pytestconfig, "64"))


def test_an_empty_heldout_file_is_also_refused(tmp_path):
    (tmp_path / "scales").mkdir()
    (tmp_path / "scales" / "heldout.yaml").write_text("languages: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not populated"):
        get_heldout(configs_dir=tmp_path)


def test_scale_configs_do_not_cross_reference_documents_that_are_not_here(pytestconfig):
    """A preset comment is public documentation, so its pointers must resolve.

    64.yaml referred a reader to a protocol document that does not exist in this
    repository. Whether a corpus a config names is one this repo can ship is a
    review question, not a testable one; whether a file it points at is present
    is testable, so it is tested.
    """
    root = Path(pytestconfig.rootpath)
    # Only repo-relative pointers: a path with a directory component and a
    # source or documentation suffix. Bare filenames in these configs are
    # corpus members (archives, index files), which live in $OPENSLR_ROOT and
    # are not supposed to be in the tree.
    pointer = re.compile(r"\b(?:[\w.-]+/)+[\w.-]+\.(?:md|py|sh|yaml)\b")

    for path in sorted((root / "configs" / "scales").glob("*.yaml")):
        referenced = pointer.findall(path.read_text(encoding="utf-8"))
        missing = [r for r in referenced if not (root / r).exists()]
        assert not missing, f"{path.name} points at {missing}, which are not in the tree"
