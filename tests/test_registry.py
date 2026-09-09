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


@pytest.mark.parametrize("scale", ["3", "16"])
def test_commonvoice_presets_still_parse(pytestconfig, scale):
    specs = get_preset(scale, configs_dir=Path(pytestconfig.rootpath) / "configs")
    assert specs and {s.source for s in specs} == {"commonvoice"}


def test_word_boundary_defaults_to_true_and_is_declarable():
    assert _cv().word_boundary is True
    assert (
        LangSpec(code="ja", source="commonvoice", hf_config="ja", word_boundary=False).word_boundary
        is False
    )


@pytest.mark.parametrize("scale", ["3", "16"])
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
