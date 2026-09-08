"""datasets.py is a dispatcher over local corpora — no HF `datasets` anywhere."""

import ast
import inspect

import pytest

from svb.data import datasets
from svb.data.openslr_local import OpenSLRLocal
from svb.data.registry import LangSpec

SLR = LangSpec(
    code="malayalam",
    source="openslr",
    slr=63,
    archives=("ml_in_female.zip", "ml_in_male.zip"),
    index_files=("line_index_female.tsv", "line_index_male.tsv"),
    license="CC-BY-SA-4.0",
)


def test_the_dead_hf_path_is_gone():
    tree = ast.parse(inspect.getsource(datasets))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert "datasets" not in imported
    assert "huggingface_hub" not in imported
    assert not hasattr(datasets, "AudioTextDataset")
    assert not hasattr(datasets, "_resolve_split")


def test_openslr_dispatches_to_the_local_loader(slr_root, monkeypatch):
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))
    ds = datasets.load_language(SLR, "validation")
    assert isinstance(ds, OpenSLRLocal)
    assert len(ds) > 0


def test_load_texts_dispatches_without_decoding_audio(slr_root, monkeypatch):
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))
    for wav in (slr_root / "SLR63").glob("*.wav"):
        wav.unlink()
    assert datasets.load_texts(SLR, "train")


def test_max_samples_still_truncates_through_the_dispatcher(slr_root, monkeypatch):
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))
    ds = datasets.load_language(SLR, "train", 32)
    assert ds[0][0].shape[0] == 32


def test_commonvoice_dispatches_to_the_local_loader(tmp_path, monkeypatch):
    lang = tmp_path / "hi"
    (lang / "clips").mkdir(parents=True)
    (lang / "train.tsv").write_text("path\tsentence\na.mp3\tnamaste\n", encoding="utf-8")
    monkeypatch.setenv("CV_ROOT", str(tmp_path))

    spec = LangSpec(code="hi", source="commonvoice", hf_dataset="common_voice_25", hf_config="hi")
    assert datasets.load_texts(spec, "train") == ["namaste"]
    assert len(datasets.load_language(spec, "train")) == 1


def test_unknown_source_is_rejected_at_dispatch():
    spec = LangSpec.__new__(LangSpec)  # bypass __post_init__ to reach the dispatch guard
    object.__setattr__(spec, "source", "kaldi")
    object.__setattr__(spec, "code", "x")
    with pytest.raises(ValueError, match="kaldi"):
        datasets.load_language(spec, "train")
