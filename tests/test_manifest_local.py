"""The manifest layer: one reader for every corpus that is not Common Voice.

A preparer turns a corpus into `$CORPORA_ROOT/<corpus>/<lang>/manifest.tsv` and
this reads it. The point is that the fourteen new corpora need fourteen
preparers but only one loader, and only one place where a split can be got
wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from svb.data.manifest_local import ManifestLocal, load_manifest_texts
from svb.data.registry import LangSpec
from tests.conftest import ManifestRow

SPEC = LangSpec(code="xx", source="manifest", corpus="demo_corpus", hf_config="xx")


def _shipped(n_train: int = 4, n_dev: int = 2, n_test: int = 2) -> list[ManifestRow]:
    rows = []
    for split, count in (("train", n_train), ("validation", n_dev), ("test", n_test)):
        for i in range(count):
            key = f"{split}{i}"
            rows.append(ManifestRow(key, f"audio/{key}.wav", f"text {key}", f"spk_{split}", split))
    return rows


def _unsplit(n_speakers: int = 6, per_speaker: int = 3) -> list[ManifestRow]:
    rows = []
    for s in range(n_speakers):
        for i in range(per_speaker):
            key = f"s{s}u{i}"
            rows.append(ManifestRow(key, f"audio/{key}.wav", f"text {key}", f"spk{s}", ""))
    return rows


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def test_a_shipped_split_is_used_as_shipped(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    """Samrómur states its split has no speaker overlap and ParlaSpeech ships
    one too. Re-deriving over the top of that would throw away better work than
    this repository can do."""
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(_shipped())))

    dataset = ManifestLocal(SPEC, "train")

    assert len(dataset) == 4
    assert dataset.split_policy == "shipped"
    assert len(ManifestLocal(SPEC, "validation")) == 2
    assert len(ManifestLocal(SPEC, "test")) == 2


def test_an_absent_split_is_derived_speaker_disjointly(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(_unsplit())))

    parts = {split: ManifestLocal(SPEC, split) for split in ("train", "validation", "test")}

    assert parts["train"].split_policy == "speaker"
    assert all(len(part) for part in parts.values())
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert not speakers["train"] & speakers["test"]
    assert not speakers["train"] & speakers["validation"]


def test_a_row_with_no_speaker_drops_the_whole_corpus_to_utterance_level(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    """One missing speaker is enough. A partition that is speaker-disjoint for
    most rows is not speaker-disjoint, and reporting it as such would be the
    kind of claim nobody could check afterwards."""
    rows = _unsplit()
    rows[0] = rows[0]._replace(speaker="")
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(rows)))

    assert ManifestLocal(SPEC, "train").split_policy == "utterance"


def test_a_partly_split_manifest_is_refused_rather_than_half_derived(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    """Two corpora ship an incomplete partition — Kannada has no dev split and
    Zeroth's test is under the evaluation bar. Filling the gaps by deriving over
    the leftovers would produce a partition that is half shipped and half
    derived, which no write-up could describe. The preparer chooses: use the
    shipped split whole, or clear it and re-derive.
    """
    rows = _shipped()
    rows[0] = rows[0]._replace(split="")
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(rows)))

    with pytest.raises(ValueError, match="some rows"):
        ManifestLocal(SPEC, "train")


def test_the_test_utterance_list_is_digested_so_provenance_can_pin_it(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(_unsplit())))

    first = ManifestLocal(SPEC, "test")
    second = ManifestLocal(SPEC, "test")

    assert len(first.test_files_sha1) == 40
    assert first.test_files_sha1 == second.test_files_sha1


def test_a_shipped_split_is_digested_too(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    """A shipped split still needs pinning: the corpus can be re-released."""
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(_shipped())))

    assert len(ManifestLocal(SPEC, "test").test_files_sha1) == 40


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_an_item_carries_its_language_like_every_other_loader(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(_shipped())))

    wav, text, lang = ManifestLocal(SPEC, "train")[0]

    assert wav.dim() == 1 and wav.shape[0] > 0
    assert text.startswith("text ")
    assert lang == "xx"


def test_audio_is_resampled_to_the_target_rate(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    """Faroese is 48 kHz and one Kazakh option is 22 or 44 kHz, so this path is
    not hypothetical."""
    import wave as wavemod

    root = make_manifest_corpus(_shipped(), frames=48000)
    for path in (root / "demo_corpus" / "xx" / "audio").glob("*.wav"):
        with wavemod.open(str(path), "rb") as reader:
            frames = reader.readframes(reader.getnframes())
        with wavemod.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(48000)
            writer.writeframes(frames)
    monkeypatch.setenv("CORPORA_ROOT", str(root))

    wav, _, _ = ManifestLocal(SPEC, "train")[0]

    assert wav.shape[0] == 16000  # one second at 48 kHz, resampled


def test_the_audio_guard_truncates_like_the_other_loaders(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(_shipped())))

    wav, _, _ = ManifestLocal(SPEC, "train", max_samples=32)[0]

    assert wav.shape[0] == 32


def test_transcripts_read_without_touching_the_audio(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    """Vocabulary construction reads every training transcript. Decoding the
    audio to do it would make a vocab build as expensive as an epoch."""
    root = make_manifest_corpus(_shipped())
    monkeypatch.setenv("CORPORA_ROOT", str(root))
    for wav in (root / "demo_corpus" / "xx" / "audio").glob("*.wav"):
        wav.unlink()

    assert len(load_manifest_texts(SPEC, "train")) == 4


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_missing_root_says_what_to_set_and_what_writes_it(monkeypatch) -> None:
    monkeypatch.delenv("CORPORA_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="CORPORA_ROOT"):
        ManifestLocal(SPEC, "train")


def test_a_missing_manifest_names_the_preparer_that_writes_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="prepare_"):
        ManifestLocal(SPEC, "train")


def test_a_manifest_missing_a_column_names_it(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv(
        "CORPORA_ROOT",
        str(make_manifest_corpus(_shipped(), columns=("utt_id", "path", "text", "speaker"))),
    )

    with pytest.raises(ValueError, match="split"):
        ManifestLocal(SPEC, "train")


def test_an_unknown_split_name_is_rejected(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(_shipped())))

    with pytest.raises(ValueError, match="dev"):
        ManifestLocal(SPEC, "dev")


def test_a_manifest_declaring_a_split_name_the_loader_does_not_know_is_rejected(
    make_manifest_corpus: Callable[..., Path], monkeypatch
) -> None:
    """`dev` is the Common Voice file name, and a preparer author will write it
    out of habit. Silently dropping those rows would shrink the corpus."""
    rows = _shipped()
    rows = [r._replace(split="dev") if r.split == "validation" else r for r in rows]
    monkeypatch.setenv("CORPORA_ROOT", str(make_manifest_corpus(rows)))

    with pytest.raises(ValueError, match="dev"):
        ManifestLocal(SPEC, "train")


def test_a_spec_without_a_corpus_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="corpus"):
        LangSpec(code="xx", source="manifest")
