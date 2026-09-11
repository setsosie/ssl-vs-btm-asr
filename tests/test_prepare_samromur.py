"""Samrómur preparer.

The archive is built in the test and served by a fake urlopen, reproducing the
layout read off the real tgz's first members: `<split>/<speaker>/<speaker>-<audio
id>.flac`, plus a tab-separated metadata file carrying the read prompt. This is
the one corpus in this group whose split is used exactly as published.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

from svb.data.manifest_local import ManifestLocal
from svb.data.registry import LangSpec
from tests.conftest import _load_script

#: `{split directory: {speaker: [audio ids]}}`, shaped like the real corpus:
#: speaker-disjoint across the three subsets.
TREE = {
    "train": {"003040": ["0022916", "0022917"], "001277": ["0009138", "0009139"]},
    "dev": {"002611": ["0019472"]},
    "test": {"003189": ["0024048"]},
}
PROMPTS = {
    "0022916": "Halló heimur",
    "0022917": "Þetta er önnur setning",
    "0009138": "Þriðja setningin hér",
    "0009139": "Fjórða setningin",
    "0019472": "Fimmta setningin",
    "0024048": "Sjötta setningin",
}


def _metadata(
    audio_ids: list[str], *, id_column: str = "audio_id", text_column: str = "normalized_text"
) -> bytes:
    body = f"{id_column}\tspeaker_id\tgender\tage\tduration\t{text_column}\n"
    for audio_id in audio_ids:
        body += f"{audio_id}\tx\tfemale\t30\t2.5\t{PROMPTS[audio_id]}\n"
    return body.encode("utf-8")


def _archive(
    *,
    tree: dict[str, dict[str, list[str]]] | None = None,
    metadata_name: str = "metadata.tsv",
    metadata_ids: list[str] | None = None,
    id_column: str = "audio_id",
    text_column: str = "normalized_text",
    with_metadata: bool = True,
) -> bytes:
    tree = TREE if tree is None else tree
    ids = [i for speakers in tree.values() for group in speakers.values() for i in group]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        for split, speakers in tree.items():
            for speaker, audio_ids in speakers.items():
                for audio_id in audio_ids:
                    add(
                        f"{split}/{speaker}/{speaker}-{audio_id}.flac",
                        b"not really flac, never decoded here",
                    )
        if with_metadata:
            add(
                metadata_name,
                _metadata(
                    ids if metadata_ids is None else metadata_ids,
                    id_column=id_column,
                    text_column=text_column,
                ),
            )
    return buffer.getvalue()


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body, self._offset, self.status = body, 0, 200
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n: int) -> bytes:
        block = self._body[self._offset : self._offset + n]
        self._offset += len(block)
        return block

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        """None, not False: a bool return would imply it can swallow exceptions."""


@pytest.fixture
def prepare_samromur(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "prepare_samromur")


@pytest.fixture
def serve(monkeypatch):
    def install(body: bytes) -> None:
        import corpus_fetch

        monkeypatch.setattr(corpus_fetch, "urlopen", lambda request, timeout=None: _Response(body))

    return install


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# --------------------------------------------------------------------------- #
# The shipped split
# --------------------------------------------------------------------------- #


def test_the_published_split_is_used_exactly_as_it_ships(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    """The page states the three subsets have no speaker overlap. It is the
    cleanest partition of any corpus here and re-deriving would replace it with
    a worse one."""
    serve(_archive())
    rows = _rows(prepare_samromur.prepare(tmp_path))

    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["speaker"])
    assert by_split == {
        "train": {"003040", "001277"},
        "validation": {"002611"},
        "test": {"003189"},
    }


def test_the_corpus_directory_dev_becomes_the_loader_word_validation(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    splits = {row["split"] for row in _rows(prepare_samromur.prepare(tmp_path))}

    assert "dev" not in splits
    assert splits == {"train", "validation", "test"}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("dev/003040/003040-0022916.flac", ("validation", "003040", "0022916")),
        ("train/001277/001277-0009138.flac", ("train", "001277", "0009138")),
        ("test/003189/003189-0024048.flac", ("test", "003189", "0024048")),
        ("other/003189/003189-0024048.flac", None),
        ("003189-0024048.flac", None),
    ],
)
def test_split_speaker_and_id_all_come_out_of_the_path(
    prepare_samromur: ModuleType, tmp_path: Path, path: str, expected
) -> None:
    """The page states the `{speaker_ID}-{utterance_ID}.flac` convention, so
    nothing about the identity of a row is inferred from the metadata."""
    assert prepare_samromur.parse_audio_path(tmp_path / path, tmp_path) == expected


def test_an_incomplete_shipped_partition_is_refused_rather_than_written(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    """resources/112/info.txt calls the tgz "whole corpus (includes dev and train
    sets)" and does not mention test. A manifest missing a split is one the
    loader accepts and scores on nothing."""
    serve(_archive(tree={k: v for k, v in TREE.items() if k != "test"}))

    with pytest.raises(ValueError, match="partition is incomplete"):
        prepare_samromur.prepare(tmp_path)


def test_derive_splits_is_the_documented_way_out_of_an_incomplete_partition(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(tree={k: v for k, v in TREE.items() if k != "test"}))
    rows = _rows(prepare_samromur.prepare(tmp_path, derive_splits=True))

    assert {row["split"] for row in rows} == {""}
    assert {row["speaker"] for row in rows} == {"003040", "001277", "002611"}


# --------------------------------------------------------------------------- #
# Finding the metadata
# --------------------------------------------------------------------------- #


def test_the_transcript_is_the_read_prompt_from_the_metadata(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    rows = _rows(prepare_samromur.prepare(tmp_path))

    text = next(row["text"] for row in rows if row["utt_id"] == "003040-0022916")
    assert text == "Halló heimur"


@pytest.mark.parametrize("name", ["metadata.tsv", "files/metadata_train.tsv", "corpus/meta.tsv"])
def test_the_metadata_file_is_found_by_shape_not_by_name(
    prepare_samromur: ModuleType, serve, tmp_path: Path, name: str
) -> None:
    """The source page says the corpus "is distributed with a metadata file"
    without naming it, so a hard-coded name would be a guess."""
    serve(_archive(metadata_name=name))
    manifest = prepare_samromur.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert Path(name).name in payload["metadata_files"]
    assert len(_rows(manifest)) == 6


@pytest.mark.parametrize(
    ("id_column", "text_column"),
    [("audio_id", "normalized_text"), ("filename", "sentence"), ("utt_id", "text")],
)
def test_several_plausible_column_namings_are_accepted(
    prepare_samromur: ModuleType, serve, tmp_path: Path, id_column: str, text_column: str
) -> None:
    serve(_archive(id_column=id_column, text_column=text_column))
    manifest = prepare_samromur.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["metadata_files"]["metadata.tsv"]["id_column"] == id_column
    assert payload["metadata_files"]["metadata.tsv"]["text_column"] == text_column


def test_an_archive_with_no_recognisable_metadata_is_refused_with_the_columns_tried(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(with_metadata=False))

    with pytest.raises(FileNotFoundError, match="no tab-separated metadata"):
        prepare_samromur.prepare(tmp_path)


def test_audio_the_metadata_does_not_cover_is_dropped_rather_than_written(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(metadata_ids=[i for i in PROMPTS if i != "0022917"]))
    manifest = prepare_samromur.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert "003040-0022917" not in {row["utt_id"] for row in _rows(manifest)}
    assert payload["missing_transcript"] == 1


# --------------------------------------------------------------------------- #
# Refusals, idempotence, and the loader
# --------------------------------------------------------------------------- #


def test_the_manifest_paths_resolve_to_the_extracted_audio(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    manifest = prepare_samromur.prepare(tmp_path)

    for row in _rows(manifest):
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_a_corrupt_archive_is_deleted_and_reported_rather_than_extracted(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(b"not a tarball at all")

    with pytest.raises(OSError, match="re-run"):
        prepare_samromur.prepare(tmp_path)


def test_preparing_twice_gives_the_same_manifest(
    prepare_samromur: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    first = prepare_samromur.prepare(tmp_path).read_text(encoding="utf-8")
    serve(_archive())
    second = prepare_samromur.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second


def test_the_loader_reports_the_split_as_shipped_rather_than_derived(
    prepare_samromur: ModuleType, serve, tmp_path: Path, monkeypatch
) -> None:
    serve(_archive())
    prepare_samromur.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(
        code="is", source="manifest", corpus=prepare_samromur.CORPUS_IDS[0], hf_config="is"
    )

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "shipped"
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert not speakers["train"] & speakers["test"]
    assert not speakers["train"] & speakers["validation"]
