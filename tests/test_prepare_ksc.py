"""Kazakh Speech Corpus preparer.

The archive is built in the test and served by a fake urlopen, so this exercises
the whole path — fetch, verify, extract, Meta, manifest — without the network and
without the 19 GB the real corpus would take. The synthetic tree reproduces the
layout the corpus author's own ESPnet recipe reads: `Audios_flac/<uttID>.flac`,
`Transcriptions/<uttID>.txt` and one `Meta/<split>.csv` per split.
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

ROOT = "ISSAI_KSC_335RS_v1.1_flac"

#: Two speakers' worth of utterances across the three shipped Meta files. The
#: opaque hex-ish ids are the shape the real corpus uses; nothing is parsed out
#: of them, which is the point.
META = {
    "train": ["5f318346ebd7f", "5f591bb5a80c8", "5f2cf82209eb8"],
    "dev": ["5f1ba9feb3124"],
    "test": ["5f5af8479c85c", "5efc5e726df42"],
}
ALL_UTTS = [utt for utts in META.values() for utt in utts]


@pytest.fixture
def prepare_ksc(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "prepare_ksc")


def _archive(
    *,
    meta: dict[str, list[str]] | None = None,
    header: str = "uttID duration",
    delimiter: str = " ",
    drop_audio: tuple[str, ...] = (),
    drop_transcript: tuple[str, ...] = (),
    extra_meta: str | None = None,
) -> bytes:
    meta = META if meta is None else meta
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        for split, utts in meta.items():
            body = header + "\n"
            for utt in utts:
                body += delimiter.join([utt, "3.5"]) + "\n"
            add(f"{ROOT}/Meta/{split}.csv", body.encode("utf-8"))
        for utt in [u for utts in meta.values() for u in utts]:
            if utt not in drop_audio:
                add(f"{ROOT}/Audios_flac/{utt}.flac", b"not really flac, never decoded here")
            if utt not in drop_transcript:
                add(f"{ROOT}/Transcriptions/{utt}.txt", f"  сөйлем {utt}  \n".encode())
        if extra_meta is not None:
            add(f"{ROOT}/Meta/{extra_meta}", b"uttID\n")
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
def serve(monkeypatch):
    def install(body: bytes) -> None:
        import corpus_fetch

        monkeypatch.setattr(corpus_fetch, "urlopen", lambda request, timeout=None: _Response(body))

    return install


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_the_shipped_split_is_passed_through_not_derived(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    """The source page says nothing about a split, but Meta/*.csv is one and the
    corpus author's recipe reads it. Deriving over it would throw it away."""
    serve(_archive())
    rows = _rows(prepare_ksc.prepare(tmp_path))

    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["utt_id"])
    assert by_split == {
        "train": set(META["train"]),
        "validation": set(META["dev"]),
        "test": set(META["test"]),
    }


def test_the_corpus_word_dev_becomes_the_loader_word_validation(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    """The loader refuses a manifest carrying `dev`, so the mapping is the whole
    reason this preparer cannot just copy the Meta file stem."""
    serve(_archive())
    splits = {row["split"] for row in _rows(prepare_ksc.prepare(tmp_path))}

    assert "dev" not in splits
    assert splits == {"train", "validation", "test"}


def test_transcripts_come_from_one_file_per_utterance_whitespace_collapsed(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    rows = _rows(prepare_ksc.prepare(tmp_path))

    text = next(row["text"] for row in rows if row["utt_id"] == "5f318346ebd7f")
    assert text == "сөйлем 5f318346ebd7f"


def test_the_speaker_column_is_empty_because_the_corpus_publishes_no_speaker(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    """The reference recipe writes `utt2spk` as `uttID uttID`. Filling this from
    the utterance id would claim a disjointness the corpus never established."""
    serve(_archive())

    assert {row["speaker"] for row in _rows(prepare_ksc.prepare(tmp_path))} == {""}


def test_a_meta_column_actually_named_as_a_speaker_is_used(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    """If a Meta file does carry a speaker, it should reach the manifest — but
    only when it is named as one, never inferred from the shape of the values."""
    serve(_archive(header="uttID speaker_id", delimiter=" "))
    rows = _rows(prepare_ksc.prepare(tmp_path))

    assert {row["speaker"] for row in rows} == {"3.5"}


def test_the_manifest_paths_resolve_to_the_extracted_audio(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    manifest = prepare_ksc.prepare(tmp_path)

    for row in _rows(manifest):
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_preparing_twice_gives_the_same_manifest(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    first = prepare_ksc.prepare(tmp_path).read_text(encoding="utf-8")
    serve(_archive())
    second = prepare_ksc.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second


# --------------------------------------------------------------------------- #
# Reading the Meta files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("uttID duration", " "),
        ("uttID\tduration", "\t"),
        ("uttID,duration", ","),
    ],
)
def test_the_meta_delimiter_is_sniffed_rather_than_assumed(
    prepare_ksc: ModuleType, header: str, expected: str
) -> None:
    """They are named .csv and the reference recipe reads them on whitespace, so
    which one they use is not something to assume."""
    assert prepare_ksc.sniff_delimiter(header) == expected


def test_meta_files_that_are_not_a_known_split_are_refused(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    """A new Meta file would otherwise go unread and silently shrink the corpus."""
    serve(_archive(extra_meta="heldout.csv"))

    with pytest.raises(ValueError, match="unexpected Meta file"):
        prepare_ksc.prepare(tmp_path)


# --------------------------------------------------------------------------- #
# Rows that cannot be honoured
# --------------------------------------------------------------------------- #


def test_an_utterance_without_audio_is_dropped_rather_than_written(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(drop_audio=("5f591bb5a80c8",)))
    manifest = prepare_ksc.prepare(tmp_path)

    assert "5f591bb5a80c8" not in {row["utt_id"] for row in _rows(manifest)}
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert payload["missing_audio"] == 1


def test_an_utterance_without_a_transcript_is_dropped_rather_than_written(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(drop_transcript=("5f2cf82209eb8",)))
    manifest = prepare_ksc.prepare(tmp_path)

    assert "5f2cf82209eb8" not in {row["utt_id"] for row in _rows(manifest)}
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert payload["missing_transcript"] == 1


def test_a_corrupt_archive_is_deleted_and_reported_rather_than_extracted(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    serve(b"not a tarball at all")

    with pytest.raises(OSError, match="re-run"):
        prepare_ksc.prepare(tmp_path)


# --------------------------------------------------------------------------- #
# It feeds the loader
# --------------------------------------------------------------------------- #


def test_what_the_preparer_writes_is_what_the_loader_reads(
    prepare_ksc: ModuleType, serve, tmp_path: Path, monkeypatch
) -> None:
    serve(_archive())
    prepare_ksc.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(code="kk", source="manifest", corpus=prepare_ksc.CORPUS_IDS[0], hf_config="kk")

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "shipped"
    assert sum(len(part) for part in parts.values()) == len(ALL_UTTS)
    assert {row.utt_id for row in parts["test"].rows} == set(META["test"])


def test_the_fetch_record_names_the_columns_the_meta_files_actually_had(
    prepare_ksc: ModuleType, serve, tmp_path: Path
) -> None:
    """The reference recipe reads only the first Meta column, so what the rest
    are is unrecorded anywhere. Writing them down here is how the next reader
    finds out without opening a 19 GB tarball."""
    serve(_archive())
    manifest = prepare_ksc.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["split_source"] == "shipped (Meta/*.csv)"
    assert payload["meta_files"]["train.csv"]["columns"] == ["uttID", "duration"]
    assert payload["meta_files"]["train.csv"]["speaker_column"] is None
    assert payload["meta_files"]["dev.csv"]["split"] == "validation"
    assert len(payload["archives"][0]["sha256"]) == 64
