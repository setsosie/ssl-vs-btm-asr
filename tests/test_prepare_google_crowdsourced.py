"""The one preparer implemented here: the five Google crowdsourced OpenSLR sets.

The shards are built in the test and served by a fake urlopen, so this exercises
the whole path — fetch, verify, extract, index, manifest — without the network
and without the 15 GB the real corpora would take.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from svb.data.manifest_local import ManifestLocal
from svb.data.registry import LangSpec

#: Two shards holding disjoint utterances by three speakers, plus the index
#: that repeats identically in every shard, which is how the real sets ship.
SHARD_FILES = {
    "0": ["spk1_u1", "spk1_u2", "spk2_u1"],
    "1": ["spk2_u2", "spk3_u1", "spk3_u2"],
}
ALL_UTTS = [utt for utts in SHARD_FILES.values() for utt in utts]


def _index_bytes(utts: list[str]) -> bytes:
    body = "".join(f"{utt}\t{utt.split('_')[0]}\ttranscript for {utt}\n" for utt in utts)
    return body.encode("utf-8")


def _shard(utts: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("utt_spk_text.tsv", _index_bytes(ALL_UTTS))
        for utt in utts:
            archive.writestr(f"{utt}.flac", b"not really flac, never decoded here")
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
def served(prepare_google_crowdsourced: ModuleType, monkeypatch) -> dict[str, bytes]:
    """Serve two synthetic shards over a fake urlopen; return what was served."""
    import corpus_fetch

    bodies = {shard: _shard(utts) for shard, utts in SHARD_FILES.items()}

    def fake_urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        shard = url.rsplit("_", 1)[1].removesuffix(".zip")
        return _Response(bodies[shard])

    monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)
    return bodies


def _prepare(module: ModuleType, root: Path, lang: str = "jv") -> Path:
    return module.prepare(module.BY_LANG[lang], root, shards="01")


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_every_utterance_reaches_the_manifest_with_its_speaker(
    prepare_google_crowdsourced: ModuleType, served: dict[str, bytes], tmp_path: Path
) -> None:
    """The third column of utt_spk_text.tsv is the whole reason these five are
    easy: the speaker is given rather than parsed out of the file id."""
    rows = _rows(_prepare(prepare_google_crowdsourced, tmp_path))

    assert [row["utt_id"] for row in rows] == ALL_UTTS
    assert {row["speaker"] for row in rows} == {"spk1", "spk2", "spk3"}
    assert rows[0]["text"] == "transcript for spk1_u1"


def test_the_split_column_is_left_empty_because_none_of_the_five_ships_one(
    prepare_google_crowdsourced: ModuleType, served: dict[str, bytes], tmp_path: Path
) -> None:
    rows = _rows(_prepare(prepare_google_crowdsourced, tmp_path))

    assert {row["split"] for row in rows} == {""}


def test_the_manifest_paths_resolve_to_the_extracted_audio(
    prepare_google_crowdsourced: ModuleType, served: dict[str, bytes], tmp_path: Path
) -> None:
    manifest = _prepare(prepare_google_crowdsourced, tmp_path)

    for row in _rows(manifest):
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_an_index_row_with_no_audio_is_dropped_rather_than_written(
    prepare_google_crowdsourced: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    """The index lists every utterance in the corpus, so preparing one shard
    leaves most rows without audio. Writing them anyway would fail the loader
    hours into a run instead of here."""
    import corpus_fetch

    body = _shard(SHARD_FILES["0"])
    monkeypatch.setattr(corpus_fetch, "urlopen", lambda request, timeout=None: _Response(body))

    manifest = prepare_google_crowdsourced.prepare(
        prepare_google_crowdsourced.BY_LANG["jv"], tmp_path, shards="0"
    )

    assert [row["utt_id"] for row in _rows(manifest)] == SHARD_FILES["0"]


# --------------------------------------------------------------------------- #
# It feeds the loader
# --------------------------------------------------------------------------- #


def test_what_the_preparer_writes_is_what_the_loader_reads(
    prepare_google_crowdsourced: ModuleType, served: dict[str, bytes], tmp_path: Path, monkeypatch
) -> None:
    """The contract between the two halves, checked end to end rather than
    assumed from two files that happen to name the same columns."""
    _prepare(prepare_google_crowdsourced, tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(code="jv", source="manifest", corpus="slr35_javanese", hf_config="jv")

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "speaker"
    assert sum(len(part) for part in parts.values()) == len(ALL_UTTS)
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert not speakers["train"] & speakers["test"]


# --------------------------------------------------------------------------- #
# Provenance and refusals
# --------------------------------------------------------------------------- #


def test_the_fetch_record_digests_every_shard(
    prepare_google_crowdsourced: ModuleType, served: dict[str, bytes], tmp_path: Path
) -> None:
    """No OpenSLR resource publishes a checksum, so the digest is recorded for
    later comparison rather than checked against anything."""
    manifest = _prepare(prepare_google_crowdsourced, tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["corpus"] == "slr35_javanese"
    assert [a["name"] for a in payload["archives"]] == [
        "asr_javanese_0.zip",
        "asr_javanese_1.zip",
    ]
    assert all(len(a["sha256"]) == 64 for a in payload["archives"])


def test_a_corrupt_shard_is_deleted_and_reported_rather_than_extracted(
    prepare_google_crowdsourced: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    import corpus_fetch

    monkeypatch.setattr(
        corpus_fetch, "urlopen", lambda request, timeout=None: _Response(b"not a zip at all")
    )

    with pytest.raises(OSError, match="re-run"):
        prepare_google_crowdsourced.prepare(
            prepare_google_crowdsourced.BY_LANG["jv"], tmp_path, shards="0"
        )


def test_two_shards_claiming_one_utterance_are_refused(
    prepare_google_crowdsourced: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    """Sixteen shards unpack into one directory. An overwrite there leaves the
    counts and the manifest consistent and the audio wrong."""
    import corpus_fetch

    body = _shard(SHARD_FILES["0"])
    monkeypatch.setattr(corpus_fetch, "urlopen", lambda request, timeout=None: _Response(body))

    with pytest.raises(ValueError, match="disjoint file ids"):
        prepare_google_crowdsourced.prepare(
            prepare_google_crowdsourced.BY_LANG["jv"], tmp_path, shards="01"
        )


def test_the_held_out_transfer_languages_are_not_in_this_family(
    prepare_google_crowdsourced: ModuleType,
) -> None:
    """Malayalam, Marathi, Telugu and Gujarati come from the same crowdsourced
    programme and ship in a neighbouring format, which is exactly why this is
    worth asserting rather than reading off the list."""
    from svb.data.corpora import HELD_OUT_LANGUAGES

    assert not set(prepare_google_crowdsourced.LANGUAGES) & HELD_OUT_LANGUAGES


def test_the_archive_urls_match_the_published_layout(
    prepare_google_crowdsourced: ModuleType,
) -> None:
    corpus = prepare_google_crowdsourced.BY_LANG["si"]

    assert (
        prepare_google_crowdsourced.archive_url(corpus, "f")
        == "https://www.openslr.org/resources/52/asr_sinhala_f.zip"
    )
