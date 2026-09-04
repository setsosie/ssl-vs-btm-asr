"""ParlaSpeech-HR: a METS record, a JSONL index, and a gzip stream in three slices.

The archive is built in the test and cut into slices the same way the release is,
so the part most likely to be got wrong — that ``.tgz.0/.1/.2`` are pieces of one
gzip stream rather than three archives — is exercised rather than assumed. The
METS document is the committed one, with only the checksums and sizes rewritten
to match the synthetic payloads, so the parser meets the real namespaces and the
real nested ``<mets:Local><mets:file>`` element.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from svb.data.manifest_local import ManifestLocal
from svb.data.registry import LangSpec
from tests.conftest import _load_script

JSONL = "ParlaSpeech-HR.v1.0.jsonl"
SLICES = ("ParlaSpeech-HR.flac.tgz.0", "ParlaSpeech-HR.flac.tgz.1", "ParlaSpeech-HR.flac.tgz.2")
LICENCE = "Creative Commons - Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"
METS_URL = "https://www.clarin.si/repository/xmlui/metadata/handle/11356/1494/mets.xml"


@pytest.fixture
def prepare_parlaspeech(pytestconfig: pytest.Config) -> ModuleType:
    """Loaded here rather than from a conftest fixture: several preparers are
    being written on parallel branches, and a shared fixture block is the one
    file every one of them would have to edit."""
    return _load_script(Path(pytestconfig.rootpath), "prepare_parlaspeech")


# --------------------------------------------------------------------------- #
# A fake CLARIN.SI
# --------------------------------------------------------------------------- #


def _flac(seconds: float, rate: int = 16000) -> bytes:
    """A FLAC header with a real STREAMINFO block and no audio after it.

    Only the metadata is ever read here — the preparer takes the sample rate out
    of STREAMINFO and never decodes — so the frames would be dead weight.
    """
    samples = int(seconds * rate)
    packed = (rate << 44) | (0 << 41) | (15 << 36) | samples
    streaminfo = (
        (4096).to_bytes(2, "big")  # min block size
        + (4096).to_bytes(2, "big")  # max block size
        + (0).to_bytes(3, "big")  # min frame size
        + (0).to_bytes(3, "big")  # max frame size
        + packed.to_bytes(8, "big")
        + bytes(16)  # md5 of the unencoded audio
    )
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + streaminfo


def _records(fixtures_dir: Path) -> list[dict[str, Any]]:
    path = fixtures_dir / "parlaspeech" / JSONL
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _archive(records: list[dict[str, Any]], drop: tuple[str, ...] = ()) -> bytes:
    """One gzip-compressed tar, flat, exactly as the release ships it."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for record in records:
            name = record["path"]
            if name in drop:
                continue
            body = _flac(float(record["end"]) - float(record["start"]))
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return gzip.compress(raw.getvalue())


def _slice(body: bytes, parts: int = 3) -> list[bytes]:
    """Cut a byte stream into `parts`, the way `split -b` cuts the real one."""
    step = len(body) // parts + 1
    return [body[i : i + step] for i in range(0, len(body), step)][:parts] or [body]


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body, self._offset, self.status = body, 0, 200
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n: int = -1) -> bytes:
        block = self._body[self._offset :] if n < 0 else self._body[self._offset : self._offset + n]
        self._offset += len(block)
        return block

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        """None, not False: a bool return would imply it can swallow exceptions."""


@dataclass
class _Source:
    fixtures_dir: Path
    licence: str = LICENCE
    drop_audio: tuple[str, ...] = ()
    corrupt_slice: str | None = None
    served: list[str] = field(default_factory=list)

    def payloads(self) -> dict[str, bytes]:
        records = _records(self.fixtures_dir)
        index = self.fixtures_dir.joinpath("parlaspeech", JSONL).read_bytes()
        pieces = _slice(_archive(records, self.drop_audio))
        bodies = dict(zip(SLICES, pieces, strict=True))
        bodies[JSONL] = index
        payloads: dict[str, bytes] = {}
        checksums: dict[str, tuple[str, int]] = {}
        for name, body in bodies.items():
            checksums[name] = (hashlib.md5(body, usedforsecurity=False).hexdigest(), len(body))
        if self.corrupt_slice:
            checksums[self.corrupt_slice] = ("0" * 32, len(bodies[self.corrupt_slice]))

        mets = self._mets(checksums)
        payloads[METS_URL] = mets
        for element in ElementTree.fromstring(mets).iter():
            if not element.tag.endswith("}FLocat"):
                continue
            href = next(v for k, v in element.attrib.items() if k.endswith("href"))
            url = "https://www.clarin.si" + href
            name = href.split("?", 1)[0].rsplit("/", 1)[-1]
            if name in bodies:
                payloads[url] = bodies[name]
        return payloads

    def _mets(self, checksums: dict[str, tuple[str, int]]) -> bytes:
        """The committed record with the synthetic checksums and sizes patched in."""
        root = ElementTree.parse(self.fixtures_dir / "parlaspeech" / "mets.xml").getroot()
        for element in root.iter():
            rights = element.tag.endswith("}field") and element.get("element") == "rights"
            if rights and element.get("qualifier") is None:
                element.text = self.licence
            if not element.tag.endswith("}file"):
                continue
            for child in element:
                if not child.tag.endswith("}FLocat"):
                    continue
                href = next(v for k, v in child.attrib.items() if k.endswith("href"))
                name = href.split("?", 1)[0].rsplit("/", 1)[-1]
                if name in checksums:
                    digest, size = checksums[name]
                    element.set("CHECKSUM", digest)
                    element.set("SIZE", str(size))
        return ElementTree.tostring(root, encoding="utf-8")

    def install(self, module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        import corpus_fetch

        payloads = self.payloads()

        def fake_urlopen(request: Any, timeout: float | None = None) -> _Response:
            url = request.full_url if hasattr(request, "full_url") else str(request)
            self.served.append(url)
            if url not in payloads:
                raise AssertionError(f"the test served nothing for {url}")
            return _Response(payloads[url])

        monkeypatch.setattr(module, "urlopen", fake_urlopen)
        monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)


@pytest.fixture
def source(
    prepare_parlaspeech: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> _Source:
    made = _Source(fixtures_dir)
    made.install(prepare_parlaspeech, monkeypatch)
    return made


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# --------------------------------------------------------------------------- #
# The archive is one stream in three pieces
# --------------------------------------------------------------------------- #


def test_the_three_slices_are_read_back_to_back_as_one_stream(
    prepare_parlaspeech: ModuleType, tmp_path: Path
) -> None:
    """Slice 1 and slice 2 of the real release do not begin with a gzip header,
    so decompressing them separately fails. Only the join is an archive."""
    body = b"".join(bytes([i % 251]) * 997 for i in range(40))
    paths = []
    for index, piece in enumerate(_slice(body)):
        path = tmp_path / f"slice.{index}"
        path.write_bytes(piece)
        paths.append(path)

    reader = prepare_parlaspeech.SlicedReader(paths)
    joined = reader.read()
    reader.close()

    assert joined == body


def test_a_gzip_tar_cut_into_slices_still_extracts(
    prepare_parlaspeech: ModuleType, fixtures_dir: Path, tmp_path: Path
) -> None:
    records = _records(fixtures_dir)
    paths = []
    for index, piece in enumerate(_slice(_archive(records))):
        path = tmp_path / f"ParlaSpeech-HR.flac.tgz.{index}"
        path.write_bytes(piece)
        paths.append(path)

    written = prepare_parlaspeech.extract_audio(paths, tmp_path / "audio")

    assert written == {record["path"] for record in records}


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_the_shipped_split_is_carried_through_with_dev_renamed(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The corpus calls it `dev` and the manifest layer calls it `validation`.
    Writing `dev` through would be refused by the loader as an unknown split."""
    rows = _rows(prepare_parlaspeech.prepare(tmp_path))

    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert [row["speaker"] for row in rows if row["split"] == "validation"] == ["Maras, Gordan"]


def test_a_segment_spanning_more_than_one_speaker_is_dropped(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """22,076 of them in the real release: no speaker, no split. Keeping them
    would put the whole language on an utterance-level split, and the manifest
    layer refuses a partition that is part shipped and part derived anyway."""
    rows = _rows(prepare_parlaspeech.prepare(tmp_path))

    assert "seg.NZR9CXgZNKA_6000.0-6015.0" not in {row["utt_id"] for row in rows}
    assert len(rows) == 5


def test_the_text_is_the_original_transcript_and_not_the_normalised_one(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """`norm_words` comes from an imperfect rule-based normaliser. This
    repository normalizes under its own per-language policy at collate time, and
    folding a second one in first would make Croatian the odd language out."""
    rows = {row["utt_id"]: row for row in _rows(prepare_parlaspeech.prepare(tmp_path))}

    assert rows["seg.qNpeHxO0WzA_1967.6-1985.6"]["text"] == (
        "Poštovani zastupnici, otvaram raspravu."
    )


def test_the_manifest_paths_resolve_to_the_extracted_audio(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    manifest = prepare_parlaspeech.prepare(tmp_path)

    for row in _rows(manifest):
        assert row["path"].startswith("audio/")
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_a_segment_whose_audio_is_missing_is_dropped(
    prepare_parlaspeech: ModuleType,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = "seg.k1z8behXXg_14758.4-14770.4.flac"
    _Source(fixtures_dir, drop_audio=(missing,)).install(prepare_parlaspeech, monkeypatch)

    rows = _rows(prepare_parlaspeech.prepare(tmp_path))

    assert f"audio/{missing}" not in {row["path"] for row in rows}
    assert len(rows) == 4


def test_preparing_twice_writes_the_same_manifest(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    first = prepare_parlaspeech.prepare(tmp_path).read_text(encoding="utf-8")
    second = prepare_parlaspeech.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second


def test_what_the_preparer_writes_is_what_the_loader_reads(
    prepare_parlaspeech: ModuleType,
    source: _Source,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_parlaspeech.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(code="hr", source="manifest", corpus="parlaspeech_hr", hf_config="hr")

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "shipped"
    assert [len(part) for part in parts.values()] == [3, 1, 1]
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert not speakers["train"] & speakers["test"]


# --------------------------------------------------------------------------- #
# What gets measured rather than copied
# --------------------------------------------------------------------------- #


def test_the_hours_per_split_are_measured_from_the_segment_bounds(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The repository publishes dev and test as segment counts, not hours, which
    is exactly the gap `configs/corpora.yaml` records as unverified."""
    manifest = prepare_parlaspeech.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["hours"]["validation"] == round(20.0 / 3600, 2)
    assert payload["hours"]["test"] == round(9.0 / 3600, 2)


def test_the_sample_rate_is_read_out_of_the_flac_rather_than_assumed(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The item page states no sample rate. STREAMINFO does, four fields into the
    first metadata block, and reading it decodes nothing."""
    manifest = prepare_parlaspeech.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["sample_rate_hz"] == 16000


# --------------------------------------------------------------------------- #
# The item record, and refusals
# --------------------------------------------------------------------------- #


def test_the_published_record_lists_five_bitstreams_with_their_checksums(
    prepare_parlaspeech: ModuleType, fixtures_dir: Path
) -> None:
    """Parsed from the committed copy of the real METS document, so a change in
    how CLARIN.SI shapes it shows up here rather than in a failed 116 GB fetch.
    The nested <mets:Local><mets:file> holding the README text has the same local
    name as a bitstream entry and must not be picked up as one."""
    mets = ElementTree.parse(fixtures_dir / "parlaspeech" / "mets.xml").getroot()

    found = prepare_parlaspeech.read_bitstreams(mets)

    assert set(found) == {JSONL, "ParlaSpeech-HR.v1.0.txt", *SLICES}
    assert found[JSONL]["md5"] == "271ef6589623facd86527b1e05b740f4"
    assert found[SLICES[0]]["bytes"] == 52428800000
    assert found[SLICES[2]]["bytes"] == 20321039259
    assert found[SLICES[1]]["url"].startswith("https://www.clarin.si/repository/xmlui/bitstream/")
    assert prepare_parlaspeech.read_licence(mets) == LICENCE


def test_the_published_segment_counts_are_checked_not_just_recorded(
    prepare_parlaspeech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """A silently different corpus is the failure that survives all the way into
    a results table, so the counts the item page states are asserted."""
    with pytest.raises(ValueError, match="the repository publishes"):
        prepare_parlaspeech.prepare(tmp_path, expected_counts=prepare_parlaspeech.PUBLISHED_COUNTS)


def test_the_counts_it_checks_against_are_the_ones_the_item_page_states(
    prepare_parlaspeech: ModuleType,
) -> None:
    assert prepare_parlaspeech.PUBLISHED_COUNTS == {
        "train": 380836,
        "validation": 500,
        "test": 513,
        "unassigned": 22076,
    }


def test_a_licence_that_no_longer_says_share_alike_stops_the_ingest(
    prepare_parlaspeech: ModuleType,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _Source(fixtures_dir, licence="Creative Commons Attribution-NonCommercial 4.0").install(
        prepare_parlaspeech, monkeypatch
    )

    with pytest.raises(ValueError, match="settle the difference"):
        prepare_parlaspeech.prepare(tmp_path)


def test_a_slice_failing_the_published_md5_is_deleted_and_reported(
    prepare_parlaspeech: ModuleType,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _Source(fixtures_dir, corrupt_slice=SLICES[1]).install(prepare_parlaspeech, monkeypatch)

    with pytest.raises(OSError, match="does not match"):
        prepare_parlaspeech.prepare(tmp_path)

    assert not (tmp_path / "parlaspeech_hr" / "hr" / ".archives" / SLICES[1]).exists()
