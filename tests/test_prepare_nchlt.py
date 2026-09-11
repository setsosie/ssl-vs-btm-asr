"""NCHLT: one zip per language, fetched through SADiLaR's DSpace 7 API.

Both halves are faked here — the JSON API and the bitstream — so the whole path
runs without the network and without the 4.7 GB the real archive weighs. The
transcripts are the committed fixtures under ``tests/fixtures/nchlt/``, copied
from the shape of the real ``nchlt_zul.trn.xml``, and the zip is built around
them so the archive and the transcripts cannot drift apart.
"""

from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ElementTree
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from svb.data.manifest_local import ManifestLocal
from svb.data.registry import LangSpec
from tests.conftest import _load_script

ITEM_UUID = "49959109-d0b6-4900-9292-8493d0039206"
BUNDLE_UUID = "3e4a42ad-a2ea-4e68-acf6-c74ceedbbd6d"
API = "https://repo.sadilar.org/server/api"
FIND = f"{API}/pid/find?id=hdl:20.500.12185/275"
BUNDLES = f"{API}/core/items/{ITEM_UUID}/bundles"
BITSTREAMS = f"{API}/core/bundles/{BUNDLE_UUID}/bitstreams"
CONTENT = f"{API}/core/bitstreams/5c4d5c3f-b0af-4c9a-81ed-f62601666020/content"

LICENCE = (
    "Creative Commons Attribution 3.0 Unported License (CC BY 3.0): "
    "http://creativecommons.org/licenses/by/3.0/legalcode"
)
DESCRIPTION = (
    "Orthographically transcribed broadband speech corpus of approximately 56 hours, "
    "including a test suite of 8 speakers."
)


@pytest.fixture
def prepare_nchlt(pytestconfig: pytest.Config) -> ModuleType:
    """Loaded here rather than from a conftest fixture: several preparers are
    being written on parallel branches, and a shared fixture block is the one
    file every one of them would have to edit."""
    return _load_script(Path(pytestconfig.rootpath), "prepare_nchlt")


# --------------------------------------------------------------------------- #
# A fake SADiLaR
# --------------------------------------------------------------------------- #


def _fixture_xml(fixtures_dir: Path, side: str) -> bytes:
    return (fixtures_dir / "nchlt" / f"nchlt_zul.{side}.xml").read_bytes()


def _audio_members(*documents: bytes) -> list[str]:
    """The paths the zip must hold, read out of the transcripts themselves.

    The XML names audio under a ``nchlt_zul/`` component the archive does not
    have; dropping it here is what makes the fixture archive realistic, and it is
    the behaviour the preparer has to reproduce.
    """
    members = []
    for document in documents:
        root = ElementTree.fromstring(document)
        for recording in root.iter("recording"):
            audio = recording.get("audio") or ""
            members.append(audio.split("/", 1)[1])
    return members


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
    """What the fake repository serves, so a test can spoil one piece of it."""

    trn: bytes
    tst: bytes
    licence: str = LICENCE
    md5: str | None = None
    drop_audio: tuple[str, ...] = ()
    #: Nine of the ten real archives wrap everything in `nchlt_<iso>/` and
    #: isiZulu's does not, so both shapes have to be servable.
    wrap: str = ""
    served: list[str] = field(default_factory=list)

    def archive(self) -> bytes:
        members = [m for m in _audio_members(self.trn, self.tst) if m not in self.drop_audio]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"{self.wrap}LICENSE.txt", b"Creative Commons Attribution 3.0 Unported")
            zf.writestr(f"{self.wrap}README.txt", b"README: NCHLT Speech Corpus")
            zf.writestr(f"{self.wrap}transcriptions/nchlt_zul.trn.xml", self.trn)
            zf.writestr(f"{self.wrap}transcriptions/nchlt_zul.tst.xml", self.tst)
            zf.writestr(
                f"{self.wrap}transcriptions/nchlt_zul.trn.dtd", b"<!ELEMENT corpus ( speaker+ )>"
            )
            for member in members:
                zf.writestr(f"{self.wrap}{member}", b"RIFF____WAVEfmt ")
        return buffer.getvalue()

    def payloads(self) -> dict[str, bytes]:
        import hashlib

        body = self.archive()
        digest = self.md5 or hashlib.md5(body, usedforsecurity=False).hexdigest()
        item: dict[str, Any] = {
            "uuid": ITEM_UUID,
            "name": "NCHLT isiZulu Speech Corpus",
            "handle": "20.500.12185/275",
            "metadata": {
                "dc.rights.license": [{"value": self.licence}],
                "dc.description": [{"value": DESCRIPTION}],
                "dc.language.iso": [{"value": "zul"}],
            },
            "_links": {"bundles": {"href": BUNDLES}},
        }
        bundles = {
            "_embedded": {
                "bundles": [
                    {"name": "THUMBNAIL", "_links": {"bitstreams": {"href": "unused"}}},
                    {"name": "ORIGINAL", "_links": {"bitstreams": {"href": BITSTREAMS}}},
                ]
            }
        }
        bitstreams = {
            "_embedded": {
                "bitstreams": [
                    {
                        "name": "nchlt.speech.corpus.zul.zip",
                        "sizeBytes": len(body),
                        "checkSum": {"checkSumAlgorithm": "MD5", "value": digest},
                        "_links": {"content": {"href": CONTENT}},
                    }
                ]
            }
        }
        as_json = {FIND: item, BUNDLES: bundles, BITSTREAMS: bitstreams}
        payloads = {url: json.dumps(doc).encode("utf-8") for url, doc in as_json.items()}
        payloads[CONTENT] = body
        return payloads

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
    prepare_nchlt: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> _Source:
    made = _Source(trn=_fixture_xml(fixtures_dir, "trn"), tst=_fixture_xml(fixtures_dir, "tst"))
    made.install(prepare_nchlt, monkeypatch)
    return made


def _prepare(module: ModuleType, root: Path) -> Path:
    return module.prepare(module.BY_LANG["zu"], root)


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _by_split(manifest: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _rows(manifest):
        grouped.setdefault(row["split"], []).append(row)
    return grouped


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #


def test_the_published_test_suite_becomes_the_test_split_untouched(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The .tst transcript is the corpus's own 8-speaker suite. Re-deriving over
    it would throw away the one partition the release actually publishes."""
    grouped = _by_split(_prepare(prepare_nchlt, tmp_path))

    assert {row["speaker"] for row in grouped["test"]} == {"500", "501"}
    assert len(grouped["test"]) == 4


def test_the_dev_split_is_derived_and_shares_no_speaker_with_train_or_test(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """Only .trn and .tst ship, so dev is derived from the training side — and it
    is worth nothing unless it is speaker-disjoint."""
    grouped = _by_split(_prepare(prepare_nchlt, tmp_path))
    speakers = {name: {row["speaker"] for row in rows} for name, rows in grouped.items()}

    assert speakers["validation"]
    assert not speakers["validation"] & speakers["train"]
    assert not speakers["validation"] & speakers["test"]


def test_every_row_carries_a_split_because_a_partial_partition_is_refused(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    rows = _rows(_prepare(prepare_nchlt, tmp_path))

    assert {row["split"] for row in rows} == {"train", "validation", "test"}


def test_a_speaker_in_both_published_suites_is_refused(
    prepare_nchlt: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A test speaker that also appears in training would leak into the score,
    and the derived dev split would be built on top of the leak."""
    trn = _fixture_xml(fixtures_dir, "trn")
    tst = _fixture_xml(fixtures_dir, "tst").replace(b'speaker id="500"', b'speaker id="001"')
    _Source(trn=trn, tst=tst).install(prepare_nchlt, monkeypatch)

    with pytest.raises(ValueError, match=r"both the \.trn and \.tst"):
        _prepare(prepare_nchlt, tmp_path)


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_the_audio_path_drops_the_component_the_archive_does_not_have(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The XML says `nchlt_zul/audio/001/x.wav`; the zip unpacks to `audio/001/`.
    Copying the attribute through would give the loader a path that never
    resolves."""
    manifest = _prepare(prepare_nchlt, tmp_path)
    rows = _rows(manifest)

    assert all(row["path"].startswith("audio/") for row in rows)
    for row in rows:
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_both_archive_layouts_produce_paths_that_resolve(
    prepare_nchlt: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nine of the ten real archives wrap everything in `nchlt_<iso>/` and
    isiZulu's does not — checked by reading the first local file header of all
    ten. Assuming either shape leaves nine languages, or one, with a manifest
    whose every path is wrong, so the root is detected after extraction."""
    _Source(
        trn=_fixture_xml(fixtures_dir, "trn"),
        tst=_fixture_xml(fixtures_dir, "tst"),
        wrap="nchlt_zul/",
    ).install(prepare_nchlt, monkeypatch)

    manifest = _prepare(prepare_nchlt, tmp_path)
    rows = _rows(manifest)

    assert all(row["path"].startswith("nchlt_zul/audio/") for row in rows)
    for row in rows:
        assert (manifest.parent / row["path"]).exists(), row["path"]
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert payload["archive_root"] == "nchlt_zul/"


def test_the_unwrapped_layout_records_the_root_it_found(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    manifest = _prepare(prepare_nchlt, tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["archive_root"] == "."


def test_the_transcript_and_the_speaker_come_out_of_the_xml(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    rows = {row["utt_id"]: row for row in _rows(_prepare(prepare_nchlt, tmp_path))}

    first = rows["nchlt_zul_001m_0001"]
    assert first["text"] == "ulwazi oluthile mayelana"
    assert first["speaker"] == "001"
    assert first["path"] == "audio/001/nchlt_zul_001m_0001.wav"


def test_a_transcript_row_whose_audio_is_missing_is_dropped(
    prepare_nchlt: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Writing it anyway would fail the loader hours into a run instead of here."""
    missing = "audio/001/nchlt_zul_001m_0003.wav"
    _Source(
        trn=_fixture_xml(fixtures_dir, "trn"),
        tst=_fixture_xml(fixtures_dir, "tst"),
        drop_audio=(missing,),
    ).install(prepare_nchlt, monkeypatch)

    rows = _rows(_prepare(prepare_nchlt, tmp_path))

    assert missing not in {row["path"] for row in rows}
    assert "nchlt_zul_001m_0001" in {row["utt_id"] for row in rows}


def test_preparing_twice_writes_the_same_manifest(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """Staging is interrupted and resumed on a cluster, so a second run has to
    land on the same partition rather than a freshly shuffled one."""
    first = _prepare(prepare_nchlt, tmp_path).read_text(encoding="utf-8")
    second = _prepare(prepare_nchlt, tmp_path).read_text(encoding="utf-8")

    assert first == second


# --------------------------------------------------------------------------- #
# It feeds the loader
# --------------------------------------------------------------------------- #


def test_what_the_preparer_writes_is_what_the_loader_reads(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(prepare_nchlt, tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(code="zu", source="manifest", corpus="nchlt_zulu", hf_config="zu")

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "shipped"
    assert sum(len(part) for part in parts.values()) == 24
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert not speakers["train"] & speakers["test"]


# --------------------------------------------------------------------------- #
# Provenance and refusals
# --------------------------------------------------------------------------- #


def test_the_fetch_record_keeps_the_licence_and_the_description_verbatim(
    prepare_nchlt: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """Both are read from the repository at fetch time rather than from
    configs/corpora.yaml, so the record beside the data is the primary one."""
    manifest = _prepare(prepare_nchlt, tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["licence"] == LICENCE
    assert payload["description"] == DESCRIPTION
    assert payload["handle"] == "20.500.12185/275"
    assert payload["counts"]["test"] == 4
    assert len(payload["archives"][0]["sha256"]) == 64


def test_a_licence_that_no_longer_says_cc_by_3_stops_the_ingest(
    prepare_nchlt: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _Source(
        trn=_fixture_xml(fixtures_dir, "trn"),
        tst=_fixture_xml(fixtures_dir, "tst"),
        licence="Creative Commons Attribution-NonCommercial 4.0",
    ).install(prepare_nchlt, monkeypatch)

    with pytest.raises(ValueError, match="settle the difference"):
        _prepare(prepare_nchlt, tmp_path)


def test_an_archive_failing_the_published_md5_is_deleted_and_reported(
    prepare_nchlt: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NCHLT is the only corpus here that publishes a checksum, so it is the only
    one where a digest can be checked rather than merely recorded."""
    _Source(
        trn=_fixture_xml(fixtures_dir, "trn"),
        tst=_fixture_xml(fixtures_dir, "tst"),
        md5="0" * 32,
    ).install(prepare_nchlt, monkeypatch)

    with pytest.raises(OSError, match="does not match"):
        _prepare(prepare_nchlt, tmp_path)

    staging = tmp_path / "nchlt_zulu" / "zu" / ".archives"
    assert not list(staging.glob("*.zip"))


def test_a_corrupt_archive_is_deleted_and_reported_rather_than_extracted(
    prepare_nchlt: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _Source(trn=_fixture_xml(fixtures_dir, "trn"), tst=_fixture_xml(fixtures_dir, "tst"))
    payloads = source.payloads()
    payloads[CONTENT] = b"not a zip at all"
    payloads[BITSTREAMS] = json.dumps(
        {
            "_embedded": {
                "bitstreams": [
                    {
                        "name": "nchlt.speech.corpus.zul.zip",
                        "sizeBytes": len(payloads[CONTENT]),
                        "_links": {"content": {"href": CONTENT}},
                    }
                ]
            }
        }
    ).encode("utf-8")

    import corpus_fetch

    def fake_urlopen(request: Any, timeout: float | None = None) -> _Response:
        return _Response(payloads[request.full_url])

    monkeypatch.setattr(prepare_nchlt, "urlopen", fake_urlopen)
    monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)

    with pytest.raises(OSError, match="re-run"):
        _prepare(prepare_nchlt, tmp_path)


# --------------------------------------------------------------------------- #
# The corpus table
# --------------------------------------------------------------------------- #


def test_all_ten_languages_carry_a_handle_and_an_iso_code(
    prepare_nchlt: ModuleType,
) -> None:
    """The archive names files by ISO 639-3 and the preset names languages by its
    own code; conflating the two puts isiZulu's transcripts in `zul/`."""
    assert len(prepare_nchlt.CORPORA) == 10
    for corpus in prepare_nchlt.CORPORA:
        assert corpus.handle.startswith("20.500.12185/")
        assert len(corpus.iso) == 3
        assert corpus.corpus_id.startswith("nchlt_")


def test_the_default_language_set_is_the_five_the_config_records(
    prepare_nchlt: ModuleType,
) -> None:
    """The other five ship identically but are in no preset, and each is another
    4.7 GB, so asking for them has to be deliberate."""
    assert set(prepare_nchlt.PRESET_LANGUAGES) == {"zu", "xh", "nso", "ts", "ve"}
    assert set(prepare_nchlt.PRESET_LANGUAGES) < set(prepare_nchlt.LANGUAGES)


def test_no_nchlt_language_is_a_held_out_transfer_language(
    prepare_nchlt: ModuleType,
) -> None:
    from svb.data.corpora import HELD_OUT_LANGUAGES

    assert not set(prepare_nchlt.LANGUAGES) & HELD_OUT_LANGUAGES
