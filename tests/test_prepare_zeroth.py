"""Zeroth-Korean preparer.

The archive is built in the test and served by a fake urlopen, reproducing the
layout read off the real tarball's first members: `AUDIO_INFO` at the root, then
`<dataset>/<script>/<speaker>/<speaker>_<script>_<n>.flac` beside one
`<speaker>_<script>.trans.txt`. The FLAC payloads are real STREAMINFO headers
with no audio after them, which is all `flac_duration` reads.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

from svb.data.manifest_local import ManifestLocal
from svb.data.registry import LangSpec
from tests.conftest import _load_script

#: Speakers per shipped dataset, mirroring the real 105-train / 10-test shape at
#: a size a test can hold. Three train speakers is the minimum that lets the
#: derivation stay speaker-disjoint across three splits.
SPEAKERS = {
    "train_data_01": ["106", "107", "108", "110"],
    "test_data_01": ["104", "105"],
}
SCRIPT = "003"
UTTS_PER_SPEAKER = 3


def _flac(seconds: float, sample_rate: int = 16000) -> bytes:
    """A FLAC file that is nothing but its STREAMINFO header."""
    total = int(seconds * sample_rate)
    packed = (sample_rate << 44) | (0 << 41) | (15 << 36) | total
    streaminfo = bytes(10) + packed.to_bytes(8, "big") + bytes(16)
    return b"fLaC" + bytes([0x80, 0x00, 0x00, 0x22]) + streaminfo


def _archive(
    *,
    speakers: dict[str, list[str]] | None = None,
    drop_audio: tuple[str, ...] = (),
    audio_info: str | None = None,
    seconds: float = 2.0,
) -> bytes:
    speakers = SPEAKERS if speakers is None else speakers
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        if audio_info is None:
            body = "SPEAKERID|NAME|SEX|SCRIPTID|DATASET\n"
            for dataset, ids in speakers.items():
                for speaker in ids:
                    body += f"{speaker}|Contributor{speaker}|f|{SCRIPT}|{dataset}\n"
            audio_info = body
        add("AUDIO_INFO", audio_info.encode("utf-8"))

        for dataset, ids in speakers.items():
            for speaker in ids:
                base = f"{dataset}/{SCRIPT}/{speaker}"
                lines = ""
                for n in range(UTTS_PER_SPEAKER):
                    utt = f"{speaker}_{SCRIPT}_{n:04d}"
                    lines += f"{utt} 이것은 {speaker} 화자의 문장 {n}\n"
                    if utt not in drop_audio:
                        add(f"{base}/{utt}.flac", _flac(seconds))
                add(f"{base}/{speaker}_{SCRIPT}.trans.txt", lines.encode("utf-8"))
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
def prepare_zeroth(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "prepare_zeroth")


@pytest.fixture
def serve(monkeypatch, prepare_zeroth: ModuleType):
    """Serve a body and point the published-MD5 check at it.

    The real check is against the digest openslr.org publishes for the real
    10 GB archive, which a synthetic one cannot match; a separate test covers
    the mismatch path.
    """

    def install(body: bytes, *, honour_published_md5: bool = False) -> None:
        import corpus_fetch

        monkeypatch.setattr(corpus_fetch, "urlopen", lambda request, timeout=None: _Response(body))
        if not honour_published_md5:
            digest = hashlib.md5(body, usedforsecurity=False).hexdigest()
            monkeypatch.setattr(prepare_zeroth, "PUBLISHED_MD5", digest)

    return install


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# --------------------------------------------------------------------------- #
# The re-derived split
# --------------------------------------------------------------------------- #


def test_the_split_column_is_cleared_so_all_speakers_are_repartitioned(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    """The shipped test is 1.2 h, under the 2 h evaluation bar. Passing it
    through would leave a test split too small to report a number from."""
    serve(_archive())
    rows = _rows(prepare_zeroth.prepare(tmp_path))

    assert {row["split"] for row in rows} == {""}
    assert len(rows) == 6 * UTTS_PER_SPEAKER


def test_the_shipped_test_speakers_are_kept_rather_than_discarded(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    """They enter the same pool as the train speakers; nothing is thrown away."""
    serve(_archive())
    speakers = {row["speaker"] for row in _rows(prepare_zeroth.prepare(tmp_path))}

    assert speakers == set(SPEAKERS["train_data_01"]) | set(SPEAKERS["test_data_01"])


def test_the_speaker_is_the_numeric_id_that_names_the_directory(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    rows = _rows(prepare_zeroth.prepare(tmp_path))

    for row in rows:
        assert row["utt_id"].startswith(row["speaker"] + "_")


def test_contributor_names_never_reach_the_manifest_or_the_fetch_record(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    """AUDIO_INFO carries a NAME column of real personal names. It is read only
    to find the header and must not be carried anywhere."""
    serve(_archive())
    manifest = prepare_zeroth.prepare(tmp_path)

    written = manifest.read_text(encoding="utf-8")
    record = (manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8")
    for speaker in SPEAKERS["train_data_01"]:
        assert f"Contributor{speaker}" not in written
        assert f"Contributor{speaker}" not in record


def test_audio_info_is_read_for_the_shipped_dataset_counts_only(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    manifest = prepare_zeroth.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["shipped_speakers"] == {"test_data_01": 2, "train_data_01": 4}
    assert payload["speakers"] == 6


def test_audio_info_without_the_expected_columns_is_refused(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(audio_info="ID|NAME\n1|x\n"))

    with pytest.raises(ValueError, match="SPEAKERID"):
        prepare_zeroth.prepare(tmp_path)


# --------------------------------------------------------------------------- #
# Transcripts and durations
# --------------------------------------------------------------------------- #


def test_the_transcript_is_the_rest_of_the_trans_txt_line(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    rows = _rows(prepare_zeroth.prepare(tmp_path))

    text = next(row["text"] for row in rows if row["utt_id"] == f"106_{SCRIPT}_0000")
    assert text == "이것은 106 화자의 문장 0"


@pytest.mark.parametrize(("seconds", "rate"), [(2.0, 16000), (0.5, 8000), (12.25, 44100)])
def test_flac_duration_reads_streaminfo_without_decoding(
    prepare_zeroth: ModuleType, tmp_path: Path, seconds: float, rate: int
) -> None:
    path = tmp_path / "x.flac"
    path.write_bytes(_flac(seconds, rate))

    assert prepare_zeroth.flac_duration(path) == pytest.approx(seconds, abs=1e-3)


def test_flac_duration_is_none_for_something_that_is_not_flac(
    prepare_zeroth: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "x.flac"
    path.write_bytes(b"RIFF____WAVEfmt " + bytes(40))

    assert prepare_zeroth.flac_duration(path) is None


def test_the_derived_split_hours_are_measured_and_recorded(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    """After re-deriving there is no published figure to quote, so the realised
    hours are the only record of what the run trained and scored on."""
    serve(_archive(seconds=2.0))
    manifest = prepare_zeroth.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    hours = payload["derived_hours"]
    assert set(hours) == {"train", "validation", "test"}
    # 18 utterances x 2 s = 36 s total, however the speakers fall out. Each split
    # is rounded to a millihour (3.6 s) before it is recorded, which is precision
    # to spare on a corpus of tens of hours and visible on one of 36 seconds.
    assert sum(hours.values()) == pytest.approx(36 / 3600, abs=2e-3)
    assert payload["published_hours"] == {"train": 51.6, "test": 1.2}
    assert all(value > 0 for value in hours.values())


# --------------------------------------------------------------------------- #
# Integrity, refusals, idempotence
# --------------------------------------------------------------------------- #


def test_the_published_md5_is_checked_and_a_mismatch_deletes_the_archive(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    """SLR40 is the one corpus in this family that publishes a checksum, so it
    is the one that can be verified against something rather than merely
    recorded."""
    serve(_archive(), honour_published_md5=True)

    with pytest.raises(OSError, match="does not match"):
        prepare_zeroth.prepare(tmp_path)
    assert not list((tmp_path / prepare_zeroth.CORPUS_IDS[0] / "ko" / ".archives").glob("*.gz"))


def test_the_fetch_record_carries_the_md5_and_where_it_came_from(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    manifest = prepare_zeroth.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    archive = payload["archives"][0]
    assert len(archive["md5"]) == 32
    assert archive["md5_source"].endswith("/checksum.md5")
    assert len(archive["sha256"]) == 64


def test_an_utterance_without_audio_is_dropped_rather_than_written(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    missing = f"107_{SCRIPT}_0001"
    serve(_archive(drop_audio=(missing,)))
    manifest = prepare_zeroth.prepare(tmp_path)

    assert missing not in {row["utt_id"] for row in _rows(manifest)}
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert payload["missing_audio"] == 1


def test_a_corrupt_archive_is_deleted_and_reported_rather_than_extracted(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    serve(b"not a tarball at all")

    with pytest.raises(OSError, match="re-run"):
        prepare_zeroth.prepare(tmp_path)


def test_preparing_twice_gives_the_same_manifest(
    prepare_zeroth: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    first = prepare_zeroth.prepare(tmp_path).read_text(encoding="utf-8")
    serve(_archive())
    second = prepare_zeroth.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second


# --------------------------------------------------------------------------- #
# It feeds the loader
# --------------------------------------------------------------------------- #


def test_the_loader_derives_a_speaker_disjoint_split_from_what_is_written(
    prepare_zeroth: ModuleType, serve, tmp_path: Path, monkeypatch
) -> None:
    serve(_archive())
    prepare_zeroth.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(
        code="ko", source="manifest", corpus=prepare_zeroth.CORPUS_IDS[0], hf_config="ko"
    )

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "speaker"
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert all(speakers.values())
    assert not speakers["train"] & speakers["test"]
    assert not speakers["train"] & speakers["validation"]
    assert not speakers["validation"] & speakers["test"]
