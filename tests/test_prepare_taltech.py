"""TalTech Estonian: long-form WAV plus STM, cut into one file per utterance.

The tar is built in the test with the directory shape walked out of the real
159 GB archive over HTTP Range, and the transcripts are the committed fixture,
copied from a real STM. The recordings carry a rising ramp rather than silence so
that a segment cut from the middle can be checked against the frame it should
have started at — a seek off by a segment would pass a silence test.
"""

from __future__ import annotations

import csv
import io
import json
import struct
import tarfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from svb.data.manifest_local import ManifestLocal
from svb.data.registry import LangSpec
from tests.conftest import _load_script

SOURCE_PAGE = "https://cs.taltech.ee/staff/tanel.alumae/data/est-pub-asr-data/"
ARCHIVE_URL = SOURCE_PAGE + "taltech-asr-speech-dataset-1.0.tar"
RATE = 16000
PERIOD = 20011  # a sample value that repeats slowly enough to identify a frame

PAGE = """<html><body>
<h1>TalTech Estonian Speech Dataset 1.0</h1>
<h2>Licence</h2>
<p>CC BY-SA 4.0 DEED: https://creativecommons.org/licenses/by-sa/4.0/</p>
<p>Copyright of the audio material in the dataset belongs to corresponding parties.</p>
<h2>Downloading</h2>
<p><a href="taltech-asr-speech-dataset-1.0.tar">taltech-asr-speech-dataset-1.0.tar</a> (150 GB)</p>
</body></html>"""

DEV_STM = """;; Exported from .../paevakaja/paevakaja-2015-03-09.trs using local/trs2stm.py
paevakaja-2015-03-09 1 inter_segment_gap 0.0 1.5 <o,f0,>
paevakaja-2015-03-09 1 paevakaja-2015-03-09_Mari_Tamm 1.5 6.25 <o,f0,> Tere õhtust, alustame päevakajaga.
"""

TEST_STM = """;; Exported from .../aktuaalne-kaamera/2013-01-10_AK_2100.trs using local/trs2stm.py
2013-01-10_AK_2100 1 2013-01-10_AK_2100_Jaan_Kask 2.0 8.5 <o,f0,> Aktuaalne kaamera, head õhtut.
"""


@pytest.fixture
def prepare_taltech(pytestconfig: pytest.Config) -> ModuleType:
    """Loaded here rather than from a conftest fixture: several preparers are
    being written on parallel branches, and a shared fixture block is the one
    file every one of them would have to edit."""
    return _load_script(Path(pytestconfig.rootpath), "prepare_taltech")


# --------------------------------------------------------------------------- #
# A fake taltech.ee
# --------------------------------------------------------------------------- #


def _wav(seconds: float, rate: int = RATE) -> bytes:
    """A 16-bit mono PCM WAV whose sample at frame `i` is `i % PERIOD`.

    Silence would let a seek land anywhere and still pass; a ramp makes the first
    sample of a cut identify the frame it came from.
    """
    frames = int(seconds * rate)
    payload = struct.pack(f"<{frames}h", *((i % PERIOD) for i in range(frames)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(payload)
    return buffer.getvalue()


def _first_sample(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return struct.unpack("<h", handle.readframes(1))[0]


def _frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


@dataclass
class _Source:
    """The archive, in the shape the real tar was walked into."""

    fixtures_dir: Path
    extra_members: dict[str, bytes] = field(default_factory=dict)
    drop: tuple[str, ...] = ()
    page: str = PAGE

    def members(self) -> dict[str, bytes]:
        stm = (self.fixtures_dir / "taltech" / "er-uudised.stm").read_bytes()
        members: dict[str, bytes] = {
            # The same recording name under two subsets, which the real archive
            # does too — `71_ID117_352639` is in both ERR2020 and intervjuukorpus.
            "train/transcriptions/stm/er-uudised/er-uudised.stm": stm,
            "train/transcriptions/stm/jutusaated/er-uudised.stm": stm,
            "train/audio/er-uudised/er-uudised.wav": _wav(35.0),
            "train/audio/jutusaated/er-uudised.wav": _wav(35.0),
            # The other two transcript formats, which are the same content and
            # are deliberately never read.
            "train/transcriptions/vtt/er-uudised/er-uudised.vtt": b"WEBVTT\n",
            "train/transcriptions/trs/er-uudised/er-uudised.trs": b"<Trans/>\n",
            # A .vtt sitting under the stm tree, which the real archive also has.
            "train/transcriptions/stm/er-uudised/stray.vtt": b"WEBVTT\n",
            "dev/transcriptions/stm/paevakaja/paevakaja-2015-03-09.stm": DEV_STM.encode(),
            "dev/audio/paevakaja/paevakaja-2015-03-09.wav": _wav(10.0),
            "test/transcriptions/stm/aktuaalne-kaamera/2013-01-10_AK_2100.stm": TEST_STM.encode(),
            "test/audio/aktuaalne-kaamera/2013-01-10_AK_2100.wav": _wav(12.0),
        }
        members.update(self.extra_members)
        return {name: body for name, body in members.items() if name not in self.drop}

    def archive(self) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, body in self.members().items():
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
        return buffer.getvalue()

    def install(self, module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        import corpus_fetch

        payloads = {SOURCE_PAGE: self.page.encode("utf-8"), ARCHIVE_URL: self.archive()}

        def fake_urlopen(request: Any, timeout: float | None = None) -> _Response:
            url = request.full_url if hasattr(request, "full_url") else str(request)
            if url not in payloads:
                raise AssertionError(f"the test served nothing for {url}")
            return _Response(payloads[url])

        monkeypatch.setattr(module, "urlopen", fake_urlopen)
        monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)


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


@pytest.fixture
def source(
    prepare_taltech: ModuleType, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> _Source:
    made = _Source(fixtures_dir)
    made.install(prepare_taltech, monkeypatch)
    return made


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# --------------------------------------------------------------------------- #
# Reading the STM
# --------------------------------------------------------------------------- #


def test_the_segment_bounds_are_the_utterances(
    prepare_taltech: ModuleType, fixtures_dir: Path
) -> None:
    segments = prepare_taltech.read_stm(
        fixtures_dir / "taltech" / "er-uudised.stm", "train", "er-uudised"
    )

    assert [(s.start, s.end) for s in segments] == [
        (5.409, 9.175),
        (9.175, 13.71),
        (14.2, 21.721),
        (26.5, 31.15),
        (120.0, 130.0),
    ]


def test_the_gap_rows_and_the_scoring_marker_are_not_utterances(
    prepare_taltech: ModuleType, fixtures_dir: Path
) -> None:
    """`inter_segment_gap` is the silence between segments and carries no
    transcript; IGNORE_TIME_SEGMENT_IN_SCORING is a marker, not speech."""
    segments = prepare_taltech.read_stm(
        fixtures_dir / "taltech" / "er-uudised.stm", "train", "er-uudised"
    )

    assert prepare_taltech.GAP_SPEAKER not in {s.speaker for s in segments}
    assert prepare_taltech.IGNORE_MARKER not in {s.text for s in segments}


def test_the_speaker_is_the_third_column(prepare_taltech: ModuleType, fixtures_dir: Path) -> None:
    segments = prepare_taltech.read_stm(
        fixtures_dir / "taltech" / "er-uudised.stm", "train", "er-uudised"
    )

    assert {s.speaker for s in segments} == {
        "er-uudised_Vallo_Kelmsaar",
        "er-uudised_Urmas_Paet",
    }
    assert segments[0].text.startswith("Kell on kaks, te kuulete")


def test_the_label_field_is_not_mistaken_for_the_transcript(
    prepare_taltech: ModuleType, fixtures_dir: Path
) -> None:
    """`<o,f0,>` sits between the end time and the words; a parser that split on
    whitespace alone would put it at the front of every transcript."""
    segments = prepare_taltech.read_stm(
        fixtures_dir / "taltech" / "er-uudised.stm", "train", "er-uudised"
    )

    assert not any(s.text.startswith("<") for s in segments)


# --------------------------------------------------------------------------- #
# Which members are read
# --------------------------------------------------------------------------- #


def test_only_the_stm_tree_and_the_audio_are_extracted(prepare_taltech: ModuleType) -> None:
    """VTT and TRS are the same transcripts in two other formats. Extracting
    them would write tens of gigabytes nothing reads."""
    assert prepare_taltech.is_stm("train/transcriptions/stm/er-uudised/x.stm")
    assert not prepare_taltech.is_stm("train/transcriptions/vtt/er-uudised/x.vtt")
    assert not prepare_taltech.is_stm("train/transcriptions/trs/er-uudised/x.trs")
    assert prepare_taltech.is_audio("dev/audio/paevakaja/x.wav")
    assert not prepare_taltech.is_audio("dev/transcriptions/stm/paevakaja/x.stm")


def test_a_vtt_sitting_under_the_stm_tree_is_not_parsed_as_stm(
    prepare_taltech: ModuleType,
) -> None:
    """The real archive has some: `test/transcriptions/stm/aktuaalne-kaamera/`
    holds `.vtt` files. Selecting by directory alone would feed WEBVTT to the STM
    parser and get silence rather than an error."""
    assert not prepare_taltech.is_stm("test/transcriptions/stm/aktuaalne-kaamera/x.vtt")


# --------------------------------------------------------------------------- #
# Cutting the long-form audio
# --------------------------------------------------------------------------- #


def test_each_utterance_is_cut_out_of_the_long_form_recording(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """Recordings run half an hour and the manifest addresses whole files, so the
    cut is what makes this corpus loadable at all."""
    manifest = prepare_taltech.prepare(tmp_path)
    rows = {row["utt_id"]: row for row in _rows(manifest)}

    row = rows["er-uudised_er-uudised_000005409-000009175"]
    cut = manifest.parent / row["path"]
    assert _frames(cut) == int(9.175 * RATE) - int(5.409 * RATE)
    assert _first_sample(cut) == int(5.409 * RATE) % PERIOD


def test_a_segment_starting_past_the_end_of_the_recording_is_dropped(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The fixture's last row runs 120–130 s over a 35 s recording. An empty WAV
    would fail the loader instead of this preparer."""
    manifest = prepare_taltech.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert "er-uudised_er-uudised_000120000-000130000" not in {
        row["utt_id"] for row in _rows(manifest)
    }
    assert payload["segments_past_end_of_recording"] == 2


def test_the_same_recording_name_in_two_subsets_does_not_collide(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """`71_ID117_352639` really is in both ERR2020 and intervjuukorpus, so the
    subset has to be part of the id and of the path."""
    manifest = prepare_taltech.prepare(tmp_path)
    rows = _rows(manifest)

    ids = [row["utt_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert "er-uudised_er-uudised_000005409-000009175" in ids
    assert "jutusaated_er-uudised_000005409-000009175" in ids


def test_the_sample_rate_is_recorded_from_the_file_rather_than_assumed(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The source page states no audio format at all, which is why
    configs/corpora.yaml marks it unverified."""
    manifest = prepare_taltech.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["sample_rate_hz"] == RATE


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_the_shipped_split_comes_from_the_top_level_directory(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The corpus calls it `dev` and the manifest layer calls it `validation`."""
    rows = _rows(prepare_taltech.prepare(tmp_path))
    by_split = {row["split"] for row in rows}

    assert by_split == {"train", "validation", "test"}
    assert [row["speaker"] for row in rows if row["split"] == "validation"] == [
        "paevakaja-2015-03-09_Mari_Tamm"
    ]


def test_the_manifest_paths_resolve_to_the_cut_audio(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    manifest = prepare_taltech.prepare(tmp_path)

    for row in _rows(manifest):
        assert row["path"].startswith("audio/")
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_a_recording_with_transcripts_but_no_audio_writes_no_rows(
    prepare_taltech: ModuleType,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _Source(fixtures_dir, drop=("dev/audio/paevakaja/paevakaja-2015-03-09.wav",)).install(
        prepare_taltech, monkeypatch
    )

    rows = _rows(prepare_taltech.prepare(tmp_path))

    assert not [row for row in rows if row["split"] == "validation"]


def test_preparing_twice_writes_the_same_manifest(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    first = prepare_taltech.prepare(tmp_path).read_text(encoding="utf-8")
    second = prepare_taltech.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second


def test_what_the_preparer_writes_is_what_the_loader_reads(
    prepare_taltech: ModuleType,
    source: _Source,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_taltech.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(code="et", source="manifest", corpus="taltech_estonian", hf_config="et")

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "shipped"
    assert [len(part) for part in parts.values()] == [8, 1, 1]
    waveform, text, code = parts["test"][0]
    assert code == "et"
    assert text.startswith("Aktuaalne kaamera")
    assert waveform.shape[0] == int(8.5 * RATE) - int(2.0 * RATE)


# --------------------------------------------------------------------------- #
# Provenance and refusals
# --------------------------------------------------------------------------- #


def test_the_record_says_the_publisher_offers_no_checksum(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    """The tar is 159 GB and nothing is published to compare it against, so the
    only evidence is that it read end to end. Saying so beats a digest that looks
    like a verification."""
    manifest = prepare_taltech.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert "no checksum" in payload["integrity"]
    assert len(payload["archives"][0]["sha256"]) == 64
    assert payload["archives"][0]["url"] == ARCHIVE_URL


def test_the_hours_per_split_are_measured_from_the_segment_bounds(
    prepare_taltech: ModuleType, source: _Source, tmp_path: Path
) -> None:
    manifest = prepare_taltech.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    # 3.766 + 4.535 + 7.521 + 4.65 seconds of transcribed speech, twice over,
    # because the fixture recording appears under two subsets.
    assert payload["seconds"] == {"train": 40.944, "validation": 4.75, "test": 6.5}
    assert payload["counts"] == {"train": 8, "validation": 1, "test": 1}


def test_a_licence_that_no_longer_says_share_alike_stops_the_ingest(
    prepare_taltech: ModuleType,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """There is no API and no checksum here, so the page is the only record of
    the terms — and it is the same host the 159 GB comes from."""
    _Source(fixtures_dir, page="<html><body>All rights reserved.</body></html>").install(
        prepare_taltech, monkeypatch
    )

    with pytest.raises(ValueError, match="settle the difference"):
        prepare_taltech.prepare(tmp_path)


def test_an_archive_with_no_stm_at_all_is_reported_rather_than_written_empty(
    prepare_taltech: ModuleType,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _Source(
        fixtures_dir,
        drop=(
            "train/transcriptions/stm/er-uudised/er-uudised.stm",
            "train/transcriptions/stm/jutusaated/er-uudised.stm",
            "dev/transcriptions/stm/paevakaja/paevakaja-2015-03-09.stm",
            "test/transcriptions/stm/aktuaalne-kaamera/2013-01-10_AK_2100.stm",
        ),
    ).install(prepare_taltech, monkeypatch)

    with pytest.raises(ValueError, match="layout has changed"):
        prepare_taltech.prepare(tmp_path)


def test_a_recording_name_too_long_for_a_path_component_is_shortened_uniquely(
    prepare_taltech: ModuleType,
) -> None:
    """The real tar needs GNU @LongLink entries for some of these names, so they
    are long enough to be worth guarding against."""
    long_name = "a" * 300
    other = "a" * 250 + "b" * 50

    first = prepare_taltech.safe_component(long_name)
    second = prepare_taltech.safe_component(other)

    assert len(first.encode("utf-8")) <= prepare_taltech.MAX_COMPONENT
    assert first != second
    assert prepare_taltech.safe_component("short") == "short"
