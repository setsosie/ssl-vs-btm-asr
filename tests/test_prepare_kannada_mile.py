"""IISc-MILE Kannada preparer.

Two archives are built in the test and served by a fake urlopen, reproducing the
layout read off the real tars: each carries its own top-level split directory
holding `audio_files/<stem>.wav` beside `trans_files/<stem>.txt`, with the
speaker encoded in the stem before `_UTT_`.
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

#: `{shipped directory: {speaker: [utterance numbers]}}`. Four speakers is what
#: lets the derivation put a distinct speaker in each of the three splits.
TREE = {
    "train": {"MILE_03_SP_3415": ["0038", "0102"], "MILE_03_SP_2511": ["0072", "0066"]},
    "test": {"MILE_03_SP_3386": ["0066"], "MILE_03_SP_1134": ["0059"]},
}
SENTENCE = "ಇವುಗಳಿಗಾಗಿ ಏಳುನೂರು ಕೋಟಿ ರುಗೆ ಒಪ್ಪಿಗೆ ನೀಡಲಾಗಿದೆ"


def _stems(shipped: str) -> list[str]:
    return [f"{spk}_UTT_{n}" for spk, ns in TREE[shipped].items() for n in ns]


ALL_STEMS = _stems("train") + _stems("test")


def _archive(
    shipped: str,
    *,
    drop_transcript: tuple[str, ...] = (),
    wrap: str = "",
    stems: list[str] | None = None,
) -> bytes:
    stems = _stems(shipped) if stems is None else stems
    prefix = f"{wrap}/" if wrap else ""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        for stem in stems:
            add(f"{prefix}{shipped}/audio_files/{stem}.wav", b"RIFF____WAVEfmt ")
            if stem not in drop_transcript:
                add(
                    f"{prefix}{shipped}/trans_files/{stem}.txt",
                    f"  {SENTENCE}\n  {stem}\n".encode(),
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
def prepare_kannada_mile(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "prepare_kannada_mile")


@pytest.fixture
def serve(monkeypatch):
    """Serve the two archives, dispatching on the infix in the requested URL."""

    def install(bodies: dict[str, bytes]) -> None:
        import corpus_fetch

        def fake_urlopen(request, timeout=None):
            url = request.full_url if hasattr(request, "full_url") else str(request)
            for shipped, body in bodies.items():
                if f"mile_kannada_{shipped}." in url:
                    return _Response(body)
            raise AssertionError(f"unexpected url {url}")

        monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)

    return install


@pytest.fixture
def both(serve):
    serve({shipped: _archive(shipped) for shipped in TREE})


def _rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# --------------------------------------------------------------------------- #
# The re-derived split
# --------------------------------------------------------------------------- #


def test_the_split_column_is_cleared_because_no_dev_ships(
    prepare_kannada_mile: ModuleType, both, tmp_path: Path
) -> None:
    """The loader takes a partition whole or derives one whole, so a shipped
    train/test with no dev cannot be passed through."""
    rows = _rows(prepare_kannada_mile.prepare(tmp_path))

    assert {row["split"] for row in rows} == {""}
    assert {row["utt_id"] for row in rows} == set(ALL_STEMS)


def test_both_shipped_archives_are_pooled_before_re_deriving(
    prepare_kannada_mile: ModuleType, both, tmp_path: Path
) -> None:
    """Nothing is discarded; the corpus's own train and test both enter the pool."""
    manifest = prepare_kannada_mile.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["shipped_counts"] == {"train": 4, "test": 2}
    assert [record["name"] for record in payload["archives"]] == [
        "mile_kannada_train.tar.gz",
        "mile_kannada_test.tar.gz",
    ]


def test_each_archive_keeps_its_own_top_level_split_directory(
    prepare_kannada_mile: ModuleType, both, tmp_path: Path
) -> None:
    """They cannot collide, so the layout is kept rather than flattened."""
    manifest = prepare_kannada_mile.prepare(tmp_path)
    paths = {row["path"] for row in _rows(manifest)}

    assert any(path.startswith("train/audio_files/") for path in paths)
    assert any(path.startswith("test/audio_files/") for path in paths)


# --------------------------------------------------------------------------- #
# Speakers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("MILE_03_SP_2496_UTT_0116", "MILE_03_SP_2496"),
        ("MILE_03_SP_0002_UTT_0081", "MILE_03_SP_0002"),
        ("no_separator_here", None),
        ("_UTT_0001", None),
    ],
)
def test_the_speaker_is_the_stem_before_the_utterance_separator(
    prepare_kannada_mile: ModuleType, stem: str, expected: str | None
) -> None:
    """The corpus record had speaker ids as unconfirmed. They are in the stem,
    which ESPnet's recipe for this corpus derives the same way."""
    assert prepare_kannada_mile.speaker_of(stem) == expected


def test_every_row_carries_a_speaker_so_the_split_can_be_disjoint(
    prepare_kannada_mile: ModuleType, both, tmp_path: Path
) -> None:
    manifest = prepare_kannada_mile.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["no_speaker"] == 0
    assert payload["speakers"] == 4
    for row in _rows(manifest):
        assert row["utt_id"].startswith(row["speaker"] + "_UTT_")


def test_a_stem_with_no_speaker_is_counted_rather_than_given_an_invented_one(
    prepare_kannada_mile: ModuleType, serve, tmp_path: Path
) -> None:
    serve({"test": _archive("test", stems=["odd-name-without-separator"])})
    manifest = prepare_kannada_mile.prepare(tmp_path, splits=("test",))
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["no_speaker"] == 1
    assert {row["speaker"] for row in _rows(manifest)} == {""}


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


def test_the_transcript_is_the_whole_txt_with_whitespace_collapsed(
    prepare_kannada_mile: ModuleType, both, tmp_path: Path
) -> None:
    rows = _rows(prepare_kannada_mile.prepare(tmp_path))

    stem = ALL_STEMS[0]
    text = next(row["text"] for row in rows if row["utt_id"] == stem)
    assert text == f"{SENTENCE} {stem}"


def test_audio_without_a_transcript_is_dropped_rather_than_written(
    prepare_kannada_mile: ModuleType, serve, tmp_path: Path
) -> None:
    missing = _stems("test")[0]
    serve(
        {
            "train": _archive("train"),
            "test": _archive("test", drop_transcript=(missing,)),
        }
    )
    manifest = prepare_kannada_mile.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert missing not in {row["utt_id"] for row in _rows(manifest)}
    assert payload["missing_transcript"] == 1


# --------------------------------------------------------------------------- #
# Extraction shapes, refusals, idempotence
# --------------------------------------------------------------------------- #


def test_an_extra_wrapping_directory_is_tolerated(
    prepare_kannada_mile: ModuleType, serve, tmp_path: Path
) -> None:
    """ESPnet's recipe for this corpus guards against one, so the tree is
    searched rather than assumed to be at a fixed depth."""
    serve({shipped: _archive(shipped, wrap="mile_kannada") for shipped in TREE})
    rows = _rows(prepare_kannada_mile.prepare(tmp_path))

    assert {row["utt_id"] for row in rows} == set(ALL_STEMS)


def test_a_missing_split_directory_is_refused_by_name(
    prepare_kannada_mile: ModuleType, serve, tmp_path: Path
) -> None:
    serve({"train": _archive("test")})  # the train url serves a test-shaped tree

    with pytest.raises(FileNotFoundError, match="audio_files"):
        prepare_kannada_mile.prepare(tmp_path, splits=("train",))


def test_a_corrupt_archive_is_deleted_and_reported_rather_than_extracted(
    prepare_kannada_mile: ModuleType, serve, tmp_path: Path
) -> None:
    serve({"train": b"not a tarball at all", "test": _archive("test")})

    with pytest.raises(OSError, match="re-run"):
        prepare_kannada_mile.prepare(tmp_path)


def test_preparing_twice_gives_the_same_manifest(
    prepare_kannada_mile: ModuleType, both, tmp_path: Path
) -> None:
    first = prepare_kannada_mile.prepare(tmp_path).read_text(encoding="utf-8")
    second = prepare_kannada_mile.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second


def test_the_archive_urls_match_the_published_layout(prepare_kannada_mile: ModuleType) -> None:
    assert (
        prepare_kannada_mile.archive_url("train")
        == "https://www.openslr.org/resources/126/mile_kannada_train.tar.gz"
    )


# --------------------------------------------------------------------------- #
# It feeds the loader
# --------------------------------------------------------------------------- #


def test_the_loader_derives_a_speaker_disjoint_split_from_what_is_written(
    prepare_kannada_mile: ModuleType, both, tmp_path: Path, monkeypatch
) -> None:
    prepare_kannada_mile.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(
        code="kn", source="manifest", corpus=prepare_kannada_mile.CORPUS_IDS[0], hf_config="kn"
    )

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "speaker"
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert all(speakers.values())
    assert not speakers["train"] & speakers["test"]
    assert not speakers["train"] & speakers["validation"]
    assert not speakers["validation"] & speakers["test"]
