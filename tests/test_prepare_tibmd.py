"""TIBMD@MUC preparer.

The archive is built in the test and served by a fake urlopen, reproducing the
nested layout read off the real tgz — `<Dialect>/<area>/<sub-area>/Speaker
<N>/<task>/wav/<name>.wav`, including the archive's own "mutil" spelling of its
root directory and the varying depth between the dialect and the speaker.

The transcript convention is the one thing the real archive would not give up
cheaply: 444 MB of decompressed prefix, 561 consecutive wav files under one
speaker, contained no non-audio member. So the preparer discovers it, and these
tests drive both shapes it accepts plus the refusal when neither fires.
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

ROOT = "Tibetan mutil-dialect data"

#: `{dialect: (path between dialect and speaker, {speaker: [stems]})}`. The two
#: dialects nest to different depths, which is why the preparer locates the
#: speaker and dialect components by name rather than by position.
TREE = {
    "Amdo Dialect": (
        "Pastoral areas dialect/Aba pastoral dialect",
        {"Speaker 65": ["阿坝牧区_crzm_A523", "阿坝牧区_crzm_A1709"]},
    ),
    "Kham Dialect": ("", {"Speaker 12": ["康巴_abcd_B01"], "Speaker 13": ["康巴_efgh_B02"]}),
}
SENTENCES = {
    "阿坝牧区_crzm_A523": "བཀྲ་ཤིས་བདེ་ལེགས།",
    "阿坝牧区_crzm_A1709": "ཁྱེད་རང་ག་འདྲ་འདུག",
    "康巴_abcd_B01": "ད་ལྟ་ཆུ་ཚོད་ག་ཚོད་རེད།",
    "康巴_efgh_B02": "ང་བོད་པ་ཡིན།",
}
ALL_STEMS = [stem for _, speakers in TREE.values() for stems in speakers.values() for stem in stems]


def _wav_paths() -> list[str]:
    paths: list[str] = []
    for dialect, (middle, speakers) in TREE.items():
        for speaker, stems in speakers.items():
            parent = "/".join(part for part in (ROOT, dialect, middle, speaker, "Reading") if part)
            paths.extend(f"{parent}/wav/{stem}.wav" for stem in stems)
    return paths


def _archive(*, transcripts: str = "index", extra: dict[str, bytes] | None = None) -> bytes:
    """`transcripts`: "index", "per_file", "both" or "none"."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        for path in _wav_paths():
            add(path, b"RIFF____WAVEfmt ")

        if transcripts in {"index", "both"}:
            body = "".join(f"{stem}\t{text}\n" for stem, text in SENTENCES.items())
            add(f"{ROOT}/transcript.txt", body.encode("utf-8"))
        if transcripts in {"per_file", "both"}:
            for path in _wav_paths():
                stem = Path(path).stem
                text = SENTENCES[stem] + (" per-file" if transcripts == "both" else "")
                add(path.replace("/wav/", "/txt/").replace(".wav", ".txt"), text.encode("utf-8"))
        for name, payload in (extra or {}).items():
            add(name, payload)
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
def prepare_tibmd(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "prepare_tibmd")


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
# Speaker and dialect out of the path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (
            f"{ROOT}/Amdo Dialect/Pastoral areas dialect/Aba pastoral dialect/"
            "Speaker 65/Reading/wav/x.wav",
            ("Amdo_Dialect-speaker_65", "Amdo_Dialect"),
        ),
        (
            f"{ROOT}/Kham Dialect/Speaker 12/Reading/wav/x.wav",
            ("Kham_Dialect-speaker_12", "Kham_Dialect"),
        ),
        (f"{ROOT}/Kham Dialect/speaker_7/wav/x.wav", ("Kham_Dialect-speaker_7", "Kham_Dialect")),
        (f"{ROOT}/Kham Dialect/Reading/wav/x.wav", None),
    ],
)
def test_the_speaker_and_dialect_are_located_by_name_not_by_depth(
    prepare_tibmd: ModuleType, tmp_path: Path, relative: str, expected
) -> None:
    """The depth between the dialect directory and the Speaker one differs across
    the three dialect trees, so counting components would work for Amdo and break
    for the others.

    The first case also covers two traps at once: the archive's own root is
    "Tibetan mutil-dialect data" (its spelling), so a substring match on
    "dialect" would take the root for the dialect on every path in the corpus;
    and "Pastoral areas dialect" and "Aba pastoral dialect" end in the word too,
    so it is the first matching component that counts.
    """
    assert prepare_tibmd.parse_audio_path(tmp_path / relative, tmp_path) == expected


def test_two_dialects_speaker_3_do_not_collapse_into_one_speaker(
    prepare_tibmd: ModuleType, tmp_path: Path
) -> None:
    """The corpus is not stated to number its 87 speakers globally, and merging
    two speakers would put the same voice in train and test."""
    amdo = prepare_tibmd.parse_audio_path(
        tmp_path / f"{ROOT}/Amdo Dialect/Speaker 3/x.wav", tmp_path
    )
    kham = prepare_tibmd.parse_audio_path(
        tmp_path / f"{ROOT}/Kham Dialect/Speaker 3/x.wav", tmp_path
    )

    assert amdo is not None and kham is not None
    assert amdo[0] != kham[0]


def test_every_row_carries_its_speaker_and_dialect(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    manifest = prepare_tibmd.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert payload["speakers"] == 3
    assert payload["dialects"] == ["Amdo_Dialect", "Kham_Dialect"]
    assert payload["no_speaker"] == 0


def test_the_dialect_travels_in_an_extra_manifest_column(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    """The loader reads the manifest with DictReader and only requires its five
    columns to be present, so an extra one stays with the data instead of being
    lost to a side file."""
    serve(_archive())
    rows = _rows(prepare_tibmd.prepare(tmp_path))

    assert {row["dialect"] for row in rows} == {"Amdo_Dialect", "Kham_Dialect"}


def test_the_loader_accepts_the_extra_column_and_still_splits_by_speaker(
    prepare_tibmd: ModuleType, serve, tmp_path: Path, monkeypatch
) -> None:
    serve(_archive())
    prepare_tibmd.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(
        code="bo", source="manifest", corpus=prepare_tibmd.CORPUS_IDS[0], hf_config="bo"
    )

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "speaker"
    assert sum(len(part) for part in parts.values()) == len(ALL_STEMS)
    speakers = {name: {row.speaker for row in part.rows} for name, part in parts.items()}
    assert not speakers["train"] & speakers["test"]


# --------------------------------------------------------------------------- #
# Discovering the transcripts
# --------------------------------------------------------------------------- #


def test_an_index_file_keyed_by_utterance_stem_is_found(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(transcripts="index"))
    manifest = prepare_tibmd.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    rows = _rows(manifest)
    assert {row["utt_id"] for row in rows} == set(ALL_STEMS)
    assert rows[0]["text"] == SENTENCES[rows[0]["utt_id"]]
    assert payload["transcript_convention"] == "index file"
    assert payload["index_files"][0]["matched_rows"] == len(ALL_STEMS)


def test_one_transcript_file_per_utterance_is_found(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(transcripts="per_file"))
    manifest = prepare_tibmd.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    rows = _rows(manifest)
    assert {row["utt_id"] for row in rows} == set(ALL_STEMS)
    assert payload["transcript_convention"] == "one file per utterance"
    assert payload["per_file_transcripts"] == len(ALL_STEMS)


def test_a_per_utterance_file_wins_over_an_index_covering_the_same_stem(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    """A file named for one utterance is the more specific statement about it."""
    serve(_archive(transcripts="both"))
    rows = _rows(prepare_tibmd.prepare(tmp_path))

    assert all(row["text"].endswith("per-file") for row in rows)


def test_finding_no_transcripts_refuses_and_reports_what_was_there(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    """The source page documents no convention and the archive would not give it
    up cheaply, so a run that finds nothing has to say what it did find rather
    than fail silently."""
    serve(_archive(transcripts="none", extra={f"{ROOT}/notes.pdf": b"%PDF-1.4 binary"}))

    with pytest.raises(ValueError, match="found no transcripts"):
        prepare_tibmd.prepare(tmp_path)


def test_the_refusal_names_the_suffixes_it_saw(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(transcripts="none", extra={f"{ROOT}/labels.xml": b"<xml/>"}))

    with pytest.raises(ValueError, match=r"\.xml"):
        prepare_tibmd.prepare(tmp_path)


def test_a_non_utf8_file_is_skipped_rather_than_crashing_the_run(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(transcripts="index", extra={f"{ROOT}/blob.bin": bytes([0xFF, 0xFE, 0x00])}))

    assert len(_rows(prepare_tibmd.prepare(tmp_path))) == len(ALL_STEMS)


def test_audio_the_index_does_not_cover_is_dropped_rather_than_written(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    body = _archive(transcripts="none")
    partial = "".join(
        f"{stem}\t{text}\n" for stem, text in SENTENCES.items() if stem != ALL_STEMS[0]
    )
    serve(_archive(transcripts="none", extra={f"{ROOT}/transcript.txt": partial.encode("utf-8")}))
    assert body  # the "none" archive is the base the extra index is added to

    manifest = prepare_tibmd.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert ALL_STEMS[0] not in {row["utt_id"] for row in _rows(manifest)}
    assert payload["missing_transcript"] == 1


# --------------------------------------------------------------------------- #
# The split, refusals and idempotence
# --------------------------------------------------------------------------- #


def test_the_split_column_is_empty_because_the_corpus_ships_none(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())

    assert {row["split"] for row in _rows(prepare_tibmd.prepare(tmp_path))} == {""}


def test_the_manifest_paths_resolve_to_the_extracted_audio(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    """The paths carry spaces and non-Latin characters, which is worth walking
    through the join rather than assuming."""
    serve(_archive())
    manifest = prepare_tibmd.prepare(tmp_path)

    for row in _rows(manifest):
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_a_corrupt_archive_is_deleted_and_reported_rather_than_extracted(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(b"not a tarball at all")

    with pytest.raises(OSError, match="re-run"):
        prepare_tibmd.prepare(tmp_path)


def test_preparing_twice_gives_the_same_manifest(
    prepare_tibmd: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    first = prepare_tibmd.prepare(tmp_path).read_text(encoding="utf-8")
    serve(_archive())
    second = prepare_tibmd.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second
