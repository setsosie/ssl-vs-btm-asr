"""Armenian Speech Crowdsourcing Data preparer.

The archive is built in the test and served by a fake urlopen. The synthetic
`pitched.jsonl` carries exactly the three keys the real one does — `text`,
`audio_filepath` and `duration` — including the absolute `/data/pitched/...`
path from the corpus builder's machine, which is the reason only the file name
is used.
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

ROOT = "armenian_speech_crowdsourcing_data"

UTTERANCES = [
    ("eu.1619b5a6-8837-46cb-bf8a-6c037b2eace9", "Լավ էլ ապրում են իրանց համար։", 5.33),
    ("eu.0d16c4ad-db4c-4a44-b56c-29ecbe78e22a", "Գևորգ Էմինի հուշարձան։", 5.33),
    ("eu.72d798a0-4965-4766-baea-74d79b99b49a", "Երևանի կենտրոնում։", 11.39),
    ("eu.175eb22e-1292-4a88-a7ec-56cb8dbbfcef", "Չորրորդ նախադասությունը։", 3.5),
    ("eu.f920ce78-75cf-4605-a6ed-83bf80c7dfe7", "Հինգերորդ նախադասությունը։", 4.25),
]


def _archive(
    *,
    utterances=None,
    drop_audio: tuple[str, ...] = (),
    wrap: str | None = ROOT,
    extra_lines: str = "",
) -> bytes:
    utterances = UTTERANCES if utterances is None else utterances
    prefix = f"{wrap}/" if wrap else ""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        body = ""
        for stem, text, duration in utterances:
            body += (
                json.dumps(
                    {
                        "text": text,
                        "audio_filepath": f"/data/pitched/{stem}.wav",
                        "duration": duration,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if stem not in drop_audio:
                add(f"{prefix}pitched/{stem}.wav", b"RIFF____WAVEfmt ")
        add(f"{prefix}pitched.jsonl", (body + extra_lines).encode("utf-8"))
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
def prepare_armenian(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "prepare_armenian")


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
# No speaker, and that is the corpus's design
# --------------------------------------------------------------------------- #


def test_the_speaker_column_is_empty_because_the_voices_were_anonymized(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    """The page states the voices "could not be easily identified or matched to
    individual speakers". pitched.jsonl carries text, audio_filepath and
    duration and nothing else, so there is no speaker to carry."""
    serve(_archive())

    assert {row["speaker"] for row in _rows(prepare_armenian.prepare(tmp_path))} == {""}


def test_the_loader_therefore_reports_an_utterance_level_split(
    prepare_armenian: ModuleType, serve, tmp_path: Path, monkeypatch
) -> None:
    """A result on this language is not speaker-independent, and the policy the
    loader reports is what carries that caveat into a run's record."""
    serve(_archive())
    prepare_armenian.prepare(tmp_path)
    monkeypatch.setenv("CORPORA_ROOT", str(tmp_path))
    spec = LangSpec(
        code="hy", source="manifest", corpus=prepare_armenian.CORPUS_IDS[0], hf_config="hy"
    )

    parts = {name: ManifestLocal(spec, name) for name in ("train", "validation", "test")}

    assert parts["train"].split_policy == "utterance"
    assert sum(len(part) for part in parts.values()) == len(UTTERANCES)


def test_the_fetch_record_says_why_there_is_no_speaker(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    manifest = prepare_armenian.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert "not speaker-independent" in payload["speaker_ids"]
    assert payload["split_source"] == "derived"


def test_this_corpus_does_not_share_the_google_crowdsourced_layout(
    prepare_armenian: ModuleType, prepare_google_crowdsourced: ModuleType
) -> None:
    """Both are crowdsourced read speech on OpenSLR, so it is worth asserting
    rather than assuming: those five ship sixteen asr_<language>_*.zip shards
    and a three-column utt_spk_text.tsv whose middle column is the speaker,
    which is the one field this corpus does not have."""
    google = prepare_google_crowdsourced

    assert "hy" not in google.LANGUAGES
    assert prepare_armenian.INDEX == "pitched.jsonl"
    assert prepare_armenian.INDEX != google.INDEX


# --------------------------------------------------------------------------- #
# Reading pitched.jsonl
# --------------------------------------------------------------------------- #


def test_only_the_file_name_of_the_absolute_audio_path_is_used(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    """audio_filepath is /data/pitched/... — an absolute path on the machine that
    built the corpus, which resolves to nothing here."""
    serve(_archive())
    manifest = prepare_armenian.prepare(tmp_path)

    for row in _rows(manifest):
        assert not row["path"].startswith("/")
        assert (manifest.parent / row["path"]).exists(), row["path"]


def test_the_utterance_id_is_the_audio_stem(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    rows = _rows(prepare_armenian.prepare(tmp_path))

    assert {row["utt_id"] for row in rows} == {stem for stem, _, _ in UTTERANCES}


def test_the_transcript_comes_through_unicode_intact(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    rows = _rows(prepare_armenian.prepare(tmp_path))

    text = next(row["text"] for row in rows if row["utt_id"] == UTTERANCES[0][0])
    assert text == "Լավ էլ ապրում են իրանց համար։"


def test_a_malformed_jsonl_line_is_skipped_rather_than_losing_the_corpus(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(extra_lines='not json at all\n{"text": "", "audio_filepath": ""}\n[]\n'))

    assert len(_rows(prepare_armenian.prepare(tmp_path))) == len(UTTERANCES)


def test_an_indexed_utterance_without_audio_is_dropped_rather_than_written(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(drop_audio=(UTTERANCES[1][0],)))
    manifest = prepare_armenian.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    assert UTTERANCES[1][0] not in {row["utt_id"] for row in _rows(manifest)}
    assert payload["missing_audio"] == 1
    assert payload["index_rows"] == len(UTTERANCES)


def test_the_indexed_hours_are_summed_from_the_duration_field(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    manifest = prepare_armenian.prepare(tmp_path)
    payload = json.loads((manifest.parent / "fetch_manifest.json").read_text(encoding="utf-8"))

    total = sum(duration for _, _, duration in UTTERANCES)
    assert payload["indexed_hours"] == pytest.approx(total / 3600, abs=1e-3)


def test_the_index_is_found_even_if_the_archive_stops_wrapping_everything(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    """A fixed depth would fail with a path error instead of a name."""
    serve(_archive(wrap=None))

    assert len(_rows(prepare_armenian.prepare(tmp_path))) == len(UTTERANCES)


def test_an_archive_without_the_index_is_refused_by_name(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive(utterances=[]))

    with pytest.raises((FileNotFoundError, ValueError), match=r"pitched\.jsonl"):
        prepare_armenian.prepare(tmp_path)


# --------------------------------------------------------------------------- #
# Refusals and idempotence
# --------------------------------------------------------------------------- #


def test_a_corrupt_archive_is_deleted_and_reported_rather_than_extracted(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(b"not a tarball at all")

    with pytest.raises(OSError, match="re-run"):
        prepare_armenian.prepare(tmp_path)


def test_preparing_twice_gives_the_same_manifest(
    prepare_armenian: ModuleType, serve, tmp_path: Path
) -> None:
    serve(_archive())
    first = prepare_armenian.prepare(tmp_path).read_text(encoding="utf-8")
    serve(_archive())
    second = prepare_armenian.prepare(tmp_path).read_text(encoding="utf-8")

    assert first == second
