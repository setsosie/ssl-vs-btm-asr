"""`scripts/corpus_fetch.py` — the part of corpus preparation that is the same
for every corpus. Nothing here touches the network: the archives are built in
the test, which is the only way to exercise a truncated one.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def _tar(path: Path, files: dict[str, bytes]) -> Path:
    import io

    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


CORPUS = {"lang/a.wav": b"aaaa", "lang/b.wav": b"bbbb", "lang/index.tsv": b"a\tone\nb\ttwo\n"}


def _make_archive(path: Path, wav_ids: list[str], index_name: str = "line_index.tsv") -> Path:
    """An archive shaped like a real OpenSLR one, for the checks that need a
    realistic payload rather than four bytes."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(index_name, "".join(f"{i}\ttranscript {i}\n" for i in wav_ids))
        for i in wav_ids:
            zf.writestr(f"{i}.wav", b"RIFF____WAVEfmt ")
    return path


# --------------------------------------------------------------------------- #
# Archive kinds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("x.zip", "zip"),
        ("x.tar.gz", "tar"),
        ("x.tgz", "tar"),
        ("x.tar", "tar"),
        ("x.tar.xz", "tar"),
    ],
)
def test_the_archive_kind_is_read_off_the_name(
    corpus_fetch: ModuleType, name: str, kind: str
) -> None:
    assert corpus_fetch.archive_kind(Path(name)) == kind


def test_something_that_is_not_an_archive_is_named_rather_than_guessed(
    corpus_fetch: ModuleType,
) -> None:
    with pytest.raises(ValueError, match=r"corpus\.rar"):
        corpus_fetch.archive_kind(Path("corpus.rar"))


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


def test_an_intact_zip_and_an_intact_tar_both_verify(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    assert corpus_fetch.verify_archive(_zip(tmp_path / "a.zip", CORPUS)) is None
    assert corpus_fetch.verify_archive(_tar(tmp_path / "a.tar.gz", CORPUS)) is None


def test_a_truncated_archive_is_reported_not_extracted(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    """The failure a dropped download actually produces."""
    path = _tar(tmp_path / "a.tar.gz", CORPUS)
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])

    assert corpus_fetch.verify_archive(path) is not None


def test_a_zip_with_a_corrupted_member_fails_its_crc(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    """A tar has no checksum, so this is the one archive kind where a silent
    bit flip inside an otherwise complete file is caught."""
    path = _zip(tmp_path / "a.zip", {"x.bin": b"0" * 4096})
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    path.write_bytes(bytes(raw))

    assert corpus_fetch.verify_archive(path) is not None


def test_the_record_carries_a_digest_because_no_corpus_publishes_one(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    record = corpus_fetch.archive_record(_zip(tmp_path / "a.zip", CORPUS), "https://example/a.zip")

    assert record["name"] == "a.zip"
    assert len(record["sha256"]) == 64
    assert record["bytes"] > 0


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_extraction_keeps_the_layout_by_default(
    corpus_fetch: ModuleType, tmp_path: Path, suffix: str
) -> None:
    """A corpus whose directories carry meaning — Kannada's audio_files and
    trans_files, or a per-speaker tree — must not be flattened."""
    archive = (_zip if suffix == ".zip" else _tar)(tmp_path / f"a{suffix}", CORPUS)

    written = corpus_fetch.extract_archive(archive, tmp_path / "out")

    assert sorted(written) == ["lang/a.wav", "lang/b.wav", "lang/index.tsv"]
    assert (tmp_path / "out" / "lang" / "a.wav").read_bytes() == b"aaaa"


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_flat_extraction_drops_the_directories(
    corpus_fetch: ModuleType, tmp_path: Path, suffix: str
) -> None:
    archive = (_zip if suffix == ".zip" else _tar)(tmp_path / f"a{suffix}", CORPUS)

    written = corpus_fetch.extract_archive(archive, tmp_path / "out", flat=True)

    assert sorted(written) == ["a.wav", "b.wav", "index.tsv"]
    assert (tmp_path / "out" / "a.wav").exists()


def test_a_member_reaching_outside_the_destination_is_dropped(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    """Neither a corrupt archive nor a hostile one is worth guessing the intent
    of, so an escaping member is skipped rather than sanitised into some other
    path."""
    archive = _tar(tmp_path / "a.tar.gz", {"../escape.wav": b"x", "ok.wav": b"y"})

    written = corpus_fetch.extract_archive(archive, tmp_path / "out")

    assert written == ["ok.wav"]
    assert not (tmp_path / "escape.wav").exists()


def test_two_shards_reusing_a_name_is_refused_rather_than_overwritten(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    """The Google crowdsourced sets ship sixteen shards into one directory. An
    overwrite there leaves the row counts and the manifest perfectly consistent
    and the audio wrong, which is the worst kind of failure to have."""
    first = _zip(tmp_path / "0.zip", {"a.wav": b"aaaa"})
    second = _zip(tmp_path / "1.zip", {"a.wav": b"different"})
    seen: set[str] = set()

    corpus_fetch.extract_archive(first, tmp_path / "out", seen=seen)

    with pytest.raises(ValueError, match="already extracted"):
        corpus_fetch.extract_archive(second, tmp_path / "out", seen=seen)


def test_only_the_wanted_members_are_written(corpus_fetch: ModuleType, tmp_path: Path) -> None:
    archive = _zip(tmp_path / "a.zip", CORPUS)

    written = corpus_fetch.extract_archive(archive, tmp_path / "out", members=["lang/index.tsv"])

    assert written == ["lang/index.tsv"]
    assert not (tmp_path / "out" / "lang" / "a.wav").exists()


def test_re_extracting_leaves_an_already_complete_member_alone(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    """These archives are tens of gigabytes; a re-run after an interruption has
    to be cheap or nobody will re-run it."""
    archive = _zip(tmp_path / "a.zip", CORPUS)
    corpus_fetch.extract_archive(archive, tmp_path / "out")
    target = tmp_path / "out" / "lang" / "a.wav"
    before = target.stat().st_mtime_ns

    corpus_fetch.extract_archive(archive, tmp_path / "out")

    assert target.stat().st_mtime_ns == before


# --------------------------------------------------------------------------- #
# Provenance and the Hub
# --------------------------------------------------------------------------- #


def test_the_fetch_manifest_timestamps_what_it_records(
    corpus_fetch: ModuleType, tmp_path: Path
) -> None:
    import json

    path = corpus_fetch.write_fetch_manifest(tmp_path, {"corpus": "demo", "archives": []})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["corpus"] == "demo"
    assert payload["fetched_utc"].endswith("+00:00")


def test_the_hub_path_asks_for_a_dataset_repository_anonymously(
    corpus_fetch: ModuleType, tmp_path: Path, monkeypatch
) -> None:
    """Every corpus here serves anonymously; a gated one fails this project's
    rules and is not in the registry at all, so no token is ever passed."""
    calls: list[dict[str, object]] = []

    def fake_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path / "snap")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    corpus_fetch.snapshot_hf("org/corpus", tmp_path / "snap", allow_patterns=["*.parquet"])

    assert calls[0]["repo_id"] == "org/corpus"
    assert calls[0]["repo_type"] == "dataset"
    assert "token" not in calls[0]


def test_verify_archive_catches_a_corrupt_member(corpus_fetch, tmp_path):
    """Truncating past the end-of-central-directory only proves the file cannot
    be opened. The CRC pass is the check that matters here, because openslr.org
    publishes no checksums and it is the only evidence a member is intact.
    """
    good = _make_archive(tmp_path / "good.zip", ["a_1_1"])
    raw = bytearray(good.read_bytes())

    # Flip a byte inside the stored wav payload, leaving every header and the
    # central directory valid, so the zip opens and only the CRC disagrees.
    offset = raw.index(b"RIFF____WAVEfmt ")
    raw[offset + 4] ^= 0xFF
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(bytes(raw))

    assert "CRC mismatch on member" in corpus_fetch.verify_archive(corrupt)


# --- download and resume ------------------------------------------------------
#
# These archives are hundreds of megabytes each and are fetched unattended, so
# the resume path is the one most likely to leave a silently truncated file.
# Every response here is faked; nothing touches the network.


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = headers or {}

    def read(self, n: int) -> bytes:
        block = self._body[self._offset : self._offset + n]
        self._offset += len(block)
        return block

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        """None, not False: a bool return would imply it can swallow exceptions."""


def _capture_requests(corpus_fetch, monkeypatch, response):
    """Patch urlopen to return `response` and record the Request it was given."""
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return response() if callable(response) else response

    monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)
    return seen


def test_a_fresh_download_writes_the_whole_body(corpus_fetch, tmp_path, monkeypatch):
    seen = _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"abcdef"))
    dest = tmp_path / "a.zip"

    corpus_fetch.download("http://x/a.zip", dest, expected=6)

    assert dest.read_bytes() == b"abcdef"
    assert not dest.with_name("a.zip.part").exists()  # the .part is renamed, not left
    assert seen[0].get_header("Range") is None


def test_a_partial_file_resumes_from_where_it_stopped(corpus_fetch, tmp_path, monkeypatch):
    """A 206 means the server honoured the Range, so the body is the remainder."""
    dest = tmp_path / "a.zip"
    dest.with_name("a.zip.part").write_bytes(b"abc")
    seen = _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"def", status=206))

    corpus_fetch.download("http://x/a.zip", dest, expected=6)

    assert dest.read_bytes() == b"abcdef"
    assert seen[0].get_header("Range") == "bytes=3-"


def test_a_server_that_ignores_range_restarts_instead_of_appending(
    corpus_fetch, tmp_path, monkeypatch
):
    """200 to a Range request means the body is the whole file, not the tail.

    Appending it to the existing prefix would produce a file that is longer than
    the archive and corrupt in the middle.
    """
    dest = tmp_path / "a.zip"
    dest.with_name("a.zip.part").write_bytes(b"abc")
    _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"abcdef", status=200))

    corpus_fetch.download("http://x/a.zip", dest, expected=6)

    assert dest.read_bytes() == b"abcdef"


def test_a_short_body_is_refused_and_leaves_no_destination_file(
    corpus_fetch, tmp_path, monkeypatch
):
    _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"abc"))
    dest = tmp_path / "a.zip"

    with pytest.raises(OSError, match="expected 6"):
        corpus_fetch.download("http://x/a.zip", dest, expected=6)

    assert not dest.exists()


def test_a_complete_file_is_not_downloaded_again(corpus_fetch, tmp_path, monkeypatch):
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"abcdef")
    seen = _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"abcdef"))

    corpus_fetch.download("http://x/a.zip", dest, expected=6)

    assert seen == []  # the network was never touched


def test_an_unknown_expected_size_skips_the_check_and_says_so(
    corpus_fetch, tmp_path, monkeypatch, capsys
):
    """A blocked HEAD leaves no Content-Length, so the size assertion cannot run.

    The zip CRC is still a backstop, but the run should say which check it lost
    rather than appearing to have verified the download.
    """
    _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"abc"))
    dest = tmp_path / "a.zip"

    corpus_fetch.download("http://x/a.zip", dest, expected=None)

    assert dest.read_bytes() == b"abc"
    assert "size unknown" in capsys.readouterr().out


def test_remote_size_reads_the_content_length(corpus_fetch, monkeypatch):
    _capture_requests(
        corpus_fetch, monkeypatch, _FakeResponse(b"", headers={"Content-Length": "1234"})
    )

    assert corpus_fetch.remote_size("http://x/a.zip") == 1234


def test_remote_size_is_none_when_the_head_request_fails(corpus_fetch, monkeypatch):
    def boom(request, timeout=None):
        raise OSError("blocked")

    monkeypatch.setattr(corpus_fetch, "urlopen", boom)

    assert corpus_fetch.remote_size("http://x/a.zip") is None


def test_remote_size_is_none_when_the_server_sends_no_length(corpus_fetch, monkeypatch):
    _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"", headers={}))

    assert corpus_fetch.remote_size("http://x/a.zip") is None


def test_a_complete_part_file_is_adopted_rather_than_re_requested(
    corpus_fetch, tmp_path, monkeypatch
):
    """Killed between the last read and the rename, `.part` holds the whole
    archive and `dest` does not exist. The next run must not ask for bytes past
    the end — the server answers 416 and the fetch dies on an uncaught
    HTTPError, which is the one resume case with no way out."""
    dest = tmp_path / "a.zip"
    dest.with_name("a.zip.part").write_bytes(b"abcdef")
    seen = _capture_requests(corpus_fetch, monkeypatch, _FakeResponse(b"", status=206))

    corpus_fetch.download("http://x/a.zip", dest, expected=6)

    assert dest.read_bytes() == b"abcdef"
    assert seen == []  # nothing was requested at all


def test_a_range_the_server_rejects_restarts_the_download(corpus_fetch, tmp_path, monkeypatch):
    """416 with no expected size to compare against: discard and start over."""
    from urllib.error import HTTPError

    dest = tmp_path / "a.zip"
    dest.with_name("a.zip.part").write_bytes(b"abcdefghij")
    bodies = [
        HTTPError("http://x/a.zip", 416, "Range Not Satisfiable", {}, None),
        _FakeResponse(b"abcdef"),
    ]

    def fake_urlopen(request, timeout=None):
        item = bodies.pop(0)
        if isinstance(item, HTTPError):
            raise item
        return item

    monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)

    corpus_fetch.download("http://x/a.zip", dest)

    assert dest.read_bytes() == b"abcdef"
    assert not bodies  # both responses were used: it retried rather than gave up


def test_an_http_error_that_is_not_a_bad_range_still_propagates(
    corpus_fetch, tmp_path, monkeypatch
):
    """A 404 is not something to paper over by restarting."""
    from urllib.error import HTTPError

    def fake_urlopen(request, timeout=None):
        raise HTTPError("http://x/a.zip", 404, "Not Found", {}, None)

    monkeypatch.setattr(corpus_fetch, "urlopen", fake_urlopen)

    with pytest.raises(HTTPError):
        corpus_fetch.download("http://x/a.zip", tmp_path / "a.zip")
