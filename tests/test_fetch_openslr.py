"""fetch_openslr: extraction, manifest, verification. No network in these tests.

The real archives ship a flat layout whose index member is always named
`line_index.tsv`, so extracting a language's male and female archives into one
directory would clobber it. `extract_archive` therefore renames each archive's
index to the name declared in the config.
"""

import json
import re
import zipfile

import pytest


def _make_archive(path, wav_ids, index_name="line_index.tsv", extra=()):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(index_name, "".join(f"{i}\ttranscript {i}\n" for i in wav_ids))
        for i in wav_ids:
            zf.writestr(f"{i}.wav", b"RIFF____WAVEfmt ")
        for name, payload in extra:
            zf.writestr(name, payload)
    return path


# --- extraction ---------------------------------------------------------------


def test_extract_renames_the_index_to_the_declared_name(fetch_openslr, tmp_path):
    src = _make_archive(tmp_path / "ml_in_female.zip", ["mlf_1_1", "mlf_1_2"])
    dest = tmp_path / "SLR63"

    report = fetch_openslr.extract_archive(src, dest, "line_index_female.tsv")

    assert (dest / "line_index_female.tsv").exists()
    assert not (dest / "line_index.tsv").exists()
    assert sorted(p.name for p in dest.glob("*.wav")) == ["mlf_1_1.wav", "mlf_1_2.wav"]
    assert report["wav_files"] == 2
    assert report["index_rows"] == 2


def test_two_archives_of_one_language_coexist_in_the_same_directory(fetch_openslr, tmp_path):
    dest = tmp_path / "SLR63"
    fetch_openslr.extract_archive(
        _make_archive(tmp_path / "f.zip", ["mlf_1_1"]), dest, "line_index_female.tsv"
    )
    fetch_openslr.extract_archive(
        _make_archive(tmp_path / "m.zip", ["mlm_2_1"]), dest, "line_index_male.tsv"
    )

    assert (dest / "line_index_female.tsv").read_text().startswith("mlf_1_1")
    assert (dest / "line_index_male.tsv").read_text().startswith("mlm_2_1")
    assert len(list(dest.glob("*.wav"))) == 2


def test_extract_flattens_paths_so_a_zip_cannot_escape_the_destination(fetch_openslr, tmp_path):
    src = _make_archive(
        tmp_path / "evil.zip", ["a_1_1"], extra=[("../../escaped.wav", b"RIFF"), ("sub/", b"")]
    )
    dest = tmp_path / "SLR63"

    fetch_openslr.extract_archive(src, dest, "line_index.tsv")

    assert not (tmp_path.parent / "escaped.wav").exists()
    assert (dest / "escaped.wav").exists()


def test_extract_rejects_an_archive_without_exactly_one_index(fetch_openslr, tmp_path):
    src = tmp_path / "noindex.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("a_1_1.wav", b"RIFF")
    with pytest.raises(ValueError, match="index"):
        fetch_openslr.extract_archive(src, tmp_path / "SLR63", "line_index.tsv")


def test_extract_is_idempotent(fetch_openslr, tmp_path):
    src = _make_archive(tmp_path / "f.zip", ["mlf_1_1", "mlf_1_2"])
    dest = tmp_path / "SLR63"
    first = fetch_openslr.extract_archive(src, dest, "line_index_female.tsv")
    second = fetch_openslr.extract_archive(src, dest, "line_index_female.tsv")
    assert first == second
    assert len(list(dest.glob("*.wav"))) == 2


# --- verification -------------------------------------------------------------


def test_verify_archive_accepts_a_good_zip_and_rejects_a_truncated_one(fetch_openslr, tmp_path):
    good = _make_archive(tmp_path / "good.zip", ["a_1_1"])
    assert fetch_openslr.verify_archive(good) is None

    bad = tmp_path / "bad.zip"
    bad.write_bytes(good.read_bytes()[:-40])
    assert fetch_openslr.verify_archive(bad) is not None


def test_sha256_is_stable(fetch_openslr, tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"openslr")
    digest = fetch_openslr.sha256_file(p)
    assert len(digest) == 64
    assert digest == fetch_openslr.sha256_file(p)


# --- manifest -----------------------------------------------------------------


def test_manifest_records_provenance_and_metadata_row_counts(fetch_openslr, tmp_path):
    dest = tmp_path / "SLR63"
    src = _make_archive(tmp_path / "ml_in_female.zip", ["mlf_1_1", "mlf_1_2", "mlf_2_1"])
    report = fetch_openslr.extract_archive(src, dest, "line_index_female.tsv")

    fetch_openslr.write_manifest(
        dest,
        code="malayalam",
        slr=63,
        license="CC-BY-SA-4.0",
        archives=[fetch_openslr.archive_record(src, "https://example.invalid/ml_in_female.zip")],
        index_files=["line_index_female.tsv"],
        reports=[report],
    )

    data = json.loads((dest / "manifest.json").read_text())
    assert data["slr"] == 63
    assert data["code"] == "malayalam"
    assert data["license"] == "CC-BY-SA-4.0"
    assert data["index_files"] == ["line_index_female.tsv"]
    assert data["index_rows"] == {"line_index_female.tsv": 3}
    assert data["wav_files"] == 3
    assert data["archives"][0]["name"] == "ml_in_female.zip"
    assert len(data["archives"][0]["sha256"]) == 64
    assert data["archives"][0]["bytes"] == src.stat().st_size
    assert data["downloaded_utc"].endswith("Z")


def test_language_is_complete_only_once_every_index_and_the_manifest_exist(fetch_openslr, tmp_path):
    dest = tmp_path / "SLR63"
    names = ["line_index_female.tsv", "line_index_male.tsv"]
    assert not fetch_openslr.is_complete(dest, names)

    dest.mkdir()
    for n in names:
        (dest / n).write_text("a_1_1\tx\n")
    assert not fetch_openslr.is_complete(dest, names)

    (dest / "manifest.json").write_text("{}")
    assert fetch_openslr.is_complete(dest, names)


# --- config ------------------------------------------------------------------


def test_config_reader_yields_only_openslr_languages(fetch_openslr, pytestconfig):
    cfg = pytestconfig.rootpath / "configs" / "scales" / "heldout.yaml"
    langs = fetch_openslr.read_config(cfg)
    assert [entry["code"] for entry in langs] == ["malayalam", "marathi", "telugu", "gujarati"]
    assert all(len(e["archives"]) == len(e["index_files"]) for e in langs)


def test_archive_url_uses_the_openslr_resources_layout(fetch_openslr):
    url = fetch_openslr.archive_url(63, "ml_in_female.zip")
    assert url == "https://openslr.trmal.net/resources/63/ml_in_female.zip"


# --- cross-archive collisions -------------------------------------------------


def test_a_wav_name_in_two_archives_of_one_language_is_refused(fetch_openslr, tmp_path):
    """The flat layout only works because the archives use disjoint prefixes.

    Male and female archives of a language extract into one directory. The
    design assumes the index is the only name they share, which happens to hold
    because the FileIDs carry an `mlf_`/`mlm_` prefix — but nothing checked it.
    A shared wav name would silently overwrite one archive's audio with the
    other's, leaving the row counts and the manifest entirely consistent.
    """
    dest = tmp_path / "SLR63"
    seen: set[str] = set()
    fetch_openslr.extract_archive(
        _make_archive(tmp_path / "f.zip", ["shared_1_1"]), dest, "line_index_female.tsv", seen=seen
    )

    with pytest.raises(ValueError, match=re.escape("shared_1_1.wav")):
        fetch_openslr.extract_archive(
            _make_archive(tmp_path / "m.zip", ["shared_1_1"]),
            dest,
            "line_index_male.tsv",
            seen=seen,
        )


def test_disjoint_archives_still_share_a_directory(fetch_openslr, tmp_path):
    """The collision check must not reject the layout the real corpora use."""
    dest = tmp_path / "SLR63"
    seen: set[str] = set()
    fetch_openslr.extract_archive(
        _make_archive(tmp_path / "f.zip", ["mlf_1_1"]), dest, "line_index_female.tsv", seen=seen
    )
    fetch_openslr.extract_archive(
        _make_archive(tmp_path / "m.zip", ["mlm_2_1"]), dest, "line_index_male.tsv", seen=seen
    )

    assert len(list(dest.glob("*.wav"))) == 2
    assert seen == {"mlf_1_1.wav", "mlm_2_1.wav"}


def test_the_renamed_index_never_counts_as_a_collision(fetch_openslr, tmp_path):
    """Every archive holds `line_index.tsv`; renaming is the whole point."""
    dest = tmp_path / "SLR63"
    seen: set[str] = set()
    for archive, index in (("f.zip", "line_index_female.tsv"), ("m.zip", "line_index_male.tsv")):
        ids = ["mlf_1_1"] if archive == "f.zip" else ["mlm_2_1"]
        fetch_openslr.extract_archive(
            _make_archive(tmp_path / archive, ids), dest, index, seen=seen
        )

    assert (dest / "line_index_female.tsv").exists()
    assert (dest / "line_index_male.tsv").exists()


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


def _capture_requests(fetch_openslr, monkeypatch, response):
    """Patch urlopen to return `response` and record the Request it was given."""
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return response() if callable(response) else response

    monkeypatch.setattr(fetch_openslr, "urlopen", fake_urlopen)
    return seen


def test_a_fresh_download_writes_the_whole_body(fetch_openslr, tmp_path, monkeypatch):
    seen = _capture_requests(fetch_openslr, monkeypatch, _FakeResponse(b"abcdef"))
    dest = tmp_path / "a.zip"

    fetch_openslr.download("http://x/a.zip", dest, expected=6)

    assert dest.read_bytes() == b"abcdef"
    assert not dest.with_name("a.zip.part").exists()  # the .part is renamed, not left
    assert seen[0].get_header("Range") is None


def test_a_partial_file_resumes_from_where_it_stopped(fetch_openslr, tmp_path, monkeypatch):
    """A 206 means the server honoured the Range, so the body is the remainder."""
    dest = tmp_path / "a.zip"
    dest.with_name("a.zip.part").write_bytes(b"abc")
    seen = _capture_requests(fetch_openslr, monkeypatch, _FakeResponse(b"def", status=206))

    fetch_openslr.download("http://x/a.zip", dest, expected=6)

    assert dest.read_bytes() == b"abcdef"
    assert seen[0].get_header("Range") == "bytes=3-"


def test_a_server_that_ignores_range_restarts_instead_of_appending(
    fetch_openslr, tmp_path, monkeypatch
):
    """200 to a Range request means the body is the whole file, not the tail.

    Appending it to the existing prefix would produce a file that is longer than
    the archive and corrupt in the middle.
    """
    dest = tmp_path / "a.zip"
    dest.with_name("a.zip.part").write_bytes(b"abc")
    _capture_requests(fetch_openslr, monkeypatch, _FakeResponse(b"abcdef", status=200))

    fetch_openslr.download("http://x/a.zip", dest, expected=6)

    assert dest.read_bytes() == b"abcdef"


def test_a_short_body_is_refused_and_leaves_no_destination_file(
    fetch_openslr, tmp_path, monkeypatch
):
    _capture_requests(fetch_openslr, monkeypatch, _FakeResponse(b"abc"))
    dest = tmp_path / "a.zip"

    with pytest.raises(OSError, match="expected 6"):
        fetch_openslr.download("http://x/a.zip", dest, expected=6)

    assert not dest.exists()


def test_a_complete_file_is_not_downloaded_again(fetch_openslr, tmp_path, monkeypatch):
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"abcdef")
    seen = _capture_requests(fetch_openslr, monkeypatch, _FakeResponse(b"abcdef"))

    fetch_openslr.download("http://x/a.zip", dest, expected=6)

    assert seen == []  # the network was never touched


def test_an_unknown_expected_size_skips_the_check_and_says_so(
    fetch_openslr, tmp_path, monkeypatch, capsys
):
    """A blocked HEAD leaves no Content-Length, so the size assertion cannot run.

    The zip CRC is still a backstop, but the run should say which check it lost
    rather than appearing to have verified the download.
    """
    _capture_requests(fetch_openslr, monkeypatch, _FakeResponse(b"abc"))
    dest = tmp_path / "a.zip"

    fetch_openslr.download("http://x/a.zip", dest, expected=None)

    assert dest.read_bytes() == b"abc"
    assert "size unknown" in capsys.readouterr().out


def test_remote_size_reads_the_content_length(fetch_openslr, monkeypatch):
    _capture_requests(
        fetch_openslr, monkeypatch, _FakeResponse(b"", headers={"Content-Length": "1234"})
    )

    assert fetch_openslr.remote_size("http://x/a.zip") == 1234


def test_remote_size_is_none_when_the_head_request_fails(fetch_openslr, monkeypatch):
    def boom(request, timeout=None):
        raise OSError("blocked")

    monkeypatch.setattr(fetch_openslr, "urlopen", boom)

    assert fetch_openslr.remote_size("http://x/a.zip") is None


def test_remote_size_is_none_when_the_server_sends_no_length(fetch_openslr, monkeypatch):
    _capture_requests(fetch_openslr, monkeypatch, _FakeResponse(b"", headers={}))

    assert fetch_openslr.remote_size("http://x/a.zip") is None
