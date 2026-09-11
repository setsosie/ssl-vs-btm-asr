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

    # A manifest that does not say how much audio it wrote cannot vouch for the
    # directory, so it is refetched rather than trusted.
    (dest / "manifest.json").write_text("{}")
    assert not fetch_openslr.is_complete(dest, names)

    (dest / "manifest.json").write_text(json.dumps({"wav_files": 1}))
    assert not fetch_openslr.is_complete(dest, names)

    (dest / "a_1_1.wav").write_bytes(b"RIFF")
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


# --- config and completeness --------------------------------------------------


def test_two_archives_declaring_one_index_name_are_refused(fetch_openslr, tmp_path):
    """A typo here overwrites one index with the other and collapses the
    manifest's per-index row counts to a single key — the same silent corruption
    the cross-archive name check exists to prevent."""
    config = tmp_path / "heldout.yaml"
    config.write_text(
        "languages:\n"
        "  - code: malayalam\n"
        "    source: openslr\n"
        "    slr: 63\n"
        "    archives: [ml_in_female.zip, ml_in_male.zip]\n"
        "    index_files: [line_index.tsv, line_index.tsv]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index_files"):
        fetch_openslr.read_config(config)


def test_a_language_missing_its_audio_is_not_reported_complete(fetch_openslr, tmp_path):
    """The manifest records how many wavs were written, and nothing compared it
    to the directory — so a pruned corpus was skipped by the fetcher and blew up
    mid-training instead."""
    dest = tmp_path / "SLR63"
    dest.mkdir()
    (dest / "line_index.tsv").write_text("a\tb\n", encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps({"wav_files": 2}), encoding="utf-8")

    assert not fetch_openslr.is_complete(dest, ["line_index.tsv"])

    for name in ("one.wav", "two.wav"):
        (dest / name).write_bytes(b"RIFF")
    assert fetch_openslr.is_complete(dest, ["line_index.tsv"])
