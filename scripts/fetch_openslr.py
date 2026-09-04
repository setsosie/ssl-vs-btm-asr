#!/usr/bin/env python3
"""Download and extract the held-out OpenSLR corpora described in a scale config.

    python scripts/fetch_openslr.py --root $OPENSLR_ROOT
    python scripts/fetch_openslr.py --root $OPENSLR_ROOT --langs malayalam marathi

Standalone by design: it imports nothing from `svb`, only the standard library
and PyYAML, so the data can be staged on a machine that has no project install.

Layout it produces, matching what `svb.data.openslr_local` expects::

    $OPENSLR_ROOT/
        .archives/SLR63/ml_in_female.zip          # kept, so a rerun is free
        SLR63/line_index_female.tsv
        SLR63/line_index_male.tsv
        SLR63/mlf_02879_01795762363.wav ...
        SLR63/manifest.json

Two details of the real archives drive the design. Every archive is flat and
holds its own index member named `line_index.tsv`, so extracting a language's
male and female archives into one directory would clobber it; each archive's
index is therefore written under the name the config declares for it, and
`archives` and `index_files` are parallel lists. And openslr.org publishes no
checksums for these resources, so an archive is verified by its byte length
against the server's Content-Length and by a full zip CRC check; the SHA-256 is
recorded in the manifest for later reference rather than compared to a
published value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import UTC, datetime  # `datetime.UTC` needs 3.11; this repo targets 3.10
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml  # type: ignore[import-untyped]  # drop once types-PyYAML is a dev dep

DEFAULT_MIRROR = "https://openslr.trmal.net"
# Alternates if the primary is slow or down; same paths under /resources/<slr>/.
MIRRORS = (DEFAULT_MIRROR, "https://openslr.elda.org", "https://openslr.magicdatatech.com")
CHUNK = 1 << 20
_UA = {"User-Agent": "ssl-vs-btm-asr/0.1 (openslr fetch script)"}


# --- config -------------------------------------------------------------------


def default_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "scales" / "heldout.yaml"


def read_config(path: Path) -> list[dict[str, Any]]:
    """Return the `openslr` language entries of a scale config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = [e for e in raw.get("languages", []) if e.get("source") == "openslr"]
    for entry in entries:
        if len(entry.get("archives", [])) != len(entry.get("index_files", [])):
            raise ValueError(
                f"{entry.get('code')}: archives and index_files must be parallel lists"
            )
    return entries


def archive_url(slr: int, name: str, mirror: str = DEFAULT_MIRROR) -> str:
    return f"{mirror.rstrip('/')}/resources/{slr}/{name}"


# --- verification -------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path) -> str | None:
    """None if the zip is intact, else a one-line reason it is not."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
    except (zipfile.BadZipFile, OSError) as exc:
        return f"unreadable zip: {exc}"
    return None if bad is None else f"CRC mismatch on member {bad}"


def archive_record(path: Path, url: str) -> dict[str, Any]:
    return {
        "name": path.name,
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


# --- download -----------------------------------------------------------------


def remote_size(url: str) -> int | None:
    try:
        with urlopen(Request(url, headers=_UA, method="HEAD"), timeout=60) as resp:
            length = resp.headers.get("Content-Length")
    except (HTTPError, OSError):
        return None
    return int(length) if length else None


def download(url: str, dest: Path, *, expected: int | None = None) -> Path:
    """Stream `url` to `dest`, resuming a partial `.part` file when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected is not None and dest.exists() and dest.stat().st_size == expected:
        print(f"    have {dest.name} ({expected} bytes)")
        return dest

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = Request(url, headers=dict(_UA))
    if have:
        request.add_header("Range", f"bytes={have}-")

    with urlopen(request, timeout=120) as resp:
        # A server that ignores Range answers 200 with the whole body; restart.
        resuming = have > 0 and resp.status == 206
        if have and not resuming:
            have = 0
        with open(part, "ab" if resuming else "wb") as out:
            print(f"    {'resuming' if resuming else 'downloading'} {dest.name} from byte {have}")
            while block := resp.read(CHUNK):
                out.write(block)

    if expected is not None and part.stat().st_size != expected:
        raise OSError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected}")
    part.replace(dest)
    return dest


# --- extraction ---------------------------------------------------------------


def extract_archive(
    archive: Path, dest: Path, index_name: str, seen: set[str] | None = None
) -> dict[str, Any]:
    """Extract one archive flat into `dest`, renaming its index to `index_name`.

    Only member basenames are used, so a crafted archive cannot write outside
    `dest`. Returns the counts the manifest reports.

    `seen` accumulates the names written for one language across its archives.
    A language's male and female archives share a directory, and the whole flat
    layout rests on the index being the only name they have in common — true of
    the real corpora only because the FileIDs carry an archive prefix. Pass a
    shared set and a second archive reusing a name is refused instead of
    overwriting the first archive's audio, which would leave the row counts and
    the manifest perfectly consistent and the corpus wrong.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and PurePosixPath(m.filename).name]
        indices = [
            m
            for m in members
            if PurePosixPath(m.filename).name.startswith("line_index")
            and m.filename.endswith(".tsv")
        ]
        if len(indices) != 1:
            raise ValueError(
                f"{archive.name}: expected exactly one line_index*.tsv member, found "
                f"{[m.filename for m in indices]}"
            )

        index_member = indices[0]
        wavs = 0
        for member in members:
            name = PurePosixPath(member.filename).name
            # The index is renamed per archive, so it is expected to repeat and
            # is never a collision; every other name must be unique.
            written = index_name if member is index_member else name
            if seen is not None and member is not index_member:
                if written in seen:
                    raise ValueError(
                        f"{archive.name}: {written!r} was already extracted from another "
                        "archive of this language; the archives are supposed to use disjoint "
                        "file ids, and overwriting would silently replace that audio"
                    )
                seen.add(written)
            target = dest / written
            if target.exists() and target.stat().st_size == member.file_size:
                wavs += name.endswith(".wav")
                continue
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            wavs += name.endswith(".wav")

    rows = sum(1 for line in (dest / index_name).read_text(encoding="utf-8").splitlines() if line)
    return {
        "archive": archive.name,
        "index_file": index_name,
        "index_rows": rows,
        "wav_files": wavs,
    }


# --- manifest -----------------------------------------------------------------


def write_manifest(
    dest: Path,
    *,
    code: str,
    slr: int,
    license: str,
    archives: list[dict[str, Any]],
    index_files: list[str],
    reports: list[dict[str, Any]],
) -> Path:
    """Record what was fetched, so `check_data.sh` never has to decode audio."""
    manifest = {
        "code": code,
        "slr": slr,
        "license": license,
        "source": f"https://openslr.org/{slr}/",
        "downloaded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "archives": archives,
        "index_files": list(index_files),
        "index_rows": {r["index_file"]: r["index_rows"] for r in reports},
        "wav_files": sum(r["wav_files"] for r in reports),
    }
    path = dest / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def is_complete(dest: Path, index_files: list[str]) -> bool:
    """True when every declared index and the manifest are already on disk."""
    return (dest / "manifest.json").exists() and all((dest / n).exists() for n in index_files)


# --- driver -------------------------------------------------------------------


def fetch_language(entry: dict[str, Any], root: Path, *, mirror: str, keep_archives: bool) -> None:
    slr, code = int(entry["slr"]), str(entry["code"])
    index_files = list(entry["index_files"])
    dest = root / f"SLR{slr}"

    if is_complete(dest, index_files):
        print(f"  {code} (SLR{slr}): already extracted, skipping")
        return

    print(f"  {code} (SLR{slr}): {entry.get('license', 'licence unstated')}")
    archive_dir = root / ".archives" / f"SLR{slr}"
    records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    # Shared across this language's archives so a name written by one is
    # refused by the next rather than overwritten.
    seen: set[str] = set()

    for name, index_name in zip(entry["archives"], index_files, strict=True):
        url = archive_url(slr, name, mirror)
        path = download(url, archive_dir / name, expected=remote_size(url))
        # openslr.org publishes no checksums for these resources, so the zip's
        # own CRCs are the integrity check.
        problem = verify_archive(path)
        if problem is not None:
            path.unlink(missing_ok=True)
            raise SystemExit(f"{name}: {problem} — deleted; rerun to download again")
        records.append(archive_record(path, url))
        reports.append(extract_archive(path, dest, index_name, seen=seen))
        print(f"    extracted {reports[-1]['wav_files']} wavs, {reports[-1]['index_rows']} rows")

    write_manifest(
        dest,
        code=code,
        slr=slr,
        license=str(entry.get("license", "")),
        archives=records,
        index_files=index_files,
        reports=reports,
    )
    if not keep_archives:
        shutil.rmtree(archive_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--root",
        default=os.environ.get("OPENSLR_ROOT"),
        help="extraction root; defaults to $OPENSLR_ROOT",
    )
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--langs", nargs="*", help="language codes; default all in the config")
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, choices=MIRRORS)
    parser.add_argument(
        "--keep-archives", action="store_true", help="do not delete the zips after extracting"
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = parser.parse_args(argv)

    if not args.root:
        parser.error("pass --root or set $OPENSLR_ROOT")

    entries = read_config(args.config)
    if args.langs:
        wanted = set(args.langs)
        unknown = wanted - {e["code"] for e in entries}
        if unknown:
            parser.error(f"unknown language(s): {sorted(unknown)}")
        entries = [e for e in entries if e["code"] in wanted]

    root = Path(args.root)
    print(f"openslr root: {root}")
    if args.dry_run:
        for entry in entries:
            for name in entry["archives"]:
                print(f"  would fetch {archive_url(int(entry['slr']), name, args.mirror)}")
        return 0

    for entry in entries:
        fetch_language(entry, root, mirror=args.mirror, keep_archives=args.keep_archives)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
