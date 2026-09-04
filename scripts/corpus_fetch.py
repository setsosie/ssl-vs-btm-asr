"""Fetching primitives shared by every corpus preparer.

Downloading tens of gigabytes over a link that drops is the part of corpus
preparation that goes wrong, and it goes wrong the same way for every corpus.
This module holds that part once: resumable download, integrity checking,
archive extraction for zip and tar, and a Hugging Face snapshot path for the
corpora distributed that way.

It is deliberately dependency-light — the standard library plus, only on the
Hugging Face path, ``huggingface_hub``. A preparer can then stage data on a
machine with no project install, which is usually the machine with the disk.

What it does **not** do is decide what a corpus means. Turning archives into
``manifest.tsv`` is each ``scripts/prepare_<corpus>.py``'s job, because that is
where the per-corpus knowledge belongs. See ``docs/data.md``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CHUNK = 1 << 20
UA = {"User-Agent": "ssl-vs-btm-asr/0.1 (corpus fetch)"}

#: Suffixes handled by :func:`extract_archive`, longest first so that
#: ``.tar.gz`` is matched before ``.gz``.
ZIP_SUFFIXES = (".zip",)
TAR_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar")


def archive_kind(path: Path) -> str:
    """``"zip"``, ``"tar"``, or a failure naming what was passed."""
    name = path.name.lower()
    if name.endswith(ZIP_SUFFIXES):
        return "zip"
    if name.endswith(TAR_SUFFIXES):
        return "tar"
    raise ValueError(
        f"{path.name}: not an archive this fetcher opens; expected one of "
        f"{[*ZIP_SUFFIXES, *TAR_SUFFIXES]}"
    )


# --- integrity ----------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path) -> str | None:
    """None if the archive reads end to end, else a one-line reason it does not.

    A zip carries a CRC per member, so this is a real integrity check. A tar
    carries no checksum at all, so the most that can be said is that every
    member header and body could be read — worth doing, because a truncated
    download fails it, but it is not a CRC and the caller should not report it
    as one.
    """
    try:
        if archive_kind(path) == "zip":
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
            return None if bad is None else f"CRC mismatch on member {bad}"
        with tarfile.open(path) as tar:
            for member in tar:
                if member.isfile():
                    handle = tar.extractfile(member)
                    if handle is None:
                        return f"unreadable member {member.name}"
                    while handle.read(CHUNK):
                        pass
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as exc:
        # EOFError is what a truncated .tar.gz raises: gzip hits the end before
        # the stream marker. It is not an OSError, so leaving it out turned the
        # commonest download failure into a traceback instead of a report.
        return f"unreadable archive: {exc}"
    return None


def archive_record(path: Path, url: str) -> dict[str, Any]:
    """What the fetch manifest records about one downloaded archive."""
    return {
        "name": path.name,
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


# --- download -----------------------------------------------------------------


def remote_size(url: str) -> int | None:
    """Content-Length, or None when the server does not send one."""
    try:
        with urlopen(Request(url, headers=UA, method="HEAD"), timeout=60) as response:
            length = response.headers.get("Content-Length")
    except (HTTPError, OSError):
        return None
    return int(length) if length else None


#: A resume offset at or past the end of the resource. Reachable without any
#: mistake: an interrupted run can leave a `.part` that is already complete.
_RANGE_NOT_SATISFIABLE = 416


def _ranged_request(url: str, have: int) -> Request:
    request = Request(url, headers=dict(UA))
    if have:
        request.add_header("Range", f"bytes={have}-")
    return request


def download(url: str, dest: Path, *, expected: int | None = None) -> Path:
    """Stream `url` to `dest`, resuming a partial `.part` file when possible.

    Three resume outcomes, because all three happen on a cluster: the server
    honours the Range (206, append), ignores it (200, restart), or rejects the
    offset as past the end (416, discard the `.part` and start over).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected is not None and dest.exists() and dest.stat().st_size == expected:
        print(f"    have {dest.name} ({expected} bytes)")
        return dest

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    if expected is not None and have >= expected:
        # Killed between the last read and the rename: `.part` is the whole
        # archive and `dest` does not exist. Asking for bytes from the end would
        # get a 416, so adopt what is already here and let the integrity check
        # judge it.
        print(f"    {dest.name}: complete .part adopted ({have} bytes)")
        part.replace(dest)
        return dest

    try:
        response_ctx = urlopen(_ranged_request(url, have), timeout=120)
    except HTTPError as exc:
        if exc.code != _RANGE_NOT_SATISFIABLE or not have:
            raise
        print(f"    {dest.name}: server rejected the resume offset; restarting")
        part.unlink(missing_ok=True)
        have = 0
        response_ctx = urlopen(_ranged_request(url, 0), timeout=120)

    with response_ctx as response:
        # A server that ignores Range answers 200 with the whole body; restart.
        resuming = have > 0 and response.status == 206
        if have and not resuming:
            have = 0
        with open(part, "ab" if resuming else "wb") as out:
            print(f"    {'resuming' if resuming else 'downloading'} {dest.name} from byte {have}")
            while block := response.read(CHUNK):
                out.write(block)

    if expected is None:
        # No Content-Length, so the length check cannot run and the archive's own
        # integrity check is the only evidence left. Say which check was lost
        # rather than letting the download look fully verified.
        print(f"    {dest.name}: size unknown (no Content-Length), relying on the archive check")
    elif part.stat().st_size != expected:
        raise OSError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected}")
    part.replace(dest)
    return dest


# --- extraction ---------------------------------------------------------------


def _safe_relative(name: str, flat: bool) -> str | None:
    """The path a member is written to, or None to skip it.

    Absolute paths and ``..`` segments are dropped rather than sanitised: a
    member that tries to escape the destination is a corrupt or hostile archive,
    and neither is worth guessing the intent of.
    """
    pure = PurePosixPath(name)
    if flat:
        return pure.name or None
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        return None
    return str(pure) or None


def extract_archive(
    archive: Path,
    dest: Path,
    *,
    flat: bool = False,
    members: Iterable[str] | None = None,
    seen: set[str] | None = None,
) -> list[str]:
    """Extract `archive` into `dest`, returning the relative paths written.

    Args:
        flat: Write every member by basename, discarding its directories. The
            OpenSLR crowdsourced sets need this; corpora whose directory layout
            carries meaning must not have it.
        members: Restrict to these archive-relative names, for the corpora where
            one archive holds far more than the preparer wants.
        seen: Accumulates names across the shards of one language, so a second
            shard reusing a name is refused instead of overwriting the first.
            Silently overwriting would leave the counts and the manifest
            perfectly consistent and the corpus wrong.

    A member already present at the expected size is left alone, which makes a
    re-run after an interrupted extraction cheap.
    """
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(members) if members is not None else None
    written: list[str] = []

    def record(name: str, size: int, read: Any) -> None:
        relative = _safe_relative(name, flat)
        if relative is None or (wanted is not None and name not in wanted):
            return
        if seen is not None:
            if relative in seen:
                raise ValueError(
                    f"{archive.name}: {relative!r} was already extracted from another archive "
                    "of this language; the archives are supposed to use disjoint names, and "
                    "overwriting would silently replace that audio"
                )
            seen.add(relative)
        target = dest / relative
        written.append(relative)
        if target.exists() and target.stat().st_size == size:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with read() as source, open(target, "wb") as out:
            shutil.copyfileobj(source, out, CHUNK)

    if archive_kind(archive) == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    record(info.filename, info.file_size, lambda i=info: zf.open(i))
    else:
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                record(member.name, member.size, lambda h=handle: h)
    return written


# --- Hugging Face -------------------------------------------------------------


def snapshot_hf(repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None) -> Path:
    """Download a non-gated Hugging Face dataset repository to `dest`.

    Only for repositories that serve anonymously. A gated repository fails
    criterion (a) of this project's corpus rules and is not in
    ``configs/corpora.yaml``; if one of these ever becomes gated, the right
    response is to drop the language, not to add a token.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised by the message, not the path
        raise ImportError(
            "huggingface_hub is needed for corpora distributed through the Hub. "
            "Install the project (`uv sync`) or `pip install 'huggingface-hub>=0.23,<1'`."
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
    )
    return Path(path)


# --- provenance ---------------------------------------------------------------


def write_fetch_manifest(dest: Path, payload: dict[str, Any]) -> Path:
    """Record what was fetched, beside what was fetched.

    ``fetched_utc`` and the per-archive SHA-256 are the only evidence later that
    two runs read the same bytes: none of these corpora publishes a checksum, so
    the digest is recorded for comparison rather than checked against anything.
    """
    path = dest / "fetch_manifest.json"
    payload = {"fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
