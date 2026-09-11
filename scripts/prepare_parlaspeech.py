"""Prepare ParlaSpeech-HR v1.0 (Croatian) into the manifest layer.

    python scripts/prepare_parlaspeech.py --root $CORPORA_ROOT

One language, `hr`, from CLARIN.SI handle 11356/1494. Publicly available, no
login: the repository labels the item ``PUB`` and every bitstream serves to an
anonymous client.

**What is fetched**, resolved from the item's METS record at
``/repository/xmlui/metadata/handle/11356/1494/mets.xml`` rather than from
hard-coded URLs, because that document also carries a published MD5 and byte
size for each file:

===============================  ===============  =========================
File                             Bytes            What it is
===============================  ===============  =========================
``ParlaSpeech-HR.v1.0.jsonl``        712,525,343  every segment's metadata
``ParlaSpeech-HR.flac.tgz.0``     52,428,800,000  audio, slice 0
``ParlaSpeech-HR.flac.tgz.1``     52,428,800,000  audio, slice 1
``ParlaSpeech-HR.flac.tgz.2``     20,321,039,259  audio, slice 2
===============================  ===============  =========================

The three ``.tgz.N`` are **byte slices of one gzip stream**, not three archives:
slice 1 and slice 2 do not begin with a gzip header, and the first two are
exactly 50,000 MiB each. So they are read back to back through one reader and
the tar is extracted from that stream, which is also why the 116 GB never has to
be written out a second time as a joined file.

**The transcripts** are one JSON object per line. From the README the repository
ships beside them::

    path: name of the file with the segment recording
    words: list of words from the original transcript
    norm_words: list of words normalized with an imperfect rule-based normaliser
    speaker_info: list of speaker attributes from ParlaMint 2.1, if single
                  speaker (null otherwise)
    split: either "train", "dev", or "test", or "null" if multiple speakers

and one real record's `path` is ``seg.qNpeHxO0WzA_1967.6-1986.11.flac``, which is
also the member name in the tar — the archive is flat, so the two line up without
any mapping. ``text`` is joined from ``words``, the original transcript, not from
``norm_words``: this repository normalizes text under its own per-language policy
at collate time, and folding a corpus-specific normaliser in first would make
Croatian the one language whose transcripts had been through two.

**Split.** All three ship and are used as shipped. Dev is 500 segments from the
five most frequent speakers, test is 513 from three male and three female
speakers, and the repository states there are no segments from the six test
speakers anywhere else. The 22,076 segments that span more than one speaker carry
no speaker and no split; they are dropped rather than trained on, because the
manifest layer refuses a partition that is part shipped and part derived, and an
empty speaker on any row would drop the whole language to an utterance-level
split.

Audio is FLAC, 16 kHz 16-bit mono — read out of one file's STREAMINFO block, as
the repository does not state it — and the preparer records the rate it actually
found. Segment durations come from the ``start`` and ``end`` fields, so the hours
per split are measured rather than copied, and no audio is decoded to get them.

**Disk.** About 116 GB of archive plus about 116 GB of extracted FLAC. Pass
``--keep-archives`` only if there is room for both afterwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, cast
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from corpus_fetch import CHUNK, UA, archive_record, download, write_fetch_manifest

CORPUS_ID = "parlaspeech_hr"
CORPUS_IDS = (CORPUS_ID,)
LANGUAGES = ("hr",)
HANDLE = "11356/1494"
REPOSITORY = "https://www.clarin.si"
METS_URL = f"{REPOSITORY}/repository/xmlui/metadata/handle/{HANDLE}/mets.xml"
MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")

JSONL = "ParlaSpeech-HR.v1.0.jsonl"
SLICES = ("ParlaSpeech-HR.flac.tgz.0", "ParlaSpeech-HR.flac.tgz.1", "ParlaSpeech-HR.flac.tgz.2")

#: What the repository says the licence is. A change is a decision for a person.
EXPECTED_LICENCE = "Attribution-ShareAlike 4.0"

#: The corpus's own split names, mapped onto the manifest's.
SPLIT_NAMES = {"train": "train", "dev": "validation", "test": "test"}

#: Segment counts the item page publishes: 380,836 train, 500 dev over the five
#: most frequent speakers, 513 test over three male and three female speakers,
#: and 22,076 multi-speaker segments in no split at all. Checked rather than
#: recorded, because a silently different corpus is the failure that survives
#: all the way into a results table.
PUBLISHED_COUNTS = {"train": 380836, "validation": 500, "test": 513, "unassigned": 22076}


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    path: str
    text: str
    speaker: str
    split: str
    seconds: float


# --- the CLARIN.SI item record ------------------------------------------------


def fetch_mets(url: str = METS_URL) -> str:
    request = Request(url, headers={**UA, "Accept": "application/xml"})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_licence(mets: ElementTree.Element) -> str:
    """``dim:field element="rights"``, which is where CLARIN.SI states it."""
    for field in mets.iter():
        rights = _local(field.tag) == "field" and field.get("element") == "rights"
        if rights and field.get("qualifier") is None and (field.text or "").strip():
            return str(field.text).strip()
    return ""


def href_of(element: ElementTree.Element) -> str | None:
    """An element's ``xlink:href``, whatever prefix the document bound it to.

    This repository declares xlink as ``http://www.w3.org/TR/xlink/`` rather than
    the usual ``1999`` namespace, so matching the qualified name would work here
    and break on the next repository. The local name is the stable part.
    """
    for name, value in element.attrib.items():
        if _local(name) == "href":
            return value
    return None


def read_bitstreams(mets: ElementTree.Element) -> dict[str, dict[str, Any]]:
    """``{file name: {url, bytes, md5}}`` for every bitstream in the item.

    Matched on carrying a location rather than on the tag alone: the README
    bitstream nests its own text inside a ``<mets:Local><mets:file>`` element,
    which has the same local name as a real entry and no href. The href is
    relative and carries a ``sequence`` query, so the file name comes from the
    path rather than from the whole URL.
    """
    found: dict[str, dict[str, Any]] = {}
    for element in mets.iter():
        if _local(element.tag) != "file":
            continue
        href = href_of(element)
        for child in element:
            if _local(child.tag) == "FLocat":
                href = href_of(child) or href
        if not href:
            continue
        url = urljoin(REPOSITORY, href)
        name = PurePosixPath(url.split("?", 1)[0]).name
        size = element.get("SIZE")
        found[name] = {
            "url": url,
            "bytes": int(size) if size and size.isdigit() else None,
            "md5": (element.get("CHECKSUM") or "").lower()
            if (element.get("CHECKSUMTYPE") or "").upper() == "MD5"
            else None,
        }
    return found


def check_licence(licence: str) -> str:
    if EXPECTED_LICENCE not in licence:
        raise ValueError(
            f"CLARIN.SI now states the licence as {licence!r}, which does not contain "
            f"{EXPECTED_LICENCE!r}. configs/corpora.yaml records CC BY-SA 4.0 for this "
            "corpus; settle the difference before ingesting it."
        )
    return licence


def md5_file(path: Path) -> str:
    # MD5 because that is what CLARIN.SI publishes. Compared against a value from
    # the same repository, so it catches a truncated or corrupted transfer and
    # claims nothing beyond that.
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_bitstream(name: str, record: dict[str, Any], staging: Path) -> Path:
    """Download one bitstream and check it against the published MD5."""
    path = download(record["url"], staging / name, expected=record.get("bytes"))
    published = record.get("md5")
    if published:
        got = md5_file(path)
        if got != published:
            path.unlink(missing_ok=True)
            raise OSError(
                f"{name}: MD5 {got} does not match the {published} CLARIN.SI publishes; "
                "deleted, re-run to fetch it again"
            )
    return path


# --- the audio ----------------------------------------------------------------


class SlicedReader:
    """The three ``.tgz.N`` slices read back to back as one stream.

    They are pieces of a single gzip stream — only the first carries a gzip
    header — so decompressing them separately fails and joining them into one
    116 GB file before extracting would double the disk this corpus needs.
    ``tarfile`` in stream mode asks only for ``read``.
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = list(paths)
        self._index = 0
        self._handle: IO[bytes] | None = None

    def _current(self) -> IO[bytes] | None:
        if self._handle is None and self._index < len(self._paths):
            # Deliberately not a context manager: the handle has to outlive this
            # call and is closed either when its slice runs out or in `close`.
            self._handle = open(self._paths[self._index], "rb")  # noqa: SIM115
        return self._handle

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining != 0:
            handle = self._current()
            if handle is None:
                break
            block = handle.read(remaining if remaining > 0 else CHUNK)
            if not block:
                handle.close()
                self._handle = None
                self._index += 1
                if remaining < 0 and self._index >= len(self._paths):
                    break
                continue
            chunks.append(block)
            if remaining > 0:
                remaining -= len(block)
        return b"".join(chunks)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def extract_audio(slices: list[Path], audio_dir: Path) -> set[str]:
    """Stream the joined archive into `audio_dir`, returning the names written.

    The tar is flat, so every member is written by basename; a second member
    claiming a name already taken is refused rather than overwritten, which would
    leave the counts and the manifest consistent and the audio wrong.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    reader = SlicedReader(slices)
    try:
        # `r|` is tarfile's stream mode and asks the file object for nothing but
        # `read`; the stub's file-object type still wants the whole IO surface.
        stream = cast("IO[bytes]", reader)
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name).name
                if not name:
                    continue
                if name in written:
                    raise ValueError(
                        f"{name!r} appears twice in the archive; the segment names are "
                        "supposed to be unique and overwriting would silently replace audio"
                    )
                written.add(name)
                target = audio_dir / name
                if target.exists() and target.stat().st_size == member.size:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                with open(target, "wb") as out:
                    shutil.copyfileobj(source, out, CHUNK)
    finally:
        reader.close()
    return written


def flac_sample_rate(path: Path) -> int | None:
    """The rate in a FLAC file's STREAMINFO block, without decoding anything.

    The repository does not state a sample rate, and this is four fields into the
    first metadata block, so it is cheaper to read than to assume.
    """
    with open(path, "rb") as handle:
        if handle.read(4) != b"fLaC":
            return None
        handle.read(4)  # metadata block header
        streaminfo = handle.read(34)
    if len(streaminfo) < 18:
        return None
    return int.from_bytes(streaminfo[10:18], "big") >> 44


# --- the transcripts ----------------------------------------------------------


def read_jsonl(path: Path) -> tuple[list[Utterance], dict[str, int]]:
    """Every segment that carries both a speaker and a split, and why the rest did not."""
    utterances: list[Utterance] = []
    skipped = {"unassigned": 0, "no_speaker": 0, "no_text": 0}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            split = SPLIT_NAMES.get(str(record.get("split") or ""))
            speaker_info = record.get("speaker_info") or {}
            speaker = str(speaker_info.get("Speaker_name") or "").strip()
            text = " ".join(" ".join(record.get("words") or []).split())
            if split is None:
                skipped["unassigned"] += 1
                continue
            if not speaker:
                skipped["no_speaker"] += 1
                continue
            if not text:
                skipped["no_text"] += 1
                continue
            name = str(record["path"])
            utterances.append(
                Utterance(
                    utt_id=PurePosixPath(name).stem,
                    path=f"audio/{name}",
                    text=text,
                    speaker=speaker,
                    split=split,
                    seconds=max(
                        float(record.get("end", 0.0)) - float(record.get("start", 0.0)), 0.0
                    ),
                )
            )
    return utterances, skipped


def check_counts(
    utterances: list[Utterance], skipped: dict[str, int], expected: dict[str, int] | None
) -> dict[str, int]:
    counts = {name: sum(1 for u in utterances if u.split == name) for name in SPLIT_NAMES.values()}
    counts["unassigned"] = skipped["unassigned"]
    if expected is not None and counts != expected:
        raise ValueError(
            f"ParlaSpeech-HR: read {counts} segments, but the repository publishes {expected}. "
            "Either the release changed or the download is incomplete; both need a person."
        )
    return counts


def write_manifest(dest: Path, utterances: list[Utterance]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for utterance in sorted(utterances, key=lambda u: u.utt_id):
            writer.writerow(
                [
                    utterance.utt_id,
                    utterance.path,
                    utterance.text,
                    utterance.speaker,
                    utterance.split,
                ]
            )
    return path


# --- the preparer -------------------------------------------------------------


def prepare(
    root: Path,
    *,
    keep_archives: bool = False,
    expected_counts: dict[str, int] | None = None,
) -> Path:
    """Fetch, extract and write the manifest for Croatian."""
    dest = root / CORPUS_ID / "hr"
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] hr: {CORPUS_ID} (CLARIN.SI {HANDLE})")
    mets = ElementTree.fromstring(fetch_mets())
    licence = check_licence(read_licence(mets))
    print(f"    licence: {licence}")

    bitstreams = read_bitstreams(mets)
    missing_files = [name for name in (JSONL, *SLICES) if name not in bitstreams]
    if missing_files:
        raise LookupError(
            f"the item record no longer lists {missing_files}; it lists "
            f"{sorted(bitstreams)}. The release layout has changed."
        )

    index = fetch_bitstream(JSONL, bitstreams[JSONL], staging)
    slices = [fetch_bitstream(name, bitstreams[name], staging) for name in SLICES]
    records = [archive_record(path, bitstreams[path.name]["url"]) for path in [index, *slices]]

    written = extract_audio(slices, dest / "audio")
    print(f"    {len(written)} segment recordings extracted")
    if not keep_archives:
        for path in slices:
            path.unlink(missing_ok=True)

    utterances, skipped = read_jsonl(index)
    counts = check_counts(utterances, skipped, expected_counts)
    if not keep_archives:
        index.unlink(missing_ok=True)

    # Drop rows whose audio is not on disk before writing, rather than letting
    # the loader discover an unresolvable path hours into a run.
    kept = [u for u in utterances if PurePosixPath(u.path).name in written]
    manifest = write_manifest(dest, kept)
    # Seconds is what was measured; hours is the readable form of it. Both are
    # recorded because rounding hours to two places loses a small corpus
    # entirely, and the audit is supposed to survive a small corpus.
    seconds = {
        name: round(sum(u.seconds for u in kept if u.split == name), 3)
        for name in SPLIT_NAMES.values()
    }
    hours = {name: round(value / 3600, 2) for name, value in seconds.items()}
    print(
        f"    {len(kept)} utterances written, hours {hours}, {len(utterances) - len(kept)} "
        "rows without audio"
    )

    sample = next(iter(sorted(written)), None)
    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_ID,
            "language": "hr",
            "handle": HANDLE,
            "source_page": f"https://www.clarin.si/repository/xmlui/handle/{HANDLE}",
            "licence": licence,
            "split_policy": "shipped",
            "published_counts": PUBLISHED_COUNTS,
            "counts": counts,
            "seconds": seconds,
            "hours": hours,
            "speakers": {
                name: len({u.speaker for u in kept if u.split == name})
                for name in SPLIT_NAMES.values()
            },
            "rows_without_audio": len(utterances) - len(kept),
            "rows_skipped": skipped,
            "sample_rate_hz": flac_sample_rate(dest / "audio" / sample) if sample else None,
            "transcript_field": "words (the original transcript, not norm_words)",
            "archives": records,
        },
    )
    if not keep_archives and staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="$CORPORA_ROOT")
    parser.add_argument(
        "--langs", nargs="*", choices=LANGUAGES, help="only 'hr'; the flag exists for symmetry"
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep the slices so a re-extract is free"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for _ in args.langs or LANGUAGES:
        prepare(args.root, keep_archives=args.keep_archives, expected_counts=PUBLISHED_COUNTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
