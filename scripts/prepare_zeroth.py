"""Prepare Zeroth-Korean into the manifest layer.

Corpus id: slr40_zeroth_korean. Language: ko. Source: https://www.openslr.org/40/
Licence: Attribution 4.0 International (CC BY 4.0), verbatim from the page.

One archive, `zeroth_korean.tar.gz` (10 G). Layout, read directly off the first
tar members::

    AUDIO_INFO
    train_data_01/<script>/<speaker>/<speaker>_<script>_<n>.flac
    train_data_01/<script>/<speaker>/<speaker>_<script>.trans.txt
    test_data_01/...

`AUDIO_INFO` is pipe-separated with the header
`SPEAKERID|NAME|SEX|SCRIPTID|DATASET`, where `DATASET` is `train_data_01` or
`test_data_01`. The transcripts are LibriSpeech-shaped: one `.trans.txt` per
speaker-and-script directory, each line `<utterance id> <transcript>`.

**`NAME` holds contributors' personal names.** It is read only to locate the
header and is never written to the manifest, the fetch record or any log; the
numeric `SPEAKERID` is the only speaker identity this preparer carries.

**The split is re-derived, and the published figures stop describing it.** The
corpus ships 51.6 h of train over 105 speakers and 1.2 h of test over 10 — the
test side is under this project's 2 h evaluation bar. So `split` is left empty
for every row and the shared speaker-disjoint derivation repartitions all 115
speakers, which puts validation and test both over the bar. The shipped test
speakers are not discarded; they simply enter the same pool. The cost is that
roughly a fifth of the 51.6 published training hours now sit in validation and
test, and any write-up has to say the split was re-derived rather than quote
51.6/1.2.

The realised hours per derived split are measured here and recorded in the fetch
manifest, because after re-deriving there is no published number to quote. FLAC
durations are read out of the STREAMINFO header — no decode, no dependency.

    python scripts/prepare_zeroth.py --root $CORPORA_ROOT
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

from corpus_fetch import (
    archive_record,
    download,
    extract_archive,
    remote_size,
    verify_archive,
    write_fetch_manifest,
)

CORPUS_IDS = ("slr40_zeroth_korean",)
LANGUAGES = ("ko",)

SLR = 40
ARCHIVE = "zeroth_korean.tar.gz"
ARCHIVE_URL = f"https://www.openslr.org/resources/{SLR}/{ARCHIVE}"
CHECKSUM_URL = f"https://www.openslr.org/resources/{SLR}/checksum.md5"
SOURCE_PAGE = f"https://www.openslr.org/{SLR}/"

AUDIO_INFO = "AUDIO_INFO"
#: SLR40 is the one corpus in this family that publishes a checksum, at
#: resources/40/checksum.md5. Recorded so the operator can check it; not
#: fetched automatically, which would make preparation depend on a second URL.
PUBLISHED_MD5 = "8e0a4268bb8e80db3773c331025ef1e2"

MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")
_FLAC_MAGIC = b"fLaC"


def read_audio_info(path: Path) -> dict[str, str]:
    """`{SPEAKERID: DATASET}` from AUDIO_INFO.

    Only those two fields are read. `NAME` is a contributor's personal name and
    is deliberately not returned, so it cannot reach the manifest by accident.
    """
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty {AUDIO_INFO}")
    columns = [name.strip().upper() for name in lines[0].split("|")]
    for required in ("SPEAKERID", "DATASET"):
        if required not in columns:
            raise ValueError(f"{path}: {AUDIO_INFO} has no {required} column; found {columns}")
    speaker_at, dataset_at = columns.index("SPEAKERID"), columns.index("DATASET")

    datasets: dict[str, str] = {}
    for line in lines[1:]:
        fields = [field.strip() for field in line.split("|")]
        if len(fields) <= max(speaker_at, dataset_at):
            continue
        if fields[speaker_at]:
            datasets[fields[speaker_at]] = fields[dataset_at]
    return datasets


def read_trans(path: Path) -> list[tuple[str, str]]:
    """`(utterance id, transcript)` pairs from one `.trans.txt`.

    LibriSpeech shape: the id is the first whitespace-delimited token and the
    transcript is the rest of the line.
    """
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        utt, _, text = line.strip().partition(" ")
        text = " ".join(text.split())
        if utt and text:
            rows.append((utt, text))
    return rows


def speaker_of(utt_id: str) -> str | None:
    """`149_003_0057` -> `149`, the SPEAKERID that names the containing directory."""
    speaker = utt_id.split("_", 1)[0]
    return speaker or None


def flac_duration(path: Path) -> float | None:
    """Seconds, read from the FLAC STREAMINFO block, or None if it is not FLAC.

    STREAMINFO is the first metadata block: after the 4-byte magic and a 4-byte
    block header, bytes 10..18 pack a 20-bit sample rate, 3-bit channel count,
    5-bit depth and 36-bit total sample count. Reading it costs 26 bytes and no
    decoder.
    """
    with open(path, "rb") as handle:
        head = handle.read(26)
    if len(head) < 26 or head[:4] != _FLAC_MAGIC:
        return None
    packed = int.from_bytes(head[18:26], "big")
    sample_rate = packed >> 44
    total_samples = packed & ((1 << 36) - 1)
    if not sample_rate or not total_samples:
        return None
    return total_samples / sample_rate


def collect(base: Path) -> tuple[list[list[str]], dict[str, Any]]:
    """Manifest rows for every utterance with audio, plus what was skipped.

    `split` is left empty on every row on purpose: see the module docstring.
    """
    info_path = base / AUDIO_INFO
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}; the archive did not extract as expected")
    shipped = read_audio_info(info_path)

    rows: list[list[str]] = []
    report: dict[str, Any] = {
        "shipped_speakers": {
            dataset: sum(1 for value in shipped.values() if value == dataset)
            for dataset in sorted(set(shipped.values()))
        },
        "missing_audio": 0,
        "speakers": 0,
    }
    speakers: set[str] = set()

    for trans in sorted(base.rglob("*.trans.txt")):
        for utt, text in read_trans(trans):
            audio = trans.parent / f"{utt}.flac"
            if not audio.exists():
                report["missing_audio"] += 1
                continue
            speaker = speaker_of(utt)
            if speaker is None:
                report["missing_audio"] += 1
                continue
            speakers.add(speaker)
            relative = audio.relative_to(base).as_posix()
            rows.append([utt, relative, text, speaker, ""])

    if not rows:
        raise ValueError(f"{base}: no .trans.txt line had matching audio")
    report["speakers"] = len(speakers)
    return rows, report


def derived_hours(base: Path, rows: list[list[str]]) -> dict[str, float]:
    """Hours per derived split, measured from the FLAC headers.

    Re-deriving leaves no published figure to quote, so this is the only record
    of how large the splits a run actually used were.

    Returns an empty mapping when `svb` is not importable. Preparers are meant to
    run on the machine with the disk, which need not have the project installed,
    and losing a reported number there is better than refusing to stage 10 GB.
    """
    try:
        from svb.data.splits import derive_splits
    except ImportError:
        print("    svb not importable here; skipping the derived-hours measurement")
        return {}

    parts, _, _ = derive_splits(rows, lambda row: row[3] or None, lambda row: row[0])
    hours = {}
    for name, part in parts.items():
        seconds = sum(flac_duration(base / row[1]) or 0.0 for row in part)
        hours[name] = round(seconds / 3600.0, 3)
    return hours


def write_manifest(dest: Path, rows: list[list[str]]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(sorted(rows))
    return path


def md5_file(path: Path) -> str:
    # MD5 because that is what SLR40 published; nothing here depends on it being
    # collision-resistant, and `usedforsecurity` keeps it working on FIPS builds.
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root: Path, *, keep_archives: bool = False, measure_hours: bool = True) -> Path:
    """Fetch, extract and write the manifest for Korean."""
    dest = root / CORPUS_IDS[0] / LANGUAGES[0]
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] ko: {CORPUS_IDS[0]} (SLR{SLR})")
    archive = download(ARCHIVE_URL, staging / ARCHIVE, expected=remote_size(ARCHIVE_URL))
    problem = verify_archive(archive)
    if problem:
        archive.unlink(missing_ok=True)
        raise OSError(f"{archive.name}: {problem}; deleted, re-run to fetch it again")

    # Unlike the rest of this family, SLR40 publishes an MD5. Check it.
    digest = md5_file(archive)
    if digest != PUBLISHED_MD5:
        archive.unlink(missing_ok=True)
        raise OSError(
            f"{archive.name}: md5 {digest} does not match the {PUBLISHED_MD5} published at "
            f"{CHECKSUM_URL}; deleted, re-run to fetch it again"
        )
    record = archive_record(archive, ARCHIVE_URL) | {"md5": digest, "md5_source": CHECKSUM_URL}
    extract_archive(archive, dest)
    if not keep_archives:
        archive.unlink(missing_ok=True)

    rows, report = collect(dest)
    manifest = write_manifest(dest, rows)
    hours = derived_hours(dest, rows) if measure_hours else {}
    print(f"    {len(rows)} utterances over {report['speakers']} speakers; split re-derived")
    if hours:
        print("    derived hours: " + ", ".join(f"{k}={v}" for k, v in hours.items()))

    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_IDS[0],
            "language": LANGUAGES[0],
            "source_page": SOURCE_PAGE,
            "split_source": "re-derived; the shipped 1.2 h test is under the 2 h bar",
            "published_hours": {"train": 51.6, "test": 1.2},
            "derived_hours": hours,
            "utterances": len(rows),
            "archives": [record],
            **report,
        },
    )
    if not keep_archives and staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="$CORPORA_ROOT")
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep the tarball so a re-extract is free"
    )
    parser.add_argument(
        "--no-measure-hours",
        action="store_true",
        help="skip reading every FLAC header for the derived split hours",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(args.root, keep_archives=args.keep_archives, measure_hours=not args.no_measure_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
