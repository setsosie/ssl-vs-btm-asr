"""Prepare Samrómur 21.05 (Icelandic) into the manifest layer.

Corpus id: slr112_samromur. Language: is. Source: https://www.openslr.org/112/
Licence: CC BY 4.0, verbatim from the page.

One archive, `samromur_21.05.tgz` (7.0 G): 100,000 utterances, 8,392 speakers,
145 h, FLAC at 16 kHz, 16-bit linear PCM, mono. Layout, read off the first tar
members::

    dev/003040/003040-0022916.flac
    dev/001277/001277-0009138.flac
    ...

The top-level directory is the split, the directory under it is the speaker id,
and the file stem is `{speaker_id}-{audio_id}` — the naming convention the source
page states. So both the split and the speaker come out of the path and neither
is inferred.

**The split ships and passes through untouched.** The page says the corpus is
"split into train, dev, and test subsets with no speaker overlap" — the cleanest
partition of any corpus here. Re-deriving it would throw away a published,
speaker-disjoint split and replace it with a worse one.

The transcript is the read prompt, which lives in the metadata rather than the
path. The page says only that the corpus "is distributed with a metadata file"
without naming it, so the file is *discovered*: any tab-separated file in the
extracted tree with a column naming the utterance and a column holding text.
What it found is recorded in the fetch manifest.

**One thing to check on the real archive.** The page describes three subsets, but
`resources/112/info.txt` describes the tgz as "whole corpus (includes dev and
train sets)" and does not mention test. If a split directory is missing, this
preparer refuses rather than writing a manifest whose test side is empty — the
loader would accept that and score on nothing. `--derive-splits` is the
documented way out: it clears the column and lets the shared speaker-disjoint
derivation run over whatever did ship, at the cost of no longer using the
published partition.

    python scripts/prepare_samromur.py --root $CORPORA_ROOT
"""

from __future__ import annotations

import argparse
import csv
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

CORPUS_IDS = ("slr112_samromur",)
LANGUAGES = ("is",)

SLR = 112
ARCHIVE = "samromur_21.05.tgz"
ARCHIVE_URL = f"https://www.openslr.org/resources/{SLR}/{ARCHIVE}"
SOURCE_PAGE = f"https://www.openslr.org/{SLR}/"

#: Corpus directory name -> the split name the loader uses.
SPLIT_OF_DIR = {"train": "train", "dev": "validation", "test": "test"}
AUDIO_SUFFIX = ".flac"

#: Metadata column names to look for, in preference order. The Hugging Face
#: repackaging of this corpus documents `audio_id`, `speaker_id`, `gender`,
#: `age`, `duration` and `normalized_text`; the OpenSLR tgz is not the same
#: packaging, so the names are looked for rather than assumed.
ID_COLUMNS = ("audio_id", "utterance_id", "utt_id", "id", "filename", "file")
TEXT_COLUMNS = ("sentence_norm", "normalized_text", "sentence", "text", "prompt")

MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")


def parse_audio_path(path: Path, base: Path) -> tuple[str, str, str] | None:
    """`(split, speaker, audio_id)` for one FLAC, or None if it is not in a split.

    The path carries all three: `<split>/<speaker>/<speaker>-<audio_id>.flac`.
    """
    try:
        parts = path.relative_to(base).parts
    except ValueError:
        return None
    if len(parts) < 3:
        return None
    split = SPLIT_OF_DIR.get(parts[-3].lower())
    if split is None:
        return None
    speaker = parts[-2]
    stem = path.stem
    audio_id = stem.split("-", 1)[1] if "-" in stem else stem
    if not speaker or not audio_id:
        return None
    return split, speaker, audio_id


def find_metadata(base: Path) -> list[Path]:
    """Tab-separated files in the tree that look like utterance metadata.

    The source page does not name the metadata file, so it is found by shape: a
    header carrying both an id column and a text column.
    """
    found = []
    for candidate in sorted(base.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in {".tsv", ".csv"}:
            continue
        with candidate.open(encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
        columns = [name.strip().lower() for name in header.rstrip("\n").split("\t")]
        if pick_column(columns, ID_COLUMNS) and pick_column(columns, TEXT_COLUMNS):
            found.append(candidate)
    return found


def pick_column(columns: list[str], wanted: tuple[str, ...]) -> str | None:
    """The first of `wanted` present in `columns`, or None."""
    lowered = {name.lower().strip(): name for name in columns}
    for candidate in wanted:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_metadata(paths: list[Path]) -> tuple[dict[str, str], dict[str, Any]]:
    """`{audio_id: transcript}` merged over every metadata file found."""
    texts: dict[str, str] = {}
    report: dict[str, Any] = {"metadata_files": {}}
    for path in paths:
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = [name.strip() for name in reader.fieldnames or []]
            id_key = pick_column(columns, ID_COLUMNS)
            text_key = pick_column(columns, TEXT_COLUMNS)
            if id_key is None or text_key is None:  # pragma: no cover - find_metadata filters
                continue
            rows = 0
            for record in reader:
                audio_id = (record.get(id_key) or "").strip()
                text = " ".join((record.get(text_key) or "").split())
                if not audio_id or not text:
                    continue
                # An id may be a bare id or a file name; both are accepted.
                texts.setdefault(Path(audio_id).stem, text)
                rows += 1
        report["metadata_files"][path.name] = {
            "rows": rows,
            "id_column": id_key,
            "text_column": text_key,
            "columns": columns,
        }
    return texts, report


def collect(base: Path, *, derive_splits: bool = False) -> tuple[list[list[str]], dict[str, Any]]:
    """Manifest rows for every FLAC whose transcript the metadata carries."""
    metadata_paths = find_metadata(base)
    if not metadata_paths:
        raise FileNotFoundError(
            f"{base}: found no tab-separated metadata carrying an id column "
            f"{ID_COLUMNS} and a text column {TEXT_COLUMNS}. The source page does not "
            "name the metadata file, so this preparer looks for it by shape; if the "
            "archive names its columns differently, extend ID_COLUMNS/TEXT_COLUMNS."
        )
    texts, report = read_metadata(metadata_paths)

    rows: list[list[str]] = []
    seen_splits: set[str] = set()
    report["missing_transcript"] = 0
    for audio in sorted(base.rglob(f"*{AUDIO_SUFFIX}")):
        parsed = parse_audio_path(audio, base)
        if parsed is None:
            continue
        split, speaker, audio_id = parsed
        text = texts.get(audio_id) or texts.get(audio.stem)
        if not text:
            report["missing_transcript"] += 1
            continue
        seen_splits.add(split)
        rows.append(
            [
                audio.stem,
                audio.relative_to(base).as_posix(),
                text,
                speaker,
                "" if derive_splits else split,
            ]
        )

    if not rows:
        raise ValueError(f"{base}: no FLAC under a train/dev/test directory had a transcript")

    missing = sorted(set(SPLIT_OF_DIR.values()) - seen_splits)
    if missing and not derive_splits:
        raise ValueError(
            f"{base}: the shipped partition is incomplete — nothing under {missing}. "
            "Writing it anyway would give the loader a split it accepts and a test set "
            "that scores on nothing. Re-run with --derive-splits to clear the column and "
            "derive a speaker-disjoint partition over what did ship, and say in the "
            "write-up that the published split was not used."
        )
    report["shipped_splits"] = sorted(seen_splits)
    report["split_source"] = "re-derived" if derive_splits else "shipped (directory layout)"
    return rows, report


def write_manifest(dest: Path, rows: list[list[str]]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(sorted(rows))
    return path


def prepare(root: Path, *, keep_archives: bool = False, derive_splits: bool = False) -> Path:
    """Fetch, extract and write the manifest for Icelandic."""
    dest = root / CORPUS_IDS[0] / LANGUAGES[0]
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] is: {CORPUS_IDS[0]} (SLR{SLR})")
    archive = download(ARCHIVE_URL, staging / ARCHIVE, expected=remote_size(ARCHIVE_URL))
    problem = verify_archive(archive)
    if problem:
        archive.unlink(missing_ok=True)
        raise OSError(f"{archive.name}: {problem}; deleted, re-run to fetch it again")
    record = archive_record(archive, ARCHIVE_URL)
    extract_archive(archive, dest)
    if not keep_archives:
        archive.unlink(missing_ok=True)

    rows, report = collect(dest, derive_splits=derive_splits)
    manifest = write_manifest(dest, rows)
    print(f"    {len(rows)} utterances; split {report['split_source']}")

    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_IDS[0],
            "language": LANGUAGES[0],
            "source_page": SOURCE_PAGE,
            "utterances": len(rows),
            "speakers": len({row[3] for row in rows}),
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
        "--derive-splits",
        action="store_true",
        help="clear the shipped split and derive a speaker-disjoint one instead",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(args.root, keep_archives=args.keep_archives, derive_splits=args.derive_splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
