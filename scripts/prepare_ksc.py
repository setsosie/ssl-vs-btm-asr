"""Prepare the Kazakh Speech Corpus (KSC) into the manifest layer.

Corpus id: slr102_ksc_kazakh. Language: kk. Source: https://www.openslr.org/102/
Licence: Attribution 4.0 International (CC BY 4.0), verbatim from the page.

One archive, `ISSAI_KSC_335RS_v1.1_flac.tar.gz` (19 G), ~332 h over 153,000+
utterances. Inside::

    ISSAI_KSC_335RS_v1.1_flac/
        Audios_flac/<uttID>.flac
        Transcriptions/<uttID>.txt
        Meta/{train,dev,test}.csv

**A split ships**, which the source page does not say and which the corpus
record had recorded as `none`. The evidence is the corpus author's own ESPnet
recipe (`egs2/ksc/asr1/local/prepare_data.sh`, ISSAI, Yerbolat Khassanov — first
author of the KSC paper), which builds one Kaldi directory per `Meta/*.csv`,
reads each utterance's transcript from `Transcriptions/<uttID>.txt` and its audio
from `Audios_flac/<uttID>.flac`. So the manifest fills `split` from the Meta file
an utterance is listed in and nothing is derived.

The Meta files carry a header and more columns than the id; that recipe reads
only the first (`read` to skip the header, then `IFS=" " read -r uttID others`),
so the remaining columns are unnamed there. This preparer reads the header
properly, records the column names in the fetch manifest so the next reader does
not have to guess, and fills `speaker` from a column only if one is literally
named as a speaker id.

**No speaker id is used in practice.** That same recipe writes `utt2spk` as
`uttID uttID` — every utterance is its own speaker. So unless a Meta column turns
out to be a speaker, `speaker` stays empty and the shipped split's
speaker-disjointness is *unknown*, not established. Say that in a write-up
rather than implying a speaker-held-out result.

    python scripts/prepare_ksc.py --root $CORPORA_ROOT
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

CORPUS_IDS = ("slr102_ksc_kazakh",)
LANGUAGES = ("kk",)

SLR = 102
ARCHIVE = "ISSAI_KSC_335RS_v1.1_flac.tar.gz"
ARCHIVE_URL = f"https://www.openslr.org/resources/{SLR}/{ARCHIVE}"
SOURCE_PAGE = f"https://www.openslr.org/{SLR}/"

ROOT_DIR = "ISSAI_KSC_335RS_v1.1_flac"
AUDIO_DIR = f"{ROOT_DIR}/Audios_flac"
TRANS_DIR = f"{ROOT_DIR}/Transcriptions"
META_DIR = f"{ROOT_DIR}/Meta"

#: Meta file stem -> the split name the loader uses. "dev" is the corpus's word
#: and "validation" is the loader's; mapping it here is what stops a manifest
#: from reaching the loader with a split value it refuses.
SPLIT_OF_META = {"train": "train", "dev": "validation", "test": "test"}

#: A Meta column is treated as the speaker only if it is named as one. Guessing
#: from the shape of the values would invent disjointness the corpus never
#: claimed.
SPEAKER_COLUMNS = ("speaker", "speaker_id", "speakerid", "spk", "spk_id", "spkid")

MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")
#: Tried in order; the one that splits the header into the most fields wins.
_DELIMITERS = ("\t", ",", " ")


def sniff_delimiter(header: str) -> str:
    """The delimiter that splits `header` into the most fields.

    The Meta files are named `.csv` but the reference recipe reads them on
    whitespace, so which one they actually use is not something to assume.
    """
    return max(_DELIMITERS, key=lambda d: (len(header.split(d)), -_DELIMITERS.index(d)))


def read_meta(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """`(column names, rows)` of one Meta file, keyed by the header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty Meta file")
    delimiter = sniff_delimiter(lines[0])
    reader = csv.reader(lines, delimiter=delimiter, skipinitialspace=True)
    columns = [name.strip() for name in next(reader)]
    rows = [
        {name: value.strip() for name, value in zip(columns, record, strict=False)}
        for record in reader
        if record and record[0].strip()
    ]
    return columns, rows


def speaker_column(columns: list[str]) -> str | None:
    """The column holding a speaker id, or None if the Meta file has none."""
    lowered = {name.lower().replace("-", "_"): name for name in columns}
    for candidate in SPEAKER_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_transcript(path: Path) -> str:
    """One `Transcriptions/<uttID>.txt`, whitespace collapsed to single spaces."""
    return " ".join(path.read_text(encoding="utf-8").split())


def collect(base: Path) -> tuple[list[list[str]], dict[str, Any]]:
    """Manifest rows for every Meta-listed utterance, plus what was skipped.

    An utterance listed in a Meta file but missing its audio or its transcript is
    dropped rather than written with a path that does not resolve: the loader
    would fail on it mid-epoch, hours into a run, instead of here.
    """
    meta_dir = base / META_DIR
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"missing {meta_dir}; the archive did not extract as expected")

    rows: list[list[str]] = []
    report: dict[str, Any] = {"meta_files": {}, "missing_audio": 0, "missing_transcript": 0}
    seen: set[str] = set()

    for meta in sorted(meta_dir.glob("*.csv")):
        split = SPLIT_OF_META.get(meta.stem.lower())
        if split is None:
            raise ValueError(
                f"{meta.name}: unexpected Meta file; known splits are "
                f"{sorted(SPLIT_OF_META)}. A new one would silently go unread."
            )
        columns, records = read_meta(meta)
        speaker_key = speaker_column(columns)
        report["meta_files"][meta.name] = {
            "split": split,
            "rows": len(records),
            "columns": columns,
            "speaker_column": speaker_key,
        }

        for record in records:
            utt = record.get(columns[0], "").strip()
            if not utt or utt in seen:
                continue
            audio = base / AUDIO_DIR / f"{utt}.flac"
            transcript = base / TRANS_DIR / f"{utt}.txt"
            if not audio.exists():
                report["missing_audio"] += 1
                continue
            if not transcript.exists():
                report["missing_transcript"] += 1
                continue
            text = read_transcript(transcript)
            if not text:
                report["missing_transcript"] += 1
                continue
            seen.add(utt)
            speaker = record.get(speaker_key, "") if speaker_key else ""
            rows.append([utt, f"{AUDIO_DIR}/{utt}.flac", text, speaker, split])

    if not rows:
        raise ValueError(f"{base}: no utterance had both audio and a transcript")
    return rows, report


def write_manifest(dest: Path, rows: list[list[str]]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)
    return path


def prepare(root: Path, *, keep_archives: bool = False) -> Path:
    """Fetch, extract and write the manifest for Kazakh."""
    dest = root / CORPUS_IDS[0] / LANGUAGES[0]
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] kk: {CORPUS_IDS[0]} (SLR{SLR})")
    archive = download(ARCHIVE_URL, staging / ARCHIVE, expected=remote_size(ARCHIVE_URL))
    problem = verify_archive(archive)
    if problem:
        archive.unlink(missing_ok=True)
        raise OSError(f"{archive.name}: {problem}; deleted, re-run to fetch it again")
    record = archive_record(archive, ARCHIVE_URL)
    extract_archive(archive, dest)
    if not keep_archives:
        archive.unlink(missing_ok=True)

    rows, report = collect(dest)
    manifest = write_manifest(dest, rows)
    print(f"    {len(rows)} utterances written; splits taken from {META_DIR}")

    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_IDS[0],
            "language": LANGUAGES[0],
            "source_page": SOURCE_PAGE,
            "split_source": "shipped (Meta/*.csv)",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(args.root, keep_archives=args.keep_archives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
