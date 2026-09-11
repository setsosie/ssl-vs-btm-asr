"""Prepare the Armenian Speech Crowdsourcing Data into the manifest layer.

Corpus id: slr160_armenian. Language: hy. Source: https://www.openslr.org/160/
Licence: CC BY 4.0, verbatim from the page.

One archive, `armenian_speech_crowdsourcing_data.tar.gz` (5.6 G), 70 h of
crowdsourced read speech (Toloka, Yerevan City Magazine text). Layout, read off
the archive's own members::

    armenian_speech_crowdsourcing_data/
        pitched.jsonl
        pitched/eu.<uuid>.wav

`pitched.jsonl` is one JSON object per line and carries exactly three keys, read
off the real file::

    {"text": "...", "audio_filepath": "/data/pitched/eu.<uuid>.wav", "duration": 5.33}

`audio_filepath` is an absolute path from the machine that built the corpus, so
only its file name is used and the file is resolved under the extracted tree.

**There is no speaker field, and there is not meant to be one.** The page says
"the voices themselves were anonymized so that they could not be easily
identified or matched to individual speakers". So `speaker` is left empty on
every row, the shared derivation degrades to an utterance-level split, and a
result on this language is **not speaker-independent** — the same contributor
can appear in train and test. That belongs in any write-up of an Armenian
number; the loader reports it as `split_policy == "utterance"`.

This corpus is *not* one of the Google crowdsourced sets, despite also being
crowdsourced read speech on OpenSLR. Those ship sixteen `asr_<language>_*.zip`
shards and a three-column `utt_spk_text.tsv` whose middle column is the speaker,
which is exactly the field this one does not have, so there is nothing to share
with `prepare_google_crowdsourced.py`.

    python scripts/prepare_armenian.py --root $CORPORA_ROOT
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from corpus_fetch import (
    archive_record,
    download,
    extract_archive,
    remote_size,
    verify_archive,
    write_fetch_manifest,
)

CORPUS_IDS = ("slr160_armenian",)
LANGUAGES = ("hy",)

SLR = 160
ARCHIVE = "armenian_speech_crowdsourcing_data.tar.gz"
ARCHIVE_URL = f"https://www.openslr.org/resources/{SLR}/{ARCHIVE}"
SOURCE_PAGE = f"https://www.openslr.org/{SLR}/"

ROOT_DIR = "armenian_speech_crowdsourcing_data"
INDEX = "pitched.jsonl"
MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")


def find_index(base: Path) -> Path:
    """The `pitched.jsonl` under the extracted tree.

    Located rather than assumed to be at a fixed depth: the archive wraps
    everything in one directory today, and an archive that stopped doing so
    would otherwise fail with a path error instead of a name.
    """
    direct = base / ROOT_DIR / INDEX
    if direct.exists():
        return direct
    found = sorted(base.rglob(INDEX))
    if not found:
        raise FileNotFoundError(
            f"missing {INDEX} under {base}; the archive did not extract as expected"
        )
    return found[0]


def read_index(path: Path) -> list[tuple[str, str, float | None]]:
    """`(audio file name, transcript, duration)` per JSONL line.

    Lines that are not JSON, or that lack a transcript or an audio path, are
    skipped: one bad line should not lose the other 70 hours.
    """
    rows: list[tuple[str, str, float | None]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            text = " ".join(str(record.get("text") or "").split())
            # An absolute path from the corpus builder's machine; take the name.
            name = PurePosixPath(str(record.get("audio_filepath") or "")).name
            if not text or not name:
                continue
            duration = record.get("duration")
            rows.append(
                (name, text, float(duration) if isinstance(duration, int | float) else None)
            )
    return rows


def collect(base: Path) -> tuple[list[list[str]], dict[str, Any]]:
    """Manifest rows for every indexed utterance whose audio is on disk."""
    index = find_index(base)
    rows = read_index(index)
    if not rows:
        raise ValueError(f"{index}: no line carried both a transcript and an audio path")

    audio = {path.name: path for path in base.rglob("*.wav")}
    written: list[list[str]] = []
    report: dict[str, Any] = {
        "index_file": index.relative_to(base).as_posix(),
        "index_rows": len(rows),
        "missing_audio": 0,
    }
    seconds = 0.0
    for name, text, duration in rows:
        path = audio.get(name)
        if path is None:
            report["missing_audio"] += 1
            continue
        # `speaker` is empty by design; see the module docstring.
        written.append([Path(name).stem, path.relative_to(base).as_posix(), text, "", ""])
        seconds += duration or 0.0

    if not written:
        raise ValueError(f"{base}: no indexed utterance had audio on disk")
    report["indexed_hours"] = round(seconds / 3600.0, 3)
    return written, report


def write_manifest(dest: Path, rows: list[list[str]]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(sorted(rows))
    return path


def prepare(root: Path, *, keep_archives: bool = False) -> Path:
    """Fetch, extract and write the manifest for Armenian."""
    dest = root / CORPUS_IDS[0] / LANGUAGES[0]
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] hy: {CORPUS_IDS[0]} (SLR{SLR})")
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
    print(f"    {len(rows)} utterances, {report['indexed_hours']} h; no speaker field")

    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_IDS[0],
            "language": LANGUAGES[0],
            "source_page": SOURCE_PAGE,
            "split_source": "derived",
            "speaker_ids": (
                "none — the page states the voices were anonymized so that they could not "
                "be matched to individual speakers, so the split is utterance-level and "
                "results on this language are not speaker-independent"
            ),
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
