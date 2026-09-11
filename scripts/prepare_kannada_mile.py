"""Prepare the IISc-MILE Kannada ASR Corpus into the manifest layer.

Corpus id: slr126_kannada. Language: kn. Source: https://www.openslr.org/126/
Licence: Attribution 2.0 Generic (CC BY 2.0), verbatim from the page — the only
CC BY 2.0 corpus in this set.

Two archives, `mile_kannada_train.tar.gz` (22 G) and `mile_kannada_test.tar.gz`
(6.1 G): ~350 h of read speech from 915 speakers, 16 kHz 16-bit mono PCM WAV.
Each archive carries its own top-level split directory, read off the tar
members::

    train/audio_files/MILE_03_SP_3415_UTT_0038.wav
    train/trans_files/MILE_03_SP_3415_UTT_0038.txt
    test/audio_files/...   test/trans_files/...

The transcript is the whole `.txt`, UTF-8, whitespace collapsed.

**Speaker ids are present**, which the corpus record had as unconfirmed: the file
stem is `MILE_03_SP_<speaker>_UTT_<n>`, so everything before `_UTT_` identifies
the speaker. Confirmed against the archive members and against ESPnet's
`egs2/kn_openslr126/asr1/local/data_prep.py`, which derives the speaker the same
way.

**The split is re-derived and the published division is not the one a run uses.**
Train and test ship; there is no dev split. The loader takes a partition whole or
derives one whole — a manifest that is part shipped and part derived could not be
written down in a results file — so `split` is cleared on every row and the shared
speaker-disjoint derivation produces all three. A write-up has to say so rather
than describe the corpus's own train/test division.

    python scripts/prepare_kannada_mile.py --root $CORPORA_ROOT
    python scripts/prepare_kannada_mile.py --root $CORPORA_ROOT --splits test
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

CORPUS_IDS = ("slr126_kannada",)
LANGUAGES = ("kn",)

SLR = 126
SOURCE_PAGE = f"https://www.openslr.org/{SLR}/"
#: The corpus's own directory names, which are also the archive infixes. They are
#: *not* the manifest's split values: see the module docstring.
SHIPPED = ("train", "test")

AUDIO_DIR = "audio_files"
TRANS_DIR = "trans_files"
SPEAKER_SEPARATOR = "_UTT_"
MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")


def archive_url(shipped: str) -> str:
    return f"https://www.openslr.org/resources/{SLR}/mile_kannada_{shipped}.tar.gz"


def speaker_of(utt_id: str) -> str | None:
    """`MILE_03_SP_2496_UTT_0116` -> `MILE_03_SP_2496`.

    None when the stem does not carry the separator, which drops the language to
    an utterance-level split rather than inventing a speaker.
    """
    speaker, separator, _ = utt_id.partition(SPEAKER_SEPARATOR)
    return speaker if separator and speaker else None


def read_transcript(path: Path) -> str:
    """One `trans_files/<stem>.txt`, whitespace collapsed to single spaces."""
    return " ".join(path.read_text(encoding="utf-8").split())


def find_split_dir(base: Path, shipped: str) -> Path | None:
    """Where a shipped split's `audio_files` lives.

    Normally `<base>/<shipped>/`, but the ESPnet recipe for this corpus guards
    against an extra wrapping directory, so the tree is searched rather than
    assumed.
    """
    direct = base / shipped
    if (direct / AUDIO_DIR).is_dir():
        return direct
    for candidate in sorted(base.rglob(AUDIO_DIR)):
        if candidate.is_dir() and shipped in candidate.parent.parts:
            return candidate.parent
    return None


def collect(base: Path, shipped: tuple[str, ...] = SHIPPED) -> tuple[list[list[str]], Any]:
    """Manifest rows for every wav that has a transcript, `split` left empty."""
    rows: list[list[str]] = []
    report: dict[str, Any] = {"shipped_counts": {}, "missing_transcript": 0, "no_speaker": 0}

    for name in shipped:
        split_dir = find_split_dir(base, name)
        if split_dir is None:
            raise FileNotFoundError(
                f"{base}: no {name}/{AUDIO_DIR} directory; the archive did not extract as expected"
            )
        count = 0
        for audio in sorted((split_dir / AUDIO_DIR).glob("*.wav")):
            transcript = split_dir / TRANS_DIR / f"{audio.stem}.txt"
            if not transcript.exists():
                report["missing_transcript"] += 1
                continue
            text = read_transcript(transcript)
            if not text:
                report["missing_transcript"] += 1
                continue
            speaker = speaker_of(audio.stem)
            if speaker is None:
                report["no_speaker"] += 1
            # `split` is empty on purpose: no dev ships, so all three are derived.
            rows.append([audio.stem, audio.relative_to(base).as_posix(), text, speaker or "", ""])
            count += 1
        report["shipped_counts"][name] = count

    if not rows:
        raise ValueError(f"{base}: no wav had a matching transcript")
    report["speakers"] = len({row[3] for row in rows if row[3]})
    return rows, report


def write_manifest(dest: Path, rows: list[list[str]]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(sorted(rows))
    return path


def prepare(root: Path, *, keep_archives: bool = False, splits: tuple[str, ...] = SHIPPED) -> Path:
    """Fetch, extract and write the manifest for Kannada."""
    dest = root / CORPUS_IDS[0] / LANGUAGES[0]
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] kn: {CORPUS_IDS[0]} (SLR{SLR})")
    records = []
    for name in splits:
        url = archive_url(name)
        archive = download(url, staging / f"mile_kannada_{name}.tar.gz", expected=remote_size(url))
        problem = verify_archive(archive)
        if problem:
            archive.unlink(missing_ok=True)
            raise OSError(f"{archive.name}: {problem}; deleted, re-run to fetch it again")
        records.append(archive_record(archive, url))
        # Each archive carries its own top-level split directory, so they cannot
        # collide and the layout is kept rather than flattened.
        extract_archive(archive, dest)
        if not keep_archives:
            archive.unlink(missing_ok=True)

    rows, report = collect(dest, splits)
    manifest = write_manifest(dest, rows)
    print(f"    {len(rows)} utterances over {report['speakers']} speakers; split re-derived")

    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_IDS[0],
            "language": LANGUAGES[0],
            "source_page": SOURCE_PAGE,
            "split_source": "re-derived; train and test ship but there is no dev split",
            "utterances": len(rows),
            "archives": records,
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
        "--splits",
        nargs="*",
        choices=SHIPPED,
        help="shipped archives to fetch; default both. They are pooled and re-split.",
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep the tarballs so a re-extract is free"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(args.root, keep_archives=args.keep_archives, splits=tuple(args.splits or SHIPPED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
