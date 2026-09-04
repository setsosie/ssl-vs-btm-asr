"""Prepare TIBMD@MUC (Tibetan) into the manifest layer.

Corpus id: slr124_tibetan. Language: bo. Source: https://www.openslr.org/124/
Licence: CC BY-SA 4.0, verbatim from the page.

One archive, `Tibetan_speech_data.tgz` (29 G): 84.33 h over three dialects
(Ü-Tsang 52.53 h, Kham 5.87 h, Amdo 25.93 h), 87 native speakers, 10,390 unique
sentences, 16 kHz 16-bit audio.

Layout, read off the archive's own members::

    Tibetan mutil-dialect data/          <- the archive's own spelling
        Amdo Dialect/
            Pastoral areas dialect/
                Aba pastoral dialect/
                    Speaker 65/
                        Reading/
                            wav/阿坝牧区_crzm_A523.wav

**Speaker ids are present**, which the corpus record had as unconfirmed: each
recording sits under a `Speaker <N>` directory. The dialect is the `... Dialect`
component near the top. Both are taken from the path, so the derived split is
speaker-disjoint. The path depth between them varies by dialect, so the two
components are located by name rather than by position.

**The transcript convention is the one thing that could not be verified before
writing this.** The source page does not document it, and 444 MB of decompressed
archive prefix — 561 consecutive wav files under one speaker — contained no
non-audio member at all, so it is certainly not one `.txt` beside each `.wav`.
Rather than guess, this preparer *discovers* it after extraction, accepting the
two shapes the rest of this corpus family uses:

* one transcript file per utterance, matched by stem, anywhere in the tree;
* an index file whose lines start with an utterance stem, matched the same way.

If neither yields a single match it refuses and reports the non-audio files it
did find, so one run establishes the real convention instead of a guess failing
silently. Whichever fired, and how many rows it produced, goes in the fetch
manifest.

The dialect is carried in an extra `dialect` column. The loader reads the
manifest with `csv.DictReader` and only requires its five columns to be present,
so extra ones travel with the data rather than being lost to `manifest.json`.

Tibetan is written without word boundaries, so the preset entry carries
`word_boundary: false` and character error rate is the primary metric.

    python scripts/prepare_tibmd.py --root $CORPORA_ROOT
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
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

CORPUS_IDS = ("slr124_tibetan",)
LANGUAGES = ("bo",)

SLR = 124
ARCHIVE = "Tibetan_speech_data.tgz"
ARCHIVE_URL = f"https://www.openslr.org/resources/{SLR}/{ARCHIVE}"
SOURCE_PAGE = f"https://www.openslr.org/{SLR}/"

AUDIO_SUFFIX = ".wav"
#: `Speaker 65`, `speaker_12`, `Speaker12` — the separator varies across the
#: dialect trees, so it is matched rather than assumed.
SPEAKER_PATTERN = re.compile(r"^speaker[\s_-]*(\S+)$", re.IGNORECASE)
#: Anchored at the end on purpose. The archive's own root directory is
#: "Tibetan mutil-dialect data" (its spelling), so a substring match would take
#: the root for the dialect on every path in the corpus. The three dialect
#: directories end in "Dialect"; the sub-area directories under them do too, so
#: the *first* such component wins.
DIALECT_PATTERN = re.compile(r"dialect$", re.IGNORECASE)
#: Enough of a candidate index file to tell whether its lines start with an
#: utterance stem, without reading a large file into memory.
_INDEX_PROBE_BYTES = 1 << 20

MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split", "dialect")


def _slug(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip())


def parse_audio_path(path: Path, base: Path) -> tuple[str, str] | None:
    """`(speaker, dialect)` for one wav, or None when the path carries neither.

    Both are located by name: the depth between the dialect directory and the
    `Speaker <N>` one differs across the three dialect trees, so counting path
    components would work for Amdo and break for the others.
    """
    try:
        parts = path.relative_to(base).parts[:-1]
    except ValueError:
        return None

    speaker = next(
        (match.group(1) for part in reversed(parts) if (match := SPEAKER_PATTERN.match(part))),
        None,
    )
    dialect = next((part for part in parts if DIALECT_PATTERN.search(part.strip())), "")
    if speaker is None:
        return None
    # Prefixing with the dialect keeps two dialects' "Speaker 3" apart without
    # assuming the corpus numbers its 87 speakers globally.
    key = f"speaker_{_slug(speaker)}"
    if dialect:
        key = f"{_slug(dialect)}-{key}"
    return key, _slug(dialect)


def read_text(path: Path) -> str | None:
    """A file's text, or None when it is not UTF-8."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return None


def _split_index_line(line: str) -> tuple[str, str] | None:
    """`(stem, transcript)` from an index line, tab-separated or whitespace."""
    head, separator, tail = line.partition("\t")
    if not separator:
        head, separator, tail = line.partition(" ")
    if not separator:
        return None
    stem = Path(head.strip()).stem
    text = " ".join(tail.split())
    return (stem, text) if stem and text else None


def discover_transcripts(base: Path, stems: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Find the transcripts, whichever of the two shapes the archive uses.

    The source page documents neither, so both are tried and what worked is
    reported. Per-utterance files win over an index for a stem they both cover:
    a file named for one utterance is the more specific statement about it.
    """
    from_index: dict[str, str] = {}
    per_file: dict[str, str] = {}
    report: dict[str, Any] = {"index_files": [], "per_file_transcripts": 0, "candidates": []}

    for candidate in sorted(base.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() == AUDIO_SUFFIX:
            continue
        report["candidates"].append(candidate.relative_to(base).as_posix())

        if candidate.stem in stems:
            text = read_text(candidate)
            if text and text.strip():
                per_file[candidate.stem] = " ".join(text.split())
                continue

        text = read_text(candidate)
        if text is None:
            continue
        hits = 0
        for line in text[:_INDEX_PROBE_BYTES].splitlines():
            parsed = _split_index_line(line)
            if parsed and parsed[0] in stems:
                from_index.setdefault(parsed[0], parsed[1])
                hits += 1
        if hits:
            report["index_files"].append(
                {"file": candidate.relative_to(base).as_posix(), "matched_rows": hits}
            )

    report["per_file_transcripts"] = len(per_file)
    return from_index | per_file, report


def collect(base: Path) -> tuple[list[list[str]], dict[str, Any]]:
    """Manifest rows for every wav whose transcript could be found."""
    audio = sorted(base.rglob(f"*{AUDIO_SUFFIX}"))
    if not audio:
        raise FileNotFoundError(f"{base}: no {AUDIO_SUFFIX} files; the archive did not extract")

    counts = Counter(path.stem for path in audio)
    stems = set(counts)
    if len(stems) != len(audio):
        duplicates = sorted(stem for stem, count in counts.items() if count > 1)
        raise ValueError(
            f"{base}: {len(audio) - len(stems)} wav files share a stem "
            f"(for example {duplicates[:3]}). The stem is the manifest's utt_id and has to be "
            "unique within a language, so the id would have to include the dialect path."
        )

    transcripts, report = discover_transcripts(base, stems)
    if not transcripts:
        suffixes = sorted({Path(name).suffix or "(none)" for name in report["candidates"]})
        raise ValueError(
            f"{base}: found no transcripts. The source page does not document the convention, "
            "so both a per-utterance file matched by stem and an index file keyed by stem were "
            f"tried. Non-audio files present: {len(report['candidates'])}, suffixes {suffixes}, "
            f"for example {report['candidates'][:5]}. Extend discover_transcripts for whatever "
            "shape that turns out to be."
        )

    rows: list[list[str]] = []
    report["missing_transcript"] = 0
    report["no_speaker"] = 0
    for path in audio:
        text = transcripts.get(path.stem)
        if not text:
            report["missing_transcript"] += 1
            continue
        parsed = parse_audio_path(path, base)
        if parsed is None:
            report["no_speaker"] += 1
            speaker, dialect = "", ""
        else:
            speaker, dialect = parsed
        # `split` is empty: the corpus ships none, so one is derived.
        rows.append([path.stem, path.relative_to(base).as_posix(), text, speaker, "", dialect])

    if not rows:
        raise ValueError(f"{base}: no wav matched a discovered transcript")
    report["speakers"] = len({row[3] for row in rows if row[3]})
    report["dialects"] = sorted({row[5] for row in rows if row[5]})
    report.pop("candidates")
    return rows, report


def write_manifest(dest: Path, rows: list[list[str]]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(sorted(rows))
    return path


def prepare(root: Path, *, keep_archives: bool = False) -> Path:
    """Fetch, extract and write the manifest for Tibetan."""
    dest = root / CORPUS_IDS[0] / LANGUAGES[0]
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] bo: {CORPUS_IDS[0]} (SLR{SLR})")
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
    print(
        f"    {len(rows)} utterances over {report['speakers']} speakers, "
        f"dialects {report['dialects']}"
    )

    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_IDS[0],
            "language": LANGUAGES[0],
            "source_page": SOURCE_PAGE,
            "split_source": "derived; the corpus ships none",
            "transcript_convention": (
                "index file" if report["index_files"] else "one file per utterance"
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
