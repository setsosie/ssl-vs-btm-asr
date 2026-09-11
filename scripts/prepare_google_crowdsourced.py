"""Prepare the Google crowdsourced OpenSLR corpora into the manifest layer.

Five languages, one archive layout, one preparer: Javanese (SLR35), Sundanese
(SLR36), Sinhala (SLR52), Bengali (SLR53) and Nepali (SLR54).

Each is published as **sixteen** zip shards named ``asr_<language>_{0..9,a..f}``
plus a single ``utt_spk_text.tsv`` of ``FileID``, anonymized ``UserID`` and
transcript. That third column is the important difference from the held-out four
(SLR63/64/66/78), which ship a two-column ``line_index*.tsv`` and force the
speaker out of the FileID: here the speaker is given, so the derived split is
speaker-disjoint without parsing anything. Verified on the openslr.org pages for
35, 36, 52, 53 and 54.

None of the five ships a train/dev/test split, so the manifest leaves ``split``
empty and the loader derives one. Audio is FLAC; the pages say only "wave
files", so the sample rate is read from the files rather than assumed.

Bengali is also a Common Voice locale, at 31.5 trainable hours — below the rule.
This corpus, at 229 hours, is what puts Bengali in the preset, so the language is
read from here and not from Common Voice.

    python scripts/prepare_google_crowdsourced.py --root $CORPORA_ROOT --langs jv
    python scripts/prepare_google_crowdsourced.py --root $CORPORA_ROOT   # all five

The held-out four are **not** here and must never be: they are the transfer set,
and ``scripts/fetch_openslr.py`` stages them separately under ``$OPENSLR_ROOT``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from corpus_fetch import (
    archive_record,
    download,
    extract_archive,
    remote_size,
    verify_archive,
    write_fetch_manifest,
)

#: Shard suffixes, in the order the pages list them.
SHARDS = "0123456789abcdef"
INDEX = "utt_spk_text.tsv"
MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")


@dataclass(frozen=True)
class Corpus:
    corpus_id: str
    lang: str
    slr: int
    slug: str  # the archive infix, e.g. "javanese" in asr_javanese_0.zip


CORPORA = (
    Corpus("slr35_javanese", "jv", 35, "javanese"),
    Corpus("slr36_sundanese", "su", 36, "sundanese"),
    Corpus("slr52_sinhala", "si", 52, "sinhala"),
    Corpus("slr53_bengali", "bn", 53, "bengali"),
    Corpus("slr54_nepali", "ne", 54, "nepali"),
)
CORPUS_IDS = tuple(c.corpus_id for c in CORPORA)
LANGUAGES = tuple(c.lang for c in CORPORA)
BY_LANG = {c.lang: c for c in CORPORA}


def archive_url(corpus: Corpus, shard: str) -> str:
    return f"https://www.openslr.org/resources/{corpus.slr}/asr_{corpus.slug}_{shard}.zip"


def read_index(path: Path) -> list[tuple[str, str, str]]:
    """``(FileID, UserID, transcript)`` rows from ``utt_spk_text.tsv``.

    Rows missing any of the three are skipped: the corpora are hand-checked but
    the pages warn that errors remain, and a row with no speaker would drop the
    whole language to an utterance-level split.
    """
    rows: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for record in csv.reader(handle, delimiter="\t"):
            if len(record) < 3:
                continue
            utt, speaker, text = record[0].strip(), record[1].strip(), record[2].strip()
            if utt and speaker and text:
                rows.append((utt, speaker, text))
    return rows


def write_manifest(dest: Path, rows: list[tuple[str, str, str]], audio: dict[str, str]) -> Path:
    """Write ``manifest.tsv``, one row per utterance that has audio on disk.

    An index row whose audio is missing is dropped rather than written with a
    path that does not resolve: the loader would fail on it mid-epoch, hours
    into a run, instead of here.
    """
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for utt, speaker, text in rows:
            relative = audio.get(utt)
            if relative is None:
                continue
            # `split` is empty: none of these five publishes a partition, so the
            # loader derives a speaker-disjoint one from the speaker column.
            writer.writerow([utt, relative, text, speaker, ""])
    return path


def prepare(
    corpus: Corpus, root: Path, *, keep_archives: bool = False, shards: str = SHARDS
) -> Path:
    """Fetch, extract and write the manifest for one language."""
    dest = root / corpus.corpus_id / corpus.lang
    audio_dir = dest / "audio"
    staging = dest / ".archives"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] {corpus.lang}: {corpus.corpus_id} (SLR{corpus.slr})")
    records = []
    seen: set[str] = set()
    index_path: Path | None = None
    for shard in shards:
        url = archive_url(corpus, shard)
        archive = download(
            url, staging / f"asr_{corpus.slug}_{shard}.zip", expected=remote_size(url)
        )
        problem = verify_archive(archive)
        if problem:
            archive.unlink(missing_ok=True)
            raise OSError(f"{archive.name}: {problem}; deleted, re-run to fetch it again")
        records.append(archive_record(archive, url))

        # Flat, because the audio is what matters and the shard directories only
        # say which shard a file arrived in. The index repeats identically in
        # every shard, so it is extracted once and excluded from the collision
        # guard that protects the audio.
        written = extract_archive(archive, audio_dir, flat=True, members=None, seen=None)
        for name in written:
            if name == INDEX:
                index_path = audio_dir / INDEX
                continue
            if name in seen:
                raise ValueError(
                    f"{archive.name}: {name!r} was already extracted from another shard of "
                    f"{corpus.lang}; the shards are supposed to hold disjoint file ids"
                )
            seen.add(name)
        if not keep_archives:
            archive.unlink(missing_ok=True)

    if index_path is None or not index_path.exists():
        raise FileNotFoundError(f"{corpus.lang}: no {INDEX} in any shard of {corpus.corpus_id}")

    audio = {Path(name).stem: f"audio/{name}" for name in seen}
    rows = read_index(index_path)
    manifest = write_manifest(dest, rows, audio)
    missing = sum(1 for utt, _, _ in rows if utt not in audio)
    print(f"    {len(rows) - missing} utterances written, {missing} index rows without audio")

    write_fetch_manifest(
        dest,
        {
            "corpus": corpus.corpus_id,
            "language": corpus.lang,
            "source_page": f"https://www.openslr.org/{corpus.slr}/",
            "index_rows": len(rows),
            "rows_without_audio": missing,
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
        "--langs", nargs="*", choices=LANGUAGES, help="default: every language in this family"
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep the zips so a re-extract is free"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for lang in args.langs or LANGUAGES:
        prepare(BY_LANG[lang], args.root, keep_archives=args.keep_archives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
