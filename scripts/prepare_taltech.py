"""Prepare the TalTech Estonian Speech Dataset 1.0 into the manifest layer.

    python scripts/prepare_taltech.py --root $CORPORA_ROOT

One language, ``et``, one 159 GB tar, no registration:
https://cs.taltech.ee/staff/tanel.alumae/data/est-pub-asr-data/

**Layout**, walked over HTTP Range rather than downloaded — the tar's header
chain was followed from byte 0 and sampled to the end::

    <split>/transcriptions/stm/<subset>/<recording>.stm
    <split>/transcriptions/vtt/<subset>/<recording>.vtt
    <split>/transcriptions/trs/<subset>/<recording>.trs
    <split>/audio/<subset>/<recording>.wav

``<split>`` is ``train``, ``dev`` or ``test``. ``<subset>`` is one of the
collections the page names — ERR2020, aktuaalne2021, intervjuukorpus, paevakaja,
MKK, aktuaalne-kaamera, er-uudised, jutusaated, konverentsid, veebiseminarid,
Riigikogu_salvestused, podcastid. Only the STM tree is read: the three transcript
formats are the same content, and STM is the one that carries the speaker and the
segment bounds on a single line. A few members under ``.../stm/`` carry a ``.vtt``
extension, so the selection is by suffix rather than by directory.

**The audio is long form.** Recordings run half an hour and more — one sampled
member is 74,092,622 bytes, about 38 minutes — and the manifest layer addresses
whole files, with no offset or duration column. So the preparer *cuts* each
recording at the STM segment bounds and writes one WAV per utterance. That is the
whole reason this preparer is expensive rather than a transcript reshuffle.

Audio is 16 kHz 16-bit mono PCM WAV, read out of a sampled member's RIFF header;
the page states no format at all, which is why ``configs/corpora.yaml`` marks it
unverified. Nothing is resampled here, and the rate found is recorded.

**The STM format**, from a real file in the archive::

    ;; Exported from .../er-uudised/20051206-1430ER_uudised....trs using local/trs2stm.py
    <file_id> 1 inter_segment_gap 0.0 5.409 <o,f0,>
    <file_id> 1 <file_id>_Vallo_Kelmsaar 5.409 9.175 <o,f0,> Kell on kaks, te kuulete ...

Six whitespace-separated fields then the transcript: recording id, channel,
speaker, start and end in seconds, an angle-bracketed label. ``inter_segment_gap``
rows carry no transcript and are not utterances.

**Split.** All three ship and are used as shipped, so nothing is derived. The
speaker ids are *per recording* — the recording id is a prefix of every speaker id
in its file — so the same person appearing in two recordings looks like two
speakers. That does not affect this manifest, whose split is shipped rather than
speaker-derived, but it does mean the shipped split cannot be claimed to be
speaker-disjoint.

**Disk.** About 159 GB of tar, about the same again extracted, and roughly 130 GB
of cut segments. Without ``--keep-archives`` the tar is deleted as soon as
extraction finishes and each long-form recording as soon as its segments are
written, which keeps the steady state near the segments alone; the peak is still
both the tar and the extracted tree at once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import tarfile
import wave
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from corpus_fetch import (
    UA,
    archive_record,
    download,
    extract_archive,
    remote_size,
    verify_archive,
    write_fetch_manifest,
)

CORPUS_ID = "taltech_estonian"
CORPUS_IDS = (CORPUS_ID,)
LANGUAGES = ("et",)
SOURCE_PAGE = "https://cs.taltech.ee/staff/tanel.alumae/data/est-pub-asr-data/"
ARCHIVE = "taltech-asr-speech-dataset-1.0.tar"
ARCHIVE_URL = SOURCE_PAGE + ARCHIVE
MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")

#: What the page states. No API and no checksum, so the page is the only record.
EXPECTED_LICENCE = "CC BY-SA 4.0"

#: The corpus's own split directories, mapped onto the manifest's split names.
SPLIT_NAMES = {"train": "train", "dev": "validation", "test": "test"}
SHIPPED_SPLIT = {mapped: shipped for shipped, mapped in SPLIT_NAMES.items()}

#: A pseudo-speaker the exporter emits for the silence between segments. Its rows
#: carry no transcript, so the empty-transcript rule would drop them anyway; it is
#: named here because a future export that gave them a label should still not
#: produce utterances.
GAP_SPEAKER = "inter_segment_gap"

#: The NIST convention for a region excluded from scoring. It is a marker, not a
#: transcript.
IGNORE_MARKER = "IGNORE_TIME_SEGMENT_IN_SCORING"

STM_LINE = re.compile(
    r"^(?P<recording>\S+)\s+(?P<channel>\S+)\s+(?P<speaker>\S+)\s+"
    r"(?P<start>\d+(?:\.\d+)?)\s+(?P<end>\d+(?:\.\d+)?)\s*"
    r"(?:<(?P<label>[^>]*)>)?\s*(?P<text>.*)$"
)

#: Longest single path component written. Recording names in this corpus are long
#: enough that the tar needs GNU @LongLink entries for them, and a component over
#: the filesystem's limit would fail thousands of segments into the cut.
MAX_COMPONENT = 200


@dataclass(frozen=True)
class Segment:
    utt_id: str
    split: str
    subset: str
    recording: str
    speaker: str
    text: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return max(self.end - self.start, 0.0)


# --- the source page ----------------------------------------------------------


def fetch_page(url: str = SOURCE_PAGE) -> str:
    with urlopen(Request(url, headers=dict(UA)), timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def check_licence(page: str) -> str:
    """Refuse to ingest if the page no longer states CC BY-SA 4.0.

    TalTech publishes no API and no checksum, so this page is the only record of
    the terms, and it is the same host the 159 GB is about to come from. A page
    that cannot be read, or that now says something else, is a question for a
    person rather than something to download through.
    """
    if EXPECTED_LICENCE not in page:
        raise ValueError(
            f"{SOURCE_PAGE} no longer states {EXPECTED_LICENCE!r}. configs/corpora.yaml "
            "records CC BY-SA 4.0 DEED for this corpus; settle the difference before "
            "ingesting it."
        )
    return EXPECTED_LICENCE


# --- members --------------------------------------------------------------


def _parts(name: str) -> tuple[str, ...]:
    return PurePosixPath(name).parts


def is_stm(name: str) -> bool:
    """``<split>/transcriptions/stm/<subset>/<recording>.stm``.

    Selected by suffix as well as by directory: some members under the ``stm``
    tree carry a ``.vtt`` extension, and parsing one of those as STM would
    produce nonsense rather than an error.
    """
    parts = _parts(name)
    return (
        len(parts) >= 5
        and parts[0] in SPLIT_NAMES
        and parts[1] == "transcriptions"
        and parts[2] == "stm"
        and name.lower().endswith(".stm")
    )


def is_audio(name: str) -> bool:
    """``<split>/audio/<subset>/<recording>.wav``."""
    parts = _parts(name)
    return (
        len(parts) >= 4
        and parts[0] in SPLIT_NAMES
        and parts[1] == "audio"
        and name.lower().endswith(".wav")
    )


def wanted_members(archive: Path) -> list[str]:
    """The STM and audio members, which is a third of what the tar holds.

    The VTT and TRS trees are the same transcripts in two other formats and are
    left in the archive rather than written to disk.
    """
    with tarfile.open(archive) as tar:
        return [name for name in tar.getnames() if is_stm(name) or is_audio(name)]


def safe_component(name: str) -> str:
    """A path component short enough to write, keeping it recognisable.

    Truncating alone would collide two recordings that share a long prefix, so a
    digest of the full name goes on the end.
    """
    if len(name.encode("utf-8")) <= MAX_COMPONENT:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name.encode('utf-8')[: MAX_COMPONENT - 16].decode('utf-8', 'ignore')}-{digest}"


# --- transcripts --------------------------------------------------------------


def read_stm(path: Path, split: str, subset: str) -> list[Segment]:
    """Utterances of one STM file, in the order the exporter wrote them.

    Rows with no transcript are dropped: that is the silence between segments,
    which the exporter emits under a pseudo-speaker, and the training split is
    documented to contain material that is not speech at all.
    """
    recording = path.stem
    segments: list[Segment] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(";;"):
            continue
        match = STM_LINE.match(line)
        if match is None:
            continue
        speaker = match["speaker"]
        text = " ".join(match["text"].split())
        if speaker == GAP_SPEAKER or not text or text == IGNORE_MARKER:
            continue
        start, end = float(match["start"]), float(match["end"])
        if end <= start:
            continue
        segments.append(
            Segment(
                # The subset has to be in the id: recording names repeat across
                # subsets — `71_ID117_352639` is in both ERR2020 and
                # intervjuukorpus — and utt_id has to be unique per language.
                utt_id=f"{subset}_{recording}_{round(start * 1000):09d}-{round(end * 1000):09d}",
                split=split,
                subset=subset,
                recording=recording,
                speaker=speaker,
                text=text,
                start=start,
                end=end,
            )
        )
    return segments


def read_transcripts(source: Path) -> list[Segment]:
    """Every utterance in the extracted tree, across all three splits."""
    segments: list[Segment] = []
    for shipped, name in SPLIT_NAMES.items():
        root = source / shipped / "transcriptions" / "stm"
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.stm")):
            segments.extend(read_stm(path, name, path.parent.name))
    return segments


# --- cutting the long-form audio ---------------------------------------------


def segment_path(segment: Segment) -> str:
    """Where one utterance's audio goes, relative to the language directory."""
    return (
        f"audio/{segment.split}/{safe_component(segment.subset)}/"
        f"{safe_component(segment.recording)}/"
        f"{round(segment.start * 1000):09d}-{round(segment.end * 1000):09d}.wav"
    )


def cut_recording(
    source_wav: Path, segments: list[Segment], dest: Path
) -> tuple[dict[str, str], int, int]:
    """Write one WAV per segment, returning ``{utt_id: relative path}``.

    The manifest layer addresses whole files, so a half-hour recording has to
    become its utterances on disk. Frames are copied verbatim: same rate, same
    width, same channel count, no resampling and no decode beyond PCM.
    """
    written: dict[str, str] = {}
    past_end = 0
    with wave.open(str(source_wav), "rb") as source:
        rate = source.getframerate()
        channels, width, frames = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getnframes(),
        )
        for segment in segments:
            start = min(int(segment.start * rate), frames)
            end = min(int(segment.end * rate), frames)
            if end <= start:
                # The transcript runs past the end of the recording. Writing an
                # empty WAV would fail the loader instead of this preparer.
                past_end += 1
                continue
            relative = segment_path(segment)
            target = dest / relative
            written[segment.utt_id] = relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source.setpos(start)
            payload = source.readframes(end - start)
            with wave.open(str(target), "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(payload)
    return written, rate, past_end


def cut_all(
    source: Path, segments: list[Segment], dest: Path, *, keep_sources: bool
) -> tuple[dict[str, str], int | None, int]:
    """Cut every recording that has both audio and transcripts."""
    by_recording: dict[tuple[str, str, str], list[Segment]] = {}
    for segment in segments:
        by_recording.setdefault((segment.split, segment.subset, segment.recording), []).append(
            segment
        )

    (dest / "audio").mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    rate: int | None = None
    past_end = 0
    for (split, subset, recording), group in sorted(by_recording.items()):
        source_wav = source / SHIPPED_SPLIT[split] / "audio" / subset / f"{recording}.wav"
        if not source_wav.exists():
            continue
        part, found, skipped = cut_recording(source_wav, group, dest)
        written.update(part)
        past_end += skipped
        rate = rate or found
        if not keep_sources:
            source_wav.unlink(missing_ok=True)
    return written, rate, past_end


def write_manifest(dest: Path, segments: list[Segment], audio: dict[str, str]) -> Path:
    path = dest / "manifest.tsv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for segment in sorted(segments, key=lambda s: s.utt_id):
            relative = audio.get(segment.utt_id)
            if relative is None:
                continue
            writer.writerow(
                [segment.utt_id, relative, segment.text, segment.speaker, segment.split]
            )
    return path


# --- the preparer -------------------------------------------------------------


def prepare(root: Path, *, keep_archives: bool = False) -> Path:
    """Fetch, extract, cut and write the manifest for Estonian."""
    dest = root / CORPUS_ID / "et"
    staging = dest / ".archives"
    source = dest / ".source"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[svb] et: {CORPUS_ID}")
    licence = check_licence(fetch_page())
    print(f"    licence: {licence}")

    archive = download(ARCHIVE_URL, staging / ARCHIVE, expected=remote_size(ARCHIVE_URL))
    problem = verify_archive(archive)
    if problem:
        archive.unlink(missing_ok=True)
        raise OSError(f"{archive.name}: {problem}; deleted, re-run to fetch it again")
    record = archive_record(archive, ARCHIVE_URL)

    members = wanted_members(archive)
    print(f"    extracting {len(members)} of the tar's members (STM and audio only)")
    extract_archive(archive, source, members=members)
    if not keep_archives:
        archive.unlink(missing_ok=True)

    segments = read_transcripts(source)
    if not segments:
        raise ValueError(
            f"no STM transcripts under {source}; the archive layout has changed "
            "(expected <split>/transcriptions/stm/<subset>/*.stm)"
        )

    audio, rate, past_end = cut_all(source, segments, dest, keep_sources=keep_archives)
    manifest = write_manifest(dest, segments, audio)
    counts = {
        name: sum(1 for s in segments if s.split == name and s.utt_id in audio)
        for name in SPLIT_NAMES.values()
    }
    hours = {
        name: round(
            sum(s.seconds for s in segments if s.split == name and s.utt_id in audio) / 3600, 2
        )
        for name in SPLIT_NAMES.values()
    }
    print(f"    {len(audio)} utterances cut {counts}, hours {hours}")

    write_fetch_manifest(
        dest,
        {
            "corpus": CORPUS_ID,
            "language": "et",
            "source_page": SOURCE_PAGE,
            "licence": licence,
            "split_policy": "shipped",
            "counts": counts,
            "hours": hours,
            "speakers": {
                name: len({s.speaker for s in segments if s.split == name and s.utt_id in audio})
                for name in SPLIT_NAMES.values()
            },
            "sample_rate_hz": rate,
            "segments_without_audio": len(segments) - len(audio),
            "segments_past_end_of_recording": past_end,
            "integrity": "tar read end to end; the publisher publishes no checksum",
            "speaker_ids": "per recording — the recording id prefixes every speaker id in its STM",
            "archives": [record],
        },
    )
    if not keep_archives:
        shutil.rmtree(source, ignore_errors=True)
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="$CORPORA_ROOT")
    parser.add_argument(
        "--langs", nargs="*", choices=LANGUAGES, help="only 'et'; the flag exists for symmetry"
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep the tar and the long-form recordings so a re-cut is free",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for _ in args.langs or LANGUAGES:
        prepare(args.root, keep_archives=args.keep_archives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
