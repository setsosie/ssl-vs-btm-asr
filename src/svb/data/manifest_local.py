"""One reader for every corpus that is not Common Voice.

The fourteen corpora the large preset draws on ship in about ten different
shapes — per-utterance XML, JSONL with word alignments, Kaldi-ish flat
directories, STM and VTT, one `.txt` per `.wav`. Writing ten dataset classes
would mean ten places to get a split wrong. Instead each corpus gets a
*preparer* (``scripts/prepare_<corpus>.py``) that converts it once into a
manifest, and this module is the only thing that reads audio::

    $CORPORA_ROOT/
        <corpus id>/
            <language code>/
                manifest.tsv
                <audio, wherever the manifest's path column says>

``manifest.tsv`` is tab-separated with a header and exactly these columns:

``utt_id``
    Stable id for the utterance. It is the identity used to order the rows, to
    digest the test set, and as the fallback grouping key when there is no
    speaker. It must be unique within a language.
``path``
    Audio path, relative to the language directory.
``text``
    The transcript, already extracted from whatever the corpus shipped.
``speaker``
    Speaker id, or empty. Empty on *any* row drops the whole language to an
    utterance-level split, because a partition that is speaker-disjoint for most
    rows is not speaker-disjoint.
``split``
    ``train``, ``validation``, ``test``, or empty. Either every row carries one
    or none does; see :func:`_partition`.

Audio is read with ``soundfile`` where it can be (wav, flac) and through
``torchaudio`` otherwise (mp3, m4a — which needs an ffmpeg backend), then
downmixed to mono and resampled to 16 kHz. Items are ``(waveform, text,
language code)``, the same shape the Common Voice and OpenSLR loaders return, so
the collate can normalize each transcript under its own language's policy.
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .registry import LangSpec
from .splits import SPLITS
from .splits import derive_splits as _derive_splits

TARGET_SR = 16000
MANIFEST = "manifest.tsv"
COLUMNS = ("utt_id", "path", "text", "speaker", "split")
#: Extensions ``soundfile`` reads directly. Anything else goes through
#: torchaudio, which needs an ffmpeg backend for mp3 and m4a.
_SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg", ".opus", ".aiff", ".au"}


@dataclass(frozen=True)
class ManifestRow:
    utt_id: str
    path: str
    text: str
    speaker: str
    split: str


def _corpora_root(root: str | None) -> Path:
    root = root or os.environ.get("CORPORA_ROOT")
    if not root:
        raise RuntimeError(
            "Corpora outside Common Voice are read from a local directory prepared by "
            "scripts/prepare_<corpus>.py. Set $CORPORA_ROOT to the preparation root "
            "(see docs/data.md)."
        )
    return Path(root)


def _language_dir(spec: LangSpec, root: str | None) -> Path:
    return _corpora_root(root) / spec.corpus / spec.hf_config


def read_manifest(path: Path) -> list[ManifestRow]:
    """Rows of one ``manifest.tsv``, with the column check done up front."""
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. A corpus reaches this loader only through its preparer; "
            f"run the matching scripts/prepare_<corpus>.py first (see docs/data.md)."
        )
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = sorted(set(COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path}: manifest is missing columns {missing}")
        rows = [
            ManifestRow(
                utt_id=(record.get("utt_id") or "").strip(),
                path=(record.get("path") or "").strip(),
                text=(record.get("text") or "").strip(),
                speaker=(record.get("speaker") or "").strip(),
                split=(record.get("split") or "").strip(),
            )
            for record in reader
        ]
    usable = [row for row in rows if row.utt_id and row.path and row.text]
    if not usable:
        raise ValueError(f"{path}: no rows with an id, a path and a transcript")

    unknown = sorted({row.split for row in usable if row.split and row.split not in SPLITS})
    if unknown:
        # "dev" is the Common Voice file name and the mistake a preparer author
        # makes from habit. Dropping those rows would silently shrink the corpus.
        raise ValueError(f"{path}: split column has unknown values {unknown}; expected {SPLITS}")
    return usable


def _partition(
    rows: list[ManifestRow], path: Path
) -> tuple[dict[str, list[ManifestRow]], str, str]:
    """Split the rows, either as the corpus shipped them or by deriving one.

    A manifest where only some rows carry a split is refused. Deriving over the
    leftovers would produce a partition that is half shipped and half derived,
    which no write-up could describe; the two corpora with an incomplete
    published split — Kannada has no dev, Zeroth's test is under the evaluation
    bar — resolve it in their preparer by clearing the column and re-deriving,
    which is a decision that belongs there and not here.
    """
    shipped = [row for row in rows if row.split]
    if shipped and len(shipped) != len(rows):
        raise ValueError(
            f"{path}: some rows carry a split and some do not ({len(shipped)} of {len(rows)}). "
            "Either fill the column for every row or clear it for every row; a partition that "
            "is partly shipped and partly derived cannot be described in a results file."
        )

    if shipped:
        parts = {
            name: sorted((r for r in rows if r.split == name), key=lambda r: r.utt_id)
            for name in SPLITS
        }
        listing = "\n".join(row.utt_id for row in parts["test"])
        return parts, "shipped", hashlib.sha1(listing.encode("utf-8")).hexdigest()

    return _derive_splits(
        rows,
        lambda row: row.speaker or None,
        lambda row: row.utt_id,
    )


def _rows_for_split(
    spec: LangSpec, split: str, root: str | None
) -> tuple[Path, list[ManifestRow], str, str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    base = _language_dir(spec, root)
    parts, policy, digest = _partition(read_manifest(base / MANIFEST), base / MANIFEST)
    return base, parts[split], policy, digest


class ManifestLocal(Dataset):
    """One split of one prepared corpus as (waveform@16k, text, language code)."""

    def __init__(
        self,
        spec: LangSpec,
        split: str,
        root: str | None = None,
        max_samples: int | None = None,
    ):
        self.base, self.rows, self.split_policy, self.test_files_sha1 = _rows_for_split(
            spec, split, root
        )
        self.spec = spec
        self.split = split
        self._max = max_samples

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        row = self.rows[idx]
        wav, sr = _read_audio(self.base / row.path)
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if sr != TARGET_SR:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if self._max is not None and wav.shape[0] > self._max:
            wav = wav[: self._max]
        return wav.float(), row.text, self.spec.code


def _read_audio(path: Path) -> tuple[torch.Tensor, int]:
    """``(samples, rate)``, from soundfile where it can read the container."""
    if path.suffix.lower() in _SOUNDFILE_SUFFIXES:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return torch.as_tensor(data, dtype=torch.float32), int(sr)

    import torchaudio

    wav, sr = torchaudio.load(str(path))  # (channels, time)
    return wav.transpose(0, 1), int(sr)


def load_manifest_texts(spec: LangSpec, split: str, root: str | None = None) -> list[str]:
    """Transcripts of one split, with no audio decode."""
    _, rows, _, _ = _rows_for_split(spec, split, root)
    return [row.text for row in rows]
