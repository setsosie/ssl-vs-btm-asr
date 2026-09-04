"""Common Voice (local) loader.

As of October 2025 Common Voice is distributed only via Mozilla Data Collective
(no longer on the HF Hub past v17). So CV25 is read from a **local directory** —
the standard release layout after you download and extract it:

    $CV_ROOT/
        <lang>/
            train.tsv  dev.tsv  test.tsv   (tab-separated; cols incl. path, sentence)
            validated.tsv                  (every validated clip)
            clips/<file>.mp3

Set ``CV_ROOT`` (or pass ``root``). mp3 decoding uses torchaudio's ffmpeg
backend — install ffmpeg if clips fail to load.

Two training sources
====================

``train``
    The official ``train.tsv``. What this repository read until now.

``validated_minus_eval`` (default)
    Every validated clip that is not in ``dev.tsv`` or ``test.tsv`` and does not
    belong to a speaker who appears in either. This is the standard Common Voice
    recipe and it is much larger: the official ``train.tsv`` is roughly one clip
    per sentence, so across the release it holds about a third of the validated
    audio, and it is what makes most Common Voice languages look too small to
    train on.

Why the speaker guard is not a formality
----------------------------------------

The release documentation says only this about how the partition is made:

    We use the Corpora Creator tool to parse through metadata to generate
    train, dev, and test sets. The Corpora Creator eliminates duplication in
    clips and maximizes for speaker diversity.

    Each train/dev/test set is generated non-deterministically, meaning they
    will vary from release to release even for minor updates. [...] Note that
    total clips in these sets will most probably not add up to the total
    validated clips because of this limitation.

    -- github.com/common-voice/cv-dataset, datasets/scripted-speech/README.md

That is not a guarantee about speakers, and the guarantee that does exist does
not cover this mode. In Corpora Creator (``src/corporacreator/corpus.py``, blob
``da6233c3d76d13627e569763cd6cdcfb32540b80``) the split label is assigned one
whole ``client_id`` at a time, so ``train``, ``dev`` and ``test`` are indeed
speaker-disjoint from each other — but the labels are assigned over the
*deduplicated* validated frame, and the test split is then further truncated
with ``.head(test_size)``. Both leave clips in ``validated.tsv`` that are in no
split at all, and some of those belong to the dev and test speakers. Subtracting
dev and test by clip path alone would therefore train the model on the voices it
is about to be evaluated on. Hence the guard, and hence the per-language count
of what it removed.

The same fact is what makes ``validated_minus_eval`` a superset of ``train``:
no training speaker is a dev or test speaker, so the guard never removes a row
the official split had kept.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

TARGET_SR = 16000
_SPLIT_FILE = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
#: Every clip with two or more validations and more up-votes than down-votes.
VALIDATED_FILE = "validated.tsv"
#: The split files a training clip must be kept clear of, by path and by speaker.
_EVAL_FILES = ("dev.tsv", "test.tsv")

TrainSource = Literal["train", "validated_minus_eval"]
TRAIN_SOURCES: tuple[TrainSource, ...] = ("train", "validated_minus_eval")
DEFAULT_TRAIN_SOURCE: TrainSource = "validated_minus_eval"


def _cv_root(root: str | None) -> Path:
    root = root or os.environ.get("CV_ROOT")
    if not root:
        raise RuntimeError(
            "Common Voice is local-only (Mozilla Data Collective). "
            "Set $CV_ROOT to the extracted CV25 directory (see README)."
        )
    return Path(root)


@dataclass(frozen=True)
class CvRow:
    """One row of a Common Voice split file."""

    path: str  # clip filename
    text: str  # transcript
    client_id: str  # hashed contributor id; "" when the release omits the column


def _read_full_rows(tsv: Path, text_column: str) -> list[CvRow]:
    """Rows of a CV tsv, dropping any with no transcript or no clip path."""
    rows: list[CvRow] = []
    with open(tsv, encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            text = (record.get(text_column) or "").strip()
            path = (record.get("path") or "").strip()
            if text and path:
                rows.append(
                    CvRow(path=path, text=text, client_id=(record.get("client_id") or "").strip())
                )
    return rows


def _read_rows(tsv: Path, text_column: str) -> list[tuple[str, str]]:
    """Return (clip_filename, sentence) rows from a CV split tsv."""
    return [(row.path, row.text) for row in _read_full_rows(tsv, text_column)]


@dataclass(frozen=True)
class TrainSelection:
    """The rows a training run reads for one language, and what was held back.

    The three counts are disjoint and sum with ``len(rows)`` to ``n_validated``,
    so a reader can check the accounting rather than take it on trust. Under
    ``train_source="train"`` nothing is filtered and they are all zero.
    """

    rows: tuple[tuple[str, str], ...]  # (clip filename, transcript)
    train_source: str
    #: Rows read from ``validated.tsv``.
    n_validated: int = 0
    #: Clips dropped because they are themselves in ``dev.tsv`` or ``test.tsv``.
    in_eval_splits: tuple[str, ...] = ()
    #: Clips dropped because the *speaker* appears in ``dev.tsv`` or
    #: ``test.tsv``, though the clip is in neither. These are the leak the guard
    #: exists for. Named rather than counted so the duration audit can price
    #: them against the release's own clip-duration manifest.
    eval_speaker_clips: tuple[str, ...] = ()
    #: Distinct speakers making up ``eval_speaker_clips``.
    n_eval_speakers: int = 0

    @property
    def n_in_eval_splits(self) -> int:
        return len(self.in_eval_splits)

    @property
    def n_eval_speaker_clips(self) -> int:
        return len(self.eval_speaker_clips)


def _require(tsv: Path) -> Path:
    if not tsv.exists():
        raise FileNotFoundError(f"missing CV split: {tsv}")
    return tsv


def select_train_rows(
    lang: str,
    root: str | None = None,
    text_column: str = "sentence",
    train_source: str = DEFAULT_TRAIN_SOURCE,
) -> TrainSelection:
    """The training rows for one language under the chosen source.

    Raises:
        ValueError: for an unknown source, or when the release omits
            ``client_id`` and the speaker guard therefore cannot run.
        FileNotFoundError: when a file the source needs is not there.
    """
    if train_source not in TRAIN_SOURCES:
        raise ValueError(
            f"{lang}: unknown train_source {train_source!r}; expected one of {TRAIN_SOURCES}"
        )
    base = _cv_root(root) / lang

    if train_source == "train":
        rows = _read_full_rows(_require(base / _SPLIT_FILE["train"]), text_column)
        return TrainSelection(
            rows=tuple((row.path, row.text) for row in rows), train_source=train_source
        )

    validated = _read_full_rows(_require(base / VALIDATED_FILE), text_column)
    evaluation = [
        row for name in _EVAL_FILES for row in _read_full_rows(_require(base / name), text_column)
    ]

    if not all(row.client_id for row in (*validated, *evaluation)):
        raise ValueError(
            f"{lang}: {VALIDATED_FILE} or a split file has no client_id column, so the "
            "speaker guard cannot run. Training on validated_minus_eval without it would "
            "put the dev and test speakers in the training set; use train_source='train' "
            "or a release that ships client_id."
        )

    eval_paths = {row.path for row in evaluation}
    eval_speakers = {row.client_id for row in evaluation}

    kept: list[tuple[str, str]] = []
    in_eval_splits: list[str] = []
    eval_speaker_clips: list[str] = []
    leaked_speakers: set[str] = set()
    for row in validated:
        if row.path in eval_paths:
            in_eval_splits.append(row.path)
        elif row.client_id in eval_speakers:
            eval_speaker_clips.append(row.path)
            leaked_speakers.add(row.client_id)
        else:
            kept.append((row.path, row.text))

    return TrainSelection(
        rows=tuple(kept),
        train_source=train_source,
        n_validated=len(validated),
        in_eval_splits=tuple(in_eval_splits),
        eval_speaker_clips=tuple(eval_speaker_clips),
        n_eval_speakers=len(leaked_speakers),
    )


def _rows_for_split(
    lang: str,
    split: str,
    root: str | None,
    text_column: str,
    train_source: str,
) -> list[tuple[str, str]]:
    """The single row reader every entry point below goes through.

    Training rows depend on the source and evaluation rows never do, and that
    distinction lives here so the dataset, the transcript reader and the
    duration accounting cannot disagree about which rows a split contains.
    """
    if split == "train":
        return list(select_train_rows(lang, root, text_column, train_source).rows)
    base = _cv_root(root) / lang
    return _read_rows(_require(base / _SPLIT_FILE[split]), text_column)


class CommonVoiceLocal(Dataset):
    """One CV language split as (waveform@16k, text, language code)."""

    def __init__(
        self,
        lang: str,
        split: str,
        root: str | None = None,
        text_column: str = "sentence",
        max_samples: int | None = None,
        train_source: str = DEFAULT_TRAIN_SOURCE,
    ):
        # Carried on the item so the collate can normalize each transcript
        # under its own language's policy.
        self.lang = lang
        self._clips = _cv_root(root) / lang / "clips"
        self._rows = _rows_for_split(lang, split, root, text_column, train_source)
        self._max = max_samples

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        import torchaudio

        fname, text = self._rows[idx]
        wav, sr = torchaudio.load(str(self._clips / fname))  # (channels, time)
        wav = wav.mean(dim=0)  # mono
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if self._max is not None and wav.shape[0] > self._max:
            wav = wav[: self._max]
        return wav.float(), text, self.lang


def load_cv_texts(
    lang: str,
    split: str,
    root: str | None = None,
    text_column: str = "sentence",
    train_source: str = DEFAULT_TRAIN_SOURCE,
) -> list[str]:
    return [text for _, text in _rows_for_split(lang, split, root, text_column, train_source)]


def cv_split_dir(lang: str, root: str | None = None) -> Path:
    """The directory holding one language's split tsvs and ``clips/``."""
    return _cv_root(root) / lang


def load_cv_rows(
    lang: str,
    split: str,
    root: str | None = None,
    text_column: str = "sentence",
    train_source: str = DEFAULT_TRAIN_SOURCE,
) -> list[tuple[str, str]]:
    """``(clip filename, transcript)`` for one split.

    The audio accounting needs the filenames, which ``load_cv_texts`` discards.
    Both go through the same reader so the two cannot disagree about which rows
    a split contains.
    """
    return _rows_for_split(lang, split, root, text_column, train_source)
