"""Common Voice (local) loader.

As of October 2025 Common Voice is distributed only via Mozilla Data Collective
(no longer on the HF Hub past v17). So CV25 is read from a **local directory** —
the standard release layout after you download and extract it:

    $CV_ROOT/
        <lang>/
            train.tsv  dev.tsv  test.tsv   (tab-separated; cols incl. path, sentence)
            clips/<file>.mp3

Set ``CV_ROOT`` (or pass ``root``). mp3 decoding uses torchaudio's ffmpeg
backend — install ffmpeg if clips fail to load.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

TARGET_SR = 16000
_SPLIT_FILE = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}


def _cv_root(root: str | None) -> Path:
    root = root or os.environ.get("CV_ROOT")
    if not root:
        raise RuntimeError(
            "Common Voice is local-only (Mozilla Data Collective). "
            "Set $CV_ROOT to the extracted CV25 directory (see README)."
        )
    return Path(root)


def _read_rows(tsv: Path, text_column: str) -> list[tuple[str, str]]:
    """Return (clip_filename, sentence) rows from a CV split tsv."""
    rows: list[tuple[str, str]] = []
    with open(tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            text = (r.get(text_column) or "").strip()
            path = (r.get("path") or "").strip()
            if text and path:
                rows.append((path, text))
    return rows


class CommonVoiceLocal(Dataset):
    """One CV language split as (waveform@16k, text)."""

    def __init__(
        self,
        lang: str,
        split: str,
        root: str | None = None,
        text_column: str = "sentence",
        max_samples: int | None = None,
    ):
        base = _cv_root(root) / lang
        tsv = base / _SPLIT_FILE[split]
        if not tsv.exists():
            raise FileNotFoundError(f"missing CV split: {tsv}")
        self._clips = base / "clips"
        self._rows = _read_rows(tsv, text_column)
        self._max = max_samples

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        import torchaudio

        fname, text = self._rows[idx]
        wav, sr = torchaudio.load(str(self._clips / fname))  # (channels, time)
        wav = wav.mean(dim=0)  # mono
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if self._max is not None and wav.shape[0] > self._max:
            wav = wav[: self._max]
        return wav.float(), text


def load_cv_texts(
    lang: str, split: str, root: str | None = None, text_column: str = "sentence"
) -> list[str]:
    base = _cv_root(root) / lang
    return [t for _, t in _read_rows(base / _SPLIT_FILE[split], text_column)]
