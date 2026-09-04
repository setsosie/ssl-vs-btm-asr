"""Thin, source-dispatching dataset loader over public corpora.

Returns a torch ``Dataset`` of ``(waveform: 1-D float tensor @16kHz, text: str)``
pairs. Both Common Voice (via HF ``datasets``) and OpenSLR (HF mirrors) expose
an ``audio`` column with ``{array, sampling_rate}`` and a text column, so one
adapter covers both; the ``source`` field is kept for provenance and for
per-corpus handling if it ever diverges.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .registry import LangSpec

TARGET_SR = 16000


class AudioTextDataset(Dataset):
    """Wraps a HF dataset split as (waveform@16k, normalized text)."""

    def __init__(self, hf_split, spec: LangSpec, max_samples: int | None = None):
        self._ds = hf_split
        self._spec = spec
        self._max = max_samples

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self._ds[idx]
        audio = row[self._spec.audio_column]
        wav = torch.as_tensor(audio["array"], dtype=torch.float32)
        sr = int(audio["sampling_rate"])
        if sr != TARGET_SR:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if wav.dim() > 1:  # stereo -> mono
            wav = wav.mean(dim=0)
        if self._max is not None and wav.shape[0] > self._max:
            wav = wav[: self._max]
        text = str(row[self._spec.text_column]).strip()
        return wav, text


# Deterministic partition for corpora that ship a single split (e.g. OpenSLR).
_DERIVE_FRACTIONS = {"train": (0.0, 0.8), "validation": (0.8, 0.9), "test": (0.9, 1.0)}
_DERIVE_SEED = 1234


def _resolve_split(spec: LangSpec, split: str):
    """Return a HF split, deriving a deterministic partition if it's missing.

    Corpora with named train/validation/test splits return them directly. Single
    -split corpora are shuffled once with a fixed seed and sliced 80/10/10 so the
    test set is disjoint and reproducible.
    """
    from datasets import get_dataset_split_names, load_dataset

    canonical = {"validation": ["validation", "dev", "valid"], "test": ["test"], "train": ["train"]}
    try:
        available = set(get_dataset_split_names(spec.hf_dataset, spec.hf_config))
    except Exception:
        available = set()

    for name in canonical.get(split, [split]):
        if name in available:
            return load_dataset(spec.hf_dataset, spec.hf_config, split=name)

    # Derive from the single available split (prefer "train", else the first).
    source = "train" if "train" in available else (sorted(available)[0] if available else "train")
    full = load_dataset(spec.hf_dataset, spec.hf_config, split=source)
    full = full.shuffle(seed=_DERIVE_SEED)
    lo, hi = _DERIVE_FRACTIONS[split]
    n = len(full)
    return full.select(range(int(lo * n), int(hi * n)))


def load_language(spec: LangSpec, split: str, max_samples: int | None = None):
    """Load one language split as a torch Dataset of (waveform@16k, text).

    Dispatches on ``spec.source``:
      - ``commonvoice`` → local CV directory ($CV_ROOT), the only path post-Oct-2025.
      - ``openslr`` → HF Hub (open), with deterministic split derivation.

    Args:
        spec: Language spec.
        split: "train", "validation", or "test".
        max_samples: Per-utterance audio truncation guard (samples @16k).
    """
    if spec.source == "commonvoice":
        from .commonvoice_local import CommonVoiceLocal

        return CommonVoiceLocal(
            lang=spec.hf_config, split=split, text_column=spec.text_column, max_samples=max_samples
        )

    from datasets import Audio

    ds = _resolve_split(spec, split)
    try:
        ds = ds.cast_column(spec.audio_column, Audio(sampling_rate=TARGET_SR))
    except Exception:
        pass  # already decoded / non-Audio column
    return AudioTextDataset(ds, spec, max_samples=max_samples)


def load_texts(spec: LangSpec, split: str) -> list[str]:
    """Load just the transcripts for a split (no audio decode) — for vocab build."""
    if spec.source == "commonvoice":
        from .commonvoice_local import load_cv_texts

        return load_cv_texts(spec.hf_config, split, text_column=spec.text_column)

    ds = _resolve_split(spec, split)
    keep = spec.text_column
    drop = [c for c in ds.column_names if c != keep]
    ds = ds.remove_columns(drop)
    return [str(t).strip() for t in ds[keep]]
