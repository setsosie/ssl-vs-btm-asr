"""CTC collate: pad waveforms + attention mask, encode + pad labels.

Labels are padded with ``-100`` so the model's CTC loss can recover per-sample
target lengths via ``(labels != -100).sum(-1)``.

The batch also carries ``texts``, the raw transcripts. Scoring must use those
rather than decoding the label ids back to characters: any character the
training vocab never saw encodes to ``<unk>`` and would vanish from the
reference, quietly shortening the transcript the model is measured against.
Consumers that forward the batch into the model must therefore pass only its
tensor entries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..model.ctc_vocab import CtcVocab
from ..model.xeus_standalone import max_label_len_for_samples


def make_ctc_collate(
    vocab: CtcVocab,
    *,
    max_audio_samples: int | None = None,
    drop_overlong: bool = False,
) -> Callable:
    """Return a collate_fn closed over a vocab.

    Args:
        vocab: Character vocab used to encode targets.
        max_audio_samples: The training-time truncation guard, in samples. Only
            used to count how many utterances reach it; the truncation itself
            happens in the dataset. Pass ``None`` for evaluation, where no
            truncation is applied and the count is meaningless.
        drop_overlong: Training only. Drop pairs whose transcript is longer than
            the waveform's encoder frame budget. Such a pair cannot be aligned
            by CTC, so it would score as a free zero instead of a large loss.
            Never enable this for evaluation: dropping test utterances would
            quietly shrink the test set.

    The batch reports ``n_dropped`` and ``n_at_audio_guard`` per call rather
    than accumulating them on the collate object, because DataLoader workers run
    in separate processes and their counters would never reach the caller.
    """

    def collate(batch: list[tuple[torch.Tensor, str]]) -> dict[str, Any]:
        kept: list[tuple[torch.Tensor, str, list[int]]] = []
        n_dropped = 0
        n_at_audio_guard = 0
        for wav, text in batch:
            n_samples = int(wav.shape[0])
            ids = vocab.encode(text)
            if max_audio_samples is not None and n_samples >= max_audio_samples:
                n_at_audio_guard += 1
            if drop_overlong and len(ids) > max_label_len_for_samples(n_samples):
                n_dropped += 1
                continue
            kept.append((wav, text, ids))

        wavs = [w for w, _, _ in kept]
        max_len = max((int(w.shape[0]) for w in wavs), default=0)
        padded = torch.zeros(len(kept), max_len, dtype=torch.float32)
        attn = torch.zeros(len(kept), max_len, dtype=torch.long)
        for i, w in enumerate(wavs):
            padded[i, : w.shape[0]] = w
            attn[i, : w.shape[0]] = 1

        max_lab = max(1, max((len(ids) for _, _, ids in kept), default=1))
        labels = torch.full((len(kept), max_lab), -100, dtype=torch.long)
        for i, (_, _, ids) in enumerate(kept):
            labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {
            "input_values": padded,
            "attention_mask": attn,
            "labels": labels,
            "texts": [t for _, t, _ in kept],
            "n_dropped": n_dropped,
            "n_at_audio_guard": n_at_audio_guard,
        }

    return collate
