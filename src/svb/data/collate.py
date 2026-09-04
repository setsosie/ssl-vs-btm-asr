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


def make_ctc_collate(vocab: CtcVocab) -> Callable:
    """Return a collate_fn closed over a vocab."""

    def collate(batch: list[tuple[torch.Tensor, str]]) -> dict[str, Any]:
        wavs, texts = zip(*batch, strict=True)
        lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
        max_len = int(lengths.max())
        padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
        attn = torch.zeros(len(wavs), max_len, dtype=torch.long)
        for i, w in enumerate(wavs):
            padded[i, : w.shape[0]] = w
            attn[i, : w.shape[0]] = 1

        label_ids = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        max_lab = max(1, max(len(x) for x in label_ids))
        labels = torch.full((len(batch), max_lab), -100, dtype=torch.long)
        for i, ids in enumerate(label_ids):
            labels[i, : len(ids)] = ids

        return {
            "input_values": padded,
            "attention_mask": attn,
            "labels": labels,
            "texts": list(texts),
        }

    return collate
