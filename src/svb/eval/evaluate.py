"""Evaluation: greedy CTC decode over a dataset → WER / CER.

Saves a per-utterance predictions sidecar (ref/hyp) so bootstrap CIs and paired
permutation tests can be recomputed from disk without re-running the model.
``hybrid`` substitutes CER for no-space scripts (Thai/CJK) where word-level WER
is meaningless under character-concatenation tokenization.

Batching is part of the protocol, not an implementation detail. The encoder's
two time-axis convolutions do not mask padded frames (see the note in
``xeus_standalone``), so an utterance's encoding depends slightly on what was
padded beside it. Utterances are therefore evaluated in descending length
order: batch membership becomes a fixed function of the split and the batch
size rather than of the order rows happen to sit in on disk. Results remain
comparable only between runs that used the same batch size.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..model.ctc_vocab import CtcVocab
from ..model.xeus_ctc import XeusCTC

# Languages without word boundaries — report CER, not WER.
NO_SPACE_LANGS = {"thai", "mandarin", "cantonese", "japanese", "yue", "zh", "ja", "th"}


@dataclass
class EvalResult:
    wer: float
    cer: float
    n: int
    refs: list[str] = field(default_factory=list)
    hyps: list[str] = field(default_factory=list)

    def primary(self, lang_code: str) -> float:
        return self.cer if lang_code in NO_SPACE_LANGS else self.wer


def _wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    return 100.0 * jiwer.wer(refs, hyps)


def _cer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    return 100.0 * jiwer.cer(refs, hyps)


def _length_sorted_indices(dataset: Dataset) -> list[int]:
    """Dataset indices in descending waveform length, ties by original index.

    Costs one extra pass over the split, which for the audio corpora means
    decoding each clip twice. That is the price of batch membership being
    reproducible; pass ``sort_by_length=False`` to skip it.
    """
    lengths = [int(dataset[i][0].shape[0]) for i in range(len(dataset))]  # type: ignore[arg-type]
    return sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))


@torch.no_grad()
def evaluate(
    model: XeusCTC,
    dataset: Dataset,
    vocab: CtcVocab,
    collate: Callable,
    device: str = "cuda",
    batch_size: int = 8,
    save_predictions: Path | None = None,
    sort_by_length: bool = True,
) -> EvalResult:
    """Greedy-decode ``dataset`` and compute corpus WER/CER.

    Args:
        sort_by_length: Group utterances of similar length into the same batch,
            so results do not depend on the split's row order. Refs and hyps are
            reported back in dataset order either way, which keeps the sidecar
            stable across the setting.
    """
    model.to(device).eval()
    order = _length_sorted_indices(dataset) if sort_by_length else None
    batched: Dataset = Subset(dataset, order) if order is not None else dataset
    loader = DataLoader(batched, batch_size=batch_size, shuffle=False, collate_fn=collate)
    refs: list[str] = []
    hyps: list[str] = []
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_ids, pred_lens = model.greedy_decode(input_values, attention_mask=attn)  # (B, T')
        for row, valid in zip(pred_ids, pred_lens.tolist(), strict=True):
            # Slice off the padded tail before collapsing: frames past the
            # utterance's own length belong to whatever else shared the batch.
            hyps.append(vocab.decode(row[: int(valid)].tolist()))
        # References are the corpus transcripts, never a round-trip through the
        # label ids: characters absent from the training vocab encode to <unk>
        # and would be silently deleted from the reference.
        refs.extend(batch["texts"])

    if order is not None:
        # Undo the length sort so the sidecar lines up with the split on disk.
        restored_refs = [""] * len(order)
        restored_hyps = [""] * len(order)
        for position, index in enumerate(order):
            restored_refs[index] = refs[position]
            restored_hyps[index] = hyps[position]
        refs, hyps = restored_refs, restored_hyps

    result = EvalResult(
        wer=_wer(refs, hyps), cer=_cer(refs, hyps), n=len(refs), refs=refs, hyps=hyps
    )
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        save_predictions.write_text(
            json.dumps(
                {
                    "wer": result.wer,
                    "cer": result.cer,
                    "n": result.n,
                    "pairs": list(zip(refs, hyps)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return result
