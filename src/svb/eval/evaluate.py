"""Evaluation: greedy CTC decode over a dataset → WER / CER.

Saves a per-utterance predictions sidecar (ref/hyp) so bootstrap CIs and paired
permutation tests can be recomputed from disk without re-running the model.

This is the third of the three sites that touch text. References are the
normalized transcripts the collate produced, so the model is measured against
exactly the string it was trained to emit; hypotheses go through the same
policy, which is close to a no-op except that it collapses the leading,
trailing and doubled spaces a greedy decode does produce and that would
otherwise score as word errors the model had no way to avoid.

Which of WER and CER is primary follows the language's ``word_boundary``, since
whitespace tokenization of a script written without spaces yields one token per
sentence. Both are always computed and reported; only the choice of primary is
per language.

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
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..data.registry import LangSpec
from ..model.ctc_vocab import CtcVocab
from ..model.xeus_ctc import XeusCTC
from ..text.normalize import normalize_text
from ..text.stats import collect_text_stats


@dataclass
class EvalResult:
    wer: float
    cer: float
    n: int
    # Utterances whose reference normalized to nothing. Excluded from both
    # metrics: an empty reference adds nothing to the denominator and its whole
    # hypothesis to the numerator, so keeping it inflates corpus WER with no
    # weight behind it. Reported because dropping test utterances silently is a
    # change to the test set.
    n_empty_refs: int = 0
    refs: list[str] = field(default_factory=list)
    hyps: list[str] = field(default_factory=list)

    def primary(self, spec: LangSpec) -> float:
        return self.wer if spec.word_boundary else self.cer


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
    spec: LangSpec | None = None,
) -> EvalResult:
    """Greedy-decode ``dataset`` and compute corpus WER/CER.

    Args:
        sort_by_length: Group utterances of similar length into the same batch,
            so results do not depend on the split's row order. Refs and hyps are
            reported back in dataset order either way, which keeps the sidecar
            stable across the setting.
        spec: The language being evaluated. Supplying it checks the preset's
            ``word_boundary`` declaration against the transcripts themselves, so
            a mislabelled language is caught by the data rather than by a reader
            noticing that a word error rate looks impossible.
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
            hyps.append(normalize_text(vocab.decode(row[: int(valid)].tolist()), vocab.policy))
        # References are the normalized corpus transcripts, never a round-trip
        # through the label ids: characters absent from the training vocab
        # encode to <unk> and would be silently deleted from the reference.
        refs.extend(batch["texts"])

    if order is not None:
        # Undo the length sort so the sidecar lines up with the split on disk.
        # The mapping is positional, so a collate that dropped anything would
        # silently pair hypothesis i with reference j. The training collate does
        # drop rows; passing it here instead of the evaluation one is the way
        # this happens, so name it rather than scoring a shuffled corpus.
        if len(refs) != len(order):
            raise ValueError(
                f"the collate returned {len(refs)} utterances for {len(order)} inputs, so "
                "rows were dropped; evaluation must use a collate with drop_overlong and "
                "drop_empty off, or the length sort cannot be undone"
            )
        restored_refs = [""] * len(order)
        restored_hyps = [""] * len(order)
        for position, index in enumerate(order):
            restored_refs[index] = refs[position]
            restored_hyps[index] = hyps[position]
        refs, hyps = restored_refs, restored_hyps

    scoreable = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    n_empty_refs = len(refs) - len(scoreable)
    if not scoreable:
        raise ValueError(
            f"every one of {len(refs)} references normalized to nothing, so there is "
            "nothing to score; the transcripts or the text column are wrong"
        )
    refs = [r for r, _ in scoreable]
    hyps = [h for _, h in scoreable]

    if spec is not None:
        _check_word_boundary(spec, refs, vocab)

    result = EvalResult(
        wer=_wer(refs, hyps),
        cer=_cer(refs, hyps),
        n=len(refs),
        n_empty_refs=n_empty_refs,
        refs=refs,
        hyps=hyps,
    )
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        save_predictions.write_text(
            json.dumps(
                {
                    "wer": result.wer,
                    "cer": result.cer,
                    "n": result.n,
                    "n_empty_refs": result.n_empty_refs,
                    "pairs": list(zip(refs, hyps, strict=True)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return result


def _check_word_boundary(spec: LangSpec, refs: list[str], vocab: CtcVocab) -> None:
    """Warn when a language's declared ``word_boundary`` contradicts its text."""
    stats = collect_text_stats(refs, vocab.policy)
    if spec.word_boundary and stats.looks_unspaced:
        warnings.warn(
            f"{spec.code}: word_boundary is declared true but the median utterance has "
            f"{stats.median_tokens_per_utt:g} whitespace token(s), so word error rate is "
            "scoring whole sentences as single words; set word_boundary: false in the preset",
            UserWarning,
            stacklevel=2,
        )
    elif not spec.word_boundary and not stats.looks_unspaced:
        warnings.warn(
            f"{spec.code}: word_boundary is declared false but the median utterance has "
            f"{stats.median_tokens_per_utt:g} whitespace tokens, so character error rate is "
            "being reported as primary for a language that does separate words",
            UserWarning,
            stacklevel=2,
        )
