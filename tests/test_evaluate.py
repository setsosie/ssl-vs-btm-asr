"""Evaluation: padding trim, reference fidelity, and sidecar output."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from svb.data.collate import make_ctc_collate
from svb.eval.evaluate import evaluate
from svb.model.ctc_vocab import build_vocab_from_texts

from .conftest import FakeXeusCTC, wav_for


class ListDataset(Dataset):
    """In-memory (waveform, text) pairs."""

    def __init__(self, items: list[tuple[torch.Tensor, str]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        return self.items[idx]


def test_hypothesis_is_independent_of_padding() -> None:
    """An utterance must decode the same alone as it does beside a longer one.

    Without trimming to ``input_lengths`` the shorter row picks up whatever the
    encoder emits over the zero-padded tail, so every WER in the repo would
    depend on which utterances happened to share a batch.
    """
    vocab, _ = build_vocab_from_texts(["ab", "abc"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size, junk_id=vocab.char_to_id["a"])

    padded = evaluate(
        model,
        ListDataset([(wav_for(vocab, "ab"), "ab"), (wav_for(vocab, "abc"), "abc")]),
        vocab,
        collate,
        device="cpu",
        batch_size=2,
    )
    alone = evaluate(
        model,
        ListDataset([(wav_for(vocab, "ab"), "ab")]),
        vocab,
        collate,
        device="cpu",
        batch_size=1,
    )

    assert alone.hyps == ["ab"]
    assert padded.hyps[padded.refs.index("ab")] == alone.hyps[0]
    assert padded.wer == 0.0


def test_reference_keeps_characters_absent_from_the_vocab() -> None:
    """A test-set character the training vocab never saw must stay in the ref.

    Reconstructing the reference from label ids drops every id that mapped to
    ``<unk>``, so the model is scored against a shortened transcript. Here the
    hypothesis is missing a whole word and the honest WER is 33.3%; the
    id round-trip reports a perfect 0%.
    """
    vocab, _ = build_vocab_from_texts(["a b"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset([(wav_for(vocab, "a b"), "a b c")])

    result = evaluate(model, dataset, vocab, collate, device="cpu", batch_size=1)

    assert result.refs == ["a b c"]
    assert result.hyps == ["a b"]
    assert result.wer == pytest.approx(100.0 / 3.0)


def test_predictions_sidecar_carries_every_utterance(tmp_path) -> None:
    """The sidecar is the input to bootstrap CIs, so it needs the pairs.

    Corpus-level WER alone cannot be resampled; the per-utterance (ref, hyp)
    pairs are what the statistics layer reads back off disk.
    """
    import json

    vocab, _ = build_vocab_from_texts(["ab", "abc"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset([(wav_for(vocab, "ab"), "ab"), (wav_for(vocab, "abc"), "abc")])
    sidecar = tmp_path / "predictions" / "xx.json"

    result = evaluate(
        model, dataset, vocab, collate, device="cpu", batch_size=2, save_predictions=sidecar
    )

    saved = json.loads(sidecar.read_text())
    assert saved["n"] == result.n == 2
    assert [tuple(p) for p in saved["pairs"]] == list(zip(result.refs, result.hyps, strict=True))


def test_cli_predictions_path_is_per_language() -> None:
    from svb.cli import _predictions_path

    assert _predictions_path(Path("/runs/seed0"), "hi") == Path("/runs/seed0/predictions/hi.json")


def test_evaluation_batches_by_length_and_reports_in_dataset_order() -> None:
    """Batch mates are chosen by length, so results do not ride on file order.

    The encoder's two time-axis convolutions are unmasked, mirroring the
    ESPnet reference, so an utterance's frames are influenced by whatever was
    padded beside it. Grouping by length makes that influence a fixed function
    of the split rather than of the order rows happen to appear in.
    """
    vocab, _ = build_vocab_from_texts(["abcd"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset(
        [
            (wav_for(vocab, "a"), "a"),
            (wav_for(vocab, "abcd"), "abcd"),
            (wav_for(vocab, "ab"), "ab"),
        ]
    )

    seen: list[list[str]] = []

    def spy(batch):
        out = collate(batch)
        seen.append(list(out["texts"]))
        return out

    result = evaluate(model, dataset, vocab, spy, device="cpu", batch_size=2)

    assert seen == [["abcd", "ab"], ["a"]]
    assert result.refs == ["a", "abcd", "ab"]  # sidecar stays in dataset order


def test_length_sorting_can_be_turned_off() -> None:
    from svb.eval.evaluate import evaluate as run_eval

    vocab, _ = build_vocab_from_texts(["abcd"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset([(wav_for(vocab, "a"), "a"), (wav_for(vocab, "abcd"), "abcd")])

    seen: list[list[str]] = []

    def spy(batch):
        out = collate(batch)
        seen.append(list(out["texts"]))
        return out

    run_eval(model, dataset, vocab, spy, device="cpu", batch_size=2, sort_by_length=False)

    assert seen == [["a", "abcd"]]
