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

# Every dataset item carries the language it came from, so the collate can
# normalize it under that language's policy.
LANG = "en"


class ListDataset(Dataset):
    """In-memory (waveform, text) pairs."""

    def __init__(self, items: list[tuple[torch.Tensor, str, str]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
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
        ListDataset([(wav_for(vocab, "ab"), "ab", LANG), (wav_for(vocab, "abc"), "abc", LANG)]),
        vocab,
        collate,
        device="cpu",
        batch_size=2,
    )
    alone = evaluate(
        model,
        ListDataset([(wav_for(vocab, "ab"), "ab", LANG)]),
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
    dataset = ListDataset([(wav_for(vocab, "a b"), "a b c", LANG)])

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
    dataset = ListDataset(
        [(wav_for(vocab, "ab"), "ab", LANG), (wav_for(vocab, "abc"), "abc", LANG)]
    )
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
            (wav_for(vocab, "a"), "a", LANG),
            (wav_for(vocab, "abcd"), "abcd", LANG),
            (wav_for(vocab, "ab"), "ab", LANG),
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
    dataset = ListDataset(
        [(wav_for(vocab, "a"), "a", LANG), (wav_for(vocab, "abcd"), "abcd", LANG)]
    )

    seen: list[list[str]] = []

    def spy(batch):
        out = collate(batch)
        seen.append(list(out["texts"]))
        return out

    run_eval(model, dataset, vocab, spy, device="cpu", batch_size=2, sort_by_length=False)

    assert seen == [["a", "abcd"]]


def test_hypotheses_are_normalized_before_scoring() -> None:
    """A greedy decode emits leading, trailing and doubled spaces.

    Both sides go through the same policy, so those cost nothing; without it
    they would score as word errors the model had no way to avoid.
    """
    from svb.text.normalize import normalize_text

    vocab, _ = build_vocab_from_texts(["a b"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    space = vocab.char_to_id[" "]
    a = vocab.char_to_id["a"]
    b = vocab.char_to_id["b"]
    # " a  b " once CTC collapsing is done with it.
    wav = torch.tensor([float(i) for i in (space, a, space, vocab.blank_id, space, b, space)])

    result = evaluate(model, ListDataset([(wav, "a b", LANG)]), vocab, collate, device="cpu")

    assert normalize_text(" a  b ") == "a b"
    assert result.hyps == ["a b"]
    assert result.wer == 0.0


def test_references_that_normalize_to_empty_are_excluded_and_counted() -> None:
    """An empty reference contributes nothing to the denominator and its whole
    hypothesis to the numerator, so leaving it in inflates corpus WER without
    any weight behind it. Dropping it silently would change the test set."""
    vocab, _ = build_vocab_from_texts(["a b"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset(
        [(wav_for(vocab, "a b"), "a b", LANG), (wav_for(vocab, "a b"), "…!?", LANG)]
    )

    result = evaluate(model, dataset, vocab, collate, device="cpu", batch_size=2)

    assert result.n_empty_refs == 1
    assert result.n == 1
    assert result.refs == ["a b"]
    assert result.wer == 0.0


def test_the_sidecar_reports_the_excluded_utterances(tmp_path) -> None:
    import json

    vocab, _ = build_vocab_from_texts(["a b"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset(
        [(wav_for(vocab, "a b"), "a b", LANG), (wav_for(vocab, "a b"), "!!!", LANG)]
    )
    sidecar = tmp_path / "p.json"

    evaluate(model, dataset, vocab, collate, device="cpu", batch_size=2, save_predictions=sidecar)

    saved = json.loads(sidecar.read_text())
    assert saved["n"] == 1
    assert saved["n_empty_refs"] == 1
    assert len(saved["pairs"]) == 1


def test_primary_metric_follows_the_language_spec() -> None:
    from svb.data.registry import LangSpec
    from svb.eval.evaluate import EvalResult

    result = EvalResult(wer=40.0, cer=10.0, n=5)
    spaced = LangSpec(code="de", source="commonvoice", hf_config="de")
    unspaced = LangSpec(code="ja", source="commonvoice", hf_config="ja", word_boundary=False)

    assert result.primary(spaced) == 40.0
    assert result.primary(unspaced) == 10.0


def test_a_language_declared_spaced_whose_text_is_not_warns() -> None:
    """The empirical check on the preset's declaration."""
    from svb.data.registry import LangSpec

    vocab, _ = build_vocab_from_texts(["abcd"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset([(wav_for(vocab, "abcd"), "abcd", LANG)] * 3)
    mislabelled = LangSpec(code="ja", source="commonvoice", hf_config="ja")

    with pytest.warns(UserWarning, match="word_boundary"):
        evaluate(model, dataset, vocab, collate, device="cpu", spec=mislabelled)


def test_a_correctly_declared_language_does_not_warn(recwarn) -> None:
    from svb.data.registry import LangSpec

    vocab, _ = build_vocab_from_texts(["a b"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset([(wav_for(vocab, "a b"), "a b", LANG)] * 3)
    spec = LangSpec(code="en", source="commonvoice", hf_config="en")

    evaluate(model, dataset, vocab, collate, device="cpu", spec=spec)

    assert not [w for w in recwarn if "word_boundary" in str(w.message)]


def test_a_split_where_nothing_is_scoreable_fails_loudly() -> None:
    """Reporting 0% for a split with no usable reference would be worse."""
    vocab, _ = build_vocab_from_texts(["a b"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset(
        [(wav_for(vocab, "a b"), "!!!", LANG), (wav_for(vocab, "a b"), "…", LANG)]
    )

    with pytest.raises(ValueError, match="nothing to score"):
        evaluate(model, dataset, vocab, collate, device="cpu", batch_size=2)


def test_a_collate_that_drops_rows_is_refused_when_sorting_by_length() -> None:
    """The length sort is undone by index, so every row must come back.

    A collate that drops utterances — the training one does, for transcripts
    that no longer fit their audio — would leave the restored lists misaligned:
    hypothesis i attributed to reference j. Refuse it by name rather than
    scoring a shuffled corpus.
    """
    vocab, _ = build_vocab_from_texts(["abcd"])
    collate = make_ctc_collate(vocab)
    model = FakeXeusCTC(vocab.size)
    dataset = ListDataset(
        [
            (wav_for(vocab, "a"), "a", LANG),
            (wav_for(vocab, "abcd"), "abcd", LANG),
            (wav_for(vocab, "ab"), "ab", LANG),
        ]
    )

    def dropping(batch):
        out = collate(batch[:1])  # silently discards the rest of the batch
        return out

    with pytest.raises(ValueError, match="dropped"):
        evaluate(model, dataset, vocab, dropping, device="cpu", batch_size=2)
