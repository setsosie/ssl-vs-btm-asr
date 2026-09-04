"""Collate: padding, attention mask, and -100 label padding."""

import torch

from svb.data.collate import make_ctc_collate
from svb.model.ctc_vocab import build_vocab_from_texts
from svb.model.xeus_standalone import max_label_len_for_samples


def test_collate_pads_and_masks():
    vocab = build_vocab_from_texts(["ab", "abc"])
    collate = make_ctc_collate(vocab)
    batch = [(torch.ones(3), "ab"), (torch.ones(5), "abc")]
    out = collate(batch)

    assert out["input_values"].shape == (2, 5)
    # first sample padded after 3 frames
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]
    # labels padded with -100; row 0 ("ab") has one pad vs row 1 ("abc")
    assert (out["labels"][0] == -100).sum() == 1
    assert (out["labels"][1] == -100).sum() == 0


def test_collate_drops_labels_longer_than_the_frame_budget():
    """A truncated waveform cannot host a full transcript, so drop the pair.

    CTC returns inf when the target is longer than the input, and
    zero_infinity turns that into a zero — silencing the sample's gradient and
    pulling down the mean loss that checkpoint selection reads. Dropping the
    pair makes the loss honest and the loss count auditable.
    """
    vocab = build_vocab_from_texts(["abcdefgh"])
    collate = make_ctc_collate(vocab, max_audio_samples=2000, drop_overlong=True)
    # 2000 samples buy 6 encoder frames, so "abc" fits and "abcdefgh" cannot.
    assert max_label_len_for_samples(2000) == 6
    out = collate([(torch.ones(2000), "abc"), (torch.ones(2000), "abcdefgh")])

    assert out["n_dropped"] == 1
    assert out["texts"] == ["abc"]
    assert out["input_values"].shape[0] == 1


def test_collate_counts_utterances_at_the_truncation_guard():
    vocab = build_vocab_from_texts(["ab"])
    collate = make_ctc_collate(vocab, max_audio_samples=1000)
    out = collate([(torch.ones(1000), "ab"), (torch.ones(400), "ab")])

    assert out["n_at_audio_guard"] == 1
    assert out["n_dropped"] == 0  # counting only; dropping is opt-in


def test_collate_without_a_guard_keeps_everything():
    vocab = build_vocab_from_texts(["abcde"])
    collate = make_ctc_collate(vocab)
    out = collate([(torch.ones(1000), "abcde")])

    assert out["n_dropped"] == 0
    assert out["n_at_audio_guard"] == 0
    assert out["texts"] == ["abcde"]
