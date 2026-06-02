"""Collate: padding, attention mask, and -100 label padding."""

import torch

from svb.data.collate import make_ctc_collate
from svb.model.ctc_vocab import build_vocab_from_texts


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
