"""Collate: padding, attention mask, and -100 label padding."""

import torch

from svb.data.collate import make_ctc_collate
from svb.model.ctc_vocab import build_vocab_from_texts
from svb.model.xeus_standalone import max_label_len_for_samples


def test_collate_pads_and_masks():
    vocab, _ = build_vocab_from_texts(["ab", "abc"])
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
    vocab, _ = build_vocab_from_texts(["abcdefgh"])
    collate = make_ctc_collate(vocab, max_audio_samples=2000, drop_overlong=True)
    # 2000 samples buy 6 encoder frames, so "abc" fits and "abcdefgh" cannot.
    assert max_label_len_for_samples(2000) == 6
    out = collate([(torch.ones(2000), "abc"), (torch.ones(2000), "abcdefgh")])

    assert out["n_dropped"] == 1
    assert out["texts"] == ["abc"]
    assert out["input_values"].shape[0] == 1


def test_collate_counts_utterances_at_the_truncation_guard():
    vocab, _ = build_vocab_from_texts(["ab"])
    collate = make_ctc_collate(vocab, max_audio_samples=1000)
    out = collate([(torch.ones(1000), "ab"), (torch.ones(400), "ab")])

    assert out["n_at_audio_guard"] == 1
    assert out["n_dropped"] == 0  # counting only; dropping is opt-in


def test_collate_without_a_guard_keeps_everything():
    vocab, _ = build_vocab_from_texts(["abcde"])
    collate = make_ctc_collate(vocab)
    out = collate([(torch.ones(1000), "abcde")])

    assert out["n_dropped"] == 0
    assert out["n_at_audio_guard"] == 0
    assert out["texts"] == ["abcde"]


def test_collate_normalizes_the_target_and_reports_what_it_encoded():
    """The batch's `texts` are what the model was actually trained against.

    Scoring reads them back as references, so a raw transcript here would mean
    the model is trained on one string and measured against another.
    """
    vocab, _ = build_vocab_from_texts(["Hello, World!"])
    collate = make_ctc_collate(vocab)
    out = collate([(torch.ones(4), "Hello, World!")])

    assert out["texts"] == ["hello world"]
    assert (out["labels"][0] != -100).sum() == len("hello world")
    assert vocab.unk_id not in out["labels"][0].tolist()


def test_collate_uses_the_vocabs_own_policy_by_default():
    from svb.text.normalize import NormalizerPolicy

    vocab, _ = build_vocab_from_texts(["Straße"], policy=NormalizerPolicy(case="lower"))
    out = make_ctc_collate(vocab)([(torch.ones(4), "Straße")])

    assert out["texts"] == ["straße"]


def test_collate_counts_utterances_that_normalize_to_empty():
    """A punctuation-only transcript is not a CTC target at any length."""
    vocab, _ = build_vocab_from_texts(["ab"])
    collate = make_ctc_collate(vocab)
    out = collate([(torch.ones(4), "ab"), (torch.ones(4), "…!?")])

    assert out["n_empty_text"] == 1
    assert out["texts"] == ["ab", ""]  # counted, not dropped: evaluation needs the row


def test_training_drops_utterances_that_normalize_to_empty():
    """Training cannot use a zero-length target, so the pair goes.

    Evaluation must never enable this: dropping a test utterance in the collate
    would shrink the test set without the result saying so.
    """
    vocab, _ = build_vocab_from_texts(["ab"])
    collate = make_ctc_collate(vocab, drop_empty=True)
    out = collate([(torch.ones(4), "ab"), (torch.ones(4), "…!?")])

    assert out["n_empty_text"] == 1
    assert out["texts"] == ["ab"]
    assert out["input_values"].shape[0] == 1


def test_overlong_check_measures_the_normalized_target():
    """Normalization shortens transcripts, so the budget must see the short form.

    Measuring the raw string would drop pairs that fit perfectly well once the
    punctuation is gone.
    """
    vocab, _ = build_vocab_from_texts(["abcdefgh, ..."])
    collate = make_ctc_collate(vocab, max_audio_samples=2000, drop_overlong=True)
    assert max_label_len_for_samples(2000) == 6
    out = collate([(torch.ones(2000), "abcde!!!")])

    assert out["n_dropped"] == 0
    assert out["texts"] == ["abcde"]
