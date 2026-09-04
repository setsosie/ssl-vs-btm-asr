"""Standalone XEUS encoder: frame arithmetic and checkpoint compatibility."""

from __future__ import annotations

import torch

from svb.model.xeus_standalone import (
    _Frontend,
    frontend_output_length,
    max_label_len_for_samples,
)


def test_frontend_output_length_matches_the_real_frontend() -> None:
    """The closed-form frame count must agree with the CNN it describes.

    The whole point of the helper is to predict, without running the encoder,
    how many frames a waveform will produce. If it drifts from the frontend,
    the CTC length guard it feeds is worthless.
    """
    frontend = _Frontend()
    for n_samples in (4000, 16000, 31337):
        lengths = torch.tensor([n_samples])
        _, out_lengths = frontend(torch.zeros(1, n_samples), lengths)
        assert out_lengths is not None
        assert int(out_lengths[0]) == frontend_output_length(n_samples)


def test_frame_budget_at_the_truncation_guard() -> None:
    """12.5 s of audio buys 624 frames, i.e. ~50 frames per second."""
    assert max_label_len_for_samples(200_000) == 624
    assert max_label_len_for_samples(16_000) == 49


def test_frame_budget_never_goes_negative() -> None:
    """Waveforms shorter than the frontend's receptive field yield no frames."""
    assert max_label_len_for_samples(0) == 0
    assert max_label_len_for_samples(5) == 0


def _tiny_encoder_constants(monkeypatch) -> None:
    """Shrink the architecture so a real encoder fits in a unit test."""
    from svb.model import xeus_standalone as mod

    for name, value in {
        "_HIDDEN_SIZE": 8,
        "_NUM_HEADS": 2,
        "_FFN_DIM": 16,
        "_CGMLP_DIM": 16,
        "_CGMLP_KERNEL": 3,
        "_MERGE_KERNEL": 3,
        "_NUM_BLOCKS": 2,
        "_POS_KERNEL": 4,
        "_POS_GROUPS": 2,
    }.items():
        monkeypatch.setattr(mod, name, value)


def test_gradient_checkpointing_changes_nothing_but_memory(monkeypatch) -> None:
    """Recomputing activations must reproduce the forward and the gradients.

    The config advertised grad_checkpointing, but the trainer only set a plain
    attribute that nothing read: peak memory was whatever the batch size
    implied, not what the knob promised.
    """
    from svb.model import xeus_standalone as mod

    _tiny_encoder_constants(monkeypatch)
    block_forwards = []
    plain_forward = mod._EBranchformerBlock.forward

    def counting_forward(self, x, padding_mask=None):  # type: ignore[no-untyped-def]
        block_forwards.append(1)
        return plain_forward(self, x, padding_mask)

    monkeypatch.setattr(mod._EBranchformerBlock, "forward", counting_forward)

    torch.manual_seed(0)
    encoder = mod._Encoder(dropout_rate=0.0).train()
    lengths = torch.tensor([6, 4])

    def run(enabled: bool) -> tuple[torch.Tensor, torch.Tensor, int]:
        encoder.gradient_checkpointing = enabled
        encoder.zero_grad(set_to_none=True)
        block_forwards.clear()
        x = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8) / 100.0
        x.requires_grad_(True)
        out, _ = encoder(x, lengths)
        assert isinstance(out, torch.Tensor)
        out.sum().backward()
        assert x.grad is not None
        return out.detach(), x.grad.detach(), len(block_forwards)

    plain_out, plain_grad, plain_calls = run(False)
    ckpt_out, ckpt_grad, ckpt_calls = run(True)

    # Trading compute for memory means each block runs twice: once to produce
    # the output, once to rebuild the activations the backward pass needs.
    assert plain_calls == 2  # one per block
    assert ckpt_calls == 2 * plain_calls
    assert torch.allclose(plain_out, ckpt_out, atol=1e-6)
    assert torch.allclose(plain_grad, ckpt_grad, atol=1e-6)


def test_checkpointing_is_off_in_eval_mode(monkeypatch) -> None:
    """Recompute has no purpose without a backward pass to feed."""
    from svb.model.xeus_standalone import _Encoder

    _tiny_encoder_constants(monkeypatch)
    torch.manual_seed(0)
    encoder = _Encoder(dropout_rate=0.0).eval()
    encoder.gradient_checkpointing = True
    x = torch.zeros(1, 5, 8)

    with torch.no_grad():
        out, _ = encoder(x, torch.tensor([5]))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 5, 8)
