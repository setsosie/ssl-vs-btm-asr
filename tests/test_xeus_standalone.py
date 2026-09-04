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
