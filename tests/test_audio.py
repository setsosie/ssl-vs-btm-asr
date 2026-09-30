"""The one audio reader every loader goes through.

These read real files. The Common Voice loader used to be tested against a
stub of ``torchaudio.load``, which is how it shipped calling a function that
cannot run in this project's environment: torchaudio 2.9 and later need
TorchCodec for it, and TorchCodec is not installed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from svb.data.audio import TARGET_SR, audio_seconds, load_waveform, read_audio
from svb.data.commonvoice_local import CommonVoiceLocal


@pytest.mark.parametrize("suffix", [".mp3", ".wav", ".flac"])
def test_every_container_the_corpora_ship_decodes_to_16k_mono(
    tmp_path: Path, write_clip: Callable[..., None], suffix: str
) -> None:
    clip = tmp_path / f"clip{suffix}"
    write_clip(clip, frames=48000, sr=48000)

    wav = load_waveform(clip)

    assert wav.dim() == 1
    assert wav.dtype.is_floating_point
    # One second at 48 kHz is one second at 16 kHz. mp3 pads a frame or two at
    # the edges, so the length is checked to a tolerance rather than exactly.
    assert abs(wav.shape[0] - TARGET_SR) < 0.1 * TARGET_SR


def test_stereo_is_downmixed_rather_than_returned_as_two_channels(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    clip = tmp_path / "stereo.wav"
    sf.write(str(clip), np.zeros((1600, 2), dtype="float32"), TARGET_SR)

    wav, sr = read_audio(clip)

    assert (wav.shape, sr) == ((1600,), TARGET_SR)


def test_the_truncation_guard_applies_after_resampling(
    tmp_path: Path, write_clip: Callable[..., None]
) -> None:
    """``max_samples`` is a length at 16 kHz, whatever rate the file is in."""
    clip = tmp_path / "long.wav"
    write_clip(clip, frames=48000, sr=48000)

    assert load_waveform(clip, max_samples=4000).shape[0] == 4000


def test_an_empty_clip_at_another_rate_is_returned_empty_rather_than_crashing(
    tmp_path: Path, write_clip: Callable[..., None]
) -> None:
    """One such clip in a capped subset would stop every arm and seed at the
    same utterance; the collate drops it as unalignable instead."""
    clip = tmp_path / "empty.wav"
    write_clip(clip, frames=0, sr=48000)

    assert load_waveform(clip).shape == (0,)


def test_the_header_duration_needs_no_decode(
    tmp_path: Path, write_clip: Callable[..., None]
) -> None:
    clip = tmp_path / "half.wav"
    write_clip(clip, frames=8000, sr=TARGET_SR)

    assert audio_seconds(clip) == pytest.approx(0.5)
    assert audio_seconds(tmp_path / "absent.wav") is None


def test_a_common_voice_clip_is_read_without_torchcodec(
    make_cv_lang: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: a real mp3 through the real loader, nothing stubbed."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang(clip_seconds=0.5)))

    wav, text, code = CommonVoiceLocal("en", "test")[0]

    assert (text, code) == ("six", "en")
    assert abs(wav.shape[0] - TARGET_SR // 2) < 0.1 * TARGET_SR
