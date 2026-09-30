"""Audio decoding shared by every loader.

One reader rather than one per corpus, because the three copies had drifted and
one of them could not run. Common Voice clips were decoded with
``torchaudio.load``, which since torchaudio 2.9 is a wrapper over TorchCodec and
raises ``ImportError`` without it. TorchCodec is not a dependency of this
project, so under the committed lock every Common Voice run would have stopped
on its first batch — and no test noticed, because the tests that built a Common
Voice dataset replaced ``torchaudio.load`` with a stub.

Everything the corpora ship is decoded by ``soundfile`` instead: mp3 (Common
Voice), wav (OpenSLR) and flac (the prepared corpora). Its wheels bundle a
libsndfile built with mp3 support, so there is no ffmpeg and no second decoding
library to keep in step with the torch version. A container libsndfile cannot
read — m4a, for one — raises from ``soundfile`` with the path in the message,
which is the right outcome for a corpus this project does not yet claim to read.

Only resampling still goes through torchaudio, and that is pure tensor code.
"""

from __future__ import annotations

from pathlib import Path

import torch

TARGET_SR = 16000


def read_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Mono float32 samples at the file's own sample rate."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = torch.as_tensor(data, dtype=torch.float32)
    if wav.dim() > 1:  # soundfile yields (frames, channels)
        wav = wav.mean(dim=1)
    return wav, int(sr)


def load_waveform(path: str | Path, max_samples: int | None = None) -> torch.Tensor:
    """One utterance as the model takes it: mono, 16 kHz, optionally truncated.

    Args:
        max_samples: The training-time truncation guard, in samples at 16 kHz.
            ``None`` reads the whole file, which is what evaluation must do.
    """
    wav, sr = read_audio(path)
    # An empty clip stays empty: the resampler cannot reshape zero samples and
    # raises. Training drops such a pair as unalignable, which is the right end
    # for it; a crash here would stop every run at the same utterance.
    if sr != TARGET_SR and wav.numel() > 0:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if max_samples is not None and wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    return wav


def audio_seconds(path: str | Path) -> float | None:
    """Duration from the file header, or None if it cannot be read.

    Never raises: the callers are sweeps over whole corpora, where one
    unreadable clip must be counted as missing rather than end the sweep.
    """
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:
        return None
    return float(info.frames) / float(info.samplerate) if info.samplerate else None
