"""Shared test fixtures and CPU test doubles.

Two families live here:

* **Synthetic data fixtures** for the data package. Everything is generated
  locally: tiny silent WAVs plus the committed ``tests/fixtures/openslr/*.tsv``
  line indices, which reproduce the real OpenSLR layout (flat directory,
  ``<FileID>.wav``, two-column tab-separated index). No test touches the
  network or a real corpus.
* **A model double.** The real model is a 577M-parameter encoder, so every
  unit test runs against a stand-in that keeps the public ``XeusCTC`` surface
  (``forward`` returning ``logits``/``input_lengths``/``loss``,
  ``greedy_decode``, ``save``, ``load``) and nothing else. Subclassing the real
  class rather than duck-typing it means the tests exercise the inherited
  methods under test, not a copy.
"""

from __future__ import annotations

import importlib.util
import wave
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from svb.model.ctc_vocab import CtcVocab
from svb.model.xeus_ctc import XeusCTC

TARGET_SR = 16000


# --------------------------------------------------------------------------- #
# Synthetic data fixtures
# --------------------------------------------------------------------------- #


def _write_silent_wav(path: Path, frames: int, sr: int = TARGET_SR, channels: int = 1) -> None:
    """Write a 16-bit PCM WAV of `frames` silent samples."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(2 * channels * frames))


@pytest.fixture
def write_silent_wav() -> Callable[..., None]:
    return _write_silent_wav


@pytest.fixture
def fixtures_dir(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.rootpath) / "tests" / "fixtures"


@pytest.fixture
def slr_root(tmp_path: Path, fixtures_dir: Path) -> Path:
    """An extracted two-archive OpenSLR language dir under a fake $OPENSLR_ROOT."""
    root = tmp_path / "openslr"
    dest = root / "SLR63"
    dest.mkdir(parents=True)
    for name in ("line_index_female.tsv", "line_index_male.tsv"):
        text = (fixtures_dir / "openslr" / name).read_text(encoding="utf-8")
        (dest / name).write_text(text, encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                _write_silent_wav(dest / f"{line.split(chr(9))[0]}.wav", frames=160)
    return root


@pytest.fixture
def fetch_openslr(pytestconfig: pytest.Config) -> ModuleType:
    """Load `scripts/fetch_openslr.py` as a module without touching sys.path."""
    path = Path(pytestconfig.rootpath) / "scripts" / "fetch_openslr.py"
    spec = importlib.util.spec_from_file_location("fetch_openslr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Model double
# --------------------------------------------------------------------------- #


class FakeXeusCTC(XeusCTC):
    """Encoder-free double: one output frame per input sample.

    The waveform *is* the transcript — sample ``i`` of the waveform holds the
    vocab id the encoder should emit at frame ``i``. Frames beyond
    ``attention_mask`` emit ``junk_id`` instead, which is what a real encoder
    does over the zero-padded tail: it predicts *something*, and that something
    is only excluded from the hypothesis if the decoder honours the lengths.
    """

    def __init__(self, vocab_size: int, junk_id: int = 2) -> None:
        nn.Module.__init__(self)
        self.hidden_size = 1
        self.ctc_norm = nn.LayerNorm(1)
        self.ctc_proj = nn.Linear(1, vocab_size)
        self.junk_id = junk_id

    def forward(  # type: ignore[override]
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, frames = input_values.shape
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long()
        else:
            lengths = torch.full((bsz,), frames, dtype=torch.long)

        ids = input_values.round().long().clamp(min=0, max=self.vocab_size - 1)
        valid = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
        ids = torch.where(valid, ids, torch.full_like(ids, self.junk_id))
        logits = F.one_hot(ids, self.vocab_size).float() * 10.0
        return {"logits": logits, "input_lengths": lengths}


def wav_for(vocab: CtcVocab, text: str) -> torch.Tensor:
    """Waveform that makes :class:`FakeXeusCTC` emit ``text`` verbatim."""
    return torch.tensor([float(i) for i in vocab.encode(text)], dtype=torch.float32)


@pytest.fixture
def fake_model_cls() -> type[FakeXeusCTC]:
    return FakeXeusCTC
