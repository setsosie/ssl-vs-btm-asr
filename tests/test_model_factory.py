"""One construction site for XeusCTC, and one source of truth for its path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from svb.config import ExperimentConfig, load_config


class _StubEncoder(nn.Module):
    """Stands in for the 577M encoder so construction is free."""

    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False


@pytest.fixture
def scratch_cfg() -> ExperimentConfig:
    return load_config(
        "C_btm_scratch",
        "3",
        seed=0,
        overrides={"model": {"hidden_size": 12, "blank_bias_init": -3.5}},
    )


def test_make_model_honours_hidden_size_and_blank_bias(monkeypatch, scratch_cfg) -> None:
    """Three of the five call sites dropped these, so the config was a fiction.

    hidden_size and blank_bias_init were passed only by the BTM pipeline; the
    merged model and every arm-A model silently used the defaults.
    """
    from svb.model import make_model, xeus_ctc

    monkeypatch.setattr(xeus_ctc, "StandaloneXEUS", _StubEncoder)
    model = make_model(scratch_cfg, vocab_size=7)

    assert model.ctc_proj.in_features == 12
    assert model.ctc_proj.out_features == 7
    assert model.ctc_proj.bias[0].item() == pytest.approx(-3.5)


def test_transfer_loads_the_checkpoint_the_trainer_returned(monkeypatch, tmp_path) -> None:
    """Re-deriving out_dir/"best.pt" is a second source of truth for one path.

    Any change to the trainer's naming would leave transfer silently loading a
    stale file, or crashing after a full fine-tune.
    """
    from svb.data.registry import LangSpec
    from svb.eval import transfer as mod
    from svb.eval.evaluate import EvalResult
    from svb.train.trainer import TrainResult

    written = tmp_path / "epoch_7.pt"
    loaded: list[Path] = []

    class _Model:
        def load(self, path: Path) -> None:
            loaded.append(Path(path))

        def expand_head(self, size: int, seed: int = 0) -> None:
            pass

    monkeypatch.setattr(mod, "load_texts", lambda lang, split: ["ab"])
    monkeypatch.setattr(mod, "load_language", lambda lang, split, max_samples: [])
    monkeypatch.setattr(mod, "make_model", lambda cfg, vocab_size: _Model())
    monkeypatch.setattr(
        mod,
        "train",
        lambda *a, **k: TrainResult(
            best_val_loss=1.0, best_epoch=0, epochs_run=1, checkpoint=written
        ),
    )
    monkeypatch.setattr(mod, "evaluate", lambda *a, **k: EvalResult(wer=0.0, cer=0.0, n=0))

    cfg = load_config("A_ssl", "3", seed=0)
    vocab_stub: Any = mod.CtcVocab(id_to_char=["<blank>", "<unk>", "a", "b"])
    spec = LangSpec(code="xx", source="openslr", hf_dataset="x", hf_config="x")

    mod.transfer_one(cfg, None, vocab_stub, spec, tmp_path, device="cpu")

    assert loaded == [written]


def test_gradient_checkpointing_flag_reaches_the_encoder(monkeypatch, scratch_cfg) -> None:
    from svb.model import make_model, xeus_ctc

    monkeypatch.setattr(xeus_ctc, "StandaloneXEUS", _StubEncoder)
    model = make_model(scratch_cfg, vocab_size=3)

    model.set_gradient_checkpointing(True)

    assert model.encoder.gradient_checkpointing is True
    assert isinstance(model.ctc_norm, torch.nn.LayerNorm)
