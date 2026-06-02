"""XEUS encoder loading — reference (ESPnet) backend or standalone fallback.

``load_xeus_encoder`` returns an object exposing the standalone ``encode``
contract:

    encode(waveforms, wav_lengths, use_mask=False, use_final_output=True)
        -> (features_or_layerlist, output_lengths)

Backends:
  - **espnet** (reference): the canonical XEUS via the wanchichen/espnet@ssl fork
    (``SSLTask.build_model_from_file``). This is the model the paper should rest
    on. XEUS is not yet in stock ESPnet — see pyproject's ``espnet`` extra.
  - **standalone**: the fork-free PyTorch reimplementation (``xeus_standalone``).
    Used as a fallback and for scratch (arm C) init, where there are no
    pretrained weights to match.
  - **auto**: espnet if importable, else standalone.

The cross-check test asserts the two backends produce equivalent features on the
same audio (run on the server with the real checkpoint), retiring the
reimplementation-correctness risk.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .xeus_standalone import StandaloneXEUS, load_xeus_from_checkpoint

Backend = Literal["auto", "espnet", "standalone"]


def _espnet_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("espnet2") is not None


def resolve_backend(backend: str) -> str:
    """Resolve ``auto`` to a concrete backend based on what's installed."""
    if backend not in ("auto", "espnet", "standalone"):
        raise ValueError(f"unknown backend: {backend!r}")
    if backend == "auto":
        return "espnet" if _espnet_available() else "standalone"
    return backend


class ESPnetXEUS(nn.Module):
    """Adapter wrapping the reference ESPnet XEUS model to the standalone API."""

    def __init__(self, espnet_model: nn.Module) -> None:
        super().__init__()
        self.model = espnet_model

    def encode(self, waveforms, wav_lengths, use_mask: bool = False, use_final_output: bool = True):
        # ESPnet's SSL model exposes the same encode() signature the standalone
        # mirrors; it returns (output, lengths) where output is the final tensor
        # (use_final_output=True) or the per-layer list (False).
        return self.model.encode(
            waveforms, wav_lengths, use_mask=use_mask, use_final_output=use_final_output
        )


def _load_espnet(checkpoint: str, device: str) -> ESPnetXEUS:
    from espnet2.tasks.ssl import SSLTask

    model, _ = SSLTask.build_model_from_file(None, checkpoint, device)
    return ESPnetXEUS(model)


def load_xeus_encoder(
    init: Literal["ssl", "scratch"],
    checkpoint: str | None = None,
    backend: str = "auto",
    device: str = "cpu",
) -> nn.Module:
    """Load the XEUS encoder.

    Args:
        init: "ssl" loads pretrained weights; "scratch" is random init (arm C),
            always via the standalone architecture (no weights to match).
        checkpoint: XEUS checkpoint path (required for ``init='ssl'``).
        backend: "auto" | "espnet" | "standalone".
        device: device for the reference backend's build step.
    """
    if init == "scratch":
        return StandaloneXEUS()

    if not checkpoint:
        raise ValueError("init='ssl' requires a XEUS checkpoint path")

    chosen = resolve_backend(backend)
    if chosen == "espnet":
        return _load_espnet(checkpoint, device)
    return load_xeus_from_checkpoint(checkpoint, device)
