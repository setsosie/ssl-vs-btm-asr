"""Backend resolution (runs anywhere) + the server-only equivalence cross-check.

The cross-check is the point of the hybrid design: it asserts the reference
ESPnet backend and the standalone reimplementation produce the same encoder
features on the same audio, retiring the reimplementation-correctness risk. It
needs the ESPnet fork installed and $XEUS_CHECKPOINT set, so it's skipped in CI
and run on the GPU server.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from svb.model.xeus_backend import resolve_backend


def test_resolve_backend_explicit():
    assert resolve_backend("standalone") == "standalone"
    assert resolve_backend("espnet") == "espnet"


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_backend("nope")


def test_resolve_auto_matches_install_state():
    expected = "espnet" if importlib.util.find_spec("espnet2") else "standalone"
    assert resolve_backend("auto") == expected


_HAVE_ESPNET = importlib.util.find_spec("espnet2") is not None
_HAVE_CKPT = bool(os.environ.get("XEUS_CHECKPOINT"))


@pytest.mark.skipif(
    not (_HAVE_ESPNET and _HAVE_CKPT),
    reason="needs ESPnet fork + $XEUS_CHECKPOINT (run on the GPU server)",
)
def test_espnet_standalone_feature_equivalence():
    """Both backends must yield equivalent final-layer features on the same audio."""
    import torch

    from svb.model.xeus_backend import load_xeus_encoder

    ckpt = os.environ["XEUS_CHECKPOINT"]
    espnet_enc = load_xeus_encoder("ssl", ckpt, backend="espnet").eval()
    standalone_enc = load_xeus_encoder("ssl", ckpt, backend="standalone").eval()

    torch.manual_seed(0)
    wav = torch.randn(1, 16000)  # 1s of audio
    lengths = torch.tensor([16000])
    with torch.no_grad():
        a, _ = espnet_enc.encode(wav, lengths, use_final_output=True)
        b, _ = standalone_enc.encode(wav, lengths, use_final_output=True)

    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    max_abs = (a - b).abs().max().item()
    assert max_abs < 1e-3, f"feature mismatch: max|Δ|={max_abs}"
