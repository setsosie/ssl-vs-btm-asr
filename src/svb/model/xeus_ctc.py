"""XEUS encoder + CTC head — the lean model for all three arms.

A thin wrapper over the fork-free ``StandaloneXEUS`` encoder:

  - ``init="ssl"``   loads the XEUS SSL checkpoint (arms A, B)
  - ``init="scratch"`` uses random weights (arm C)

The head is ``LayerNorm(hidden) -> Linear(hidden, vocab)``. The LayerNorm is
load-bearing: SSL-pretrained XEUS features have a large scale that destabilises
a bare linear CTC head. ``expand_head`` grows the vocab dimension while
preserving trained rows — used for held-out-language transfer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from .xeus_standalone import StandaloneXEUS, load_xeus_from_checkpoint

HIDDEN_SIZE = 1024


class XeusCTC(nn.Module):
    """XEUS E-Branchformer encoder with a character CTC head."""

    def __init__(
        self,
        vocab_size: int,
        init: Literal["ssl", "scratch"] = "ssl",
        checkpoint: str | None = None,
        hidden_size: int = HIDDEN_SIZE,
        blank_bias_init: float | None = None,
    ) -> None:
        super().__init__()
        if init == "ssl":
            if not checkpoint:
                raise ValueError("init='ssl' requires a XEUS checkpoint path")
            self.encoder: StandaloneXEUS = load_xeus_from_checkpoint(checkpoint, device="cpu")
        else:
            self.encoder = StandaloneXEUS()  # random init
        self.hidden_size = hidden_size
        self.ctc_norm = nn.LayerNorm(hidden_size)
        self.ctc_proj = nn.Linear(hidden_size, vocab_size)
        if blank_bias_init is not None:
            with torch.no_grad():
                self.ctc_proj.bias.data[0] = blank_bias_init

    @property
    def vocab_size(self) -> int:
        return self.ctc_proj.out_features

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            input_values: Raw waveforms, (batch, time).
            attention_mask: (batch, time) padding mask; sum gives wav lengths.
            labels: (batch, seq) CTC targets, padded with -100.

        Returns:
            dict with ``logits`` (B, T', V), ``input_lengths`` (B,), and
            ``loss`` when labels are given.
        """
        bsz = input_values.shape[0]
        if attention_mask is not None:
            wav_lengths = attention_mask.sum(dim=1).long()
        else:
            wav_lengths = torch.full(
                (bsz,), input_values.shape[1], device=input_values.device, dtype=torch.long
            )

        feats, out_lengths = self.encoder.encode(input_values, wav_lengths, use_final_output=True)
        feats = self.ctc_norm(feats)
        logits = self.ctc_proj(feats)

        result: dict[str, Any] = {"logits": logits, "input_lengths": out_lengths}

        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
            target_lengths = (labels != -100).sum(dim=-1)
            targets = labels.clone()
            targets[targets == -100] = 0
            result["loss"] = F.ctc_loss(
                log_probs,
                targets,
                out_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
        return result

    @torch.no_grad()
    def greedy_decode(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return per-frame argmax ids, (batch, time')."""
        logits = self.forward(input_values, attention_mask=attention_mask)["logits"]
        return logits.argmax(dim=-1)

    def expand_head(self, new_vocab_size: int, seed: int = 0) -> None:
        """Grow the CTC head to ``new_vocab_size``, preserving trained rows.

        Existing rows keep their weights; the appended rows (the new
        characters, always added at the end by ``expand_vocab``) are randomly
        initialised with a seeded generator for reproducibility.
        """
        old = self.ctc_proj
        if new_vocab_size <= old.out_features:
            return
        new = nn.Linear(self.hidden_size, new_vocab_size)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            new.weight.normal_(0.0, 0.02, generator=gen)
            new.bias.zero_()
            new.weight[: old.out_features].copy_(old.weight)
            new.bias[: old.out_features].copy_(old.bias)
        self.ctc_proj = new.to(old.weight.device)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location: str = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))
