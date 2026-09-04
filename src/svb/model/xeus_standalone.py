"""Standalone XEUS encoder (fork-free reimplementation).

Reimplements the XEUS E-Branchformer architecture so the published checkpoint
can be loaded with PyTorch alone — no ESPnet dependency. XEUS is the work of
Chen et al.; this file only re-expresses its architecture to load the public
weights.

  Model:    XEUS — "Towards Robust Speech Representation Learning for Thousands
            of Languages", Chen et al. (arXiv:2407.00837); weights: espnet/xeus.
  Origin:   Adapted from the author's FLAIME project (same author as this repo).
            All credit for the XEUS model and weights belongs to its authors.

Architecture (577M params):
  - Frontend: 7-layer wav2vec2-style CNN (512-dim)
  - Preencoder: Linear projection (512 -> 1024)
  - Encoder: 19-block E-Branchformer (1024-dim, 8 heads)
  - Convolutional positional encoding (kernel=128, groups=16)

NOTE: a bespoke reimplementation is a known correctness risk (an earlier version
had E-Branchformer forward deviations from the ESPnet reference). See the README
"ESPnet" note for the planned migration to the reference implementation.

Padding and the time-axis convolutions
--------------------------------------
Self-attention is masked, but the two convolutions over time — the cgMLP's
depthwise ``_CSGU.conv`` and the branch-fusion ``depthwise_conv_fusion``, both
kernel 31 — are not. Padded frames therefore bleed up to 15 positions into
valid ones, so an utterance's encoding depends slightly on what shared its
batch.

This is deliberate, and it is not a bug to be fixed here. The ESPnet reference
does not mask either convolution: ``ConvolutionalSpatialGatingUnit.forward``
normalizes and convolves with no mask, ``ConvolutionalGatingMLP.forward``
accepts a ``mask`` argument and never uses it, and
``EBranchformerEncoderLayer.forward`` passes its mask only to the attention
module. (Verified against the pinned fork at
raw.githubusercontent.com/wanchichen/espnet/5b52d57b4f872ff7babded35316a79642e9e6c12,
espnet2/asr/layers/cgmlp.py and espnet2/asr/encoder/e_branchformer_encoder.py.)
The published weights were pretrained under exactly these conditions, so
masking here would diverge from the checkpoint rather than correct it.

The consequence belongs to the evaluation protocol instead: batch composition
is part of a reported number. ``eval.evaluate`` therefore batches in descending
length order at a fixed batch size, which makes membership a function of the
split rather than of row order, and results are comparable only across runs
that used the same batch size.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

# wav2vec2-style CNN frontend specifications
_FRONTEND_KERNELS = [10, 3, 3, 3, 3, 2, 2]
_FRONTEND_STRIDES = [5, 2, 2, 2, 2, 2, 2]
_FRONTEND_DIM = 512

# E-Branchformer encoder constants (from model/config.yaml)
_HIDDEN_SIZE = 1024
_NUM_HEADS = 8
_FFN_DIM = 4096
_CGMLP_DIM = 4096
_CGMLP_KERNEL = 31
_MERGE_KERNEL = 31
_NUM_BLOCKS = 19
_POS_KERNEL = 128
_POS_GROUPS = 16


def frontend_output_length(n_samples: int) -> int:
    """Encoder frames produced by an ``n_samples`` waveform at 16 kHz.

    Closed form of the 7-layer CNN frontend's downsampling, so callers can size
    a CTC target without running the encoder. The convolutions are unpadded, so
    each layer maps ``n`` to ``(n - kernel) // stride + 1``, floored at zero for
    waveforms shorter than the receptive field.
    """
    n = int(n_samples)
    for kernel, stride in zip(_FRONTEND_KERNELS, _FRONTEND_STRIDES, strict=True):
        n = max(0, (n - kernel) // stride + 1)
    return n


def max_label_len_for_samples(n_samples: int) -> int:
    """Longest CTC target that ``n_samples`` of audio can still align to.

    CTC needs at least one input frame per target symbol. A target longer than
    the frame budget cannot be aligned at all: the loss is ``inf``, and the
    ``zero_infinity`` guard rewrites that to zero — which silences the sample's
    gradient *and* pulls down the mean loss that checkpoint selection reads.
    Callers use this to drop such pairs rather than let them score as perfect.
    """
    return frontend_output_length(n_samples)


# ---------------------------------------------------------------------------
# Frontend: wav2vec2 CNN feature extractor
# ---------------------------------------------------------------------------


class _FrontendLayer(nn.Module):
    """Single CNN layer: Conv1d + LayerNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C) for LayerNorm
        x = self.layer_norm(x)
        x = F.gelu(x)
        return x.transpose(1, 2)  # (B, C, T)


class _Frontend(nn.Module):
    """7-layer wav2vec2-style CNN feature extractor (1-channel -> 512-dim)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # Layer 0: mono waveform -> 512, kernel=10, stride=5
        self.layers.append(
            _FrontendLayer(1, _FRONTEND_DIM, _FRONTEND_KERNELS[0], _FRONTEND_STRIDES[0])
        )
        # Layers 1-6: 512 -> 512
        for k, s in zip(_FRONTEND_KERNELS[1:], _FRONTEND_STRIDES[1:], strict=False):
            self.layers.append(_FrontendLayer(_FRONTEND_DIM, _FRONTEND_DIM, k, s, bias=True))

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Process raw waveforms through CNN frontend.

        Args:
            x: Raw waveforms (batch, time)
            lengths: Waveform lengths (batch,)

        Returns:
            (features, output_lengths) where features is (batch, time', 512)
        """
        # Per-utterance zero-mean unit-variance normalization (matches
        # ESPnet wav2vec_cnn normalize_audio=True).  When lengths are known,
        # exclude padding zeros from the statistics so short utterances
        # padded to the batch max aren't distorted.
        if lengths is not None:
            mask = torch.arange(x.shape[-1], device=x.device) < lengths.unsqueeze(1)  # (B, T)
            mask_f = mask.float()
            n = lengths.float().unsqueeze(1).clamp(min=1)
            mean = (x * mask_f).sum(dim=-1, keepdim=True) / n
            x_centered = (x - mean) * mask_f
            var = (x_centered**2).sum(dim=-1, keepdim=True) / n
            x = x_centered / (var.sqrt() + 1e-5)
        else:
            x = F.layer_norm(x, (x.shape[-1],))
        x = x.unsqueeze(1)  # (B, 1, T)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)  # (B, T', 512)

        # Compute output lengths
        if lengths is not None:
            out_lengths = lengths.clone()
            for k, s in zip(_FRONTEND_KERNELS, _FRONTEND_STRIDES, strict=False):
                out_lengths = (out_lengths - k) // s + 1
        else:
            out_lengths = None

        return x, out_lengths


# ---------------------------------------------------------------------------
# Preencoder: linear projection 512 -> 1024
# ---------------------------------------------------------------------------


class _Preencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_out = nn.Linear(_FRONTEND_DIM, _HIDDEN_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(x)


# ---------------------------------------------------------------------------
# Convolutional positional encoding
# ---------------------------------------------------------------------------


class _ConvPositionalEncoding(nn.Module):
    """Convolutional positional encoding (wav2vec2-style, groups=16)."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        _HIDDEN_SIZE,
                        _HIDDEN_SIZE,
                        _POS_KERNEL,
                        padding=_POS_KERNEL // 2,
                        groups=_POS_GROUPS,
                    ),
                    dim=2,
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (B, T, C)."""
        seq_len = x.size(1)
        x_conv = x.transpose(1, 2)  # (B, C, T)
        x_conv = self.convs[0](x_conv)
        x_conv = x_conv[..., :seq_len]  # trim padding overshoot
        x_conv = x_conv.transpose(1, 2)  # (B, T, C)
        return x + F.gelu(x_conv)


# ---------------------------------------------------------------------------
# E-Branchformer components
# ---------------------------------------------------------------------------


class _MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (no relative position bias)."""

    def __init__(self, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.linear_q = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE)
        self.linear_k = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE)
        self.linear_v = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE)
        self.linear_out = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE)
        self.num_heads = _NUM_HEADS
        self.d_k = _HIDDEN_SIZE // _NUM_HEADS
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-attention with optional key-padding mask.

        Args:
            x: (B, T, hidden_size).
            padding_mask: (B, T) bool tensor. True marks valid positions;
                False marks padding that must not be attended to. When
                None (single utterance or no padding), behaves identically
                to the unmasked implementation.
        """
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Broadcast (B, T) -> (B, 1, 1, T) over heads and query positions.
        # SDPA treats bool `attn_mask` as "True = attend, False = mask".
        attn_mask = padding_mask.view(B, 1, 1, T) if padding_mask is not None else None

        # Use PyTorch SDPA (auto-selects Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, _HIDDEN_SIZE)
        return self.linear_out(out)


class _FeedForward(nn.Module):
    """Positionwise feed-forward: Linear -> SiLU -> Dropout -> Linear."""

    def __init__(self, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(_HIDDEN_SIZE, _FFN_DIM)
        self.w_2 = nn.Linear(_FFN_DIM, _HIDDEN_SIZE)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.silu(self.w_1(x))))


class _CSGU(nn.Module):
    """Convolutional Spatial Gating Unit (gate_activation=identity)."""

    def __init__(self) -> None:
        super().__init__()
        half = _CGMLP_DIM // 2  # 2048
        self.norm = nn.LayerNorm(half)
        self.conv = nn.Conv1d(
            half, half, _CGMLP_KERNEL, groups=half, padding=(_CGMLP_KERNEL - 1) // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 4096) -> (B, T, 2048)."""
        x1, x2 = x.chunk(2, dim=-1)
        x2 = self.norm(x2)
        x2 = x2.transpose(1, 2)
        x2 = self.conv(x2)
        x2 = x2.transpose(1, 2)
        return x1 * x2  # identity gate activation


class _CGMLP(nn.Module):
    """Convolutional Gating MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_proj1 = nn.Sequential(
            nn.Linear(_HIDDEN_SIZE, _CGMLP_DIM),
            nn.GELU(),
        )
        self.csgu = _CSGU()
        self.channel_proj2 = nn.Linear(_CGMLP_DIM // 2, _HIDDEN_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj1(x)
        x = self.csgu(x)
        return self.channel_proj2(x)


class _EBranchformerBlock(nn.Module):
    """Single E-Branchformer encoder block.

    Architecture:
      1. Macaron FFN (half-step residual)
      2. Parallel branches: MHSA + CGMLP
      3. Depthwise conv fusion + merge projection
      4. FFN (half-step residual)
    """

    def __init__(self, dropout_rate: float = 0.1) -> None:
        super().__init__()
        # Attention branch
        self.attn = _MultiHeadSelfAttention(dropout_rate)
        # CGMLP branch
        self.cgmlp = _CGMLP()
        # Feed-forward (macaron + standard)
        self.feed_forward = _FeedForward(dropout_rate)
        self.feed_forward_macaron = _FeedForward(dropout_rate)
        # Branch fusion
        self.depthwise_conv_fusion = nn.Conv1d(
            _HIDDEN_SIZE * 2,
            _HIDDEN_SIZE * 2,
            _MERGE_KERNEL,
            groups=_HIDDEN_SIZE * 2,
            padding=(_MERGE_KERNEL - 1) // 2,
        )
        self.merge_proj = nn.Linear(_HIDDEN_SIZE * 2, _HIDDEN_SIZE)
        # Layer norms (5 per block)
        self.norm_ff = nn.LayerNorm(_HIDDEN_SIZE)
        self.norm_ff_macaron = nn.LayerNorm(_HIDDEN_SIZE)
        self.norm_mha = nn.LayerNorm(_HIDDEN_SIZE)
        self.norm_mlp = nn.LayerNorm(_HIDDEN_SIZE)
        self.norm_final = nn.LayerNorm(_HIDDEN_SIZE)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Macaron FFN (half-step)
        residual = x
        x = self.norm_ff_macaron(x)
        x = residual + 0.5 * self.dropout(self.feed_forward_macaron(x))

        # Parallel branches
        x_att = self.dropout(self.attn(self.norm_mha(x), padding_mask))
        x_mlp = self.dropout(self.cgmlp(self.norm_mlp(x)))

        # Merge branches. The depthwise conv smooths across time; the ESPnet
        # reference passes x_concat + x_tmp (pre- and post-smoothed) into
        # merge_proj so the un-smoothed signal is preserved. Dropping that
        # residual causes each block to apply extra temporal smoothing, which
        # saturates adj_cos → 1.0 after 4-5 blocks.
        x_concat = torch.cat([x_att, x_mlp], dim=-1)  # (B, T, 2048)
        x_tmp = x_concat.transpose(1, 2)  # (B, 2048, T)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)  # (B, T, 2048)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))

        # FFN (half-step) — must come BEFORE norm_final in ESPnet's
        # EBranchformerEncoderLayer, so the pretrained norm_final γ/β are
        # applied to the post-FFN output they were trained to normalize.
        residual = x
        x = self.norm_ff(x)
        x = residual + 0.5 * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        return x


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    """E-Branchformer encoder with convolutional positional encoding."""

    def __init__(self, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.ModuleList([_ConvPositionalEncoding()])
        self.encoders = nn.ModuleList(
            [_EBranchformerBlock(dropout_rate) for _ in range(_NUM_BLOCKS)]
        )
        self.after_norm = nn.LayerNorm(_HIDDEN_SIZE)
        # Trade compute for activation memory: when set, each block's
        # activations are discarded after the forward pass and rebuilt during
        # backward. Plain attribute, not a buffer — it must not enter
        # state_dict and change checkpoint compatibility.
        self.gradient_checkpointing = False

    def _run_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """One encoder block, recomputed on backward when checkpointing is on.

        ``use_reentrant=False`` is the non-deprecated implementation and the
        one that tolerates a block returning a plain tensor while some of its
        inputs (the padding mask) need no gradient.
        """
        if self.gradient_checkpointing and self.training:
            out = checkpoint(block, x, padding_mask, use_reentrant=False)
            return cast(torch.Tensor, out)
        return cast(torch.Tensor, block(x, padding_mask))

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None,
        use_final_output: bool = True,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | None]:
        """Run encoder.

        Args:
            x: Input features (batch, time, hidden_size)
            lengths: Sequence lengths (batch,)
            use_final_output: If True, return final output only.
                If False, return list of all layer outputs (for intermediate losses).

        Returns:
            (output, lengths) where output is a tensor or list of tensors
        """
        x = self.embed[0](x)

        # Build boolean key-padding mask from lengths so self-attention
        # ignores padded positions. True = valid, False = padding. When
        # lengths is None (e.g. single utterance), the mask is not built
        # and attention falls back to unmasked behavior.
        padding_mask: torch.Tensor | None = None
        if lengths is not None:
            time_dim = x.size(1)
            arange = torch.arange(time_dim, device=x.device)
            padding_mask = arange.unsqueeze(0) < lengths.unsqueeze(1)

        if use_final_output:
            for block in self.encoders:
                x = self._run_block(block, x, padding_mask)
            return self.after_norm(x), lengths

        # Collect intermediate layer outputs (no after_norm)
        layer_outputs: list[torch.Tensor] = []
        for block in self.encoders:
            x = self._run_block(block, x, padding_mask)
            layer_outputs.append(x)
        return layer_outputs, lengths


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class StandaloneXEUS(nn.Module):
    """Standalone XEUS model (frontend + preencoder + encoder).

    Provides the same ``encode()`` interface as the ESPnet SSL model,
    so it can be used as a drop-in replacement in XEUSASRModel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.frontend = _Frontend()
        self.preencoder = _Preencoder()
        self.encoder = _Encoder()
        # Optional SpecAugment between preencoder and encoder. Left as
        # None by default; XEUSASRModel attaches an instance when
        # SpecAug is enabled in the training config. Applied only in
        # .train() mode (the module self-gates on self.training).
        self.spec_augment: nn.Module | None = None

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.encoder.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.encoder.gradient_checkpointing = bool(enabled)

    def encode(
        self,
        waveforms: torch.Tensor,
        wav_lengths: torch.Tensor,
        use_mask: bool = False,
        use_final_output: bool = True,
    ) -> tuple[Any, torch.Tensor | None]:
        """Encode raw waveforms to hidden representations.

        Args:
            waveforms: Raw audio (batch, time)
            wav_lengths: Waveform lengths (batch,)
            use_mask: Unused (kept for API compatibility)
            use_final_output: If True return final encoder output,
                otherwise return list of per-layer outputs.

        Returns:
            (features, output_lengths)
        """
        features, lengths = self.frontend(waveforms, wav_lengths)
        features = self.preencoder(features)
        if self.spec_augment is not None:
            features = self.spec_augment(features)
        return self.encoder(features, lengths, use_final_output)


def load_xeus_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> StandaloneXEUS:
    """Load XEUS model from HuggingFace checkpoint.

    Filters out the SSL training head (losses.*, util_modules.*, global_step)
    and loads only the frontend, preencoder, and encoder weights.

    Args:
        checkpoint_path: Path to xeus_checkpoint_new.pth
        device: Device to load weights onto

    Returns:
        Loaded StandaloneXEUS model
    """
    model = StandaloneXEUS()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Filter out SSL training head and metadata
    skip_prefixes = ("losses.", "util_modules.", "global_step")
    filtered = {k: v for k, v in ckpt.items() if not k.startswith(skip_prefixes)}

    model.load_state_dict(filtered, strict=True)
    return model
