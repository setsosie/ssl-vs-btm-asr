"""Lean single-GPU CTC trainer.

AdamW + linear warmup→linear decay, gradient accumulation, bf16 autocast,
gradient clipping, and val-loss early stopping (patience). Deliberately minimal:
no DDP, no W&B, no schedulers beyond linear — the bare minimum needed to produce
honest numbers. Multi-GPU is left to the launcher running independent
(arm, scale, seed) jobs in parallel rather than DDP within a job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from ..model.xeus_ctc import XeusCTC


@dataclass
class TrainResult:
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    checkpoint: Path


def _model_inputs(batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    """Tensor entries of a collated batch, on ``device``.

    The batch also carries the raw transcripts (``texts``), which the model does
    not take and which cannot be moved to a device.
    """
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def train(
    model: XeusCTC,
    cfg: ExperimentConfig,
    train_ds: Dataset,
    val_ds: Dataset,
    collate,
    max_epochs: int,
    out_dir: Path,
    device: str = "cuda",
) -> TrainResult:
    """Train ``model`` on ``train_ds``, selecting the best-val checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best.pt"
    model.to(device)
    if cfg.train.grad_checkpointing:
        model.encoder.gradient_checkpointing = True  # honored if encoder supports it

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg.optim.accum_steps)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * cfg.optim.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, total_steps, warmup_steps)
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if cfg.train.bf16 and device == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    epoch = -1
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        n_at_guard = 0
        n_dropped = 0
        for i, batch in enumerate(train_loader):
            n_at_guard += int(batch.get("n_at_audio_guard", 0))
            n_dropped += int(batch.get("n_dropped", 0))
            if batch["input_values"].shape[0] == 0:
                continue  # every pair in this batch was dropped as unalignable
            with autocast:
                out = model(**_model_inputs(batch, device))
                loss = out["loss"] / cfg.optim.accum_steps
            loss.backward()
            if (i + 1) % cfg.optim.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        if n_at_guard or n_dropped:
            print(
                f"[svb] epoch {epoch}: {n_at_guard} utterances reached the audio "
                f"truncation guard, {n_dropped} dropped as unalignable"
            )

        val_loss = _validate(model, val_loader, device, autocast)
        if val_loss < best_val:
            best_val, best_epoch, bad = val_loss, epoch, 0
            model.save(ckpt)
        else:
            bad += 1
            if bad >= cfg.train.patience:
                break

    return TrainResult(
        best_val_loss=best_val,
        best_epoch=best_epoch,
        epochs_run=epoch + 1,
        checkpoint=ckpt,
    )


@torch.no_grad()
def _validate(model: XeusCTC, loader: DataLoader, device: str, autocast) -> float:
    """Mean validation loss per *utterance*, not per batch.

    The val loader keeps its last short batch, so averaging batch means would
    let a two-sample tail count as much as a full batch in the number that
    checkpoint selection reads.
    """
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        n_in_batch = int(batch["input_values"].shape[0])
        if n_in_batch == 0:
            continue
        with autocast:
            loss = model(**_model_inputs(batch, device))["loss"]
        total += float(loss) * n_in_batch
        n += n_in_batch
    return total / max(1, n)
