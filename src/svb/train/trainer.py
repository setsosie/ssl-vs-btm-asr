"""Lean single-GPU CTC trainer.

AdamW + linear warmup→linear decay, gradient accumulation, bf16 autocast,
gradient clipping, and val-loss early stopping (patience). Deliberately minimal:
no DDP, no W&B, no schedulers beyond linear — the bare minimum needed to produce
honest numbers. Multi-GPU is left to the launcher running independent
(arm, scale, seed) jobs in parallel rather than DDP within a job.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
    #: Training pairs removed because the transcript could not be aligned to the
    #: audio, and utterances the audio guard truncated. Counted over the first
    #: epoch, which is the size of the effect on the split; later epochs see the
    #: same data and would multiply it by the epoch count. Reported so the
    #: reader knows what the run actually trained on rather than what the split
    #: contains.
    n_dropped_unalignable: int = 0
    n_at_audio_guard: int = 0


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Give each DataLoader worker its own reproducible random stream.

    Workers are forked after the parent has been seeded, so without this they
    all inherit the same Python and NumPy state. Torch reseeds its own
    generator per worker; ``random`` and ``numpy`` are the two it leaves alone.
    """

    def init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))

    return init


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
    collate: Callable[..., dict[str, Any]],
    max_epochs: int,
    out_dir: Path,
    device: str = "cuda",
) -> TrainResult:
    """Train ``model`` on ``train_ds``, selecting the best-val checkpoint.

    Selection is on validation loss, not WER; with CTC the two can diverge, so
    the choice is part of the reported protocol. An epoch-0 checkpoint is always
    written as a floor, so ``TrainResult.checkpoint`` names a file that exists
    even for a run whose loss never becomes finite.

    Early stopping needs ``patience`` consecutive non-improving epochs, so it
    cannot fire at all when ``patience >= max_epochs``. That is the case for
    phase 0 and the experts under the shipped config, and the run log says so
    rather than leaving the impression that a stop was possible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best.pt"
    model.to(device)
    if cfg.train.grad_checkpointing:
        model.set_gradient_checkpointing(True)

    n_train = len(train_ds)  # type: ignore[arg-type]
    if n_train == 0:
        raise ValueError(
            "empty training split: the loader would yield no batches and the run "
            "would report a completed training that touched no data"
        )
    if cfg.train.patience >= max_epochs:
        print(
            f"[svb] early stopping is inert: patience={cfg.train.patience} "
            f">= max_epochs={max_epochs}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
        # Dropping the tail batch is a throughput choice, not a correctness one.
        # A split smaller than one batch would otherwise yield no batches at
        # all, collapsing the LR schedule and training on nothing.
        drop_last=n_train >= cfg.optim.batch_size,
        generator=torch.Generator().manual_seed(cfg.seed),
        worker_init_fn=make_worker_init_fn(cfg.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
        worker_init_fn=make_worker_init_fn(cfg.seed),
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
    first_epoch_dropped = 0
    first_epoch_at_guard = 0
    bad = 0
    epoch = -1
    # Floor: a run whose validation loss never becomes finite would otherwise
    # leave no best.pt, and the caller loads that path unconditionally.
    model.save(ckpt)
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
        if epoch == 0:
            # The first pass over the split is the size of the effect. Summing
            # every epoch would report one dropped pair as `max_epochs` of them.
            first_epoch_dropped, first_epoch_at_guard = n_dropped, n_at_guard

        val_loss = _validate(model, val_loader, device, autocast)
        # NaN never compares less than anything, so an unguarded `<` would
        # already reject it — the explicit check states the intent, and keeps
        # a NaN from being adopted should best_val ever start out non-finite.
        if math.isfinite(val_loss) and val_loss < best_val:
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
        n_dropped_unalignable=first_epoch_dropped,
        n_at_audio_guard=first_epoch_at_guard,
    )


@torch.no_grad()
def _validate(
    model: XeusCTC,
    loader: DataLoader,
    device: str,
    autocast: AbstractContextManager[Any],
) -> float:
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
