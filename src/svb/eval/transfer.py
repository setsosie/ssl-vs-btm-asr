"""Held-out-language transfer (finding #3).

Given a starting point — the SSL encoder (arm A), or a merged BTM checkpoint
(arms B/C) — adapt to an unseen language:

  1. expand the char vocab with the new language's characters,
  2. grow the CTC head (trained rows preserved, new rows seeded-random),
  3. fine-tune on the new language,
  4. evaluate on its disjoint test split.

This is the experiment behind "does the encoder transfer to scripts it never
saw?" — zero-shot is ~100% WER (vocabulary mismatch), and the question is how
low fine-tuning drives it, and whether SSL alone matches the full BTM pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..config import ExperimentConfig
from ..data.collate import make_ctc_collate
from ..data.datasets import load_language, load_texts
from ..data.registry import LangSpec
from ..model.ctc_vocab import CtcVocab, expand_vocab
from ..model.xeus_ctc import XeusCTC
from ..train.trainer import train
from .evaluate import EvalResult, evaluate


@dataclass
class TransferResult:
    lang: str
    wer: float
    cer: float
    n: int


def transfer_one(
    cfg: ExperimentConfig,
    init_ckpt: Path | None,
    init: Literal["ssl", "scratch"],
    base_vocab: CtcVocab,
    lang: LangSpec,
    out_dir: Path,
    device: str = "cuda",
) -> EvalResult:
    """Adapt to one held-out language and evaluate.

    Args:
        init_ckpt: Checkpoint to start from. ``None`` means start from the bare
            encoder (arm A loads the SSL encoder via ``init='ssl'``).
        init: "ssl" or "scratch" for encoder weights when no full checkpoint.
        base_vocab: The training vocab whose head rows we preserve.
    """
    new_vocab, _ = expand_vocab(base_vocab, load_texts(lang, "train"))
    model = XeusCTC(
        vocab_size=base_vocab.size,
        init=init,
        checkpoint=cfg.model.xeus_checkpoint,
        hidden_size=cfg.model.hidden_size,
    )
    if init_ckpt is not None:
        model.load(init_ckpt)
    model.expand_head(new_vocab.size, seed=cfg.seed)

    collate = make_ctc_collate(new_vocab)
    train_ds = load_language(lang, "train", cfg.train.max_audio_samples)
    val_ds = load_language(lang, "validation", cfg.train.max_audio_samples)
    train(model, cfg, train_ds, val_ds, collate, cfg.train.finetune_epochs, out_dir, device)
    model.load(out_dir / "best.pt")

    test_ds = load_language(lang, "test", cfg.train.max_audio_samples)
    return evaluate(
        model,
        test_ds,
        new_vocab,
        collate,
        device=device,
        batch_size=cfg.optim.batch_size,
        save_predictions=out_dir / "predictions.json",
    )
