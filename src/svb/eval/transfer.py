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

from ..config import ExperimentConfig
from ..data.collate import make_ctc_collate
from ..data.datasets import load_language, load_texts
from ..data.registry import LangSpec
from ..model.ctc_vocab import CtcVocab, expand_vocab
from ..model.xeus_ctc import make_model
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
    base_vocab: CtcVocab,
    lang: LangSpec,
    out_dir: Path,
    device: str = "cuda",
) -> EvalResult:
    """Adapt to one held-out language and evaluate.

    Args:
        init_ckpt: Checkpoint to start from. ``None`` means start from the bare
            encoder, which for arm A is the SSL one (``cfg.init``).
        base_vocab: The training vocab whose head rows we preserve.
    """
    new_vocab, _, _ = expand_vocab(base_vocab, load_texts(lang, "train"))
    model = make_model(cfg, base_vocab.size)
    if init_ckpt is not None:
        model.load(init_ckpt)
    model.expand_head(new_vocab.size, seed=cfg.seed)

    train_collate = make_ctc_collate(
        new_vocab,
        max_audio_samples=cfg.train.max_audio_samples,
        drop_overlong=True,
        drop_empty=True,
    )
    eval_collate = make_ctc_collate(new_vocab)
    train_ds = load_language(lang, "train", cfg.train.max_audio_samples)
    val_ds = load_language(lang, "validation", cfg.train.max_audio_samples)
    # Load the checkpoint the trainer says it wrote, not a path re-derived here:
    # two sources of truth for one filename is how a stale model gets evaluated.
    result = train(
        model, cfg, train_ds, val_ds, train_collate, cfg.train.finetune_epochs, out_dir, device
    )
    model.load(result.checkpoint)

    # No truncation at test time: a clipped waveform scored against its full
    # transcript manufactures deletions that the model never had a chance to
    # avoid.
    test_ds = load_language(lang, "test", None)
    return evaluate(
        model,
        test_ds,
        new_vocab,
        eval_collate,
        device=device,
        batch_size=cfg.optim.batch_size,
        save_predictions=out_dir / "predictions.json",
    )
