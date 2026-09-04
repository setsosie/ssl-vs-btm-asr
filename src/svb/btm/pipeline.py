"""Branch-Train-Merge pipeline (arms B and C).

phase 0  : joint multilingual CTC training over the preset's languages
experts  : branch from phase 0, fine-tune one expert per language
merge    : combine experts (default AVERAGE) back to a single checkpoint

Arm A (SSL-only) does not use this module — it fine-tunes per language directly
from the encoder (see ``cli.run``). Arm B inits phase 0 from the XEUS SSL
checkpoint; arm C inits from random weights.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import ConcatDataset

from ..config import ExperimentConfig
from ..data.collate import make_ctc_collate
from ..data.datasets import load_language, load_texts
from ..data.registry import LangSpec
from ..merge.strategy import MERGE_STRATEGIES
from ..model.ctc_vocab import CtcVocab, build_vocab_from_texts
from ..model.xeus_ctc import make_model
from ..train.trainer import train


def build_training_vocab(specs: list[LangSpec]) -> CtcVocab:
    """Char vocab over all training transcripts across the preset languages."""
    texts: list[str] = []
    for spec in specs:
        texts.extend(load_texts(spec, "train"))
    return build_vocab_from_texts(texts)


def run_phase0(
    cfg: ExperimentConfig,
    specs: list[LangSpec],
    vocab: CtcVocab,
    out_dir: Path,
    device: str = "cuda",
) -> Path:
    """Joint multilingual CTC training; returns the best checkpoint path."""
    collate = make_ctc_collate(
        vocab, max_audio_samples=cfg.train.max_audio_samples, drop_overlong=True
    )
    train_ds: ConcatDataset[tuple[Tensor, str]] = ConcatDataset(
        [load_language(s, "train", cfg.train.max_audio_samples) for s in specs]
    )
    val_ds: ConcatDataset[tuple[Tensor, str]] = ConcatDataset(
        [load_language(s, "validation", cfg.train.max_audio_samples) for s in specs]
    )
    model = make_model(cfg, vocab.size)
    result = train(model, cfg, train_ds, val_ds, collate, cfg.train.phase0_epochs, out_dir, device)
    return result.checkpoint


def train_experts(
    cfg: ExperimentConfig,
    phase0_ckpt: Path,
    specs: list[LangSpec],
    vocab: CtcVocab,
    out_dir: Path,
    device: str = "cuda",
) -> dict[str, Path]:
    """Fine-tune one expert per language, each branched from phase 0."""
    collate = make_ctc_collate(
        vocab, max_audio_samples=cfg.train.max_audio_samples, drop_overlong=True
    )
    experts: dict[str, Path] = {}
    for spec in specs:
        model = make_model(cfg, vocab.size)
        model.load(phase0_ckpt)
        lang_dir = out_dir / f"expert_{spec.code}"
        train_ds = load_language(spec, "train", cfg.train.max_audio_samples)
        val_ds = load_language(spec, "validation", cfg.train.max_audio_samples)
        result = train(
            model, cfg, train_ds, val_ds, collate, cfg.train.expert_epochs, lang_dir, device
        )
        experts[spec.code] = result.checkpoint
    return experts


def merge_experts(
    expert_ckpts: dict[str, Path],
    strategy: str,
    out_dir: Path,
    base_ckpt: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Merge expert state_dicts; task-vector methods need ``base_ckpt`` (phase 0)."""
    import torch

    merge_fn = MERGE_STRATEGIES[strategy]
    experts = [torch.load(p, map_location=device) for p in expert_ckpts.values()]
    base = torch.load(base_ckpt, map_location=device) if base_ckpt is not None else None
    merged = merge_fn(experts, base=base)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"merged_{strategy}.pt"
    torch.save(merged, path)
    return path
