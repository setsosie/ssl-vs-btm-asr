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

from ..config import ExperimentConfig, TextConfig
from ..data.collate import make_ctc_collate
from ..data.datasets import load_language, load_texts
from ..data.registry import LangSpec
from ..merge.strategy import MERGE_STRATEGIES
from ..model.ctc_vocab import CtcVocab, build_vocab_from_texts, require_space_token
from ..model.xeus_ctc import make_model
from ..train.trainer import TrainResult, train


def build_training_vocab(
    specs: list[LangSpec], text_cfg: TextConfig | None = None
) -> tuple[CtcVocab, dict[str, int]]:
    """Char vocab over all training transcripts across the preset languages.

    Returns the vocab and the ``{character: count}`` map of what the frequency
    floor evicted, which the run records rather than discarding: the floor
    changes what the model can emit, so it belongs in the artifacts.

    Raises when a preset containing space-separated languages produces a vocab
    with no space character — a floor tuned for a full run can evict it from a
    small corpus, and the result would be a model that emits one unbroken string
    while looking merely unconverged.
    """
    text_cfg = text_cfg or TextConfig()
    texts: list[str] = []
    for spec in specs:
        texts.extend(load_texts(spec, "train"))
    vocab, evicted = build_vocab_from_texts(
        texts, policy=text_cfg.policy, min_char_count=text_cfg.min_char_count
    )
    if any(spec.word_boundary for spec in specs):
        require_space_token(vocab)
    return vocab, evicted


def run_phase0(
    cfg: ExperimentConfig,
    specs: list[LangSpec],
    vocab: CtcVocab,
    out_dir: Path,
    device: str = "cuda",
) -> TrainResult:
    """Joint multilingual CTC training.

    Returns the whole :class:`TrainResult`, not just its checkpoint path: how
    many pairs training dropped is part of what the run did, and a function that
    returns only the path throws that away where no caller can recover it.
    """
    collate = make_ctc_collate(
        vocab, max_audio_samples=cfg.train.max_audio_samples, drop_overlong=True, drop_empty=True
    )
    train_ds: ConcatDataset[tuple[Tensor, str]] = ConcatDataset(
        [load_language(s, "train", cfg.train.max_audio_samples) for s in specs]
    )
    val_ds: ConcatDataset[tuple[Tensor, str]] = ConcatDataset(
        [load_language(s, "validation", cfg.train.max_audio_samples) for s in specs]
    )
    model = make_model(cfg, vocab.size)
    return train(model, cfg, train_ds, val_ds, collate, cfg.train.phase0_epochs, out_dir, device)


def train_experts(
    cfg: ExperimentConfig,
    phase0_ckpt: Path,
    specs: list[LangSpec],
    vocab: CtcVocab,
    out_dir: Path,
    device: str = "cuda",
) -> dict[str, TrainResult]:
    """Fine-tune one expert per language, each branched from phase 0."""
    collate = make_ctc_collate(
        vocab, max_audio_samples=cfg.train.max_audio_samples, drop_overlong=True, drop_empty=True
    )
    experts: dict[str, TrainResult] = {}
    for spec in specs:
        model = make_model(cfg, vocab.size)
        model.load(phase0_ckpt)
        lang_dir = out_dir / f"expert_{spec.code}"
        train_ds = load_language(spec, "train", cfg.train.max_audio_samples)
        val_ds = load_language(spec, "validation", cfg.train.max_audio_samples)
        experts[spec.code] = train(
            model, cfg, train_ds, val_ds, collate, cfg.train.expert_epochs, lang_dir, device
        )
    return experts


def merge_experts(
    expert_ckpts: dict[str, Path],
    strategy: str,
    out_dir: Path,
    base_ckpt: Path | None = None,
    device: str = "cpu",
    seed: int = 0,
    merge_head: bool = True,
) -> Path:
    """Merge expert state_dicts; task-vector methods need ``base_ckpt`` (phase 0).

    Args:
        seed: The run seed. DARE-TIES draws its drop mask from it, so leaving it
            at the default would give every seed of a multi-seed study the same
            mask and understate the DARE arm's variance.
        merge_head: Whether the CTC head is merged along with the encoder. See
            the merge module docstring — this is a protocol choice.
    """
    import torch

    if device != "cpu":
        raise ValueError(
            f"merging runs on CPU (got device={device!r}): the checkpoints are memory-mapped "
            "and DARE's drop mask comes from a CPU generator, so a GPU merge would need both "
            "the whole expert set resident and a different RNG stream"
        )
    merge_fn = MERGE_STRATEGIES[strategy]
    # mmap=True leaves each expert's tensors on disk until a key is touched.
    # With the strategies stacking one key at a time, resident memory is one
    # merged model plus one key across the experts, rather than the whole expert
    # set at once — the difference between merging 64 experts and not.
    experts = [torch.load(p, map_location="cpu", mmap=True) for p in expert_ckpts.values()]
    base = torch.load(base_ckpt, map_location="cpu", mmap=True) if base_ckpt is not None else None
    merged = merge_fn(experts, base=base, seed=seed, merge_head=merge_head)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"merged_{strategy}.pt"
    torch.save(merged, path)
    return path
