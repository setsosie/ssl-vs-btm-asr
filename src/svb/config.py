"""Typed, frozen configuration for a single run.

A run is fully determined by ``(arm, scale, seed)`` plus the shared defaults
here. Optimizer defaults are the XEUS-HPO-01 Trial 19 values used throughout the
paper; they are reported verbatim in every run's dumped config.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .text.normalize import NormalizerPolicy, module_sha256

Arm = Literal["A_ssl", "B_btm_ssl", "C_btm_scratch"]
Scale = Literal["3", "16", "64"]
MergeStrategy = Literal["average", "ties", "dare_ties"]


@dataclass(frozen=True)
class ModelConfig:
    name: str = "xeus"
    hidden_size: int = 1024
    # Path to the XEUS SSL checkpoint (espnet/xeus). Required for SSL-init arms
    # (A, B); ignored for scratch init (C). May be set via $XEUS_CHECKPOINT.
    xeus_checkpoint: str | None = None
    blank_bias_init: float | None = None
    # Encoder dropout. A training hyperparameter that moves every reported
    # number, so it belongs in the dumped config rather than in four
    # constructor defaults.
    dropout: float = 0.1


@dataclass(frozen=True)
class OptimConfig:
    # XEUS-HPO-01 Trial 19 (verbatim).
    lr: float = 7.446693063298197e-05
    weight_decay: float = 0.0024151339921234323
    grad_clip: float = 0.507731238072093
    warmup_ratio: float = 0.08544536055130553
    batch_size: int = 8
    accum_steps: int = 2


@dataclass(frozen=True)
class TrainConfig:
    # Epoch ceilings; scratch arms typically need the larger budget to converge.
    phase0_epochs: int = 15
    expert_epochs: int = 9
    finetune_epochs: int = 50
    patience: int = 15
    bf16: bool = True
    grad_checkpointing: bool = True
    max_audio_samples: int = 200_000  # ~12.5s at 16kHz; truncation guard
    num_workers: int = 8


@dataclass(frozen=True)
class TextConfig:
    """How transcripts are normalized, and how thin the vocabulary's tail may be.

    Changing anything here changes every reported number, so it is dumped in
    full — along with the derived digests — beside each run's results.
    """

    policy: NormalizerPolicy = field(default_factory=NormalizerPolicy)
    # Corpus-wide occurrences a character needs to earn a vocabulary slot. The
    # default keeps everything, which is right for a smoke run where a real
    # character may also be rare; configs/base.yaml raises it for a full run,
    # where the tail is stray characters from corpora that had no
    # collection-time validation.
    min_char_count: int = 1


# Written into the dumped config for the reader, derived rather than set.
_DERIVED_TEXT_KEYS = ("policy_hash", "normalizer_module_sha256")


@dataclass(frozen=True)
class ExperimentConfig:
    arm: Arm
    scale: Scale
    seed: int
    merge_strategy: MergeStrategy = "average"
    sample_rate: int = 16000
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    text: TextConfig = field(default_factory=TextConfig)
    # Held-out transfer languages (OpenSLR Indic), never in any training mix.
    heldout_langs: tuple[str, ...] = (
        "odia",
        "marathi",
        "telugu",
        "gujarati",
        "malayalam",
    )

    @property
    def init(self) -> Literal["ssl", "scratch"]:
        return "scratch" if self.arm == "C_btm_scratch" else "ssl"

    @property
    def uses_btm(self) -> bool:
        return self.arm in ("B_btm_ssl", "C_btm_scratch")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        # Derived, so that a reader can tell two runs apart without recomputing
        # anything: the policy hash identifies the settings, and the module hash
        # catches a normalizer edited without any setting changing.
        data["text"]["policy_hash"] = self.text.policy.policy_hash()
        data["text"]["normalizer_module_sha256"] = module_sha256()
        return data


def _nested_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _nested_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    arm: Arm,
    scale: Scale,
    seed: int,
    yaml_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Build an ExperimentConfig, layering an optional YAML then dict overrides."""
    cfg: dict[str, Any] = {"arm": arm, "scale": scale, "seed": seed}
    if yaml_path is not None:
        with open(yaml_path) as f:
            cfg = _nested_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _nested_update(cfg, overrides)

    model_cfg = cfg.pop("model", {})
    # Fall back to $XEUS_CHECKPOINT when the YAML leaves it unset (launcher path).
    if not model_cfg.get("xeus_checkpoint"):
        env_ckpt = os.environ.get("XEUS_CHECKPOINT")
        if env_ckpt:
            model_cfg["xeus_checkpoint"] = env_ckpt
    model = ModelConfig(**model_cfg)
    optim = OptimConfig(**cfg.pop("optim", {}))
    train = TrainConfig(**cfg.pop("train", {}))
    text = _text_config(cfg.pop("text", {}))
    if "heldout_langs" in cfg:
        cfg["heldout_langs"] = tuple(cfg["heldout_langs"])
    return ExperimentConfig(model=model, optim=optim, train=train, text=text, **cfg)


def _text_config(raw: dict[str, Any]) -> TextConfig:
    """Build a TextConfig, tolerating the digests ``to_dict`` writes.

    Dropping them here is what lets a run's own ``resolved_config.yaml`` be fed
    straight back in to reproduce it.
    """
    raw = dict(raw)
    for key in _DERIVED_TEXT_KEYS:
        raw.pop(key, None)
    policy = NormalizerPolicy(**raw.pop("policy", {}))
    return TextConfig(policy=policy, **raw)


def dump_config(cfg: ExperimentConfig, out_dir: str | Path) -> Path:
    """Write the fully-resolved config next to a run's results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return path
