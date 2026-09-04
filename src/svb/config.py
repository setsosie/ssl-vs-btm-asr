"""Typed, frozen configuration for a single run.

A run is fully determined by ``(arm, scale, seed)`` plus the shared defaults
here. Optimizer defaults are the best trial of the author's earlier
hyperparameter search on this encoder, used verbatim throughout; they are
reported in every run's dumped config.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .text.normalize import get_policy, module_sha256, policy_name
from .text.registry import policies_for_specs

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
    # Best trial of the author's earlier hyperparameter search, verbatim.
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
    # Which Common Voice rows the training split is. "train" is the official
    # train.tsv; "validated_minus_eval" is every validated clip that is not in
    # dev or test and does not belong to a dev or test speaker, which is the
    # standard recipe and roughly three times the audio. It changes the training
    # set of every Common Voice language, so it is dumped with the config.
    cv_train_source: Literal["train", "validated_minus_eval"] = "validated_minus_eval"


@dataclass(frozen=True)
class TextConfig:
    """How transcripts are normalized, and how thin the vocabulary's tail may be.

    Changing anything here changes every reported number, so it is dumped in
    full — along with the derived digests — beside each run's results.
    """

    # A policy name that replaces every language's own. Left unset, each
    # language uses the policy its preset names or its script defaults to, which
    # is the point of having a policy per script. Setting it is the switch for
    # scoring a whole run the way one published system scores — useful for
    # comparing against that system's numbers, at the cost of whatever its rules
    # do to the scripts it was not designed for.
    override: str | None = None
    # Corpus-wide occurrences a character needs to earn a vocabulary slot. The
    # default keeps everything, which is right for a smoke run where a real
    # character may also be rare; configs/base.yaml raises it for a full run,
    # where the tail is stray characters from corpora that had no
    # collection-time validation.
    min_char_count: int = 1


# Written into the dumped config for the reader, derived rather than set.
_DERIVED_TEXT_KEYS = ("policies", "normalizer_module_sha256")


@dataclass(frozen=True)
class ExperimentConfig:
    arm: Arm
    scale: Scale
    seed: int
    merge_strategy: MergeStrategy = "average"
    # Whether the CTC head is merged with the encoder, or taken from the phase-0
    # base. A protocol choice rather than a tuning knob: the experts share one
    # vocabulary so their heads are commensurable, but merging them is also the
    # part of the merge most likely to carry the penalty, and encoder-only
    # merging is the ablation that separates the two. Dumped with the config
    # because it changes every merged number.
    merge_head: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    text: TextConfig = field(default_factory=TextConfig)
    # The held-out transfer languages are NOT listed here. They live in
    # configs/scales/heldout.yaml and are read by `registry.get_heldout()`,
    # which is the single source of truth; each run records the codes it
    # actually evaluated in its results.json. A second copy on this dataclass
    # was never read, but `to_dict` still wrote it into resolved_config.yaml,
    # so it kept claiming five languages after the YAML was cut to four.

    @property
    def init(self) -> Literal["ssl", "scratch"]:
        return "scratch" if self.arm == "C_btm_scratch" else "ssl"

    @property
    def uses_btm(self) -> bool:
        return self.arm in ("B_btm_ssl", "C_btm_scratch")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        # asdict would emit every policy field, including the ones belonging to
        # the other pipeline, which reads as a list of rules that ran. The
        # policy's own record carries only what it actually runs.
        # The normalizer's own digest, which the policy hashes cannot see: it
        # catches the module being edited without any setting changing. The
        # per-language policies are added by `dump_config`, which is where the
        # language list is known.
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
    return ExperimentConfig(model=model, optim=optim, train=train, text=text, **cfg)


def _text_config(raw: dict[str, Any]) -> TextConfig:
    """Build a TextConfig, tolerating the digests ``to_dict`` writes.

    Dropping them here is what lets a run's own ``resolved_config.yaml`` be fed
    straight back in to reproduce it.
    """
    raw = dict(raw)
    for key in _DERIVED_TEXT_KEYS:
        raw.pop(key, None)
    if raw.get("override") is not None:
        # Fail here rather than at the first language: an unknown name in a
        # config is a typo, and finding out mid-run wastes the run.
        get_policy(raw["override"])
    return TextConfig(**raw)


def dump_config(cfg: ExperimentConfig, out_dir: str | Path, specs: Iterable[Any] = ()) -> Path:
    """Write the fully-resolved config next to a run's results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict()
    # Which policy each language actually ran under, by name and by hash. The
    # name is a label for the reader; the hash is what says whether two runs may
    # be pooled, and a run whose languages differ in policy cannot be compared
    # language-for-language with one whose do not.
    data["text"]["policies"] = {
        code: {"name": policy_name(policy), "hash": policy.policy_hash()}
        for code, policy in sorted(policies_for_specs(specs, cfg.text.override).items())
    }
    path = out_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path
