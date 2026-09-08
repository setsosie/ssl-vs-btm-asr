"""Language presets and dataset specs (public sources only).

A *preset* (scale "3"/"16"/"64") is a list of ``LangSpec``s read from
``configs/scales/<scale>.yaml``. Held-out transfer languages live in
``configs/scales/heldout.yaml``. Keeping these in YAML (not hard-coded) means
the exact language list is auditable and editable without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Repo root: src/svb/data/registry.py -> parents[3]
_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@dataclass(frozen=True)
class LangSpec:
    code: str  # short label used in results/paths, e.g. "hi", "telugu"
    source: str  # "commonvoice" | "openslr"
    hf_dataset: str  # HF datasets id, e.g. "mozilla-foundation/common_voice_17_0"
    hf_config: str  # config/language name, e.g. "hi"
    text_column: str = "sentence"
    audio_column: str = "audio"


def _load_specs(path: Path) -> list[LangSpec]:
    if not path.exists():
        raise FileNotFoundError(f"missing preset file: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return [LangSpec(**entry) for entry in raw.get("languages", [])]


def get_preset(scale: str, configs_dir: Path | None = None) -> list[LangSpec]:
    base = configs_dir or _CONFIGS
    return _load_specs(base / "scales" / f"{scale}.yaml")


def get_heldout(configs_dir: Path | None = None) -> list[LangSpec]:
    base = configs_dir or _CONFIGS
    return _load_specs(base / "scales" / "heldout.yaml")
