#!/bin/bash
# Pre-warm the HF cache for every configured language so training runs don't
# block on first download. Splits are resolved by the loader (single-split
# corpora are partitioned deterministically — see src/svb/data/datasets.py).
#
# Prerequisites:
#   - `huggingface-cli login` (Common Voice 17 is gated; accept terms once at
#     https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
set -euo pipefail

uv run python - <<'PY'
import yaml
from pathlib import Path
from svb.data.datasets import _resolve_split
from svb.data.registry import LangSpec

specs = []
for f in ["configs/scales/3.yaml", "configs/scales/16.yaml",
          "configs/scales/64.yaml", "configs/scales/heldout.yaml"]:
    raw = yaml.safe_load(Path(f).read_text()) or {}
    specs += [LangSpec(**e) for e in raw.get("languages", [])]

for spec in specs:
    for split in ("train", "validation", "test"):
        try:
            ds = _resolve_split(spec, split)
            print(f"OK  {spec.code:12s} {split:11s} n={len(ds)}")
        except Exception as e:  # noqa: BLE001
            print(f"ERR {spec.code:12s} {split:11s} {type(e).__name__}: {e}")
PY
