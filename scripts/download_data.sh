#!/bin/bash
# Verify data is reachable for every configured language and report split sizes.
#  - Common Voice 25: local-only. Download + extract from Mozilla Data Collective
#    (https://commonvoice.mozilla.org/en/datasets, account + ToS required) and set
#    $CV_ROOT so $CV_ROOT/<lang>/{train,dev,test}.tsv exist.
#  - OpenSLR (held-out Indic): fetched from the HF Hub (open); needs mp3/ffmpeg.
set -euo pipefail

uv run python - <<'PY'
import yaml
from pathlib import Path
from svb.data.datasets import load_language
from svb.data.registry import LangSpec

specs = []
for f in ["configs/scales/3.yaml", "configs/scales/16.yaml",
          "configs/scales/64.yaml", "configs/scales/heldout.yaml"]:
    raw = yaml.safe_load(Path(f).read_text()) or {}
    specs += [LangSpec(**e) for e in raw.get("languages", [])]

for spec in specs:
    for split in ("train", "validation", "test"):
        try:
            ds = load_language(spec, split)
            print(f"OK  {spec.code:12s} {spec.source:11s} {split:11s} n={len(ds)}")
        except Exception as e:  # noqa: BLE001
            print(f"ERR {spec.code:12s} {spec.source:11s} {split:11s} {type(e).__name__}: {e}")
PY
