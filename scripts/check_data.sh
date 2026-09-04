#!/bin/bash
# Report split sizes for every configured language, from metadata only.
#
# Nothing here decodes audio: Common Voice sizes are tsv row counts and OpenSLR
# sizes are line-index row counts run through the same split derivation the
# loader uses. The old version called load_language() for all three splits of
# every language just to print len(ds), which for the held-out set meant pulling
# several gigabytes of audio three times over to count rows.
#
#  - Common Voice 25: local-only. Download + extract from Mozilla Data Collective
#    (https://commonvoice.mozilla.org/en/datasets, account + ToS required) and set
#    $CV_ROOT so $CV_ROOT/<lang>/{train,dev,test}.tsv exist.
#  - OpenSLR (held-out Indic): run `python scripts/fetch_openslr.py --root
#    $OPENSLR_ROOT` once, then set $OPENSLR_ROOT.
#
# See docs/data.md.
set -euo pipefail

uv run python - <<'PY'
import json
import os
from pathlib import Path

from svb.data.datasets import load_texts
from svb.data.registry import get_heldout, get_preset

SPLITS = ("train", "validation", "test")


def openslr_detail(spec):
    """Split policy and manifest facts — index reads only, no audio."""
    from svb.data.openslr_local import OpenSLRLocal

    ds = OpenSLRLocal(spec, "test")
    line = f"      policy={ds.split_policy}  test_files_sha1={ds.test_files_sha1[:12]}"

    root = os.environ.get("OPENSLR_ROOT")
    manifest = Path(root) / f"SLR{spec.slr}" / "manifest.json" if root else None
    if manifest is not None and manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        line += (
            f"\n      {data.get('license', '?')}  wavs={data.get('wav_files', '?')}"
            f"  fetched={data.get('downloaded_utc', '?')}"
        )
    return line


def preset(scale):
    """Languages of one scale, or none when that scale is still a placeholder.

    An unpopulated preset raises, which is right for a run and wrong for a
    report: this command exists to say what data is on disk, so it names the
    gap and keeps sweeping.
    """
    try:
        return get_preset(scale)
    except (ValueError, FileNotFoundError) as exc:
        print(f"--  scale {scale:3s} {exc}")
        return []


specs = [s for scale in ("3", "16", "64") for s in preset(scale)] + get_heldout()

seen = set()
for spec in specs:
    if (spec.source, spec.code) in seen:
        continue
    seen.add((spec.source, spec.code))

    try:
        sizes = {split: len(load_texts(spec, split)) for split in SPLITS}
    # Broad on purpose: one unreachable corpus must not end the sweep, since
    # the point of this command is to report which ones are reachable.
    except Exception as exc:
        print(f"ERR {spec.code:12s} {spec.source:11s} {type(exc).__name__}: {exc}")
        continue

    counts = "  ".join(f"{s}={sizes[s]}" for s in SPLITS)
    print(f"OK  {spec.code:12s} {spec.source:11s} total={sum(sizes.values()):7d}  {counts}")

    if spec.source == "openslr":
        print(openslr_detail(spec))
PY
