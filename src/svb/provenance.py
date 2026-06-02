"""Run provenance — what makes results reviewer-proof.

Every run dumps an ``env.json`` recording the git SHA, key library versions, the
GPU, and a timestamp, right next to its ``results.json`` and
``resolved_config.yaml``. The paper's reproducibility appendix is generated from
these files.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod in ("torch", "torchaudio", "datasets", "jiwer", "numpy", "transformers"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "absent"
    return versions


def _gpu() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
            }
    except Exception:
        pass
    return {"available": False}


def dump_run_meta(out_dir: str | Path) -> Path:
    """Write ``env.json`` capturing the run environment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "versions": _versions(),
        "gpu": _gpu(),
        "platform": platform.platform(),
    }
    path = out_dir / "env.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
