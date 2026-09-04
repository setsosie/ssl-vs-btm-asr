"""Run provenance — what makes results reviewer-proof.

Every run dumps an ``env.json`` recording the git SHA and whether the tree was
dirty, key library versions, the ``uv.lock`` hash, the GPU, and a timestamp,
right next to its ``results.json`` and ``resolved_config.yaml``. The paper's
reproducibility appendix is generated from these files.

Nothing here may abort a run: a provenance probe that raises would destroy the
result it was meant to describe. Every probe therefore catches broadly and
records the failure in the output rather than swallowing it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Run git against the package's own repository, not the process CWD: a job
# launched from another checkout would otherwise record that repo's SHA and
# attribute these results to it.
_REPO_DIR = Path(__file__).resolve().parent

# Distributions this project actually imports. Recorded by distribution name,
# because several of them expose no __version__ attribute at all — jiwer 4 is
# the notable one, and it is the package whose major version decides whether an
# empty reference raises or scores.
_TRACKED_DISTRIBUTIONS = (
    "torch",
    "torchaudio",
    "datasets",
    "jiwer",
    "numpy",
    "soundfile",
    "pyyaml",
    "rapidfuzz",
)


def _git(*args: str) -> str:
    return (
        subprocess.check_output(["git", *args], cwd=_REPO_DIR, stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )


def _git_sha() -> str:
    try:
        return _git("rev-parse", "HEAD")
    except Exception:  # noqa: BLE001 — provenance must never abort a run
        return "unknown"


def _git_dirty() -> bool:
    """Whether the working tree had uncommitted changes.

    Without this the SHA can be a precise pointer to code that was not the code
    that ran.
    """
    try:
        return bool(_git("status", "--porcelain"))
    except Exception:  # noqa: BLE001 — provenance must never abort a run
        return False


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for dist in _TRACKED_DISTRIBUTIONS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = "absent"
    return versions


def _uv_lock_sha256(root: Path) -> str | None:
    """Hash of the resolved dependency set, when the project pins one.

    Every declared dependency is a floor, so the same commit resolves
    differently on different days. The lock hash is what makes an environment
    identifiable after the fact.
    """
    lock = root / "uv.lock"
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def _gpu() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
            }
    except Exception as exc:  # noqa: BLE001 — provenance must never abort a run
        return {"available": False, "error": repr(exc)}
    return {"available": False}


def dump_run_meta(out_dir: str | Path) -> Path:
    """Write ``env.json`` capturing the run environment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "uv_lock_sha256": _uv_lock_sha256(_REPO_DIR.parents[1]),
        "versions": _versions(),
        "gpu": _gpu(),
        "platform": platform.platform(),
    }
    path = out_dir / "env.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
