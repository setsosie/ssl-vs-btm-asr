"""Synthetic fixtures for the data package.

Everything here is generated locally: tiny silent WAVs plus the committed
`tests/fixtures/openslr/*.tsv` line indices, which reproduce the real OpenSLR
layout (flat directory, `<FileID>.wav`, two-column tab-separated index). No test
touches the network or a real corpus.
"""

from __future__ import annotations

import importlib.util
import wave
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

TARGET_SR = 16000


def _write_silent_wav(path: Path, frames: int, sr: int = TARGET_SR, channels: int = 1) -> None:
    """Write a 16-bit PCM WAV of `frames` silent samples."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(2 * channels * frames))


@pytest.fixture
def write_silent_wav() -> Callable[..., None]:
    return _write_silent_wav


@pytest.fixture
def fixtures_dir(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.rootpath) / "tests" / "fixtures"


@pytest.fixture
def slr_root(tmp_path: Path, fixtures_dir: Path) -> Path:
    """An extracted two-archive OpenSLR language dir under a fake $OPENSLR_ROOT."""
    root = tmp_path / "openslr"
    dest = root / "SLR63"
    dest.mkdir(parents=True)
    for name in ("line_index_female.tsv", "line_index_male.tsv"):
        text = (fixtures_dir / "openslr" / name).read_text(encoding="utf-8")
        (dest / name).write_text(text, encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                _write_silent_wav(dest / f"{line.split(chr(9))[0]}.wav", frames=160)
    return root


@pytest.fixture
def fetch_openslr(pytestconfig: pytest.Config) -> ModuleType:
    """Load `scripts/fetch_openslr.py` as a module without touching sys.path."""
    path = Path(pytestconfig.rootpath) / "scripts" / "fetch_openslr.py"
    spec = importlib.util.spec_from_file_location("fetch_openslr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
