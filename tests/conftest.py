"""Shared test fixtures and CPU test doubles.

Two families live here:

* **Synthetic data fixtures** for the data package. Everything is generated
  locally: tiny silent WAVs plus the committed ``tests/fixtures/openslr/*.tsv``
  line indices, which reproduce the real OpenSLR layout (flat directory,
  ``<FileID>.wav``, two-column tab-separated index). No test touches the
  network or a real corpus.
* **A model double.** The real model is a 577M-parameter encoder, so every
  unit test runs against a stand-in that keeps the public ``XeusCTC`` surface
  (``forward`` returning ``logits``/``input_lengths``/``loss``,
  ``greedy_decode``, ``save``, ``load``) and nothing else. Subclassing the real
  class rather than duck-typing it means the tests exercise the inherited
  methods under test, not a copy.
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from svb.model.ctc_vocab import CtcVocab
from svb.model.xeus_ctc import XeusCTC

TARGET_SR = 16000


# --------------------------------------------------------------------------- #
# Synthetic data fixtures
# --------------------------------------------------------------------------- #


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


class CvClip(NamedTuple):
    path: str
    client_id: str
    sentence: str
    split: str | None  # None: validated, but in no official split


#: One synthetic Common Voice language, shaped like a real release.
#:
#: ``validated.tsv`` holds every validated clip; ``train``/``dev``/``test`` hold
#: the speaker-disjoint partition Corpora Creator carved out of the
#: *deduplicated* validated frame. The interesting rows are the ones in
#: ``validated`` and in no split: ``v3`` belongs to a training speaker and is
#: fair game, while ``v5`` and ``v7`` are further recordings by the dev and test
#: speakers and are exactly what the speaker guard exists to remove.
CV_ROWS = [
    CvClip("v1.mp3", "s_train_a", "one", "train"),
    CvClip("v2.mp3", "s_train_b", "two", "train"),
    CvClip("v3.mp3", "s_train_a", "three", None),
    CvClip("v4.mp3", "s_dev", "four", "dev"),
    CvClip("v5.mp3", "s_dev", "five", None),
    CvClip("v6.mp3", "s_test", "six", "test"),
    CvClip("v7.mp3", "s_test", "seven", None),
]


def _tsv(rows: list[CvClip], columns: tuple[str, ...]) -> str:
    head = "\t".join(columns) + "\n"
    return head + "".join("\t".join(getattr(row, c) for c in columns) + "\n" for row in rows)


@pytest.fixture
def make_cv_lang(tmp_path: Path) -> Callable[..., Path]:
    """Build ``$CV_ROOT`` holding one language; returns the root.

    ``columns`` lets a test drop ``client_id`` to check what happens when the
    speaker guard has nothing to work with.
    """

    def build(
        lang: str = "en",
        rows: list[CvClip] | None = None,
        *,
        columns: tuple[str, ...] = ("client_id", "path", "sentence"),
        with_validated: bool = True,
        root: Path | None = None,
    ) -> Path:
        rows = CV_ROWS if rows is None else rows
        root = root or (tmp_path / "cv")
        base = root / lang
        (base / "clips").mkdir(parents=True, exist_ok=True)
        for split, name in (("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")):
            member = [r for r in rows if r.split == split]
            (base / name).write_text(_tsv(member, columns), encoding="utf-8")
        if with_validated:
            (base / "validated.tsv").write_text(_tsv(rows, columns), encoding="utf-8")
        return root

    return build


class ManifestRow(NamedTuple):
    utt_id: str
    path: str
    text: str
    speaker: str
    split: str  # "" when the corpus ships none and one is derived


MANIFEST_COLUMNS = ("utt_id", "path", "text", "speaker", "split")


@pytest.fixture
def make_manifest_corpus(tmp_path: Path) -> Callable[..., Path]:
    """Build ``$CORPORA_ROOT/<corpus>/<lang>/`` with a manifest and its audio.

    Returns the root. Audio is silent WAV named exactly as the manifest's
    ``path`` column says, so the loader is exercised on the same join a real
    corpus would make.
    """

    def build(
        rows: list[ManifestRow],
        corpus: str = "demo_corpus",
        lang: str = "xx",
        *,
        columns: tuple[str, ...] = MANIFEST_COLUMNS,
        write_audio: bool = True,
        frames: int = 160,
        root: Path | None = None,
    ) -> Path:
        root = root or (tmp_path / "corpora")
        base = root / corpus / lang
        (base / "audio").mkdir(parents=True, exist_ok=True)
        body = "\t".join(columns) + "\n"
        for row in rows:
            body += "\t".join(getattr(row, name) for name in columns) + "\n"
            if write_audio:
                _write_silent_wav(base / row.path, frames=frames)
        (base / "manifest.tsv").write_text(body, encoding="utf-8")
        return root

    return build


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


def _load_script(root: Path, name: str) -> ModuleType:
    """Load a file under `scripts/` as a module.

    `scripts/` goes on the path because scripts import their siblings by bare
    name — which works when one is run directly, since Python puts its own
    directory first, and does not when it is loaded by path. Every preparer
    imports `corpus_fetch`, so the alternative is an importlib shim in each.

    The module is registered in ``sys.modules`` before it executes, which the
    importlib recipe leaves out and `@dataclass` needs: it resolves a class's
    module out of ``sys.modules`` to evaluate the annotations, and fails on the
    ``None`` an unregistered module leaves there.
    """
    scripts = root / "scripts"
    # Scripts import their siblings by bare name, which works when one is run
    # directly because Python puts its directory on the path. Loading by path
    # skips that, so it is done here — `corpus_fetch` is imported by every
    # preparer.
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing. A module that defines a dataclass under
    # `from __future__ import annotations` is resolved by looking itself up in
    # sys.modules, so omitting this fails at class-creation time with an
    # AttributeError on None rather than anything that names the cause.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fetch_openslr(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "fetch_openslr")


@pytest.fixture
def run_matrix(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "run_matrix")


@pytest.fixture
def crosscheck_espnet(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "crosscheck_espnet")


@pytest.fixture
def corpus_fetch(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "corpus_fetch")


@pytest.fixture
def prepare_google_crowdsourced(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "prepare_google_crowdsourced")


@pytest.fixture
def select_languages(pytestconfig: pytest.Config) -> ModuleType:
    return _load_script(Path(pytestconfig.rootpath), "select_languages")


# --------------------------------------------------------------------------- #
# Model double
# --------------------------------------------------------------------------- #


class FakeXeusCTC(XeusCTC):
    """Encoder-free double: one output frame per input sample.

    The waveform *is* the transcript — sample ``i`` of the waveform holds the
    vocab id the encoder should emit at frame ``i``. Frames beyond
    ``attention_mask`` emit ``junk_id`` instead, which is what a real encoder
    does over the zero-padded tail: it predicts *something*, and that something
    is only excluded from the hypothesis if the decoder honours the lengths.
    """

    def __init__(self, vocab_size: int, junk_id: int = 2) -> None:
        nn.Module.__init__(self)
        self.hidden_size = 1
        self.ctc_norm = nn.LayerNorm(1)
        self.ctc_proj = nn.Linear(1, vocab_size)
        self.junk_id = junk_id

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, frames = input_values.shape
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long()
        else:
            lengths = torch.full((bsz,), frames, dtype=torch.long)

        ids = input_values.round().long().clamp(min=0, max=self.vocab_size - 1)
        valid = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
        ids = torch.where(valid, ids, torch.full_like(ids, self.junk_id))
        logits = F.one_hot(ids, self.vocab_size).float() * 10.0
        return {"logits": logits, "input_lengths": lengths}


class ScriptedLossCTC(XeusCTC):
    """Double whose validation loss follows a script, so control flow is testable.

    Training batches return a constant loss that still carries a gradient, so
    the optimizer step is exercised. Validation batches return the next entry of
    ``val_losses`` — one entry per ``_validate`` forward pass, not per epoch, so
    a test can also probe how batches are combined into the epoch's number.
    """

    def __init__(self, val_losses: list[float]) -> None:
        nn.Module.__init__(self)
        self.hidden_size = 1
        self.ctc_norm = nn.LayerNorm(1)
        self.ctc_proj = nn.Linear(1, 2)
        self.val_losses = list(val_losses)
        self.train_forwards = 0
        self.val_forwards = 0

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        anchor = self.ctc_proj.bias.sum() * 0.0  # keeps the graph alive
        if self.training:
            self.train_forwards += 1
            return {"loss": anchor + 1.0}
        index = min(self.val_forwards, len(self.val_losses) - 1)
        self.val_forwards += 1
        return {"loss": anchor.detach() + self.val_losses[index]}


def wav_for(vocab: CtcVocab, text: str) -> torch.Tensor:
    """Waveform that makes :class:`FakeXeusCTC` emit ``text`` verbatim."""
    return torch.tensor([float(i) for i in vocab.encode(text)], dtype=torch.float32)


@pytest.fixture
def fake_model_cls() -> type[FakeXeusCTC]:
    return FakeXeusCTC
