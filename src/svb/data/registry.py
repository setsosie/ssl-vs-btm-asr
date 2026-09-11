"""Language presets and dataset specs (public sources only).

A *preset* (scale "3"/"16"/"64") is a list of ``LangSpec``s read from
``configs/scales/<scale>.yaml``. Held-out transfer languages live in
``configs/scales/heldout.yaml``. Keeping these in YAML (not hard-coded) means
the exact language list is auditable and editable without touching code.

Two sources, both read from a local directory — nothing here downloads at load
time and nothing goes through the Hugging Face Hub:

``commonvoice``
    Common Voice 25 under ``$CV_ROOT/<hf_config>/``. Since October 2025 Common
    Voice is distributed only via Mozilla Data Collective, so ``hf_dataset`` is
    **not** a Hub id — it is a provenance label for the release ("common_voice_25")
    that gets recorded in run metadata. ``hf_config`` is the CV language code,
    which is also the directory name.

``openslr``
    An openslr.org corpus fetched by ``scripts/fetch_openslr.py`` and extracted
    to ``$OPENSLR_ROOT/SLR<slr>/``. Described by ``slr``, the ``archives`` to
    download, the ``index_files`` those archives are extracted with, and the
    ``license`` the corpus is published under.

The ``hf_dataset``/``hf_config`` names are historical. Renaming them to
``corpus_version``/``corpus_language`` would touch every entry of
``configs/scales/{3,16,64}.yaml``, which other work owns, so they keep their
names and this docstring carries the meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ..text.normalize import POLICIES

# Repo root: src/svb/data/registry.py -> parents[3]
_CONFIGS = Path(__file__).resolve().parents[3] / "configs"

SOURCES = ("commonvoice", "openslr")


@dataclass(frozen=True)
class LangSpec:
    """One language of one corpus. Every field after ``source`` has a default so
    that a spec only carries the fields its own source actually uses."""

    code: str  # short label used in results/paths, e.g. "hi", "telugu"
    source: str  # one of SOURCES

    # Whether the writing system separates words with spaces. False makes CER
    # the primary metric, because whitespace tokenization of a Japanese
    # transcript yields one token per sentence and WER over it is meaningless.
    # Declared beside the language rather than as a set of codes inside the
    # evaluation module, where no preset author would ever look; the run checks
    # the declaration against the transcripts and warns when they disagree.
    word_boundary: bool = True

    # The normalization policy this language's transcripts pass through, by
    # name. Every shipped preset states it, so the choice sits beside the
    # language rather than being inferred three modules away; left unset, it
    # falls back to the script default in svb.text.registry.
    normalizer: str | None = None

    # commonvoice
    hf_dataset: str = ""  # provenance label for the release, e.g. "common_voice_25"
    hf_config: str = ""  # CV language code == directory name under $CV_ROOT
    text_column: str = "sentence"  # transcript column of the CV split tsv

    # openslr
    slr: int | None = None  # openslr.org resource number, e.g. 63
    archives: tuple[str, ...] = ()  # archive file names to download
    index_files: tuple[str, ...] = ()  # extracted index name per archive (parallel to `archives`)
    license: str = ""  # licence the corpus is published under, e.g. "CC-BY-SA-4.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "archives", tuple(self.archives))
        object.__setattr__(self, "index_files", tuple(self.index_files))

        if self.normalizer is not None and self.normalizer not in POLICIES:
            known = ", ".join(sorted(POLICIES))
            raise ValueError(
                f"{self.code}: unknown normalizer {self.normalizer!r}; known policies: {known}"
            )

        if self.source not in SOURCES:
            raise ValueError(
                f"{self.code}: unknown source {self.source!r}; expected one of {SOURCES}"
            )

        if self.source == "commonvoice":
            if not self.hf_config:
                raise ValueError(f"{self.code}: commonvoice needs hf_config (the CV language code)")
        else:
            if self.slr is None:
                raise ValueError(
                    f"{self.code}: openslr needs slr (the openslr.org resource number)"
                )
            if not self.archives:
                raise ValueError(f"{self.code}: openslr needs at least one archive")
            if len(self.index_files) != len(self.archives):
                raise ValueError(
                    f"{self.code}: index_files must be parallel to archives "
                    f"({len(self.index_files)} vs {len(self.archives)}) — each archive ships its "
                    "own line_index.tsv and is extracted under the matching declared name"
                )


def _spec_from_entry(entry: dict[str, Any], path: Path) -> LangSpec:
    known = {f.name for f in fields(LangSpec)}
    unknown = sorted(set(entry) - known)
    if unknown:
        raise ValueError(
            f"{path.name}: language {entry.get('code', '?')!r} has unknown keys {unknown}"
        )
    try:
        return LangSpec(**entry)
    except TypeError as exc:  # missing required key
        raise ValueError(f"{path.name}: language {entry.get('code', '?')!r}: {exc}") from exc


def _load_specs(path: Path) -> list[LangSpec]:
    if not path.exists():
        raise FileNotFoundError(f"missing preset file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs = [_spec_from_entry(entry, path) for entry in raw.get("languages", [])]
    if not specs:
        # An empty list is a placeholder, not a preset. Left to run, it trains
        # on no languages, evaluates nothing, and still writes a results.json
        # that a reader cannot tell from a completed run.
        raise ValueError(
            f"preset {path.stem} is not populated: {path} lists no languages, "
            "so a run using it would train and evaluate on nothing"
        )
    return specs


def get_preset(scale: str, configs_dir: Path | None = None) -> list[LangSpec]:
    base = configs_dir or _CONFIGS
    return _load_specs(base / "scales" / f"{scale}.yaml")


def get_heldout(configs_dir: Path | None = None) -> list[LangSpec]:
    base = configs_dir or _CONFIGS
    return _load_specs(base / "scales" / "heldout.yaml")
