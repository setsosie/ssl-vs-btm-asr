"""How much audio each language actually contributes, per split.

A low-resource claim is a claim about hours, and until now the repository
recorded only utterance counts. An utterance count is not a proxy for duration:
the crowdsourced OpenSLR sets are single sentences of a few seconds while Common
Voice sentence lengths vary by a factor of ten across languages.

Nothing here decodes audio. Common Voice has shipped ``clip_durations.tsv``
since v16.1, which is the exact per-clip duration the release itself declares,
so the whole corpus is accounted for by joining two text files. OpenSLR ships no
such manifest, so the WAV *header* is read — a fixed-size read per file that
never touches a sample. Releases older than v16.1 fall back to header reads and
say so, because a missing convenience file is a slower path, not a failure.

Coverage is reported rather than assumed: a clip the manifest does not list is
counted in ``n_missing`` and left out of the total instead of being guessed at.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.registry import LangSpec
from .tables import fmt_value, render_table

SPLITS = ("train", "validation", "test")
#: Common Voice's own per-clip duration manifest, shipped since release 16.1.
CLIP_DURATIONS = "clip_durations.tsv"
# How a language's durations were obtained, recorded per row so a reader can see
# which numbers are the release's own and which were measured from files.
_HEADERS = "audio headers"


@dataclass(frozen=True)
class SplitDuration:
    split: str
    n_utts: int
    seconds: float
    #: Utterances whose duration could not be determined. Their audio is absent
    #: from ``seconds``, so a large value here means the total is an undercount.
    n_missing: int = 0

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


@dataclass(frozen=True)
class LanguageDuration:
    code: str
    corpus: str
    source: str  # "clip_durations.tsv" or "audio headers"
    splits: dict[str, SplitDuration]
    error: str | None = None
    below_threshold: bool = False

    @property
    def total_hours(self) -> float:
        return sum(s.seconds for s in self.splits.values()) / 3600.0

    @property
    def train_hours(self) -> float:
        train = self.splits.get("train")
        return train.hours if train else 0.0


def _audio_seconds(path: Path) -> float | None:
    """Duration from the file header, or None if it cannot be read."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:
        # One unreadable clip must not end the sweep; it is counted as missing.
        return None
    return float(info.frames) / float(info.samplerate) if info.samplerate else None


def _read_clip_durations(path: Path) -> dict[str, float]:
    """``clip`` → seconds, from Common Voice's own manifest.

    The release documents ``clip`` as the clip filename and the split tsvs'
    ``path`` as the relative path of the audio file. In every release checked
    those are the same bare filename, which is what makes the join below work.
    The tests here validate against a fixture written from that documentation
    rather than against a real release, so a release that put a directory prefix
    in ``path`` would miss every lookup — reported as ``n_missing`` and as an
    "undercount" line in the table, not silently absorbed into the totals.
    """
    durations: dict[str, float] = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            clip = (row.get("clip") or "").strip()
            raw = (row.get("duration[ms]") or "").strip()
            if not clip or not raw:
                continue
            try:
                durations[clip] = float(raw) / 1000.0
            except ValueError:
                continue
    return durations


def _commonvoice(spec: LangSpec) -> LanguageDuration:
    from ..data.commonvoice_local import cv_split_dir, load_cv_rows

    base = cv_split_dir(spec.hf_config)
    manifest = base / CLIP_DURATIONS
    declared = _read_clip_durations(manifest) if manifest.exists() else {}
    # Only worth reporting when the language is actually there. If the whole
    # directory is missing, the manifest is the least of it and the read below
    # raises with a message that says so.
    if not declared and base.is_dir():
        warnings.warn(
            f"{spec.code}: no usable {CLIP_DURATIONS} under {base}; falling back to reading "
            "audio headers, which is slower. Releases from Common Voice 16.1 ship this file.",
            UserWarning,
            stacklevel=3,
        )

    splits: dict[str, SplitDuration] = {}
    for split in SPLITS:
        rows = load_cv_rows(spec.hf_config, split, text_column=spec.text_column)
        seconds = 0.0
        missing = 0
        for clip, _ in rows:
            value = declared.get(clip) if declared else _audio_seconds(base / "clips" / clip)
            if value is None:
                missing += 1
            else:
                seconds += value
        splits[split] = SplitDuration(
            split=split, n_utts=len(rows), seconds=seconds, n_missing=missing
        )

    return LanguageDuration(
        code=spec.code,
        corpus=spec.hf_dataset or "commonvoice",
        source=CLIP_DURATIONS if declared else _HEADERS,
        splits=splits,
    )


def _openslr(spec: LangSpec) -> LanguageDuration:
    from ..data.openslr_local import OpenSLRLocal

    splits: dict[str, SplitDuration] = {}
    for split in SPLITS:
        dataset = OpenSLRLocal(spec, split)
        seconds = 0.0
        missing = 0
        for file_id, _ in dataset.rows:
            value = _audio_seconds(dataset.base / f"{file_id}.wav")
            if value is None:
                missing += 1
            else:
                seconds += value
        splits[split] = SplitDuration(
            split=split, n_utts=len(dataset.rows), seconds=seconds, n_missing=missing
        )

    return LanguageDuration(
        code=spec.code,
        corpus=f"SLR{spec.slr}",
        source=_HEADERS,
        splits=splits,
    )


def language_durations(spec: LangSpec) -> LanguageDuration:
    """Duration accounting for one language, from metadata only."""
    if spec.source == "commonvoice":
        return _commonvoice(spec)
    if spec.source == "openslr":
        return _openslr(spec)
    raise ValueError(f"{spec.code}: unknown source {spec.source!r}")


def collect_durations(
    specs: list[LangSpec], min_train_hours: float = 0.0
) -> list[LanguageDuration]:
    """Every language's accounting, with unreachable corpora recorded, not raised.

    This command exists to report what is on disk, so one missing corpus must
    not end the sweep — the whole point is to find out which ones are missing.
    """
    rows: list[LanguageDuration] = []
    for spec in specs:
        try:
            row = language_durations(spec)
        except Exception as exc:
            rows.append(
                LanguageDuration(
                    code=spec.code,
                    corpus=spec.hf_dataset or (f"SLR{spec.slr}" if spec.slr else spec.source),
                    source="—",
                    splits={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        rows.append(
            LanguageDuration(
                code=row.code,
                corpus=row.corpus,
                source=row.source,
                splits=row.splits,
                below_threshold=bool(min_train_hours) and row.train_hours < min_train_hours,
            )
        )
    return rows


def to_json(rows: list[LanguageDuration], min_train_hours: float = 0.0) -> dict[str, Any]:
    return {
        "min_train_hours": min_train_hours,
        "languages": [
            {
                "code": row.code,
                "corpus": row.corpus,
                "duration_source": row.source,
                "error": row.error,
                "below_threshold": row.below_threshold,
                "total_hours": row.total_hours,
                "splits": {
                    name: {
                        "n_utts": split.n_utts,
                        "seconds": split.seconds,
                        "hours": split.hours,
                        "n_missing": split.n_missing,
                    }
                    for name, split in row.splits.items()
                },
            }
            for row in rows
        ],
    }


def to_markdown(rows: list[LanguageDuration], min_train_hours: float = 0.0) -> str:
    lines = [
        "# Audio per language",
        "",
        "Durations are read from metadata only — Common Voice's own "
        f"`{CLIP_DURATIONS}` where the release ships one, otherwise audio file headers. "
        "No audio is decoded.",
        "",
        render_table(
            ["language", "corpus", "train h", "dev h", "test h", "total h", "utts", "source"],
            [
                [
                    row.code,
                    row.corpus,
                    *(fmt_value(row.splits[s].hours if s in row.splits else None) for s in SPLITS),
                    fmt_value(row.total_hours),
                    str(sum(s.n_utts for s in row.splits.values())),
                    row.error or row.source,
                ]
                for row in rows
            ],
            aligns=["left", "left", "right", "right", "right", "right", "right", "left"],
        ),
        "",
    ]

    flagged = [row.code for row in rows if row.below_threshold]
    if flagged:
        lines += [
            f"**Below the {fmt_value(min_train_hours)} h training threshold:** "
            f"{', '.join(flagged)}.",
            "",
        ]

    undercounted = [
        f"{row.code} ({sum(s.n_missing for s in row.splits.values())})"
        for row in rows
        if any(s.n_missing for s in row.splits.values())
    ]
    if undercounted:
        lines += [
            "**Clips with no duration available**, so the totals above undercount: "
            f"{', '.join(undercounted)}.",
            "",
        ]

    failed = [f"{row.code} — {row.error}" for row in rows if row.error]
    if failed:
        lines += ["**Unreadable:**", "", *(f"- {item}" for item in failed), ""]

    return "\n".join(lines).rstrip() + "\n"


def write_tables(
    rows: list[LanguageDuration], out_dir: Path, min_train_hours: float = 0.0
) -> list[Path]:
    """Write ``data_durations.md`` and ``.json``; return both paths."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "data_durations.md"
    json_path = out_dir / "data_durations.json"
    md_path.write_text(to_markdown(rows, min_train_hours), encoding="utf-8")
    json_path.write_text(
        json.dumps(to_json(rows, min_train_hours), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return [md_path, json_path]
