"""OpenSLR (local) loader for the held-out Indic transfer corpora.

The four crowdsourced Indic sets — Malayalam (SLR63), Marathi (SLR64), Telugu
(SLR66) and Gujarati (SLR78) — are published on openslr.org as zip archives
under CC-BY-SA-4.0. ``scripts/fetch_openslr.py`` downloads and extracts them; a
language then lives in one flat directory::

    $OPENSLR_ROOT/
        SLR63/
            line_index_female.tsv  line_index_male.tsv
            mlf_02879_01795762363.wav  ...
            manifest.json

Set ``OPENSLR_ROOT`` (or pass ``root``). Audio is 16-bit PCM WAV at 48 kHz, read
with ``soundfile`` and resampled to 16 kHz — no mp3 decoder and no ffmpeg needed
on this path.

**Index format.** Each index is a two-column tab-separated file with no header:
``FileID<TAB>transcript``, and the audio for a row is ``<FileID>.wav`` in the
same directory. Verified against the first bytes of
``openslr.trmal.net/resources/63/line_index_female.tsv`` (2103 rows, every row
exactly two tab-separated fields) and against the openslr.org page text, "The
file line_index.tsv contains a anonymized FileID and the transcription of audio
in the file."

**Splits.** These corpora ship no train/validation/test split. It is derived
here, at load time, from a stable SHA-1 of the grouping key — never from a
library's shuffle, whose output is not stable across versions.

The FileID is ``<corpus-prefix>_<speaker>_<utterance>``: in SLR63's female index
2103 utterances carry only 24 distinct middle fields (160, 142, 140 ... rows
each) while every third field is unique, so the middle field is a speaker id.
That inspection covered SLR63's female index; the other corpora are parsed the
same way at load time and report the policy they actually got. The split is
therefore **speaker-disjoint** — test speakers are never seen in training, which
is what a transfer number should measure.

Languages are small in speakers (SLR63 has 24 female + 18 male, SLR64 only 9),
so groups are assigned to whichever split has the largest remaining shortfall
rather than bucketed by hash, which is far less lumpy at these sizes. It is not
exact: whole speakers are indivisible, so the realised proportions drift from
80/10/10, and the fewer and more uneven a language's speakers the further they
drift. Nine evenly-sized speakers land near 78/11/11; nine uneven ones can put
under 3% of the utterances in the test split. Read the realised sizes from
``check_data.sh`` rather than assuming the nominal fractions.

If FileIDs do not parse as ``prefix_speaker_utterance``, or fewer than three
speakers exist, the split falls back to utterance level. That is reported as
``split_policy == "utterance"`` and carries an obvious caveat: the same speaker
then appears in train and test, so the number is not a speaker-independent one.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .registry import LangSpec

TARGET_SR = 16000
SPLITS = ("train", "validation", "test")
_FRACTIONS = (0.8, 0.1, 0.1)
_MIN_GROUPS = len(SPLITS)

Row = tuple[str, str]  # (FileID, transcript)
_FETCH_HINT = "run `python scripts/fetch_openslr.py --root $OPENSLR_ROOT` first"


def _openslr_root(root: str | None) -> Path:
    root = root or os.environ.get("OPENSLR_ROOT")
    if not root:
        raise RuntimeError(
            "Held-out OpenSLR corpora are read from a local directory. "
            f"Set $OPENSLR_ROOT to the extraction root and {_FETCH_HINT}."
        )
    return Path(root)


def read_index(path: Path) -> list[Row]:
    """Read one ``line_index*.tsv`` as ``(FileID, transcript)`` rows.

    Blank lines, lines without a tab, and rows with an empty field are skipped —
    the corpora are hand-checked but the pages warn that errors remain.
    """
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            file_id, sep, text = line.rstrip("\n").partition("\t")
            if not sep:
                continue
            file_id, text = file_id.strip(), text.strip()
            if file_id and text:
                rows.append((file_id, text))
    return rows


def speaker_of(file_id: str) -> str | None:
    """Speaker key for a FileID, or None if the id has no speaker field.

    FileIDs look like ``mlf_02879_01795762363``: corpus prefix, speaker, utterance.
    The prefix is kept in the key because it also encodes the archive (``mlf`` vs
    ``mlm``), so two archives cannot collide on a bare speaker number.
    """
    parts = file_id.split("_")
    if len(parts) != 3 or not all(parts):
        return None
    return f"{parts[0]}_{parts[1]}"


def _assign_groups(ordered: list[str], groups: dict[str, list[Row]]) -> dict[str, list[str]]:
    """Give each group to the split with the largest remaining shortfall."""
    total = sum(len(groups[k]) for k in ordered)
    targets = {name: frac * total for name, frac in zip(SPLITS, _FRACTIONS, strict=True)}
    counts = dict.fromkeys(SPLITS, 0)
    assigned: dict[str, list[str]] = {name: [] for name in SPLITS}

    for key in ordered:
        # Ties break towards the earlier split, so the choice never depends on
        # dict ordering.
        name = max(SPLITS, key=lambda n: (targets[n] - counts[n], -SPLITS.index(n)))
        assigned[name].append(key)
        counts[name] += len(groups[key])

    _fill_empty_splits(assigned, groups)
    return assigned


def _fill_empty_splits(assigned: dict[str, list[str]], groups: dict[str, list[Row]]) -> None:
    """Make sure no split is empty when there are groups enough to go round.

    One dominant group can otherwise swallow the whole target for train and leave
    validation or test with nothing. Donate the smallest group of the split that
    holds the most, which costs the donor the least mass.
    """
    if sum(len(keys) for keys in assigned.values()) < _MIN_GROUPS:
        return
    for name in SPLITS:
        if assigned[name]:
            continue
        donor = max(SPLITS, key=lambda n: (len(assigned[n]), -SPLITS.index(n)))
        if len(assigned[donor]) < 2:
            return
        key = min(assigned[donor], key=lambda k: (len(groups[k]), k))
        assigned[donor].remove(key)
        assigned[name].append(key)


def derive_splits(rows: list[Row]) -> tuple[dict[str, list[Row]], str, str]:
    """Partition rows 80/10/10.

    Returns the partition, the policy used (``"speaker"`` or ``"utterance"``),
    and the SHA-1 of the resulting test FileID list, which callers record so a
    run's provenance pins the exact held-out set it scored on.
    """
    speakers = [speaker_of(file_id) for file_id, _ in rows]
    resolved = [s for s in speakers if s is not None]
    if len(resolved) == len(rows) and len(set(resolved)) >= _MIN_GROUPS:
        policy, keys = "speaker", resolved
    else:
        policy, keys = "utterance", [file_id for file_id, _ in rows]

    groups: dict[str, list[Row]] = {}
    for key, row in zip(keys, rows, strict=True):
        groups.setdefault(key, []).append(row)

    ordered = sorted(groups, key=lambda k: hashlib.sha1(k.encode("utf-8")).hexdigest())
    assigned = _assign_groups(ordered, groups)

    # Sorting by FileID makes the result independent of the input row order.
    parts = {name: sorted(r for k in assigned[name] for r in groups[k]) for name in SPLITS}
    listing = "\n".join(file_id for file_id, _ in parts["test"])
    return parts, policy, hashlib.sha1(listing.encode("utf-8")).hexdigest()


def _language_rows(spec: LangSpec, root: str | None) -> tuple[Path, list[Row]]:
    base = _openslr_root(root) / f"SLR{spec.slr}"
    if not base.is_dir():
        raise FileNotFoundError(f"missing extracted corpus {base} — {_FETCH_HINT}")
    rows: list[Row] = []
    for name in spec.index_files:
        index = base / name
        if not index.exists():
            raise FileNotFoundError(f"missing index {index} — {_FETCH_HINT}")
        rows.extend(read_index(index))
    if not rows:
        raise ValueError(f"no usable rows in {base} indices {list(spec.index_files)}")
    return base, rows


class OpenSLRLocal(Dataset):
    """One derived split of one OpenSLR language as (waveform@16k, text)."""

    def __init__(
        self,
        spec: LangSpec,
        split: str,
        root: str | None = None,
        max_samples: int | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        self.base, all_rows = _language_rows(spec, root)
        parts, policy, digest = derive_splits(all_rows)

        self.spec = spec
        self.split = split
        self.rows: list[Row] = parts[split]
        #: How the partition was derived — record this alongside results.
        self.split_policy = policy
        #: SHA-1 of the test FileID list, so provenance pins the held-out set.
        self.test_files_sha1 = digest
        self._max = max_samples

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        import soundfile as sf

        file_id, text = self.rows[idx]
        data, sr = sf.read(str(self.base / f"{file_id}.wav"), dtype="float32", always_2d=False)
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.dim() > 1:  # soundfile yields (frames, channels)
            wav = wav.mean(dim=1)
        if sr != TARGET_SR:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if self._max is not None and wav.shape[0] > self._max:
            wav = wav[: self._max]
        return wav, text


def load_openslr_texts(spec: LangSpec, split: str, root: str | None = None) -> list[str]:
    """Transcripts of one split, read from the indices with no audio decode."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    _, rows = _language_rows(spec, root)
    parts, _, _ = derive_splits(rows)
    return [text for _, text in parts[split]]
