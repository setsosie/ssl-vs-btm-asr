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
drift. Nine evenly-sized speakers land near 78/11/11; nine uneven ones — one
dominant speaker and eight small — land near 94/3/3. Read the realised sizes
from ``check_data.py`` rather than assuming the nominal fractions.

The shortfall rule alone can leave a split empty, which one dominant speaker is
enough to do, so a second pass moves the smallest speaker out of whichever split
holds the most until none is empty. That pass, not luck, is why a nine-speaker
corpus still has a test split.

If **any** FileID does not parse as ``prefix_speaker_utterance``, or fewer than
three speakers exist, the whole language falls back to utterance level — one
malformed id is enough. That is reported as
``split_policy == "utterance"`` and carries an obvious caveat: the same speaker
then appears in train and test, so the number is not a speaker-independent one.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .registry import LangSpec
from .splits import SPLITS
from .splits import derive_splits as _derive_splits

TARGET_SR = 16000

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


def derive_splits(rows: list[Row]) -> tuple[dict[str, list[Row]], str, str]:
    """Partition rows 80/10/10 with the shared speaker-disjoint derivation.

    The FileID is both the grouping key's source and the row's identity, so the
    two callables the derivation takes are both cheap projections of it.
    """
    return _derive_splits(rows, lambda row: speaker_of(row[0]), lambda row: row[0])


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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
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
        return wav, text, self.spec.code


def load_openslr_texts(spec: LangSpec, split: str, root: str | None = None) -> list[str]:
    """Transcripts of one split, read from the indices with no audio decode."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    _, rows = _language_rows(spec, root)
    parts, _, _ = derive_splits(rows)
    return [text for _, text in parts[split]]
