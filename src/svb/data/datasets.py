"""Source-dispatching entry point for the two public corpora.

Both sources are read from a local directory and neither goes through the
Hugging Face Hub, so there is no shared "audio column" adapter to write — the
two layouts have nothing in common beyond what they return:

``commonvoice`` → :mod:`svb.data.commonvoice_local`
    ``$CV_ROOT/<lang>/{train,dev,test,validated}.tsv`` plus ``clips/*.mp3``.
    Common Voice has been distributed only via Mozilla Data Collective since
    October 2025. ``train_source`` selects between the official ``train.tsv``
    and validated-minus-evaluation; it is meaningless for ``openslr``, which
    derives its own split, and is ignored there rather than rejected.

``openslr`` → :mod:`svb.data.openslr_local`
    ``$OPENSLR_ROOT/SLR<n>/`` as extracted by ``scripts/fetch_openslr.py``, with
    an 80/10/10 split derived at load time. The held-out transfer four only.

``manifest`` → :mod:`svb.data.manifest_local`
    ``$CORPORA_ROOT/<corpus>/<lang>/manifest.tsv`` as written by
    ``scripts/prepare_<corpus>.py``. Every other corpus goes through here: the
    fourteen of them ship in about ten shapes, and converting each once beats
    fourteen dataset classes with fourteen chances to get a split wrong.

All return a torch ``Dataset`` of ``(waveform: 1-D float tensor @16kHz, text,
language code)``.

``max_hours`` caps a training or validation split at that much audio, chosen by
:mod:`svb.data.hours`. It is a parameter of both :func:`load_language` and
:func:`load_texts` and the two must be given the same value: the vocabulary and
the text statistics are built from the transcripts, and a vocabulary built over
rows the model never trains on describes a different run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset, Subset

from .commonvoice_local import DEFAULT_TRAIN_SOURCE
from .hours import CapSelection, select_within_hours
from .registry import LangSpec

if TYPE_CHECKING:
    from ..config import ExperimentConfig

TARGET_SR = 16000


def load_split(cfg: ExperimentConfig, spec: LangSpec, split: str) -> Dataset:
    """One language's split, as this run reads it.

    The training source, the hours cap and the truncation guard are all part
    of which audio a run trains on, so every stage takes them from the config
    through here rather than spelling them out at each call. The test split
    gets none of them: a clipped waveform scored against its full transcript
    manufactures deletions, and a number scored on part of a test set is a
    number about that part.
    """
    evaluating = split == "test"
    return load_language(
        spec,
        split,
        None if evaluating else cfg.train.max_audio_samples,
        train_source=cfg.train.cv_train_source,
        max_hours=None if evaluating else cfg.train.max_hours(split),
    )


def load_language(
    spec: LangSpec,
    split: str,
    max_samples: int | None = None,
    train_source: str = DEFAULT_TRAIN_SOURCE,
    max_hours: float | None = None,
) -> Dataset:
    """Load one language split as a torch Dataset of (waveform@16k, text, code).

    Args:
        spec: Language spec.
        split: "train", "validation", or "test".
        max_samples: Per-utterance audio truncation guard (samples @16k).
        train_source: Which Common Voice rows the training split is; ignored for
            other sources and for the evaluation splits.
        max_hours: Keep at most this much audio, chosen by hash. ``None`` reads
            the whole split. Refused for the test split.
    """
    dataset = _open(spec, split, max_samples, train_source)
    if max_hours is None:
        return dataset
    selection = _select(dataset, spec, split, max_hours)
    return Subset(dataset, list(selection.indices))


def hours_cap_selection(
    spec: LangSpec,
    split: str,
    max_hours: float,
    train_source: str = DEFAULT_TRAIN_SOURCE,
) -> CapSelection:
    """What ``max_hours`` keeps of one split — the record a run writes down."""
    return _select(_open(spec, split, None, train_source), spec, split, max_hours)


def _select(dataset: Dataset, spec: LangSpec, split: str, max_hours: float) -> CapSelection:
    if split == "test":
        raise ValueError(
            f"{spec.code}: the test split is never capped; a number scored on part of a "
            "test set is a number about that part"
        )
    # Every loader exposes this; it is how the cap stays out of the business of
    # knowing where each corpus keeps its audio.
    rows = dataset.utterance_seconds()  # type: ignore[attr-defined]
    return select_within_hours(rows, max_hours, salt=spec.code)


def _open(spec: LangSpec, split: str, max_samples: int | None, train_source: str) -> Dataset:
    if spec.source == "commonvoice":
        from .commonvoice_local import CommonVoiceLocal

        return CommonVoiceLocal(
            lang=spec.hf_config,
            split=split,
            text_column=spec.text_column,
            max_samples=max_samples,
            train_source=train_source,
            code=spec.code,
        )
    if spec.source == "openslr":
        from .openslr_local import OpenSLRLocal

        return OpenSLRLocal(spec, split, max_samples=max_samples)
    if spec.source == "manifest":
        from .manifest_local import ManifestLocal

        return ManifestLocal(spec, split, max_samples=max_samples)
    raise ValueError(f"{spec.code}: unknown source {spec.source!r}")


def load_texts(
    spec: LangSpec,
    split: str,
    train_source: str = DEFAULT_TRAIN_SOURCE,
    max_hours: float | None = None,
) -> list[str]:
    """Load just the transcripts for a split (no audio decode) — for vocab build.

    The vocabulary has to be built over the rows the run will actually train on,
    so this takes the same ``train_source`` and ``max_hours`` as
    :func:`load_language`. Building it from a different row set would either
    leave characters in the training targets with no id to map to, or give ids
    to characters the model never sees.
    """
    texts = _all_texts(spec, split, train_source)
    if max_hours is None:
        return texts
    selection = hours_cap_selection(spec, split, max_hours, train_source)
    if selection.n_available != len(texts):
        raise RuntimeError(
            f"{spec.code}/{split}: the transcript reader and the dataset disagree about "
            f"the split ({len(texts)} vs {selection.n_available} rows), so an index into "
            "one is not an index into the other"
        )
    return [texts[i] for i in selection.indices]


def _all_texts(spec: LangSpec, split: str, train_source: str) -> list[str]:
    if spec.source == "commonvoice":
        from .commonvoice_local import load_cv_texts

        return load_cv_texts(
            spec.hf_config, split, text_column=spec.text_column, train_source=train_source
        )
    if spec.source == "openslr":
        from .openslr_local import load_openslr_texts

        return load_openslr_texts(spec, split)
    if spec.source == "manifest":
        from .manifest_local import load_manifest_texts

        return load_manifest_texts(spec, split)
    raise ValueError(f"{spec.code}: unknown source {spec.source!r}")
