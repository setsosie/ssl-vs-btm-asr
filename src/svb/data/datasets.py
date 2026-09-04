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

Both return a torch ``Dataset`` of ``(waveform: 1-D float tensor @16kHz, text)``.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from .commonvoice_local import DEFAULT_TRAIN_SOURCE
from .registry import LangSpec

TARGET_SR = 16000


def load_language(
    spec: LangSpec,
    split: str,
    max_samples: int | None = None,
    train_source: str = DEFAULT_TRAIN_SOURCE,
) -> Dataset:
    """Load one language split as a torch Dataset of (waveform@16k, text).

    Args:
        spec: Language spec.
        split: "train", "validation", or "test".
        max_samples: Per-utterance audio truncation guard (samples @16k).
        train_source: Which Common Voice rows the training split is; ignored for
            other sources and for the evaluation splits.
    """
    if spec.source == "commonvoice":
        from .commonvoice_local import CommonVoiceLocal

        return CommonVoiceLocal(
            lang=spec.hf_config,
            split=split,
            text_column=spec.text_column,
            max_samples=max_samples,
            train_source=train_source,
        )
    if spec.source == "openslr":
        from .openslr_local import OpenSLRLocal

        return OpenSLRLocal(spec, split, max_samples=max_samples)
    if spec.source == "manifest":
        from .manifest_local import ManifestLocal

        return ManifestLocal(spec, split, max_samples=max_samples)
    raise ValueError(f"{spec.code}: unknown source {spec.source!r}")


def load_texts(spec: LangSpec, split: str, train_source: str = DEFAULT_TRAIN_SOURCE) -> list[str]:
    """Load just the transcripts for a split (no audio decode) — for vocab build.

    The vocabulary has to be built over the rows the run will actually train on,
    so this takes the same ``train_source`` as :func:`load_language`. Building it
    from a narrower row set would leave characters in the training targets with
    no id to map to.
    """
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
