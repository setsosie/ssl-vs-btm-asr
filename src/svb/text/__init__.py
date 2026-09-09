"""Transcript normalization and the text statistics a run reports."""

from .normalize import (
    DEFAULT_POLICY,
    LEGACY_POLICY,
    NORMALIZER_VERSION,
    NormalizerPolicy,
    count_arabic_marks,
    has_digits,
    module_sha256,
    normalize_batch,
    normalize_text,
)
from .stats import TextStats, collect_text_stats

__all__ = [
    "DEFAULT_POLICY",
    "LEGACY_POLICY",
    "NORMALIZER_VERSION",
    "NormalizerPolicy",
    "TextStats",
    "collect_text_stats",
    "count_arabic_marks",
    "has_digits",
    "module_sha256",
    "normalize_batch",
    "normalize_text",
]
