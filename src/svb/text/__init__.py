"""Transcript normalization and the text statistics a run reports."""

from .normalize import (
    ADDITIONAL_DIACRITICS,
    DEFAULT_POLICY,
    LEGACY_POLICY,
    NORMALIZER_VERSION,
    POLICIES,
    NormalizerPolicy,
    count_arabic_marks,
    empty_removal_counts,
    get_policy,
    has_digits,
    module_sha256,
    normalize_batch,
    normalize_text,
    normalize_with_counts,
    policy_name,
)
from .stats import TextStats, collect_text_stats

__all__ = [
    "ADDITIONAL_DIACRITICS",
    "DEFAULT_POLICY",
    "LEGACY_POLICY",
    "NORMALIZER_VERSION",
    "POLICIES",
    "NormalizerPolicy",
    "TextStats",
    "collect_text_stats",
    "count_arabic_marks",
    "empty_removal_counts",
    "get_policy",
    "has_digits",
    "module_sha256",
    "normalize_batch",
    "normalize_text",
    "normalize_with_counts",
    "policy_name",
]
