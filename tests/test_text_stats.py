"""Per-language text statistics — the file a reviewer reads to check the policy.

These numbers are how the normalization assumptions become auditable rather
than asserted: that Common Voice really does contain almost no digits, that a
language's unknown-character rate really is zero, and that the ``word_boundary``
declared for a language matches how its transcripts are actually written.
"""

from __future__ import annotations

from svb.model.ctc_vocab import CtcVocab
from svb.text.stats import collect_text_stats


def test_counts_utterances_and_characters() -> None:
    stats = collect_text_stats(["Hello, World!", "a b"])

    assert stats.n_utts == 2
    assert stats.n_chars_raw == len("Hello, World!") + len("a b")
    assert stats.n_chars_normalized == len("hello world") + len("a b")


def test_counts_utterances_that_normalize_to_empty() -> None:
    """These cannot be a CTC target and cannot be scored, so they are counted."""
    stats = collect_text_stats(["real text", "…!?", "   "])

    assert stats.n_utts == 3
    assert stats.n_utts_empty_after_norm == 2


def test_counts_digit_bearing_utterances_in_any_script() -> None:
    """Common Voice's own rule matches ASCII only, so native digits pass it."""
    stats = collect_text_stats(["no digits here", "in 2026", "मेरे पास ३ किताबें"])

    assert stats.n_utts_with_digits == 2


def test_reports_what_each_rule_removed() -> None:
    stats = collect_text_stats(["Hello, world! 50% off", "soft­hyphen", "مَرْحَبًا"])

    assert stats.removed_by_category["P"] == 3  # comma, exclamation mark, percent sign
    assert stats.removed_by_category["Cf"] == 1  # the soft hyphen
    assert stats.removed_by_category["arabic_marks"] == 4  # fatha, sukun, fatha, tanwin


def test_counts_characters_that_case_folding_rewrote() -> None:
    stats = collect_text_stats(["ABC def"])

    assert stats.case_changed_chars == 3


def test_median_tokens_per_utterance_detects_a_no_space_language() -> None:
    """The empirical check on a language's declared ``word_boundary``."""
    spaced = collect_text_stats(["the cat sat", "a dog ran here"])
    unspaced = collect_text_stats(["コーヒーを飲む", "今日はいい天気"])

    assert spaced.median_tokens_per_utt > 1
    assert unspaced.median_tokens_per_utt == 1.0
    assert unspaced.looks_unspaced
    assert not spaced.looks_unspaced


def test_unknown_character_rate_is_measured_against_a_vocab() -> None:
    """A character the vocab lacks puts an irreducible floor under the language's
    error rate, so the rate is reported rather than left for the reader to guess."""
    vocab = CtcVocab(id_to_char=["<blank>", "<unk>", " ", "a", "b"])
    stats = collect_text_stats(["a b c"], vocab=vocab)

    assert stats.unk_chars == 1
    assert stats.unk_rate == 1 / len("a b c")


def test_unknown_rate_is_zero_without_a_vocab() -> None:
    stats = collect_text_stats(["anything at all"])

    assert stats.unk_chars == 0
    assert stats.unk_rate == 0.0


def test_empty_input_does_not_divide_by_zero() -> None:
    stats = collect_text_stats([])

    assert stats.n_utts == 0
    assert stats.unk_rate == 0.0
    assert stats.median_tokens_per_utt == 0.0
    assert stats.to_dict()["n_utts"] == 0


def test_stats_serialize_for_the_run_sidecar() -> None:
    stats = collect_text_stats(["Hello!"])
    payload = stats.to_dict()

    assert payload["n_chars_raw"] == 6
    assert isinstance(payload["removed_by_category"], dict)
    assert "evicted_by_floor" in payload


def test_a_protected_apostrophe_is_not_reported_as_removed() -> None:
    """The tally must describe what happened, not what a category scan would say.

    Counting every punctuation character in the input reports the intra-word
    apostrophes the policy goes out of its way to keep as removals, which
    overstates punctuation removal for French, Italian, Ukrainian and Swahili by
    exactly the characters that survived.
    """
    stats = collect_text_stats(["ng'ombe, l'été!"])

    assert stats.removed_by_category["P"] == 2  # the comma and the exclamation mark
    assert stats.n_chars_normalized == len("ng'ombe l'été")


def test_a_rule_that_is_switched_off_removes_nothing() -> None:
    from svb.text.normalize import NormalizerPolicy

    stats = collect_text_stats(
        ["Hello, world!"], NormalizerPolicy(strip_punctuation=False, strip_symbols=False)
    )

    assert stats.removed_by_category["P"] == 0
    assert stats.removed_by_category["S"] == 0


def test_case_changes_are_not_counted_when_case_folding_is_off() -> None:
    from svb.text.normalize import NormalizerPolicy

    assert collect_text_stats(["Hello"], NormalizerPolicy(case="none")).case_changed_chars == 0
    assert collect_text_stats(["Hello"]).case_changed_chars == 1


def test_every_category_a_rule_can_delete_has_a_slot() -> None:
    """A removal with nowhere to be counted is a removal nobody sees."""
    stats = collect_text_stats(["anything"])

    assert set(stats.removed_by_category) >= {"P", "S", "Cf", "Cc", "Cs", "Co", "Cn"}
    assert "arabic_marks" in stats.removed_by_category
    assert "case_changed" not in stats.removed_by_category  # its own field
