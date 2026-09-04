"""``text_stats.json``: the per-language evidence for the normalization policy.

Written by the run itself, so the claims behind a reported number — that the
vocabulary covers the test set, that digits are rare, that a language declared
to use spaces does — are checkable from the results directory alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.cli import write_text_stats
from svb.config import TextConfig
from svb.data.registry import LangSpec
from svb.model.ctc_vocab import build_vocab_from_labelled_texts
from svb.text.registry import policies_for_specs

# Every split's characters appear in its language's training split, as a real
# corpus split of one language mostly does. Anything else would make the
# unknown-character warning fire throughout the suite and bury a real one.
CORPUS = {
    ("en", "train"): ["Hello, world!", "the cat sat", "a dog ran"],
    ("en", "validation"): ["the dog ran"],
    ("en", "test"): ["hello there", "!!!"],
    ("ja", "train"): ["コーヒーを飲む", "今日はいい天気"],
    ("ja", "validation"): ["今日は飲む"],
    ("ja", "test"): ["いい天気"],
    ("telugu", "train"): ["ఇది తెలుగు భాష."],
    ("telugu", "validation"): ["ఇది తెలుగు"],
    ("telugu", "test"): ["తెలుగు భాష"],
}


@pytest.fixture
def stub_texts(monkeypatch):
    import svb.data.datasets as datasets

    def fake_load_texts(spec: LangSpec, split: str) -> list[str]:
        return CORPUS[(spec.code, split)]

    monkeypatch.setattr(datasets, "load_texts", fake_load_texts)


def _specs() -> tuple[list[LangSpec], list[LangSpec]]:
    training = [
        LangSpec(code="en", source="commonvoice", hf_config="en", normalizer="whisper-basic"),
        LangSpec(
            code="ja",
            source="commonvoice",
            hf_config="ja",
            word_boundary=False,
            normalizer="ja-cer",
        ),
    ]
    heldout = [
        LangSpec(
            code="telugu",
            source="openslr",
            slr=66,
            archives=("te.zip",),
            index_files=("line_index.tsv",),
            normalizer="indic-vistaar",
        )
    ]
    return training, heldout


def _write(tmp_path: Path, min_char_count: int = 1) -> dict:
    training, heldout = _specs()
    text_cfg = TextConfig(min_char_count=min_char_count)
    policies = policies_for_specs(training, text_cfg.override)
    vocab, evicted = build_vocab_from_labelled_texts(
        [
            (code, t)
            for (code, split), texts in CORPUS.items()
            if code in ("en", "ja") and split == "train"
            for t in texts
        ],
        policies,
        min_char_count=min_char_count,
    )
    path = write_text_stats(
        tmp_path / "text_stats.json", training, heldout, text_cfg, vocab, evicted
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_language_and_split_is_covered(tmp_path, stub_texts) -> None:
    data = _write(tmp_path)

    assert set(data["languages"]) == {"en", "ja", "telugu"}
    for lang in data["languages"].values():
        assert set(lang["splits"]) == {"train", "validation", "test"}


def test_the_policy_is_named_so_the_file_stands_alone(tmp_path, stub_texts) -> None:
    data = _write(tmp_path)

    assert data["override"] is None
    assert data["min_char_count"] == 1
    assert data["languages"]["en"]["policy"] == "whisper-basic"
    assert data["languages"]["telugu"]["policy"] == "indic-vistaar"
    assert data["languages"]["en"]["policy_hash"] != data["languages"]["telugu"]["policy_hash"]


def test_the_declared_word_boundary_is_recorded_next_to_the_measurement(
    tmp_path, stub_texts
) -> None:
    """Side by side, so a mislabelled preset is visible without running anything."""
    data = _write(tmp_path)

    assert data["languages"]["en"]["word_boundary"] is True
    assert data["languages"]["en"]["splits"]["train"]["looks_unspaced"] is False
    assert data["languages"]["ja"]["word_boundary"] is False
    assert data["languages"]["ja"]["splits"]["train"]["looks_unspaced"] is True


def test_utterances_that_normalize_to_empty_are_counted_per_split(tmp_path, stub_texts) -> None:
    data = _write(tmp_path)

    assert data["languages"]["en"]["splits"]["test"]["n_utts_empty_after_norm"] == 1
    assert data["languages"]["en"]["splits"]["train"]["n_utts_empty_after_norm"] == 0


def test_the_training_vocab_covers_the_training_languages(tmp_path, stub_texts) -> None:
    data = _write(tmp_path)

    assert data["languages"]["en"]["splits"]["train"]["unk_rate"] == 0.0
    assert data["languages"]["ja"]["splits"]["train"]["unk_rate"] == 0.0


def test_the_heldout_language_is_measured_against_its_expanded_vocab(tmp_path, stub_texts) -> None:
    """Scoring it against the training vocab would report a Telugu unknown rate
    near 1.0 and hide the number the transfer experiment actually depends on."""
    data = _write(tmp_path)
    telugu = data["languages"]["telugu"]

    assert telugu["heldout"] is True
    assert telugu["splits"]["test"]["unk_rate"] == 0.0
    assert telugu["splits"]["train"]["unk_rate"] == 0.0


def test_what_the_floor_evicted_is_recorded(tmp_path, stub_texts) -> None:
    """A floor of two on a corpus this small evicts most of the alphabet, which
    is exactly when the unknown-character warning should be shouting: the floor
    is what put those characters out of reach."""
    with pytest.warns(UserWarning, match="absent from the vocabulary"):
        data = _write(tmp_path, min_char_count=2)

    assert data["training_vocab_evicted"]
    assert all(count < 2 for count in data["training_vocab_evicted"].values())


def test_a_vocabulary_that_cannot_spell_a_split_warns(tmp_path, monkeypatch) -> None:
    """The number always lands in the file, but a file nobody opens is not a
    warning. Unknown characters are a floor under the language's error rate."""
    import svb.data.datasets as datasets

    corpus = dict(CORPUS)
    # A Greek sentence the Latin/Japanese/Telugu training vocab has no ids for.
    corpus[("en", "test")] = ["καλημέρα κόσμε"]
    monkeypatch.setattr(datasets, "load_texts", lambda spec, split: corpus[(spec.code, split)])

    training, heldout = _specs()
    text_cfg = TextConfig()
    vocab, evicted = build_vocab_from_labelled_texts(
        [(code, t) for (code, split), texts in CORPUS.items() if split == "train" for t in texts],
        policies_for_specs([*training, *heldout]),
    )

    with pytest.warns(UserWarning, match=r"en/test:.*absent from the vocabulary"):
        write_text_stats(tmp_path / "s.json", training, heldout, text_cfg, vocab, evicted)


def test_full_coverage_says_nothing(recwarn) -> None:
    """A vocabulary that can spell the split is the normal case and is silent."""
    from svb.cli import _warn_on_unknown_characters
    from svb.text.stats import TextStats

    _warn_on_unknown_characters("en", "train", TextStats(n_chars_normalized=1000, unk_chars=0))

    assert recwarn.list == []


def test_the_warning_threshold_is_a_rate_not_a_count(recwarn) -> None:
    """One stray character in a large corpus is not a floor worth flagging; a
    tenth of a percent of them is."""
    from svb.cli import UNK_RATE_WARNING, _warn_on_unknown_characters
    from svb.text.stats import TextStats

    assert UNK_RATE_WARNING == 0.001
    below = TextStats(n_chars_normalized=100_000, unk_chars=100, unk_rate=0.001)
    _warn_on_unknown_characters("en", "train", below)
    assert recwarn.list == []

    above = TextStats(n_chars_normalized=100_000, unk_chars=101, unk_rate=0.00101)
    with pytest.warns(UserWarning, match="floor"):
        _warn_on_unknown_characters("en", "train", above)
