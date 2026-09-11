"""Vocab: build, CTC decode, head-preserving expansion, and the policy it carries.

A vocabulary is only meaningful next to the normalization policy that produced
it, so the policy travels with it: inside the object, inside the saved file, and
as a guard on expansion. Building a held-out language's characters under a
different policy than the training vocabulary used would put them in the vocab
in one form while scoring produces another, and nothing would fail loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.model.ctc_vocab import (
    CtcVocab,
    build_vocab_from_texts,
    expand_vocab,
    require_space_token,
)
from svb.text.normalize import DEFAULT_POLICY, LEGACY_POLICY, NormalizerPolicy


def test_build_and_roundtrip() -> None:
    vocab, evicted = build_vocab_from_texts(["abc", "cab"])

    assert vocab.id_to_char[:2] == ["<blank>", "<unk>"]
    assert vocab.size == 2 + 3  # blank, unk, a, b, c
    assert evicted == {}
    assert vocab.decode(vocab.encode("abc")) == "abc"


def test_ctc_decode_collapses_repeats_and_blanks() -> None:
    vocab, _ = build_vocab_from_texts(["ab"])
    a = vocab.char_to_id["a"]
    b = vocab.char_to_id["b"]
    blank = vocab.blank_id
    # a a <blank> a b b  ->  a a b  (collapse adj repeats, drop blanks, keep
    # the second 'a' because a blank separates the runs)
    assert vocab.decode([a, a, blank, a, b, b]) == "aab"


def test_expand_preserves_existing_ids() -> None:
    base, _ = build_vocab_from_texts(["abc"])
    new_vocab, added, _ = expand_vocab(base, ["abcde"])

    for c in "abc":
        assert new_vocab.char_to_id[c] == base.char_to_id[c]
    assert sorted(new_vocab.id_to_char[base.size :]) == ["d", "e"]
    assert added == [new_vocab.char_to_id["d"], new_vocab.char_to_id["e"]]


def test_build_normalizes_so_the_vocab_holds_no_punctuation() -> None:
    vocab, _ = build_vocab_from_texts(["Hello, World!"])

    assert set(vocab.id_to_char[2:]) == set("helo wrd")
    assert "," not in vocab.char_to_id
    assert "H" not in vocab.char_to_id


def test_encode_does_not_normalize_behind_the_callers_back() -> None:
    """Normalization happens at the three named sites, with an explicit policy.

    A normalizer hidden inside the encoder is how those sites drift apart: the
    vocab build would apply one policy and the target encoder another.
    """
    vocab, _ = build_vocab_from_texts(["abc"])

    assert vocab.encode("aBc") == [
        vocab.char_to_id["a"],
        vocab.unk_id,
        vocab.char_to_id["c"],
    ]


def test_frequency_floor_evicts_rare_characters_and_reports_them() -> None:
    """The floor exists for corpora with no collection-time validation, where
    stray characters from another script arrive a handful of times."""
    corpus = ["aaa bbb", "aaa bbb", "aaa bbb", "zq"]
    vocab, evicted = build_vocab_from_texts(corpus, min_char_count=3)

    assert "z" not in vocab.char_to_id
    assert "q" not in vocab.char_to_id
    assert evicted == {"z": 1, "q": 1}
    assert "a" in vocab.char_to_id


def test_the_floor_can_evict_the_space_and_the_guard_catches_it() -> None:
    """The space is an ordinary character, so a high enough floor drops it.

    That is the catastrophe ``require_space_token`` exists for: a tiny corpus
    plus a floor tuned for a full run silently produces a vocabulary that cannot
    write word boundaries.
    """
    vocab, evicted = build_vocab_from_texts(["aaaa b"], min_char_count=2)

    assert evicted == {" ": 1, "b": 1}
    with pytest.raises(ValueError, match="no space character"):
        require_space_token(vocab)


def test_a_floor_that_evicts_everything_is_refused() -> None:
    with pytest.raises(ValueError, match="evicted every character"):
        build_vocab_from_texts(["ab"], min_char_count=99)


def test_expansion_refuses_a_different_policy() -> None:
    base, _ = build_vocab_from_texts(["abc"])

    with pytest.raises(ValueError, match="policy mismatch"):
        expand_vocab(base, ["def"], policy=NormalizerPolicy(case="lower"))


def test_expansion_reuses_the_vocabs_own_policy_by_default() -> None:
    base, _ = build_vocab_from_texts(["abc"], policy=NormalizerPolicy(case="lower"))
    expanded, _, _ = expand_vocab(base, ["ÄÖ"])

    assert expanded.policies == base.policies
    assert "ä" in expanded.char_to_id  # lower, not casefold, and not upper


def test_expansion_floor_never_evicts_an_already_trained_character() -> None:
    """Dropping a character the head already has a row for would renumber ids."""
    base, _ = build_vocab_from_texts(["aaa"])
    expanded, added, evicted = expand_vocab(base, ["a", "zzz"], min_char_count=2)

    assert "a" in expanded.char_to_id
    assert "z" in expanded.char_to_id
    assert evicted == {}
    assert added == [expanded.char_to_id["z"]]


def test_save_records_the_policy_alongside_the_characters(tmp_path: Path) -> None:
    vocab, _ = build_vocab_from_texts(["abc"])
    path = tmp_path / "vocab.json"
    vocab.save(path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["policies"]["*"]["version"] == DEFAULT_POLICY.version
    assert saved["id_to_char"] == vocab.id_to_char

    reloaded = CtcVocab.load(path)
    assert reloaded.id_to_char == vocab.id_to_char
    assert reloaded.policies == vocab.policies


def test_a_legacy_vocab_file_still_loads_and_says_what_it_is(tmp_path: Path) -> None:
    """Vocabularies written before this policy existed were a bare JSON list,
    built with NFC and nothing else. They load, and describe themselves."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(["<blank>", "<unk>", "a", "b"]), encoding="utf-8")

    vocab = CtcVocab.load(path)

    assert vocab.id_to_char == ["<blank>", "<unk>", "a", "b"]
    assert vocab.policy_for("anything") == LEGACY_POLICY
    assert vocab.policy_for("anything").version == "svb-norm-0"


def test_require_space_token_catches_a_vocab_that_cannot_write_words() -> None:
    """A vocabulary without a space is a silent catastrophe: the model emits one
    unbroken string and every word-level score is meaningless."""
    spaced, _ = build_vocab_from_texts(["a b"])
    require_space_token(spaced)  # does not raise

    with pytest.raises(ValueError, match="no space character"):
        require_space_token(CtcVocab(id_to_char=["<blank>", "<unk>", "a", "b"]))


def test_require_space_token_rejects_a_space_that_is_the_blank() -> None:
    with pytest.raises(ValueError, match="reserved"):
        require_space_token(CtcVocab(id_to_char=[" ", "<unk>", "a"]))
