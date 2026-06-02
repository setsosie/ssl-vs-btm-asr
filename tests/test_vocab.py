"""Vocab: build, CTC decode, and head-preserving expansion."""

from svb.model.ctc_vocab import build_vocab_from_texts, expand_vocab


def test_build_and_roundtrip():
    vocab = build_vocab_from_texts(["abc", "cab"])
    assert vocab.id_to_char[:2] == ["<blank>", "<unk>"]
    assert vocab.size == 2 + 3  # blank, unk, a, b, c
    ids = vocab.encode("abc")
    assert vocab.decode(ids) == "abc"


def test_ctc_decode_collapses_repeats_and_blanks():
    vocab = build_vocab_from_texts(["ab"])
    a = vocab.char_to_id["a"]
    b = vocab.char_to_id["b"]
    blank = vocab.blank_id
    # a a <blank> a b b  ->  a a b  (collapse adj repeats, drop blanks, keep
    # the second 'a' because a blank separates the runs)
    assert vocab.decode([a, a, blank, a, b, b]) == "aab"


def test_expand_preserves_existing_ids():
    base = build_vocab_from_texts(["abc"])
    new_vocab, added = expand_vocab(base, ["abcde"])
    # existing ids unchanged
    for c in "abc":
        assert new_vocab.char_to_id[c] == base.char_to_id[c]
    # new chars appended at the end
    assert sorted(new_vocab.id_to_char[base.size :]) == ["d", "e"]
    assert added == [new_vocab.char_to_id["d"], new_vocab.char_to_id["e"]]
