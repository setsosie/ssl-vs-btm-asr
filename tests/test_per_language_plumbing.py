"""Two languages that need different rules, in one vocabulary and one batch.

This is the case a single global policy cannot serve and the reason the language
travels with the item: the joint phase-0 loader concatenates every language into
one dataset, so a batch holds several at once and the rules have to be chosen
per item rather than per batch.
"""

from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, Dataset

from svb.data.collate import make_ctc_collate
from svb.model.ctc_vocab import build_vocab_from_labelled_texts, expand_vocab
from svb.text.registry import policy_for_language

# Hindi keeps its combining marks; English is scored the way Whisper scores it.
HINDI = "यह हिंदी है।"
ENGLISH = "It's Bob's."
POLICIES = {
    "hi": policy_for_language("hi", "indic-vistaar"),
    "en": policy_for_language("en", "whisper-basic"),
}


class OneLanguage(Dataset):
    def __init__(self, code: str, text: str, n: int = 2) -> None:
        self.items = [(torch.ones(4), text, code) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        return self.items[idx]


def _vocab():
    return build_vocab_from_labelled_texts([("hi", HINDI), ("en", ENGLISH)], POLICIES)


def test_the_vocabulary_normalizes_each_language_under_its_own_policy() -> None:
    """Pooling the characters first and normalizing after would apply one
    language's rules to the other's script."""
    vocab, _ = _vocab()
    chars = set(vocab.id_to_char[2:])

    # Hindi matras survive, because the Indic policy keeps category M.
    assert "ि" in chars and "ं" in chars
    # The danda is deleted by that policy, and English punctuation by Whisper's.
    assert "।" not in chars
    assert "," not in chars and "." not in chars
    # Whisper does not protect the apostrophe, so English contributes none.
    assert "'" not in chars


def test_one_batch_of_two_languages_normalizes_each_item_correctly() -> None:
    vocab, _ = _vocab()
    collate = make_ctc_collate(vocab)
    mixed: ConcatDataset[tuple[torch.Tensor, str, str]] = ConcatDataset(
        [OneLanguage("hi", HINDI, 1), OneLanguage("en", ENGLISH, 1)]
    )

    batch = collate([mixed[i] for i in range(len(mixed))])

    assert batch["codes"] == ["hi", "en"]
    # The danda goes and the marks stay; the apostrophe is split, not protected.
    assert batch["texts"] == ["यह हिंदी है", "it s bob s "]


def test_the_vocabulary_reports_which_policy_each_language_uses() -> None:
    vocab, _ = _vocab()

    assert vocab.policy_for("hi").version == "indic-vistaar"
    assert vocab.policy_for("en").version == "whisper-basic"
    assert vocab.policy_for("hi") != vocab.policy_for("en")


def test_a_language_the_vocabulary_never_saw_is_refused_not_guessed() -> None:
    import pytest

    vocab, _ = _vocab()
    with pytest.raises(KeyError, match="no normalization policy for"):
        vocab.policy_for("ja")


def test_a_held_out_language_brings_its_own_policy_into_the_vocabulary() -> None:
    """Its characters were derived under its rules, so scoring must use them."""
    vocab, _ = _vocab()
    telugu = policy_for_language("telugu")

    expanded, added, _ = expand_vocab(vocab, ["ఇది తెలుగు భాష."], "telugu", telugu)

    assert expanded.policy_for("telugu").version == "indic-vistaar"
    assert expanded.policy_for("en").version == "whisper-basic"
    assert added
    # Existing rows keep their meaning.
    for char in vocab.id_to_char:
        assert expanded.char_to_id[char] == vocab.char_to_id[char]


def test_changing_a_known_languages_policy_on_expansion_is_refused() -> None:
    import pytest

    vocab, _ = _vocab()
    with pytest.raises(ValueError, match="policy mismatch for 'hi'"):
        expand_vocab(vocab, [HINDI], "hi", policy_for_language("en", "whisper-basic"))
