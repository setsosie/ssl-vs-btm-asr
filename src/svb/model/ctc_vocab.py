"""Lean character-level CTC vocabulary.

Deliberately simple and transparent: blank at index 0, unk at index 1, then one
id per character observed in the training text (NFC-normalized). No external
tokenizer — this avoids the Wav2Vec2 tokenizer's silent upper/lower-casing
quirks and makes the vocab fully inspectable for reproducibility.

Head expansion (``expand_vocab``) appends characters from a new language while
preserving the existing ids, so a trained CTC head keeps its learned rows and
only the new rows are randomly initialised — the standard recipe for adapting a
multilingual head to an unseen script.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

BLANK = "<blank>"
UNK = "<unk>"


@dataclass
class CtcVocab:
    id_to_char: list[str]
    char_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.char_to_id:
            self.char_to_id = {c: i for i, c in enumerate(self.id_to_char)}

    @property
    def size(self) -> int:
        return len(self.id_to_char)

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        """CTC greedy decode: collapse repeats, drop blanks."""
        out: list[str] = []
        prev = -1
        for i in ids:
            i = int(i)
            if i != prev and i != self.blank_id:
                if 0 <= i < self.size and i != self.unk_id:
                    out.append(self.id_to_char[i])
            prev = i
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.id_to_char, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CtcVocab:
        return cls(id_to_char=json.loads(Path(path).read_text()))


def build_vocab_from_texts(texts: Iterable[str]) -> CtcVocab:
    """Build a char vocab from training transcripts (sorted for determinism)."""
    chars: set[str] = set()
    for t in texts:
        chars.update(unicodedata.normalize("NFC", t))
    id_to_char = [BLANK, UNK] + sorted(chars)
    return CtcVocab(id_to_char=id_to_char)


def expand_vocab(vocab: CtcVocab, new_texts: Iterable[str]) -> tuple[CtcVocab, list[int]]:
    """Append unseen characters from ``new_texts``, preserving existing ids.

    Returns the expanded vocab and the list of newly added ids (so the caller
    can random-init exactly those rows of the CTC head).
    """
    existing = set(vocab.id_to_char)
    new_chars: set[str] = set()
    for t in new_texts:
        for c in unicodedata.normalize("NFC", t):
            if c not in existing:
                new_chars.add(c)
    added_chars = sorted(new_chars)
    id_to_char = list(vocab.id_to_char) + added_chars
    new_vocab = CtcVocab(id_to_char=id_to_char)
    added_ids = [new_vocab.char_to_id[c] for c in added_chars]
    return new_vocab, added_ids
