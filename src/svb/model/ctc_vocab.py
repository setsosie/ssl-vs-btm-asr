"""Lean character-level CTC vocabulary.

Deliberately simple and transparent: blank at index 0, unk at index 1, then one
id per character observed in the normalized training text. No external
tokenizer — this avoids the Wav2Vec2 tokenizer's silent upper/lower-casing
quirks and makes the vocab fully inspectable for reproducibility.

The space is an ordinary character, not a special delimiter token. For CTC there
is no functional difference, and the literal space means a decode is directly
scoreable with no post-processing step to get wrong.

A vocabulary means nothing without the :class:`~svb.text.normalize.NormalizerPolicy`
that produced it, so the policy travels with it — in the object, in the saved
file, and as a guard on expansion. Building a held-out language's characters
under a different policy than the training vocabulary used would enter them in
one form while scoring produces another, and nothing would fail loudly.

Head expansion (``expand_vocab``) appends characters from a new language while
preserving the existing ids, so a trained CTC head keeps its learned rows and
only the new rows are randomly initialised — the standard recipe for adapting a
multilingual head to an unseen script.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..text.normalize import DEFAULT_POLICY, LEGACY_POLICY, NormalizerPolicy, normalize_text

BLANK = "<blank>"
UNK = "<unk>"

# Bumped when the saved layout changes. Absent from a file means the original
# bare-list format.
VOCAB_FILE_FORMAT = 1


@dataclass
class CtcVocab:
    id_to_char: list[str]
    char_to_id: dict[str, int] = field(default_factory=dict)
    policy: NormalizerPolicy = DEFAULT_POLICY

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
        """Map already-normalized text to ids.

        This does **not** normalize. Callers normalize once, at the vocab build,
        the collate, or the scorer, with an explicit policy; a normalizer hidden
        inside the encoder is how those sites drift apart.
        """
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        """CTC greedy decode: collapse repeats, drop blanks."""
        out: list[str] = []
        prev = -1
        for i in ids:
            i = int(i)
            if i != prev and i != self.blank_id and 0 <= i < self.size and i != self.unk_id:
                out.append(self.id_to_char[i])
            prev = i
        return "".join(out)

    def save(self, path: str | Path) -> None:
        payload = {
            "format": VOCAB_FILE_FORMAT,
            "policy": self.policy.to_dict(),
            "id_to_char": self.id_to_char,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CtcVocab:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            # Written before this policy existed: NFC on the targets, nothing
            # else. Load it, and let it say so rather than claim the current
            # policy produced it.
            return cls(id_to_char=raw, policy=LEGACY_POLICY)
        return cls(
            id_to_char=raw["id_to_char"],
            policy=NormalizerPolicy.from_dict(raw["policy"]),
        )


def require_space_token(vocab: CtcVocab) -> None:
    """Fail loudly for a vocabulary that cannot separate words.

    Called for presets whose languages are written with spaces. Without the
    space character the model emits one unbroken string, every word-level score
    is meaningless, and nothing about the failure looks like a bug — it looks
    like a model that has not converged.
    """
    if " " not in vocab.char_to_id:
        raise ValueError(
            "vocabulary has no space character, so it cannot represent word "
            "boundaries; every WER computed against it would be meaningless"
        )
    if vocab.char_to_id[" "] == vocab.blank_id:
        raise ValueError("vocabulary id 0 is reserved for the CTC blank, not the space character")


def _counts(texts: Iterable[str], policy: NormalizerPolicy) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(normalize_text(text, policy))
    return counts


def build_vocab_from_texts(
    texts: Iterable[str],
    policy: NormalizerPolicy = DEFAULT_POLICY,
    min_char_count: int = 1,
) -> tuple[CtcVocab, dict[str, int]]:
    """Build a char vocab from training transcripts (sorted for determinism).

    Args:
        texts: Raw transcripts; they are normalized here, under ``policy``.
        policy: The normalization policy, stored on the returned vocab.
        min_char_count: Corpus-wide occurrences a character needs to earn a
            vocabulary slot. The default keeps everything, which is right for
            smoke runs where a legitimate character may also be rare; a full run
            raises it to evict the stray characters that arrive in corpora with
            no collection-time validation.

    Returns:
        The vocab, and the ``{character: count}`` map of what the floor evicted
        so the run can record it rather than change the vocabulary silently.
    """
    counts = _counts(texts, policy)
    evicted = {c: n for c, n in counts.items() if n < min_char_count}
    kept = sorted(c for c, n in counts.items() if n >= min_char_count)
    if not kept:
        raise ValueError(
            f"min_char_count={min_char_count} evicted every character "
            f"({len(counts)} distinct seen); refusing to build a degenerate vocab"
        )
    return CtcVocab(id_to_char=[BLANK, UNK, *kept], policy=policy), evicted


def expand_vocab(
    vocab: CtcVocab,
    new_texts: Iterable[str],
    policy: NormalizerPolicy | None = None,
    min_char_count: int = 1,
) -> tuple[CtcVocab, list[int], dict[str, int]]:
    """Append unseen characters from ``new_texts``, preserving existing ids.

    Args:
        vocab: The training vocab whose head rows must keep their meaning.
        new_texts: Raw transcripts for the new language.
        policy: Defaults to the vocab's own. Passing a different one is refused:
            the new language's characters would enter the head in one form while
            scoring produced another, and nothing would fail.
        min_char_count: Floor for *new* characters only. A character the head
            already has a row for is never evicted, since dropping it would
            renumber ids the checkpoint depends on.

    Returns:
        The expanded vocab, the newly added ids (so the caller can random-init
        exactly those rows), and what the floor evicted.
    """
    policy = policy or vocab.policy
    if policy.policy_hash() != vocab.policy.policy_hash():
        raise ValueError(
            f"policy mismatch: the vocab was built with {vocab.policy.version}/"
            f"{vocab.policy.policy_hash()} and expansion asked for {policy.version}/"
            f"{policy.policy_hash()}; the new language's characters would be stored in a "
            "different form than scoring produces"
        )

    existing = set(vocab.id_to_char)
    counts = _counts(new_texts, policy)
    fresh = {c: n for c, n in counts.items() if c not in existing}
    evicted = {c: n for c, n in fresh.items() if n < min_char_count}
    added_chars = sorted(c for c, n in fresh.items() if n >= min_char_count)

    new_vocab = CtcVocab(id_to_char=[*vocab.id_to_char, *added_chars], policy=policy)
    return new_vocab, [new_vocab.char_to_id[c] for c in added_chars], evicted
