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

# The key a single-policy vocabulary stores its policy under, and the fallback
# ``policy_for`` uses when a language has no entry of its own. Named rather than
# implicit so a vocabulary built over one language, or loaded from a file
# written before policies were per-language, says which case it is.
ANY_LANGUAGE = "*"

BLANK = "<blank>"
UNK = "<unk>"

# Bumped when the saved layout changes. Absent from a file means the original
# bare-list format.
VOCAB_FILE_FORMAT = 1


@dataclass
class CtcVocab:
    id_to_char: list[str]
    char_to_id: dict[str, int] = field(default_factory=dict)
    # One policy per language. A language's characters were derived under its
    # own policy, so encoding its targets or scoring its hypotheses under
    # another would put them in a form the vocabulary does not hold.
    policies: dict[str, NormalizerPolicy] = field(
        default_factory=lambda: {ANY_LANGUAGE: DEFAULT_POLICY}
    )

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

    def policy_for(self, code: str) -> NormalizerPolicy:
        """The policy this language's text is normalized under.

        Raises:
            KeyError: When the vocabulary has no policy for that language and no
                single-policy fallback. Loud, because the alternative is
                encoding one language's targets under another's rules.
        """
        if code in self.policies:
            return self.policies[code]
        if ANY_LANGUAGE in self.policies:
            return self.policies[ANY_LANGUAGE]
        known = ", ".join(sorted(self.policies)) or "none"
        raise KeyError(
            f"this vocabulary has no normalization policy for {code!r}; it holds: {known}"
        )

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
            "policies": {code: p.to_dict() for code, p in sorted(self.policies.items())},
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
            return cls(id_to_char=raw, policies={ANY_LANGUAGE: LEGACY_POLICY})
        if "policy" in raw:
            # One policy for the whole vocabulary, which is what files written
            # before policies were per-language carry.
            return cls(
                id_to_char=raw["id_to_char"],
                policies={ANY_LANGUAGE: NormalizerPolicy.from_dict(raw["policy"])},
            )
        return cls(
            id_to_char=raw["id_to_char"],
            policies={code: NormalizerPolicy.from_dict(p) for code, p in raw["policies"].items()},
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


def _counts(
    items: Iterable[tuple[str, str]], policies: dict[str, NormalizerPolicy]
) -> Counter[str]:
    """Count characters over (language, transcript) pairs, each under its own policy."""
    counts: Counter[str] = Counter()
    for code, text in items:
        policy = policies.get(code) or policies[ANY_LANGUAGE]
        counts.update(normalize_text(text, policy))
    return counts


def build_vocab_from_labelled_texts(
    items: Iterable[tuple[str, str]],
    policies: dict[str, NormalizerPolicy],
    min_char_count: int = 1,
) -> tuple[CtcVocab, dict[str, int]]:
    """Build a char vocab over several languages, each under its own policy.

    The languages share one character inventory but not one set of rules, so the
    text is normalized per language *before* the characters are pooled. Pooling
    first and normalizing after would apply one language's rules to another's
    script, which is the whole thing the per-script policies exist to avoid.

    Args:
        items: ``(language code, raw transcript)`` pairs.
        policies: One policy per language code. A ``"*"`` entry serves any
            language without one of its own.
        min_char_count: Corpus-wide occurrences a character needs to earn a slot.

    Returns:
        The vocab, and the ``{character: count}`` map of what the floor evicted.
    """
    counts = _counts(items, policies)
    evicted = {c: n for c, n in counts.items() if n < min_char_count}
    kept = sorted(c for c, n in counts.items() if n >= min_char_count)
    if not kept:
        raise ValueError(
            f"min_char_count={min_char_count} evicted every character "
            f"({len(counts)} distinct seen); refusing to build a degenerate vocab"
        )
    return CtcVocab(id_to_char=[BLANK, UNK, *kept], policies=dict(policies)), evicted


def build_vocab_from_texts(
    texts: Iterable[str],
    policy: NormalizerPolicy = DEFAULT_POLICY,
    min_char_count: int = 1,
) -> tuple[CtcVocab, dict[str, int]]:
    """One language, or several sharing one policy. See the labelled form."""
    return build_vocab_from_labelled_texts(
        ((ANY_LANGUAGE, t) for t in texts), {ANY_LANGUAGE: policy}, min_char_count
    )


def expand_vocab(
    vocab: CtcVocab,
    new_texts: Iterable[str],
    code: str = ANY_LANGUAGE,
    policy: NormalizerPolicy | None = None,
    min_char_count: int = 1,
) -> tuple[CtcVocab, list[int], dict[str, int]]:
    """Append unseen characters from ``new_texts``, preserving existing ids.

    Args:
        vocab: The training vocab whose head rows must keep their meaning.
        new_texts: Raw transcripts for the new language.
        code: The language being added. Its policy joins the vocabulary's map,
            so a held-out language is scored under the rules its own characters
            were derived from rather than under a training language's.
        policy: Defaults to whatever the vocab already holds for ``code``.
            Passing a different one for a language the vocab already knows is
            refused: its characters would enter the head in one form while
            scoring produced another, and nothing would fail.
        min_char_count: Floor for *new* characters only. A character the head
            already has a row for is never evicted, since dropping it would
            renumber ids the checkpoint depends on.

    Returns:
        The expanded vocab, the newly added ids, and what the floor evicted.
    """
    known = vocab.policies.get(code)
    policy = policy or known or vocab.policy_for(code)
    if known is not None and policy.policy_hash() != known.policy_hash():
        raise ValueError(
            f"policy mismatch for {code!r}: the vocab was built with {known.version}/"
            f"{known.policy_hash()} and expansion asked for {policy.version}/"
            f"{policy.policy_hash()}; that language's characters would be stored in a "
            "different form than scoring produces"
        )

    existing = set(vocab.id_to_char)
    counts = _counts(((code, t) for t in new_texts), {code: policy, ANY_LANGUAGE: policy})
    fresh = {c: n for c, n in counts.items() if c not in existing}
    evicted = {c: n for c, n in fresh.items() if n < min_char_count}
    added_chars = sorted(c for c, n in fresh.items() if n >= min_char_count)

    new_vocab = CtcVocab(
        id_to_char=[*vocab.id_to_char, *added_chars],
        policies={**vocab.policies, code: policy},
    )
    return new_vocab, [new_vocab.char_to_id[c] for c in added_chars], evicted
