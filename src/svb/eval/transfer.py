"""Held-out-language transfer (finding #3).

Given a starting point — the SSL encoder (arm A), or a merged BTM checkpoint
(arms B/C) — adapt to a language the supervised pipeline never trained on:

  1. expand the char vocab with the new language's characters,
  2. grow the CTC head (trained rows preserved, new rows seeded-random),
  3. fine-tune on the new language,
  4. evaluate on its disjoint test split.

"Held out" means held out of *this* pipeline, not unheard by the encoder: XEUS
was pretrained on unlabelled audio from thousands of languages, the held-out
four among them. What is genuinely new to the model is the supervision and the
output vocabulary, which is why zero-shot is ~100% WER before the head is
expanded. The question is how far fine-tuning drives it down from each starting
point, and whether SSL alone matches the full BTM pipeline.

Every arm fine-tunes here for the same ``finetune_epochs``, so unlike the
in-distribution comparison this one is matched on training budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..data.collate import make_ctc_collate
from ..data.datasets import load_language, load_texts
from ..data.registry import LangSpec
from ..model.ctc_vocab import CtcVocab, expand_vocab
from ..model.xeus_ctc import make_model
from ..text.registry import policy_for_language
from ..train.trainer import train
from .evaluate import EvalResult, evaluate


@dataclass
class TransferResult:
    """One held-out language's evaluation, plus how its test split was chosen.

    The held-out corpora ship no train/validation/test partition, so the loader
    derives one. That derivation is part of the reported number and cannot be
    recovered from the number, which is why the digest and the policy travel
    with the result rather than staying an attribute of a dataset object that
    goes out of scope when ``transfer_one`` returns.
    """

    lang: str
    result: EvalResult
    #: "speaker" or "utterance" — an utterance-level split puts the same
    #: speaker in train and test, so the number is not speaker-independent.
    split_policy: str | None = None
    #: SHA-1 of the test FileID list: pins the exact held-out set that was
    #: scored, so two runs can be shown to have used the same one.
    test_files_sha1: str | None = None

    def to_record(self) -> dict[str, Any]:
        """The results.json entry for this language."""
        return {
            "wer": self.result.wer,
            "cer": self.result.cer,
            "n": self.result.n,
            "n_empty_refs": self.result.n_empty_refs,
            "split_policy": self.split_policy,
            "test_files_sha1": self.test_files_sha1,
        }


def _split_provenance(dataset: object) -> tuple[str | None, str | None]:
    """Read the derived-split markers off a dataset, if it has them.

    Only the OpenSLR loader derives a split, so these are absent for any
    held-out language read from a corpus that ships its own.
    """
    policy = getattr(dataset, "split_policy", None)
    digest = getattr(dataset, "test_files_sha1", None)
    return (
        policy if isinstance(policy, str) else None,
        digest if isinstance(digest, str) else None,
    )


def transfer_one(
    cfg: ExperimentConfig,
    init_ckpt: Path | None,
    base_vocab: CtcVocab,
    lang: LangSpec,
    out_dir: Path,
    device: str = "cuda",
) -> TransferResult:
    """Adapt to one held-out language and evaluate.

    Args:
        init_ckpt: Checkpoint to start from. ``None`` means start from the bare
            encoder, which for arm A is the SSL one (``cfg.init``).
        base_vocab: The training vocab whose head rows we preserve.
    """
    # Same policy the training vocab was built with — expand_vocab refuses
    # anything else — and the same floor, so the held-out language's tail is
    # treated the way the training languages' tails were.
    new_vocab, _, _ = expand_vocab(
        base_vocab,
        load_texts(lang, "train"),
        lang.code,
        policy_for_language(lang.code, lang.normalizer),
        min_char_count=cfg.text.min_char_count,
    )
    model = make_model(cfg, base_vocab.size)
    if init_ckpt is not None:
        model.load(init_ckpt)
    model.expand_head(new_vocab.size, seed=cfg.seed)

    train_collate = make_ctc_collate(
        new_vocab,
        max_audio_samples=cfg.train.max_audio_samples,
        drop_overlong=True,
        drop_empty=True,
    )
    eval_collate = make_ctc_collate(new_vocab)
    train_ds = load_language(lang, "train", cfg.train.max_audio_samples)
    val_ds = load_language(lang, "validation", cfg.train.max_audio_samples)
    # Load the checkpoint the trainer says it wrote, not a path re-derived here:
    # two sources of truth for one filename is how a stale model gets evaluated.
    result = train(
        model, cfg, train_ds, val_ds, train_collate, cfg.train.finetune_epochs, out_dir, device
    )
    model.load(result.checkpoint)

    # No truncation at test time: a clipped waveform scored against its full
    # transcript manufactures deletions that the model never had a chance to
    # avoid.
    test_ds = load_language(lang, "test", None)
    scored = evaluate(
        model,
        test_ds,
        new_vocab,
        eval_collate,
        device=device,
        batch_size=cfg.optim.batch_size,
        save_predictions=out_dir / "predictions.json",
        spec=lang,
    )
    policy, digest = _split_provenance(test_ds)
    return TransferResult(
        lang=lang.code, result=scored, split_policy=policy, test_files_sha1=digest
    )
