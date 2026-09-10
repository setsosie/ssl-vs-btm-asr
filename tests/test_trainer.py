"""Trainer: checkpoint selection, loader construction, and worker seeding."""

from __future__ import annotations

import math
import random

import pytest
import torch

from svb.config import ExperimentConfig, load_config
from svb.data.collate import make_ctc_collate
from svb.model.ctc_vocab import build_vocab_from_texts
from svb.train.trainer import train

from .conftest import ScriptedLossCTC
from .test_evaluate import ListDataset


def _cfg(**train_overrides: object) -> ExperimentConfig:
    train: dict[str, object] = {
        "num_workers": 0,
        "patience": 1,
        "bf16": False,
        "grad_checkpointing": False,
    }
    train.update(train_overrides)
    return load_config(
        "A_ssl",
        "3",
        seed=7,
        overrides={"optim": {"batch_size": 2, "accum_steps": 1}, "train": train},
    )


def _data(n: int) -> ListDataset:
    return ListDataset([(torch.ones(2000), "abc") for _ in range(n)])


def test_a_checkpoint_exists_even_when_every_epoch_is_nan(tmp_path) -> None:
    """A run whose val loss never becomes finite must still leave a best.pt.

    Selection compares ``val_loss < best_val``, which is False for NaN, so with
    no floor the file is never written — and the caller then loads it and dies
    with FileNotFoundError after a full training run.
    """
    vocab, _ = build_vocab_from_texts(["abc"])
    model = ScriptedLossCTC([math.nan, math.nan])

    result = train(
        model,
        _cfg(),
        _data(4),
        _data(2),
        make_ctc_collate(vocab),
        max_epochs=2,
        out_dir=tmp_path,
        device="cpu",
    )

    assert result.checkpoint.exists()
    assert result.best_epoch == -1  # the floor is a fallback, not a selection


def test_finite_validation_loss_still_selects(tmp_path) -> None:
    vocab, _ = build_vocab_from_texts(["abc"])
    model = ScriptedLossCTC([5.0, 2.0])

    result = train(
        model,
        _cfg(patience=5),
        _data(2),
        _data(2),
        make_ctc_collate(vocab),
        max_epochs=2,
        out_dir=tmp_path,
        device="cpu",
    )

    assert result.best_epoch == 1
    assert result.best_val_loss == pytest.approx(2.0)


def test_validation_loss_is_weighted_by_batch_size(tmp_path) -> None:
    """Three utterances at batch size 2 give batches of 2 and 1.

    Averaging the two batch means would report 2.5; weighting by utterance
    count reports 2.0, which is the actual mean loss over the split.
    """
    vocab, _ = build_vocab_from_texts(["abc"])
    model = ScriptedLossCTC([1.0, 4.0])

    result = train(
        model,
        _cfg(),
        _data(2),
        _data(3),
        make_ctc_collate(vocab),
        max_epochs=1,
        out_dir=tmp_path,
        device="cpu",
    )

    assert result.best_val_loss == pytest.approx(2.0)


def test_a_split_smaller_than_the_batch_still_trains(tmp_path) -> None:
    """drop_last must not silently discard the only batch a language has.

    With drop_last always on, a split of fewer than batch_size utterances
    yields zero steps: the LR schedule collapses to zero and the run reports a
    completed training that touched no data.
    """
    vocab, _ = build_vocab_from_texts(["abc"])
    model = ScriptedLossCTC([1.0])

    train(
        model,
        _cfg(),
        _data(1),
        _data(1),
        make_ctc_collate(vocab),
        max_epochs=1,
        out_dir=tmp_path,
        device="cpu",
    )

    assert model.train_forwards == 1


def test_an_empty_training_split_fails_loudly(tmp_path) -> None:
    vocab, _ = build_vocab_from_texts(["abc"])
    model = ScriptedLossCTC([1.0])

    with pytest.raises(ValueError, match="empty"):
        train(
            model,
            _cfg(),
            _data(0),
            _data(1),
            make_ctc_collate(vocab),
            max_epochs=1,
            out_dir=tmp_path,
            device="cpu",
        )


def test_inert_early_stopping_is_announced(tmp_path, capsys) -> None:
    """patience >= max_epochs cannot fire; say so rather than implying it can."""
    vocab, _ = build_vocab_from_texts(["abc"])
    model = ScriptedLossCTC([1.0])

    train(
        model,
        _cfg(patience=15),
        _data(2),
        _data(2),
        make_ctc_collate(vocab),
        max_epochs=9,
        out_dir=tmp_path,
        device="cpu",
    )

    assert "early stopping" in capsys.readouterr().out


def test_worker_seeding_is_per_worker_and_reproducible() -> None:
    """Each loader worker needs its own stream, derived from the run seed.

    Without an explicit worker_init_fn every worker inherits the same base
    seed, so any per-sample randomness repeats across workers.
    """
    from svb.train.trainer import make_worker_init_fn

    init = make_worker_init_fn(7)

    init(0)
    first = random.random()
    init(1)
    second = random.random()
    init(0)
    again = random.random()

    assert first != second
    assert first == again


def test_dropped_training_pairs_reach_the_result_not_only_the_log(tmp_path) -> None:
    """A pair whose transcript cannot be aligned to its audio is removed from
    training, and an utterance clipped by the audio guard is trained on
    partially, so reading these from the log post hoc is a chore."""
    vocab, _ = build_vocab_from_texts(["a"])

    # b1: 320 samples, text "aa". Downsampling by 320 gives max label len 1.
    # "aa" has len 2 > 1, so it is unalignable and dropped.
    # b2: 640 samples, max_audio_samples 400 -> hits audio guard.
    train_ds = ListDataset([
        (torch.ones(320), "aa"),
        (torch.ones(640), "a"),
    ])
    val_ds = ListDataset([(torch.ones(320), "a")])

    collate = make_ctc_collate(
        vocab,
        drop_overlong=True,
        max_audio_samples=400,
    )

    result = train(
        ScriptedLossCTC([1.0]),
        _cfg(),
        train_ds,
        val_ds,
        collate,
        max_epochs=1,
        out_dir=tmp_path,
        device="cpu",
    )

    assert result.n_dropped_unalignable == 1
    assert result.n_at_audio_guard == 1
