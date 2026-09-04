"""Which Common Voice rows a training run reads, and what the speaker guard removes.

The synthetic language in `make_cv_lang` reproduces the shape a real release
has: `validated.tsv` holds every validated clip, the three split files hold the
speaker-disjoint partition Corpora Creator carved out of the *deduplicated*
validated frame, and three clips are in `validated` and in no split at all.
Two of those three belong to the dev and test speakers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from svb.data.commonvoice_local import (
    CommonVoiceLocal,
    load_cv_rows,
    load_cv_texts,
    select_train_rows,
)


def _clips(rows: list[tuple[str, str]]) -> set[str]:
    return {clip for clip, _ in rows}


# --------------------------------------------------------------------------- #
# The two sources
# --------------------------------------------------------------------------- #


def test_the_legacy_source_reads_the_official_train_split_and_nothing_else(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    selection = select_train_rows("en", train_source="train")

    assert _clips(list(selection.rows)) == {"v1.mp3", "v2.mp3"}
    assert selection.train_source == "train"


def test_validated_minus_eval_picks_up_the_clips_the_partition_left_out(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    """Corpora Creator splits the *deduplicated* validated frame, so a language's
    extra recordings of an already-covered sentence end up in `validated.tsv`
    and in no split. Those are the hours this mode is for."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    selection = select_train_rows("en", train_source="validated_minus_eval")

    assert _clips(list(selection.rows)) == {"v1.mp3", "v2.mp3", "v3.mp3"}


def test_a_clip_that_is_already_in_dev_or_test_is_never_trained_on(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    selection = select_train_rows("en", train_source="validated_minus_eval")

    assert not {"v4.mp3", "v6.mp3"} & _clips(list(selection.rows))
    assert selection.n_in_eval_splits == 2


def test_another_recording_by_an_evaluation_speaker_is_never_trained_on(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    """This is the whole reason the mode is not just a set difference on paths.

    `v5` and `v7` are in `validated.tsv`, are in no split, and belong to the dev
    and test speakers. Subtracting dev and test by clip path leaves them in, and
    the run would then be trained on the voices it is evaluated on.
    """
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    selection = select_train_rows("en", train_source="validated_minus_eval")

    assert not {"v5.mp3", "v7.mp3"} & _clips(list(selection.rows))
    assert selection.n_eval_speaker_clips == 2
    assert selection.n_eval_speakers == 2


def test_the_official_train_split_survives_the_guard_whole(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    """Corpora Creator labels a whole speaker at a time, so no training speaker
    is also a dev or test speaker and the new mode can only ever be a superset
    of the old one. A release where that stopped holding would show up here."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    legacy = select_train_rows("en", train_source="train")
    wide = select_train_rows("en", train_source="validated_minus_eval")

    assert _clips(list(legacy.rows)) <= _clips(list(wide.rows))


def test_the_selection_accounts_for_every_validated_clip(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    """Kept plus removed has to equal what was read, or the accounting is a
    number with no meaning attached."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    s = select_train_rows("en", train_source="validated_minus_eval")

    assert s.n_validated == 7
    assert len(s.rows) + s.n_in_eval_splits + s.n_eval_speaker_clips == s.n_validated


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_the_wide_source_says_which_file_is_missing(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang(with_validated=False)))

    with pytest.raises(FileNotFoundError, match=r"validated\.tsv"):
        select_train_rows("en", train_source="validated_minus_eval")


def test_a_release_without_client_ids_is_refused_rather_than_trained_on(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    """With no `client_id` the speaker guard cannot run.

    Carrying on without it would quietly train on the evaluation speakers, which
    is the failure the mode exists to avoid, so it stops instead.
    """
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang(columns=("path", "sentence"))))

    with pytest.raises(ValueError, match="client_id"):
        select_train_rows("en", train_source="validated_minus_eval")


def test_an_unknown_train_source_names_the_ones_that_exist(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    with pytest.raises(ValueError, match="validated_minus_eval"):
        select_train_rows("en", train_source="everything")


# --------------------------------------------------------------------------- #
# The readers all agree
# --------------------------------------------------------------------------- #


def test_dev_and_test_do_not_move_with_the_train_source(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    for source in ("train", "validated_minus_eval"):
        assert _clips(load_cv_rows("en", "validation", train_source=source)) == {"v4.mp3"}
        assert _clips(load_cv_rows("en", "test", train_source=source)) == {"v6.mp3"}


def test_the_transcripts_the_vocabulary_is_built_from_follow_the_train_source(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    """The vocabulary, the training targets and the audio have to come from one
    row set. Building the vocabulary from `train.tsv` while training on the
    wider set would leave characters in the targets with no id."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    assert sorted(load_cv_texts("en", "train", train_source="train")) == ["one", "two"]
    assert sorted(load_cv_texts("en", "train", train_source="validated_minus_eval")) == [
        "one",
        "three",
        "two",
    ]


def test_the_dataset_reads_the_rows_the_selection_chose(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))

    assert len(CommonVoiceLocal("en", "train", train_source="train")) == 2
    assert len(CommonVoiceLocal("en", "train", train_source="validated_minus_eval")) == 3


def test_a_row_the_wider_source_added_still_carries_its_language_code(
    make_cv_lang: Callable[..., Path], monkeypatch
) -> None:
    """The two things this loader does have to hold at once.

    The training rows come from validated minus the evaluation splits, and every
    item carries the language it came from so the collate can normalize it under
    that language's own policy. A change to either one can quietly drop the
    other, and `v3` — a row only the wider source selects — is where that would
    show.
    """
    import torchaudio

    monkeypatch.setenv("CV_ROOT", str(make_cv_lang()))
    monkeypatch.setattr(torchaudio, "load", lambda path: (torch.zeros(1, 160), 16000))

    dataset = CommonVoiceLocal("en", "train", train_source="validated_minus_eval")
    items = [dataset[i] for i in range(len(dataset))]

    assert {text for _, text, _ in items} == {"one", "two", "three"}
    assert {lang for _, _, lang in items} == {"en"}
