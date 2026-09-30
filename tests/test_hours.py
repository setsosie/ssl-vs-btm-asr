"""The hours cap: which utterances a capped split keeps, and what it records."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from torch.utils.data import Subset

from svb.data.datasets import hours_cap_selection, load_language, load_texts
from svb.data.hours import select_within_hours
from svb.data.registry import LangSpec

from .conftest import CvClip

HOUR = 3600.0


def _rows(n: int, seconds: float = 60.0) -> list[tuple[str, float | None]]:
    return [(f"clip{i}", seconds) for i in range(n)]


def test_the_cap_is_respected_and_reported() -> None:
    selection = select_within_hours(_rows(100), max_hours=0.5, salt="en")

    assert selection.n_selected == 30
    assert selection.seconds_selected == pytest.approx(0.5 * HOUR)
    assert selection.binding
    record = selection.to_record()
    assert record["hours_available"] == pytest.approx(100 / 60)
    assert record["n_available"] == 100


def test_a_cap_larger_than_the_split_keeps_everything() -> None:
    selection = select_within_hours(_rows(10), max_hours=5.0, salt="en")

    assert selection.indices == tuple(range(10))
    assert not selection.binding


def test_a_smaller_cap_is_a_subset_of_a_larger_one() -> None:
    """What lets a later study at more hours extend this one rather than replace it."""
    rows = [(f"clip{i}", 30.0 + (i % 7) * 11.0) for i in range(200)]

    small = select_within_hours(rows, max_hours=0.25, salt="en")
    large = select_within_hours(rows, max_hours=1.0, salt="en")

    assert set(small.indices) < set(large.indices)


def test_the_selection_ignores_the_order_rows_arrive_in() -> None:
    rows = _rows(50)
    shuffled = rows[::-1]

    forward = select_within_hours(rows, max_hours=0.2, salt="en")
    backward = select_within_hours(shuffled, max_hours=0.2, salt="en")

    assert {rows[i][0] for i in forward.indices} == {shuffled[i][0] for i in backward.indices}
    assert forward.subset_sha1 == backward.subset_sha1


def test_two_languages_with_the_same_clip_names_are_ranked_differently() -> None:
    rows = _rows(200)

    english = select_within_hours(rows, max_hours=0.5, salt="en")
    japanese = select_within_hours(rows, max_hours=0.5, salt="ja")

    assert english.indices != japanese.indices


def test_a_row_with_no_readable_duration_is_left_out_and_counted() -> None:
    rows: list[tuple[str, float | None]] = [("a", 60.0), ("b", None), ("c", 60.0)]

    selection = select_within_hours(rows, max_hours=1.0, salt="en")

    assert selection.indices == (0, 2)
    assert selection.n_unknown_duration == 1
    # Everything that could be measured was kept, so the cap did not bind.
    assert not selection.binding


def test_a_clip_of_no_length_is_not_taken_however_well_it_fits() -> None:
    """Zero seconds always fits a budget and contributes nothing but a batch
    the encoder cannot process. It is treated like a duration that could not
    be read."""
    rows: list[tuple[str, float | None]] = [("a", 60.0), ("b", 0.0), ("c", 60.0)]

    selection = select_within_hours(rows, max_hours=1.0, salt="en")

    assert selection.indices == (0, 2)
    assert selection.n_unknown_duration == 1


def test_a_cap_that_fits_nothing_is_an_error_not_an_empty_split() -> None:
    with pytest.raises(ValueError, match="nothing fits"):
        select_within_hours(_rows(5, seconds=120.0), max_hours=0.01, salt="en")


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_a_cap_must_be_positive(cap: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        select_within_hours(_rows(5), max_hours=cap, salt="en")


def test_duplicate_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="not unique"):
        select_within_hours([("a", 1.0), ("a", 1.0)], max_hours=1.0, salt="en")


# --------------------------------------------------------------------------- #
# Through the loaders
# --------------------------------------------------------------------------- #

EN = LangSpec(code="en", source="commonvoice", hf_config="en", normalizer="whisper-basic")


@pytest.fixture
def cv(make_cv_lang: Callable[..., Path], monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [CvClip(f"t{i}.mp3", f"s{i % 3}", f"sentence {i}", "train") for i in range(8)]
    rows += [CvClip("d0.mp3", "s_dev", "dev sentence", "dev")]
    rows += [CvClip("e0.mp3", "s_test", "test sentence", "test")]
    monkeypatch.setenv("CV_ROOT", str(make_cv_lang("en", rows, clip_seconds=1.0)))


def test_the_dataset_and_the_transcripts_are_cut_to_the_same_rows(cv: None) -> None:
    """The vocabulary is built from the transcripts and the model trains on the
    dataset; if the cap selected differently for the two, the vocabulary would
    describe a run that did not happen."""
    cap = 4.5 / HOUR

    dataset = load_language(EN, "train", max_hours=cap)
    texts = load_texts(EN, "train", max_hours=cap)

    assert isinstance(dataset, Subset)
    assert len(dataset) == 4
    assert [dataset[i][1] for i in range(len(dataset))] == texts


def test_durations_come_from_the_release_manifest_not_from_decoding(
    cv: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading 1.8 million mp3 headers to cap English would take longer than the cap saves."""
    import svb.data.commonvoice_local as commonvoice

    def no_headers(path: object) -> float | None:
        raise AssertionError("an audio header was read although clip_durations.tsv exists")

    monkeypatch.setattr(commonvoice, "audio_seconds", no_headers)

    assert hours_cap_selection(EN, "train", 2.5 / HOUR).n_selected == 2


def test_without_the_manifest_the_headers_are_read_instead(cv: None, tmp_path: Path) -> None:
    (tmp_path / "cv" / "en" / "clip_durations.tsv").unlink()

    selection = hours_cap_selection(EN, "train", 2.5 / HOUR)

    assert selection.n_selected == 2
    assert selection.n_unknown_duration == 0


def test_the_test_split_is_never_capped(cv: None) -> None:
    with pytest.raises(ValueError, match="never capped"):
        load_language(EN, "test", max_hours=1.0)
