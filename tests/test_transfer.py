"""Held-out transfer: the split provenance that travels with the result."""

from __future__ import annotations

import pytest

from svb.eval.evaluate import EvalResult
from svb.eval.transfer import TransferResult, _split_provenance


class _Derived:
    """Stands in for OpenSLRLocal, which derives its own split."""

    split_policy = "speaker"
    test_files_sha1 = "0" * 40


class _Shipped:
    """Stands in for a corpus that ships its own partition."""


def test_split_provenance_is_read_off_a_deriving_dataset() -> None:
    assert _split_provenance(_Derived()) == ("speaker", "0" * 40)


def test_split_provenance_is_absent_rather_than_invented() -> None:
    """A loader that ships its own split has no derivation to report."""
    assert _split_provenance(_Shipped()) == (None, None)


@pytest.mark.parametrize("bogus", [object(), 17])
def test_non_string_markers_are_treated_as_absent(bogus: object) -> None:
    class Odd:
        split_policy = bogus
        test_files_sha1 = bogus

    assert _split_provenance(Odd()) == (None, None)


def test_the_results_record_pins_which_utterances_were_scored() -> None:
    """A held-out number is only interpretable next to the split that produced it.

    The corpora ship no partition, so an utterance-level fallback — the same
    speaker in train and test — reads exactly like a speaker-disjoint result
    unless the policy is recorded beside the metric.
    """
    record = TransferResult(
        lang="telugu",
        result=EvalResult(wer=12.5, cer=3.25, n=40, n_empty_refs=2),
        split_policy="speaker",
        test_files_sha1="a" * 40,
    ).to_record()

    assert record == {
        "wer": 12.5,
        "cer": 3.25,
        "n": 40,
        "n_empty_refs": 2,
        "split_policy": "speaker",
        "test_files_sha1": "a" * 40,
    }


def test_a_missing_derivation_is_recorded_as_null_not_omitted() -> None:
    """The keys stay present so a reader can tell "not derived" from "not recorded"."""
    record = TransferResult(lang="x", result=EvalResult(wer=1.0, cer=1.0, n=1)).to_record()

    assert record["split_policy"] is None
    assert record["test_files_sha1"] is None
