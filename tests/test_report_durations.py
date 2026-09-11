"""Per-language audio accounting, from metadata only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.data.registry import LangSpec
from svb.report.durations import (
    collect_durations,
    language_durations,
    to_json,
    to_markdown,
    write_tables,
)

CV_SPEC = LangSpec(code="en", source="commonvoice", hf_config="en", hf_dataset="common_voice_25")
SLR_SPEC = LangSpec(
    code="malayalam",
    source="openslr",
    slr=63,
    archives=("ml_in_female.zip", "ml_in_male.zip"),
    index_files=("line_index_female.tsv", "line_index_male.tsv"),
    license="CC-BY-SA-4.0",
)


def make_cv_root(
    tmp_path: Path,
    fixtures_dir: Path,
    *,
    with_durations: bool = True,
    write_clips: bool = False,
    write_silent_wav=None,
) -> Path:
    root = tmp_path / "cv"
    lang = root / "en"
    (lang / "clips").mkdir(parents=True)
    # (clip, speaker, split). 0005 is validated and in no split, by the same
    # speaker as the training clips; 0006 is validated, in no split, and by the
    # test speaker — the leak the guard exists to remove.
    catalogue = [
        ("common_voice_en_0001.mp3", "s_train", "train"),
        ("common_voice_en_0002.mp3", "s_train", "train"),
        ("common_voice_en_0003.mp3", "s_dev", "dev"),
        ("common_voice_en_0004.mp3", "s_test", "test"),
        ("common_voice_en_0005.mp3", "s_train", None),
        ("common_voice_en_0006.mp3", "s_test", None),
    ]

    def body(entries):
        head = "client_id\tpath\tsentence\n"
        return head + "".join(f"{spk}\t{clip}\ta sentence\n" for clip, spk, _ in entries)

    for split, name in (("train", "train"), ("dev", "dev"), ("test", "test")):
        member = [e for e in catalogue if e[2] == split]
        (lang / f"{name}.tsv").write_text(body(member), encoding="utf-8")
    (lang / "validated.tsv").write_text(body(catalogue), encoding="utf-8")
    rows = {
        "train": [c for c, _, s in catalogue if s == "train"],
        "dev": [c for c, _, s in catalogue if s == "dev"],
        "test": [c for c, _, s in catalogue if s == "test"],
        "unsplit": [c for c, _, s in catalogue if s is None],
    }
    for _split, clips in rows.items():
        if write_clips:
            for clip in clips:
                # A real release ships mp3; the fallback path only reads a
                # header, so a WAV under the same name exercises it without
                # making the test depend on an mp3 decoder.
                write_silent_wav(lang / "clips" / clip, frames=16000)
    if with_durations:
        text = (fixtures_dir / "commonvoice" / "clip_durations.tsv").read_text(encoding="utf-8")
        (lang / "clip_durations.tsv").write_text(text, encoding="utf-8")
    return root


def test_common_voice_durations_come_from_the_shipped_manifest(tmp_path, fixtures_dir, monkeypatch):
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    lang = language_durations(CV_SPEC, train_source="train")

    # 1000 ms + 2500 ms of training audio.
    assert lang.splits["train"].seconds == pytest.approx(3.5)
    assert lang.splits["train"].n_utts == 2
    assert lang.splits["validation"].seconds == pytest.approx(0.5)
    assert lang.splits["test"].seconds == pytest.approx(3.0)
    assert lang.source == "clip_durations.tsv"
    assert lang.total_hours == pytest.approx(7.0 / 3600)


def test_the_audit_prices_the_training_set_the_run_will_actually_read(
    tmp_path, fixtures_dir, monkeypatch
):
    """A duration audit that reports `train.tsv` while the run trains on
    validated-minus-evaluation is answering a question nobody asked."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    lang = language_durations(CV_SPEC, train_source="validated_minus_eval")

    # The two train clips plus 0005, which is validated and in no split.
    assert lang.splits["train"].n_utts == 3
    assert lang.splits["train"].seconds == pytest.approx(7.5)
    assert lang.splits["validation"].seconds == pytest.approx(0.5)


def test_the_audit_prices_what_the_speaker_guard_removed(tmp_path, fixtures_dir, monkeypatch):
    """The guard's cost is the number that decides whether it is worth arguing
    about, so it is reported in hours and not only as a clip count."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    guard = language_durations(CV_SPEC, train_source="validated_minus_eval").train_source_accounting

    assert guard is not None
    assert guard.train_source == "validated_minus_eval"
    assert guard.n_validated == 6
    assert guard.n_in_eval_splits == 2  # 0003 in dev, 0004 in test
    assert guard.n_eval_speaker_clips == 1  # 0006, another recording by the test speaker
    assert guard.n_eval_speakers == 1
    assert guard.eval_speaker_seconds == pytest.approx(6.0)
    assert guard.eval_speaker_hours == pytest.approx(6.0 / 3600)


def test_the_legacy_source_reports_no_guard_accounting(tmp_path, fixtures_dir, monkeypatch):
    """Nothing was filtered, so there is nothing to account for, and a block of
    zeroes would read as a guard that ran and found nothing."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    lang = language_durations(CV_SPEC, train_source="train")

    assert lang.train_source_accounting is None


def test_openslr_has_no_train_source_because_it_derives_its_own_split(slr_root, monkeypatch):
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))

    assert language_durations(SLR_SPEC).train_source_accounting is None


def test_a_missing_manifest_falls_back_to_headers_with_a_warning(
    tmp_path, fixtures_dir, monkeypatch, write_silent_wav
):
    # A release older than v16.1 ships no clip_durations.tsv. That is a slower
    # path, not a failure.
    monkeypatch.setenv(
        "CV_ROOT",
        str(
            make_cv_root(
                tmp_path,
                fixtures_dir,
                with_durations=False,
                write_clips=True,
                write_silent_wav=write_silent_wav,
            )
        ),
    )

    with pytest.warns(UserWarning, match="clip_durations.tsv"):
        lang = language_durations(CV_SPEC, train_source="train")

    assert lang.source == "audio headers"
    assert lang.splits["train"].seconds == pytest.approx(2.0)  # two 1-second clips


def test_a_clip_absent_from_the_manifest_is_counted_not_ignored(
    tmp_path, fixtures_dir, monkeypatch
):
    root = make_cv_root(tmp_path, fixtures_dir)
    monkeypatch.setenv("CV_ROOT", str(root))
    (root / "en" / "train.tsv").write_text(
        "client_id\tpath\tsentence\nc\tcommon_voice_en_0001.mp3\ts\nc\tnot_listed.mp3\ts\n",
        encoding="utf-8",
    )

    lang = language_durations(CV_SPEC, train_source="train")

    assert lang.splits["train"].n_utts == 2
    assert lang.splits["train"].n_missing == 1
    # The known clip still contributes; the unknown one is reported, not guessed.
    assert lang.splits["train"].seconds == pytest.approx(1.0)


def test_openslr_durations_read_wav_headers(slr_root, monkeypatch):
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))

    lang = language_durations(SLR_SPEC)

    assert lang.source == "audio headers"
    # Every fixture clip is 160 frames at 16 kHz.
    total = sum(s.seconds for s in lang.splits.values())
    assert total == pytest.approx(0.01 * sum(s.n_utts for s in lang.splits.values()))
    # The derived split puts utterances in all three buckets.
    assert all(lang.splits[s].n_utts > 0 for s in ("train", "validation", "test"))


def test_a_language_that_cannot_be_read_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("CV_ROOT", str(tmp_path / "absent"))

    rows = collect_durations([CV_SPEC])

    assert rows[0].error is not None
    assert rows[0].total_hours == 0.0


def test_below_threshold_languages_are_flagged(tmp_path, fixtures_dir, monkeypatch):
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    rows = collect_durations([CV_SPEC], min_train_hours=50.0, train_source="train")

    assert rows[0].below_threshold is True
    assert "below" in to_markdown(rows, min_train_hours=50.0).lower()


def test_write_tables_emits_markdown_and_json(tmp_path, fixtures_dir, monkeypatch):
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))
    out = tmp_path / "tables"

    written = write_tables(
        collect_durations([CV_SPEC], train_source="train"), out, min_train_hours=50.0
    )

    assert set(written) == {out / "data_durations.md", out / "data_durations.json"}
    payload = json.loads((out / "data_durations.json").read_text(encoding="utf-8"))
    assert payload["min_train_hours"] == 50.0
    assert payload["languages"][0]["code"] == "en"
    assert payload["languages"][0]["splits"]["train"]["n_utts"] == 2


def test_json_reports_hours_and_seconds_so_neither_has_to_be_rederived(
    tmp_path, fixtures_dir, monkeypatch
):
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    payload = to_json(collect_durations([CV_SPEC], train_source="train"), min_train_hours=50.0)
    train = payload["languages"][0]["splits"]["train"]

    assert train["seconds"] == pytest.approx(3.5)
    assert train["hours"] == pytest.approx(3.5 / 3600)


def test_the_guard_accounting_survives_the_sweep_into_json_and_markdown(
    tmp_path, fixtures_dir, monkeypatch
):
    """`collect_durations` rebuilds each row to attach the threshold flag, which
    is exactly where a newly added field gets silently dropped."""
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    rows = collect_durations([CV_SPEC], train_source="validated_minus_eval")
    payload = to_json(rows)
    guard = payload["languages"][0]["train_source"]

    assert guard["train_source"] == "validated_minus_eval"
    assert guard["n_eval_speaker_clips"] == 1
    assert guard["eval_speaker_hours"] == pytest.approx(6.0 / 3600)
    assert "speaker" in to_markdown(rows).lower()
