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
    rows = {
        "train": ["common_voice_en_0001.mp3", "common_voice_en_0002.mp3"],
        "dev": ["common_voice_en_0003.mp3"],
        "test": ["common_voice_en_0004.mp3"],
    }
    for split, clips in rows.items():
        body = "client_id\tpath\tsentence\n" + "".join(f"c\t{c}\ta sentence\n" for c in clips)
        (lang / f"{split}.tsv").write_text(body, encoding="utf-8")
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

    lang = language_durations(CV_SPEC)

    # 1000 ms + 2500 ms of training audio.
    assert lang.splits["train"].seconds == pytest.approx(3.5)
    assert lang.splits["train"].n_utts == 2
    assert lang.splits["validation"].seconds == pytest.approx(0.5)
    assert lang.splits["test"].seconds == pytest.approx(3.0)
    assert lang.source == "clip_durations.tsv"
    assert lang.total_hours == pytest.approx(7.0 / 3600)


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
        lang = language_durations(CV_SPEC)

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

    lang = language_durations(CV_SPEC)

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

    rows = collect_durations([CV_SPEC], min_train_hours=50.0)

    assert rows[0].below_threshold is True
    assert "below" in to_markdown(rows, min_train_hours=50.0).lower()


def test_write_tables_emits_markdown_and_json(tmp_path, fixtures_dir, monkeypatch):
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))
    out = tmp_path / "tables"

    written = write_tables(collect_durations([CV_SPEC]), out, min_train_hours=50.0)

    assert set(written) == {out / "data_durations.md", out / "data_durations.json"}
    payload = json.loads((out / "data_durations.json").read_text(encoding="utf-8"))
    assert payload["min_train_hours"] == 50.0
    assert payload["languages"][0]["code"] == "en"
    assert payload["languages"][0]["splits"]["train"]["n_utts"] == 2


def test_json_reports_hours_and_seconds_so_neither_has_to_be_rederived(
    tmp_path, fixtures_dir, monkeypatch
):
    monkeypatch.setenv("CV_ROOT", str(make_cv_root(tmp_path, fixtures_dir)))

    payload = to_json(collect_durations([CV_SPEC]), min_train_hours=50.0)
    train = payload["languages"][0]["splits"]["train"]

    assert train["seconds"] == pytest.approx(3.5)
    assert train["hours"] == pytest.approx(3.5 / 3600)
