"""OpenSLR loader: index parsing, speaker-disjoint splits, audio decode."""

import hashlib

import pytest
import torch

from svb.data.openslr_local import (
    SPLITS,
    OpenSLRLocal,
    derive_splits,
    load_openslr_texts,
    read_index,
    speaker_of,
)
from svb.data.registry import LangSpec

SPEC = LangSpec(
    code="malayalam",
    source="openslr",
    slr=63,
    archives=["ml_in_female.zip", "ml_in_male.zip"],
    index_files=["line_index_female.tsv", "line_index_male.tsv"],
    license="CC-BY-SA-4.0",
)


# --- index parsing ------------------------------------------------------------


def test_read_index_returns_file_id_and_transcript(slr_root):
    rows = read_index(slr_root / "SLR63" / "line_index_female.tsv")
    assert len(rows) == 20
    assert rows[0] == ("mlf_02879_00000000001", "ithu oru cheriya vaakyam")
    assert rows[6][1] == "ഒരു മലയാളം വാക്യം"


def test_read_index_skips_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "line_index.tsv"
    p.write_text("a_1_1\tone\n\n   \nno_tab_here\nb_2_2\t\nc_3_3\ttwo\n", encoding="utf-8")
    assert read_index(p) == [("a_1_1", "one"), ("c_3_3", "two")]


@pytest.mark.parametrize(
    ("file_id", "expected"),
    [
        ("mlf_02879_00000000001", "mlf_02879"),
        ("mrt_04310_01727087719", "mrt_04310"),
        ("utt00000001", None),
        ("only_two", None),
        ("a_b_c_d", None),
    ],
)
def test_speaker_of(file_id, expected):
    assert speaker_of(file_id) == expected


# --- split derivation ---------------------------------------------------------


def test_splits_are_speaker_disjoint_when_ids_carry_a_speaker(slr_root):
    rows = read_index(slr_root / "SLR63" / "line_index_female.tsv")
    rows += read_index(slr_root / "SLR63" / "line_index_male.tsv")
    parts, policy, _ = derive_splits(rows)

    assert policy == "speaker"
    assert sum(len(v) for v in parts.values()) == len(rows)
    seen = [{speaker_of(fid) for fid, _ in parts[s]} for s in SPLITS]
    assert all(seen)  # every split got at least one speaker
    assert seen[0] & seen[1] == set()
    assert seen[0] & seen[2] == set()
    assert seen[1] & seen[2] == set()


def test_split_falls_back_to_utterances_when_ids_have_no_speaker(slr_root, fixtures_dir):
    rows = read_index(fixtures_dir / "openslr" / "line_index_flat.tsv")
    parts, policy, _ = derive_splits(rows)

    assert policy == "utterance"
    assert sum(len(v) for v in parts.values()) == len(rows)
    assert all(parts[s] for s in SPLITS)


def test_split_is_stable_under_input_order_and_repeated_calls():
    rows = [(f"ml_{i // 5:05d}_{i:09d}", f"line {i}") for i in range(100)]
    a, _, ha = derive_splits(rows)
    b, _, hb = derive_splits(list(reversed(rows)))

    assert ha == hb
    assert {s: sorted(a[s]) for s in SPLITS} == {s: sorted(b[s]) for s in SPLITS}


def test_split_hash_is_the_sha1_of_the_test_file_list():
    rows = [(f"ml_{i // 5:05d}_{i:09d}", f"line {i}") for i in range(100)]
    parts, _, digest = derive_splits(rows)
    listing = "\n".join(fid for fid, _ in parts["test"])
    assert digest == hashlib.sha1(listing.encode("utf-8")).hexdigest()
    assert len(digest) == 40


def test_split_is_roughly_80_10_10_on_many_speakers():
    rows = [(f"ml_{i // 10:05d}_{i:09d}", f"line {i}") for i in range(2000)]
    parts, policy, _ = derive_splits(rows)
    assert policy == "speaker"
    assert 0.74 <= len(parts["train"]) / 2000 <= 0.86
    assert 0.05 <= len(parts["validation"]) / 2000 <= 0.16
    assert 0.05 <= len(parts["test"]) / 2000 <= 0.16


def test_every_split_is_populated_even_when_one_speaker_dominates():
    rows = [("ml_00001_1", "a"), ("ml_00002_2", "b")]
    rows += [(f"ml_00003_{i}", "c") for i in range(200)]
    parts, _, _ = derive_splits(rows)
    assert all(parts[s] for s in SPLITS)


def test_two_speakers_cannot_be_split_three_ways_so_it_degrades_to_utterances():
    rows = [(f"ml_{i % 2:05d}_{i:09d}", f"line {i}") for i in range(30)]
    parts, policy, _ = derive_splits(rows)
    assert policy == "utterance"
    assert all(parts[s] for s in SPLITS)


# --- dataset ------------------------------------------------------------------


def test_dataset_reads_wavs_and_reports_its_split_policy(slr_root):
    ds = OpenSLRLocal(SPEC, "train", root=str(slr_root))
    assert ds.split_policy == "speaker"
    assert len(ds.test_files_sha1) == 40
    assert len(ds) == len(load_openslr_texts(SPEC, "train", root=str(slr_root)))

    wav, text = ds[0]
    assert isinstance(wav, torch.Tensor) and wav.dtype == torch.float32
    assert wav.ndim == 1 and wav.shape[0] == 160
    assert text


def test_splits_partition_the_corpus_exactly(slr_root):
    ids = {s: set() for s in SPLITS}
    for split in SPLITS:
        ds = OpenSLRLocal(SPEC, split, root=str(slr_root))
        ids[split] = {fid for fid, _ in ds.rows}
    assert sum(len(v) for v in ids.values()) == 32
    assert len(set.union(*ids.values())) == 32


def test_all_splits_agree_on_the_recorded_test_hash(slr_root):
    hashes = {OpenSLRLocal(SPEC, s, root=str(slr_root)).test_files_sha1 for s in SPLITS}
    assert len(hashes) == 1


def test_max_samples_truncates_the_waveform(slr_root):
    ds = OpenSLRLocal(SPEC, "train", root=str(slr_root), max_samples=64)
    assert ds[0][0].shape[0] == 64


def test_stereo_and_odd_sample_rate_are_normalised(slr_root, write_silent_wav):
    dest = slr_root / "SLR63"
    rows = read_index(dest / "line_index_female.tsv")
    write_silent_wav(dest / f"{rows[0][0]}.wav", frames=80, sr=8000, channels=2)

    ds = OpenSLRLocal(SPEC, "train", root=str(slr_root))
    fid_to_idx = {fid: i for i, (fid, _) in enumerate(ds.rows)}
    if rows[0][0] not in fid_to_idx:
        pytest.skip("fixture utterance did not land in train")
    wav, _ = ds[fid_to_idx[rows[0][0]]]
    assert wav.ndim == 1
    assert 150 <= wav.shape[0] <= 170  # 80 frames @8k -> ~160 @16k


def test_missing_language_directory_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="SLR63"):
        OpenSLRLocal(SPEC, "train", root=str(tmp_path))


def test_missing_index_file_names_the_fetch_script(slr_root):
    (slr_root / "SLR63" / "line_index_male.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="fetch_openslr"):
        OpenSLRLocal(SPEC, "train", root=str(slr_root))


def test_root_is_read_from_the_environment(slr_root, monkeypatch):
    monkeypatch.setenv("OPENSLR_ROOT", str(slr_root))
    assert len(OpenSLRLocal(SPEC, "test")) > 0


def test_absent_root_explains_how_to_get_the_data(monkeypatch):
    monkeypatch.delenv("OPENSLR_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="OPENSLR_ROOT"):
        OpenSLRLocal(SPEC, "train")


def test_load_texts_does_not_touch_audio(slr_root):
    for wav in (slr_root / "SLR63").glob("*.wav"):
        wav.unlink()
    texts = load_openslr_texts(SPEC, "train", root=str(slr_root))
    assert texts and all(isinstance(t, str) and t for t in texts)
