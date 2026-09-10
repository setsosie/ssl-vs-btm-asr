"""Seeding claims and run provenance — what a reviewer checks first."""

from __future__ import annotations

import json
import os
import random

import svb.seeding as seeding_mod
from svb.provenance import dump_run_meta
from svb.seeding import set_all_seeds


def test_seeding_is_reproducible() -> None:
    set_all_seeds(11)
    first = random.random()
    set_all_seeds(11)
    assert random.random() == first


def test_seeding_does_not_pretend_to_set_the_hash_seed() -> None:
    """PYTHONHASHSEED is read once at interpreter start.

    Assigning it from inside the process changes nothing, so leaving the line
    in place claimed a determinism guarantee the code could not deliver.
    """
    before = os.environ.get("PYTHONHASHSEED")

    set_all_seeds(3)

    assert os.environ.get("PYTHONHASHSEED") == before


def test_seeding_docstring_claims_only_what_holds() -> None:
    """There is no SpecAugment in this repo, and CTC backward is nondeterministic."""
    text = (seeding_mod.__doc__ or "") + (set_all_seeds.__doc__ or "")

    assert "SpecAugment" not in text
    assert "nondeterministic" in text


def test_run_meta_records_the_code_it_actually_ran(tmp_path, monkeypatch) -> None:
    """A run launched from another directory must not record that repo's SHA.

    git rev-parse in the process CWD attributes whatever repository the user
    happened to be standing in to these results.
    """
    monkeypatch.chdir(tmp_path)

    meta = json.loads(dump_run_meta(tmp_path).read_text())

    assert meta["git_sha"] != "unknown"
    assert len(meta["git_sha"]) == 40
    assert meta["git_dirty"] in (True, False)  # a real probe here, never None


def test_run_meta_records_versions_of_dependencies_that_exist(tmp_path) -> None:
    """jiwer 4 exposes no __version__, so the old probe recorded it as absent.

    That is the package whose major version decides whether an empty reference
    raises or returns a number, so "absent" was the least useful thing to say.
    """
    meta = json.loads(dump_run_meta(tmp_path).read_text())
    versions = meta["versions"]

    assert versions["jiwer"].startswith("4.")
    assert "transformers" not in versions  # never a dependency of this project
    assert versions["pyyaml"] != "absent"
    assert versions["soundfile"] != "absent"


def test_run_meta_records_a_failed_probe_instead_of_swallowing_it(tmp_path, monkeypatch) -> None:
    """The probe has to be made to fail, or this asserts nothing.

    The previous version of this test allowed `{"available": False}` as a pass,
    which is what any machine without a GPU returns without ever entering the
    error path — so on CI, and on any development box, it tested nothing.
    """
    import torch

    def boom() -> bool:
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)

    meta = json.loads(dump_run_meta(tmp_path).read_text())

    assert meta["gpu"]["available"] is False
    assert "no driver" in meta["gpu"]["error"]


def test_a_successful_gpu_probe_reports_no_error(tmp_path, monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    meta = json.loads(dump_run_meta(tmp_path).read_text())

    assert meta["gpu"] == {"available": False}


def test_an_unknown_git_state_is_recorded_as_unknown(tmp_path, monkeypatch) -> None:
    """`git_dirty: false` is a claim; failing to run git is not evidence for it.

    The field exists because a SHA can point at code that was not the code that
    ran. Reporting a clean tree when the probe never succeeded reintroduces
    exactly that, and the module promises to record a failed probe rather than
    swallow it — which `_git_sha` already does by returning "unknown".
    """
    from svb import provenance

    def boom(*args: str) -> str:
        raise OSError("git not found")

    monkeypatch.setattr(provenance, "_git", boom)

    meta = json.loads(dump_run_meta(tmp_path).read_text())

    assert meta["git_sha"] == "unknown"
    assert meta["git_dirty"] is None


def test_datasets_is_not_probed_now_that_it_is_not_a_dependency(tmp_path) -> None:
    """The held-out path stopped going through the Hugging Face Hub.

    Probing it recorded "absent" on every run forever, which reads like a
    missing install rather than a dependency that was deliberately removed.
    """
    meta = json.loads(dump_run_meta(tmp_path).read_text())

    assert "datasets" not in meta["versions"]


def test_uv_lock_hash_is_recorded_when_the_file_exists(tmp_path) -> None:
    from svb.provenance import _uv_lock_sha256

    assert _uv_lock_sha256(tmp_path) is None

    (tmp_path / "uv.lock").write_text("version = 1\n")
    digest = _uv_lock_sha256(tmp_path)

    assert digest is not None
    assert len(digest) == 64
