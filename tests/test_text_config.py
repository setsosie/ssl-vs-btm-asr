"""The normalization policy as run configuration and as run provenance.

Results produced under different policies are not comparable, so the policy has
to be recoverable from the run's own artifacts rather than from whatever the
code happened to default to on the day.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from svb.config import TextConfig, dump_config, load_config
from svb.text.normalize import NORMALIZER_VERSION, NormalizerPolicy


def test_defaults_are_the_current_policy() -> None:
    cfg = load_config("A_ssl", "3", 0)

    assert cfg.text.policy == NormalizerPolicy()
    assert cfg.text.policy.version == NORMALIZER_VERSION
    assert cfg.text.min_char_count == 1  # safe for smoke runs; raised in base.yaml


def test_a_yaml_can_override_the_policy(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"text": {"policy": {"case": "lower"}, "min_char_count": 5}}),
        encoding="utf-8",
    )

    cfg = load_config("A_ssl", "3", 0, yaml_path=path)

    assert cfg.text.policy.case == "lower"
    assert cfg.text.min_char_count == 5


def test_the_shipped_base_config_raises_the_floor_for_a_full_run(pytestconfig) -> None:
    cfg = load_config(
        "A_ssl", "16", 0, yaml_path=Path(pytestconfig.rootpath) / "configs" / "base.yaml"
    )

    assert cfg.text.min_char_count > 1


def test_resolved_config_records_the_digests_a_reader_needs(tmp_path: Path) -> None:
    """The policy hash identifies the settings; the module hash catches someone
    editing the normalizer without changing any setting."""
    cfg = load_config("A_ssl", "3", 0)

    dumped = yaml.safe_load(dump_config(cfg, tmp_path).read_text(encoding="utf-8"))

    assert dumped["text"]["policy"]["case"] == "casefold"
    assert dumped["text"]["policy_hash"] == cfg.text.policy.policy_hash()
    assert len(dumped["text"]["normalizer_module_sha256"]) == 64


def test_a_resolved_config_can_be_loaded_back(tmp_path: Path) -> None:
    """Reproducing a run means feeding its own dumped config back in."""
    original = load_config("A_ssl", "3", 0, overrides={"text": {"min_char_count": 4}})
    path = dump_config(original, tmp_path)

    reloaded = load_config("A_ssl", "3", 0, yaml_path=path)

    assert reloaded.text == original.text


def test_an_unknown_text_setting_is_refused() -> None:
    with pytest.raises(TypeError):
        load_config("A_ssl", "3", 0, overrides={"text": {"policy": {"casefold": True}}})


def test_env_json_records_what_would_change_the_text(tmp_path: Path) -> None:
    """Unicode tables decide categories and case folding, so the Unicode version
    is part of what produced a number."""
    import unicodedata

    from svb.provenance import dump_run_meta

    policy = NormalizerPolicy(case="lower")
    meta = json.loads(dump_run_meta(tmp_path, policy=policy).read_text(encoding="utf-8"))

    assert meta["normalizer"]["version"] == NORMALIZER_VERSION
    assert meta["normalizer"]["policy_hash"] == policy.policy_hash()
    assert meta["normalizer"]["unicode_version"] == unicodedata.unidata_version
    assert len(meta["normalizer"]["module_sha256"]) == 64


def test_env_json_omits_a_policy_hash_it_was_not_given(tmp_path: Path) -> None:
    """Recording the default's hash for a run that used something else would be
    worse than recording nothing."""
    from svb.provenance import dump_run_meta

    meta = json.loads(dump_run_meta(tmp_path).read_text(encoding="utf-8"))

    assert "policy_hash" not in meta["normalizer"]
    assert meta["normalizer"]["unicode_version"]


def test_text_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        TextConfig().min_char_count = 3  # type: ignore[misc]
