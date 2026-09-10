"""The job-list generator: it must not emit jobs that cannot run.

Its output is piped straight into a scheduler, so every line it prints costs a
slot. A scale whose preset is still a placeholder raises on load, so emitting it
buys a queue of guaranteed failures.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _preset(root: Path, scale: str, languages: list[dict[str, str]]) -> None:
    (root / "scales" / f"{scale}.yaml").write_text(
        yaml.safe_dump({"languages": languages}), encoding="utf-8"
    )


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    root = tmp_path / "configs"
    (root / "scales").mkdir(parents=True)
    (root / "seeds.yaml").write_text(yaml.safe_dump({"seeds": [0, 1]}), encoding="utf-8")
    _preset(root, "3", [{"code": "en", "source": "commonvoice", "hf_config": "en"}])
    _preset(root, "16", [{"code": "de", "source": "commonvoice", "hf_config": "de"}])
    _preset(root, "64", [])
    return root


def test_a_placeholder_scale_is_left_out(run_matrix: ModuleType, configs: Path) -> None:
    kept, skipped = run_matrix.populated_scales(configs)

    assert kept == ["3", "16"]
    assert [code for code, _ in skipped] == ["64"]


def test_the_omission_goes_to_stderr_not_into_the_job_list(
    run_matrix: ModuleType, configs: Path
) -> None:
    """Silently shrinking the matrix is its own failure: the reader has to be
    told the tier was skipped, but on a stream the scheduler is not reading."""
    out, err = io.StringIO(), io.StringIO()

    run_matrix.main(configs_dir=configs, stdout=out, stderr=err)

    assert "--scale 64" not in out.getvalue()
    assert "64" in err.getvalue()
    assert "not populated" in err.getvalue()


def test_every_populated_cell_is_emitted_once(run_matrix: ModuleType, configs: Path) -> None:
    out = io.StringIO()

    run_matrix.main(configs_dir=configs, stdout=out, stderr=io.StringIO())
    lines = out.getvalue().splitlines()

    # 2 scales x 3 arms x 2 seeds.
    assert len(lines) == 12
    assert len(set(lines)) == 12
    assert lines[0] == "--arm A_ssl --scale 3 --seed 0"


def test_a_matrix_with_nothing_to_run_fails_rather_than_printing_nothing(
    run_matrix: ModuleType, tmp_path: Path
) -> None:
    root = tmp_path / "configs"
    (root / "scales").mkdir(parents=True)
    (root / "seeds.yaml").write_text(yaml.safe_dump({"seeds": [0]}), encoding="utf-8")
    for scale in ("3", "16", "64"):
        _preset(root, scale, [])

    with pytest.raises(SystemExit):
        run_matrix.main(configs_dir=root, stdout=io.StringIO(), stderr=io.StringIO())


def test_a_missing_preset_file_is_skipped_like_an_empty_one(
    run_matrix: ModuleType, configs: Path
) -> None:
    """`get_preset` raises FileNotFoundError rather than ValueError for this, and
    a sweep that names the gap must survive both."""
    (configs / "scales" / "16.yaml").unlink()

    kept, skipped = run_matrix.populated_scales(configs)

    assert kept == ["3"]
    assert [code for code, _ in skipped] == ["16", "64"]


def test_the_default_configs_dir_does_not_depend_on_the_working_directory(
    run_matrix: ModuleType,
) -> None:
    """A submit-time helper runs from wherever the scheduler put the job."""
    default = run_matrix.default_configs_dir()

    assert default.is_absolute()
    assert (default / "seeds.yaml").is_file()
