"""Seeds: the committed list, and how a run turns one seed into many streams."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
import yaml

from svb.seeding import derive_seed, seed_stage
from svb.train.trainer import make_worker_init_fn


def _committed_seeds(pytestconfig: pytest.Config) -> list[int]:
    path = Path(pytestconfig.rootpath) / "configs" / "seeds.yaml"
    seeds: list[int] = yaml.safe_load(path.read_text(encoding="utf-8"))["seeds"]
    return seeds


def test_the_committed_seeds_are_usable_and_distinct(pytestconfig: pytest.Config) -> None:
    seeds = _committed_seeds(pytestconfig)

    # Three is the floor the caption rule allows a study to drop to.
    assert len(seeds) >= 3
    assert len(set(seeds)) == len(seeds)
    # Every library seeded here takes 32 bits; NumPy refuses anything wider.
    assert all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in seeds)


def test_the_makefile_does_not_keep_its_own_copy_of_the_seeds(
    pytestconfig: pytest.Config,
) -> None:
    """Two lists drift. The Makefile's used to say `0 1 2 3 4` whatever the YAML said."""
    makefile = (Path(pytestconfig.rootpath) / "Makefile").read_text(encoding="utf-8")

    assert "SEEDS" not in makefile
    assert "scripts/run_matrix.py" in makefile


def test_a_derived_seed_is_the_same_on_every_machine() -> None:
    """Pinned values: ``hash()`` is salted per process and would not survive this."""
    assert derive_seed(7, "stage", "phase0") == 3054919051
    assert derive_seed(7, "worker", 0) == 725952871


def test_derived_seeds_separate_stages_and_runs() -> None:
    assert derive_seed(7, "stage", "phase0") != derive_seed(7, "stage", "expert_en")
    assert derive_seed(7, "stage", "phase0") != derive_seed(8, "stage", "phase0")
    assert all(0 <= derive_seed(s, "stage", "x") < 2**32 for s in range(50))


def test_neighbouring_run_seeds_do_not_share_a_worker_stream() -> None:
    """``seed + worker_id`` made worker 1 of seed 7 the same stream as worker 0 of seed 8."""
    make_worker_init_fn(7)(1)
    first = random.random()
    make_worker_init_fn(8)(0)

    assert random.random() != first


def test_a_stage_draws_the_same_whatever_ran_before_it() -> None:
    """What makes a resumed run equivalent to one that never stopped."""
    seed_stage(7, "transfer_telugu")
    uninterrupted = torch.rand(3)

    torch.rand(1000)  # an earlier stage consuming the stream
    seed_stage(7, "transfer_telugu")

    assert torch.equal(torch.rand(3), uninterrupted)
