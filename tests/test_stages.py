"""The stage ledger: what may be reused when a run is resumed, and what may not."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest
import torch

from svb.stages import LEDGER, StageLedger, StageOutput


class Recorder:
    """Stages that write a file, count their calls, and return a draw."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.calls: list[str] = []

    def stage(self, name: str, seed: int) -> StageOutput:
        self.calls.append(name)
        artifact = self.run_dir / f"{name}.pt"
        artifact.write_text("weights", encoding="utf-8")
        return StageOutput(
            result={"draw": float(torch.rand(1)), "seed": seed}, artifacts=[artifact]
        )

    def run(self, ledger: StageLedger, *names: str) -> list[dict]:
        return [ledger.run(name, partial(self.stage, name)) for name in names]


def test_a_finished_stage_is_not_run_again_on_resume(tmp_path: Path) -> None:
    work = Recorder(tmp_path)
    first = work.run(StageLedger(tmp_path, seed=7), "phase0", "merge")

    again = work.run(StageLedger(tmp_path, seed=7, resume=True), "phase0", "merge")

    assert work.calls == ["phase0", "merge"]
    assert again == first


def test_without_resume_an_existing_ledger_is_discarded(tmp_path: Path) -> None:
    work = Recorder(tmp_path)
    work.run(StageLedger(tmp_path, seed=7), "phase0")

    work.run(StageLedger(tmp_path, seed=7), "phase0")

    assert work.calls == ["phase0", "phase0"]


def test_a_stage_whose_files_are_gone_is_run_again(tmp_path: Path) -> None:
    """A record of a checkpoint that no longer exists is not a result."""
    work = Recorder(tmp_path)
    work.run(StageLedger(tmp_path, seed=7), "phase0")
    (tmp_path / "phase0.pt").unlink()

    work.run(StageLedger(tmp_path, seed=7, resume=True), "phase0")

    assert work.calls == ["phase0", "phase0"]


def test_nothing_after_a_rerun_stage_is_reused(tmp_path: Path) -> None:
    """Experts branched from the old phase 0 must not be merged against the new one."""
    work = Recorder(tmp_path)
    work.run(StageLedger(tmp_path, seed=7), "phase0", "expert_en", "merge")
    (tmp_path / "expert_en.pt").unlink()

    work.run(StageLedger(tmp_path, seed=7, resume=True), "phase0", "expert_en", "merge")

    assert work.calls == ["phase0", "expert_en", "merge", "expert_en", "merge"]


def test_a_rerun_drops_the_records_of_the_stages_it_has_not_reached(tmp_path: Path) -> None:
    """Otherwise a second crash would leave stale later stages for a third attempt."""
    work = Recorder(tmp_path)
    work.run(StageLedger(tmp_path, seed=7), "phase0", "expert_en", "merge")
    (tmp_path / "expert_en.pt").unlink()

    work.run(StageLedger(tmp_path, seed=7, resume=True), "phase0", "expert_en")

    recorded = json.loads((tmp_path / LEDGER).read_text(encoding="utf-8"))["stages"]
    assert list(recorded) == ["phase0", "expert_en"]


def test_a_record_is_withdrawn_from_disk_before_its_stage_runs_again(tmp_path: Path) -> None:
    """The trainer writes a checkpoint as soon as it starts. A re-run killed in
    the middle leaves that file where the old record says the finished one is,
    and the next resume would take it for finished — unless the record went
    first."""
    work = Recorder(tmp_path)
    work.run(StageLedger(tmp_path, seed=7), "phase0", "expert_en", "merge")
    (tmp_path / "phase0.pt").unlink()

    ledger = StageLedger(tmp_path, seed=7, resume=True)

    def killed_partway(seed: int) -> StageOutput:
        (tmp_path / "phase0.pt").write_text("epoch 0", encoding="utf-8")
        raise RuntimeError("wall-clock limit")

    with pytest.raises(RuntimeError, match="wall-clock"):
        ledger.run("phase0", killed_partway)

    on_disk = json.loads((tmp_path / LEDGER).read_text(encoding="utf-8"))["stages"]
    assert on_disk == {}
    work.run(StageLedger(tmp_path, seed=7, resume=True), "phase0", "expert_en", "merge")
    assert work.calls[-3:] == ["phase0", "expert_en", "merge"]


def test_two_processes_cannot_hold_one_run_directory(tmp_path: Path) -> None:
    from svb.stages import RunLockedError, run_lock

    with run_lock(tmp_path):
        second = run_lock(tmp_path)
        with pytest.raises(RunLockedError, match="in use"):
            second.__enter__()

    # Released on exit, so the next job can have it.
    with run_lock(tmp_path):
        pass


def test_a_stage_draws_the_same_resumed_or_not(tmp_path: Path) -> None:
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    expected = Recorder(straight)
    straight.mkdir()
    resumed.mkdir()
    uninterrupted = expected.run(StageLedger(straight, seed=7), "phase0", "transfer_te")

    work = Recorder(resumed)
    work.run(StageLedger(resumed, seed=7), "phase0")
    torch.rand(100)  # a different process, a different stream position
    second = work.run(StageLedger(resumed, seed=7, resume=True), "phase0", "transfer_te")

    assert second == uninterrupted


def test_the_summary_says_what_each_stage_cost_and_which_invocation_ran_it(
    tmp_path: Path,
) -> None:
    work = Recorder(tmp_path)
    work.run(StageLedger(tmp_path, seed=7, invocation={"git_sha": "aaa"}), "phase0")
    ledger = StageLedger(tmp_path, seed=7, resume=True, invocation={"git_sha": "bbb"})
    work.run(ledger, "phase0", "merge")

    summary = ledger.summary()

    assert [entry["git_sha"] for entry in summary["invocations"]] == ["aaa", "bbb"]
    assert [entry["resumed"] for entry in summary["invocations"]] == [False, True]
    assert summary["stages"]["phase0"]["invocation"] == 0
    assert summary["stages"]["merge"]["invocation"] == 1
    assert summary["stages"]["merge"]["wall_seconds"] >= 0


def test_a_stage_name_cannot_be_used_twice(tmp_path: Path) -> None:
    """Two stages under one name would overwrite each other's record and seed."""
    work = Recorder(tmp_path)
    ledger = StageLedger(tmp_path, seed=7)
    work.run(ledger, "phase0")

    with pytest.raises(ValueError, match="already run"):
        work.run(ledger, "phase0")


def test_a_stage_is_handed_the_seed_it_was_seeded_with(tmp_path: Path) -> None:
    """What a stage passes on to a loader must be the stage's own stream."""
    from svb.seeding import derive_seed

    work = Recorder(tmp_path)
    (phase0, expert) = work.run(StageLedger(tmp_path, seed=7), "phase0", "expert_en")

    assert phase0["seed"] == derive_seed(7, "stage", "phase0")
    assert expert["seed"] == derive_seed(7, "stage", "expert_en")
    assert phase0["seed"] != expert["seed"]


def test_a_fact_survives_to_the_next_invocation_and_only_with_resume(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path, seed=7)
    ledger.remember("data", {"en": "abc"})

    assert StageLedger(tmp_path, seed=7, resume=True).recall("data") == {"en": "abc"}
    assert StageLedger(tmp_path, seed=7).recall("data") is None


def test_a_stage_that_raises_leaves_no_record(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path, seed=7)

    def dies(seed: int) -> StageOutput:
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        ledger.run("phase0", dies)

    assert json.loads((tmp_path / LEDGER).read_text(encoding="utf-8"))["stages"] == {}
