"""`svb run`, start to finish, for every arm — on a CPU, in seconds.

Until this file existed the run command had never been executed by anything:
each piece had tests and the thing that strings them together had none. These
runs use real audio files in the corpora's real layouts, the real loaders,
collate, trainer, merge, head expansion, decode and scoring, and replace only
the encoder — so what is left untested is the 577M-parameter model, not the
plumbing a first GPU run would otherwise be the first to exercise.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from svb.cli import ARMS, build_parser
from svb.data.registry import LangSpec
from svb.report.aggregate import aggregate_runs, load_runs
from svb.report.analyze import render_metric_tables

from .conftest import CvClip, TinyCTC

EN = LangSpec(code="en", source="commonvoice", hf_config="en", normalizer="whisper-basic")
JA = LangSpec(
    code="ja", source="commonvoice", hf_config="ja", word_boundary=False, normalizer="ja-cer"
)
TELUGU = LangSpec(
    code="telugu",
    source="openslr",
    slr=66,
    archives=("te.zip",),
    index_files=("line_index.tsv",),
    normalizer="indic-vistaar",
)

_EN_TEXT = ["the cat sat", "a dog ran", "hello there", "the dog sat", "a cat ran", "hello dog"]
_JA_TEXT = [
    "今日はいい天気",
    "コーヒーを飲む",
    "今日は飲む",
    "いい天気",
    "コーヒーはいい",
    "天気を飲む",
]
_TE_TEXT = ["ఇది తెలుగు భాష", "తెలుగు భాష", "ఇది తెలుగు"]


def _cv_rows(texts: list[str]) -> list[CvClip]:
    """Six training clips from two speakers, two dev, three test."""
    rows = [CvClip(f"t{i}.mp3", f"s_train_{i % 2}", text, "train") for i, text in enumerate(texts)]
    rows += [CvClip(f"d{i}.mp3", "s_dev", texts[i], "dev") for i in range(2)]
    rows += [CvClip(f"e{i}.mp3", "s_test", texts[i], "test") for i in range(3)]
    return rows


@pytest.fixture
def corpora(
    tmp_path: Path,
    make_cv_lang: Callable[..., Path],
    write_clip: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two Common Voice languages and one OpenSLR held-out language, on disk."""
    cv_root = make_cv_lang("en", _cv_rows(_EN_TEXT), clip_seconds=1.0)
    make_cv_lang("ja", _cv_rows(_JA_TEXT), clip_seconds=1.0, root=cv_root)
    monkeypatch.setenv("CV_ROOT", str(cv_root))

    slr = tmp_path / "openslr" / "SLR66"
    slr.mkdir(parents=True)
    index = []
    for speaker in range(5):
        for utt, text in enumerate(_TE_TEXT):
            file_id = f"tef_{speaker:05d}_{utt:08d}"
            write_clip(slr / f"{file_id}.wav", frames=16000)
            index.append(f"{file_id}\t{text}")
    (slr / "line_index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")
    monkeypatch.setenv("OPENSLR_ROOT", str(tmp_path / "openslr"))

    monkeypatch.setattr("svb.data.registry.get_preset", lambda scale: [EN, JA])
    monkeypatch.setattr("svb.data.registry.get_heldout", lambda: [TELUGU])
    for module in ("svb.model.xeus_ctc", "svb.btm.pipeline", "svb.eval.transfer"):
        monkeypatch.setattr(f"{module}.make_model", lambda cfg, vocab_size: TinyCTC(vocab_size))


@pytest.fixture
def run(tmp_path: Path, corpora: None) -> Callable[..., Path]:
    """Invoke `svb run` through its own parser; returns the run directory."""

    def invoke(
        arm: str, seed: int = 7, *flags: str, model: dict[str, Any] | None = None, **train: Any
    ) -> Path:
        settings = {
            "model": model or {},
            "optim": {"batch_size": 2, "accum_steps": 1},
            "train": {
                "phase0_epochs": 1,
                "expert_epochs": 1,
                "finetune_epochs": 1,
                "patience": 1,
                "bf16": False,
                "grad_checkpointing": False,
                "num_workers": 0,
                **train,
            },
        }
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(settings), encoding="utf-8")
        root = tmp_path / "results"
        argv = ["run", "--arm", arm, "--scale", "3", "--seed", str(seed), "--device", "cpu"]
        argv += ["--config", str(config), "--results-root", str(root), *flags]
        args = build_parser().parse_args(argv)
        args.func(args)
        return root / arm / "3" / f"seed{seed}"

    return invoke


def _results(run_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return payload


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_runs_end_to_end(run: Callable[..., Path], arm: str) -> None:
    run_dir = run(arm)
    results = _results(run_dir)

    assert set(results["in_distribution"]) == {"en", "ja"}
    assert set(results["transfer"]) == {"telugu"}
    for entry in (*results["in_distribution"].values(), *results["transfer"].values()):
        assert entry["n"] > 0
        assert entry["wer"] >= 0 and entry["cer"] >= 0
    assert results["transfer"]["telugu"]["split_policy"] == "speaker"

    stages = list(results["stages"])
    if arm == "A_ssl":
        assert stages == ["finetune_en", "eval_en", "finetune_ja", "eval_ja", "transfer_telugu"]
    else:
        assert stages == [
            "phase0",
            "expert_en",
            "expert_ja",
            "merge",
            "eval_en",
            "eval_ja",
            "transfer_telugu",
        ]
    # Every training stage reaches the results, the transfer fine-tune included.
    trained = {name for name in stages if not name.startswith(("eval_", "merge"))}
    assert set(results["training"]) == trained
    assert all(record["epochs_run"] == 1 for record in results["training"].values())

    for name in ("resolved_config.yaml", "env.json", "text_stats.json", "vocab.json"):
        assert (run_dir / name).is_file()
    assert (run_dir / "predictions" / "en.json").is_file()
    assert (run_dir / "transfer" / "telugu" / "predictions.json").is_file()


# Python 3.12 warns that forking a process with live threads (torch's) may
# deadlock. That is how every PyTorch DataLoader with workers starts on Linux,
# and it is the path the real config takes; the warning is not news here.
@pytest.mark.filterwarnings("ignore:This process.*is multi-threaded:DeprecationWarning")
def test_a_run_with_loader_workers_trains_on_the_same_pipeline(run: Callable[..., Path]) -> None:
    """Every other run here loads in the main process. The shipped config uses
    eight workers, which is a different code path: the datasets, the collate
    closure and the worker seeding all have to survive a process boundary."""
    results = _results(run("A_ssl", 7, num_workers=2))

    assert results["training"]["finetune_en"]["epochs_run"] == 1
    assert results["in_distribution"]["en"]["n"] == 3


def test_the_reporting_commands_read_what_a_run_wrote(
    run: Callable[..., Path], tmp_path: Path
) -> None:
    """The other half of the contract: a finished run is something the tables can use."""
    run_dir = run("B_btm_ssl")

    runs = load_runs(tmp_path / "results", "B_btm_ssl", "3")
    agg = aggregate_runs(runs)
    written = render_metric_tables(
        scale="3", runs=[run_dir], comparisons=[], out_dir=tmp_path / "tables", n_resamples=50
    )

    assert {(row.section, row.code) for row in agg.languages} == {
        ("in_distribution", "en"),
        ("in_distribution", "ja"),
        ("transfer", "telugu"),
    }
    # Japanese is scored by characters, so the in-distribution macro is mixed.
    assert agg.macro["in_distribution"].is_mixed
    assert all(path.is_file() for path in written)


def test_a_resumed_run_picks_up_where_the_last_one_died(
    run: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import svb.eval.transfer as transfer

    real = transfer.transfer_one
    monkeypatch.setattr("svb.eval.transfer.transfer_one", _raiser("out of wall-clock"))
    with pytest.raises(RuntimeError, match="out of wall-clock"):
        run("B_btm_ssl")

    # From here on, anything that trains or merges again is a failure: those
    # stages finished, and re-running them is what resume exists to avoid.
    monkeypatch.setattr("svb.eval.transfer.transfer_one", real)
    monkeypatch.setattr("svb.btm.pipeline.run_phase0", _raiser("phase 0 ran again"))
    monkeypatch.setattr("svb.btm.pipeline.train_expert", _raiser("an expert ran again"))
    monkeypatch.setattr("svb.btm.pipeline.merge_experts", _raiser("the merge ran again"))

    results = _results(run("B_btm_ssl", 7, "--resume"))

    assert set(results["transfer"]) == {"telugu"}
    assert [entry["resumed"] for entry in results["invocations"]] == [False, True]
    by_invocation = {name: stage["invocation"] for name, stage in results["stages"].items()}
    assert by_invocation["phase0"] == 0
    assert by_invocation["eval_en"] == 0
    assert by_invocation["eval_ja"] == 0
    assert by_invocation["transfer_telugu"] == 1


def test_a_finished_run_is_neither_reused_nor_discarded_without_being_told(
    run: Callable[..., Path],
) -> None:
    run("A_ssl")

    with pytest.raises(SystemExit, match=r"5 finished stage\(s\).*--resume.*--restart"):
        run("A_ssl")


def test_a_second_process_on_the_same_run_is_refused_before_it_writes(
    run: Callable[..., Path], tmp_path: Path
) -> None:
    from svb.stages import run_lock

    run_dir = tmp_path / "results" / "A_ssl" / "3" / "seed7"
    with run_lock(run_dir), pytest.raises(SystemExit, match="in use by another process"):
        run("A_ssl", 7, "--resume")

    # Nothing of the would-be second job reached the directory.
    assert not (run_dir / "resolved_config.yaml").exists()
    assert not (run_dir / "env.json").exists()


def test_a_restart_reuses_nothing_from_the_earlier_attempt(
    run: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = run("A_ssl")
    monkeypatch.setattr("svb.train.trainer.train", _raiser("trained again"))

    with pytest.raises(RuntimeError, match="trained again"):
        run("A_ssl", 7, "--restart")

    # And the attempt that died must not leave the earlier one's results behind
    # to be aggregated as if this seed were finished.
    assert not (run_dir / "results.json").exists()


def test_a_run_is_not_resumed_under_a_different_configuration(run: Callable[..., Path]) -> None:
    run("A_ssl")

    with pytest.raises(SystemExit, match=r"different configuration \(train\.finetune_epochs\)"):
        run("A_ssl", 7, "--resume", finetune_epochs=2)


def test_a_run_is_not_resumed_when_the_data_behind_its_vocabulary_changed(
    run: Callable[..., Path], tmp_path: Path
) -> None:
    """The config check cannot see the corpus. A vocabulary of the same size
    with different characters would load into the finished checkpoints and
    quietly relabel every head row."""
    run("A_ssl")
    english = tmp_path / "cv" / "en" / "validated.tsv"
    english.write_text(english.read_text(encoding="utf-8").replace("cat", "qzx"), encoding="utf-8")

    with pytest.raises(SystemExit, match="vocabulary built from the data now"):
        run("A_ssl", 7, "--resume")


# Three clips cannot spell a test set, and the run says so. That warning is the
# right behaviour and not what this test is about.
@pytest.mark.filterwarnings("ignore:.*absent from the vocabulary")
def test_an_hours_cap_gives_every_arm_and_seed_the_same_utterances(
    run: Callable[..., Path],
) -> None:
    """Three one-second clips fit under a cap of 3.5 seconds, out of six."""
    cap = {"max_train_hours": 3.5 / 3600, "max_val_hours": 1.5 / 3600}
    first = _results(run("A_ssl", 7, **cap))["data"]
    second = _results(run("C_btm_scratch", 8, **cap))["data"]

    english = first["languages"]["en"]
    assert (english["train"]["n_selected"], english["train"]["n_available"]) == (3, 6)
    assert english["train"]["binding"] is True
    assert english["validation"]["n_selected"] == 1
    # The held-out language is capped by the same rule as the training ones.
    assert first["languages"]["telugu"]["train"]["n_selected"] == 3
    assert first == second


@pytest.mark.filterwarnings("ignore:.*absent from the vocabulary")
@pytest.mark.parametrize("arm", ["A_ssl", "B_btm_ssl"])
def test_the_cap_reaches_the_trainer_not_only_the_record(
    run: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """`data` in results.json is computed on its own; this checks the datasets
    that were actually handed to `train`, in every stage, through every module
    that calls it."""
    import svb.btm.pipeline as pipeline
    import svb.eval.transfer as transfer
    import svb.train.trainer as trainer

    real = trainer.train
    sizes: list[tuple[int, int]] = []

    def spy(model: Any, cfg: Any, train_ds: Any, val_ds: Any, *rest: Any, **kw: Any) -> Any:
        sizes.append((len(train_ds), len(val_ds)))
        return real(model, cfg, train_ds, val_ds, *rest, **kw)

    for module in (trainer, pipeline, transfer):
        monkeypatch.setattr(module, "train", spy)

    run(arm, 7, max_train_hours=3.5 / 3600, max_val_hours=1.5 / 3600)

    if arm == "A_ssl":
        # en, ja, then telugu's transfer fine-tune.
        assert sizes == [(3, 1), (3, 1), (3, 1)]
    else:
        # phase 0 over both languages, one expert each, then transfer.
        assert sizes == [(6, 2), (3, 1), (3, 1), (3, 1)]


def test_a_run_that_finished_nothing_can_be_resubmitted_under_a_corrected_config(
    run: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher always passes --resume, so a job that died in its first
    stage must not pin the run directory to the settings it died under."""
    import svb.train.trainer as trainer

    real = trainer.train
    monkeypatch.setattr("svb.train.trainer.train", _raiser("died in the first stage"))
    with pytest.raises(RuntimeError, match="died in the first stage"):
        run("A_ssl", 7, "--resume")
    monkeypatch.setattr("svb.train.trainer.train", real)

    results = _results(run("A_ssl", 7, "--resume", finetune_epochs=2))

    assert results["training"]["finetune_en"]["epochs_run"] == 2


def test_a_run_is_not_resumed_when_the_capped_subset_changed(
    run: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vocabulary check cannot see a corpus change that moves the hash
    cutoff by a few rows without changing the character set. The subset
    digest the first invocation recorded can."""
    import svb.cli as cli

    run("A_ssl", 7, max_train_hours=3.5 / 3600)
    real = cli.hours_cap_records

    def shifted(cfg: Any, specs: Any) -> dict[str, Any]:
        record = real(cfg, specs)
        record["languages"]["en"]["train"]["subset_sha1"] = "0" * 40
        return record

    monkeypatch.setattr(cli, "hours_cap_records", shifted)

    with pytest.raises(SystemExit, match="data is not what its finished stages"):
        run("A_ssl", 7, "--resume", max_train_hours=3.5 / 3600)


def test_the_starting_checkpoint_is_identified_by_content_not_by_path(
    run: Callable[..., Path], tmp_path: Path
) -> None:
    """A different mount of the same file resumes; a different file at the
    same path does not. The encoder is a stand-in here, so the file is only
    ever hashed."""
    first, second = tmp_path / "a" / "xeus.pth", tmp_path / "b" / "xeus.pth"
    for path in (first, second):
        path.parent.mkdir()
        path.write_bytes(b"weights v1")
    run("A_ssl", 7, model={"xeus_checkpoint": str(first)})

    results = _results(run("A_ssl", 7, "--resume", model={"xeus_checkpoint": str(second)}))
    assert [entry["resumed"] for entry in results["invocations"]] == [False, True]

    second.write_bytes(b"weights v2")
    with pytest.raises(SystemExit, match="xeus_checkpoint_sha256 is not what"):
        run("A_ssl", 7, "--resume", model={"xeus_checkpoint": str(second)})


def test_each_stage_orders_its_data_by_its_own_seed(
    run: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two stages of one run used to shuffle with the run seed and draw the
    same permutation; now each is handed its own."""
    import svb.eval.transfer as transfer
    import svb.train.trainer as trainer

    seeds: list[int | None] = []
    real = trainer.train

    def spy(*args: Any, **kwargs: Any) -> Any:
        seeds.append(kwargs.get("data_seed"))
        return real(*args, **kwargs)

    for module in (trainer, transfer):
        monkeypatch.setattr(module, "train", spy)
    run("A_ssl")

    assert None not in seeds
    assert len(set(seeds)) == len(seeds) == 3


def test_an_uncapped_run_says_so(run: Callable[..., Path]) -> None:
    data = _results(run("A_ssl"))["data"]

    assert data == {"max_train_hours": None, "max_val_hours": None, "languages": {}}


def _raiser(message: str) -> Callable[..., Any]:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(message)

    return fail
