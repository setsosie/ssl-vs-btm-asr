"""Command-line entry point: ``svb run|aggregate|analyze|data-stats``.

``run`` executes one (arm, scale, seed) end to end and writes a single
``results.json`` plus ``resolved_config.yaml``, ``env.json``,
``text_stats.json`` and a per-language predictions sidecar. It records each
stage in ``stages.json`` as it finishes, and ``--resume`` continues from there.

``aggregate`` reads every seed's results for an (arm, scale) and reports WER,
CER and the per-language primary metric as mean ± std across seeds.

``analyze`` reads the sidecars instead, and reports what a single run's test set
leaves uncertain: utterance-level bootstrap intervals, and paired permutation
tests between two arms at the same seed. It writes ``tables/``.

The split between the last two is the point. Seed spread and test-set sampling
uncertainty are different quantities, and a five-seed standard deviation is not
a confidence interval.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .config import (
    RESOLVED_CONFIG,
    ExperimentConfig,
    TextConfig,
    TrainConfig,
    config_differences,
    dump_config,
    load_config,
    resolved_config,
)
from .data.commonvoice_local import DEFAULT_TRAIN_SOURCE, TRAIN_SOURCES
from .provenance import dump_run_meta
from .seeding import set_all_seeds
from .text.normalize import policy_name
from .text.registry import policies_for_specs
from .text.stats import TextStats, collect_text_stats

if TYPE_CHECKING:
    from .data.registry import LangSpec
    from .eval.evaluate import EvalResult
    from .model.ctc_vocab import CtcVocab
    from .model.xeus_ctc import XeusCTC
    from .stages import StageOutput
    from .train.trainer import TrainResult

DEFAULT_RESULTS_ROOT = Path("results")
ARMS = ("A_ssl", "B_btm_ssl", "C_btm_scratch")
# Scales are strings because they are directory-name keys, not counts: nothing
# does arithmetic on them and they must round-trip through paths unchanged.
SCALES = ("3", "16", "64")


def results_root(explicit: str | None) -> Path:
    """Where a run reads and writes its results.

    Resolution order: the flag, then ``$SVB_RESULTS_ROOT``, then ``./results``.
    A bare relative default is CWD-dependent, which on a cluster means results
    land wherever the job happened to start; the environment variable is how a
    launcher points them at scratch without editing the command.
    """
    if explicit:
        return Path(explicit)
    from_env = os.environ.get("SVB_RESULTS_ROOT")
    return Path(from_env) if from_env else DEFAULT_RESULTS_ROOT


def _run_dir(root: Path, arm: str, scale: str, seed: int) -> Path:
    return root / arm / scale / f"seed{seed}"


def _predictions_path(out: Path, code: str) -> Path:
    """Per-utterance sidecar for one language's evaluation.

    Bootstrap CIs and paired-permutation tests resample utterances, so they
    need these pairs on disk; a corpus-level WER cannot be resampled. Held-out
    transfer already wrote one, which left the in-distribution results — the
    ones behind the merging findings — with no way to attach an interval.
    """
    return out / "predictions" / f"{code}.json"


SPLITS = ("train", "validation", "test")


def run_manifest(
    cfg: ExperimentConfig, specs: list[LangSpec], heldout: list[LangSpec]
) -> dict[str, Any]:
    """The identifying header of a run's ``results.json``.

    The two language lists belong here rather than in ``resolved_config.yaml``.
    They are *resolved* from ``configs/scales/*.yaml`` at run time, not set by
    the run, and the dumped config is the configuration as it was rather than
    what it went on to select. Recording them makes the held-out set a fact of
    the results file, so a reader never has to infer it from which keys happen
    to appear under ``transfer``.
    """
    return {
        "arm": cfg.arm,
        "scale": cfg.scale,
        "seed": cfg.seed,
        "languages": [spec.code for spec in specs],
        "heldout_langs": [spec.code for spec in heldout],
        "in_distribution": {},
    }


# Above this share of unknown characters, the vocabulary's coverage is a fact
# about the result rather than a footnote: it is a floor under the error rate
# that no amount of training removes.
UNK_RATE_WARNING = 0.001


def training_record(result: TrainResult) -> dict[str, Any]:
    """What one training stage did, for ``results.json``.

    The dropped-pair and audio-guard counts were only ever printed. They are the
    difference between the split a reader can count and the data the model
    actually saw, so they belong in an artifact rather than in scrollback.
    """
    return {
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "n_dropped_unalignable": result.n_dropped_unalignable,
        "n_at_audio_guard": result.n_at_audio_guard,
    }


def _warn_on_unknown_characters(code: str, split: str, stats: TextStats) -> None:
    """Say so when a split carries characters the vocabulary cannot represent.

    The number always reaches ``text_stats.json``, but a file nobody opens is
    not a warning. A language whose references contain characters the model has
    no id for has an irreducible error floor, and a reader comparing its number
    to another language's needs to know that before quoting it.
    """
    if stats.unk_rate <= UNK_RATE_WARNING:
        return
    warnings.warn(
        f"{code}/{split}: {stats.unk_rate:.3%} of normalized reference characters "
        f"({stats.unk_chars} of {stats.n_chars_normalized}) are absent from the vocabulary "
        "and cannot be produced at any quality, so this language's error rate has a floor "
        "under it; see unk_rate in text_stats.json",
        UserWarning,
        stacklevel=3,
    )


def write_text_stats(
    path: Path,
    specs: list[LangSpec],
    heldout: list[LangSpec],
    text_cfg: TextConfig,
    vocab: CtcVocab,
    evicted: dict[str, int],
    train_cfg: TrainConfig | None = None,
) -> Path:
    """Write the per-language evidence for the normalization policy.

    A policy is a set of claims about the corpus — that the vocabulary covers
    the test set, that digits are rare, that a language declared to use word
    boundaries writes them. This puts each claim in the results directory as a
    number, so a reader can check it without re-running anything.

    Held-out languages are measured against their *expanded* vocabulary, not the
    training one. The expansion is recomputed here rather than shared with
    ``transfer_one``; it is deterministic and reads only text, and scoring
    Telugu against an all-Latin training vocab would report an unknown rate near
    1.0 and bury the number the transfer experiment depends on.

    ``train_cfg`` says which rows each split is — the Common Voice training
    source and the hours caps — so the statistics describe the transcripts the
    run trained and validated on rather than the splits they were cut from.
    """
    from .data.datasets import load_texts
    from .model.ctc_vocab import expand_vocab
    from .text.registry import policy_for_language

    train_cfg = train_cfg or TrainConfig()

    def texts(spec: LangSpec, split: str) -> list[str]:
        return load_texts(
            spec,
            split,
            train_source=train_cfg.cv_train_source,
            max_hours=train_cfg.max_hours(split),
        )

    languages: dict[str, Any] = {}
    for spec in [*specs, *heldout]:
        is_heldout = spec in heldout
        policy = policy_for_language(spec.code, text_cfg.override or spec.normalizer)
        lang_vocab = vocab
        lang_evicted: dict[str, int] = {}
        if is_heldout:
            lang_vocab, _, lang_evicted = expand_vocab(
                vocab,
                texts(spec, "train"),
                spec.code,
                policy,
                min_char_count=text_cfg.min_char_count,
            )
        splits: dict[str, Any] = {}
        for split in SPLITS:
            stats = collect_text_stats(texts(spec, split), policy, lang_vocab)
            if split == "train":
                stats.evicted_by_floor = lang_evicted
            _warn_on_unknown_characters(spec.code, split, stats)
            splits[split] = stats.to_dict()
        languages[spec.code] = {
            "heldout": is_heldout,
            "word_boundary": spec.word_boundary,
            # Which rules produced the numbers below. Removal tallies are not
            # comparable across languages that ran under different policies, so
            # a reader grouping them has to group by this.
            "policy": policy_name(policy),
            "policy_hash": policy.policy_hash(),
            "splits": splits,
        }

    payload = {
        "cv_train_source": train_cfg.cv_train_source,
        "max_train_hours": train_cfg.max_train_hours,
        "max_val_hours": train_cfg.max_val_hours,
        "override": text_cfg.override,
        "min_char_count": text_cfg.min_char_count,
        "training_vocab_evicted": evicted,
        "languages": languages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def hours_cap_records(cfg: ExperimentConfig, specs: list[LangSpec]) -> dict[str, Any]:
    """What the hours caps kept of each language's training and validation audio.

    A capped run trained on a subset, and "ten hours of English" does not say
    which ten. The digest does: two runs that record the same one read the same
    utterances, which is what lets a reader confirm that every arm and every
    seed of a study was given the same data.
    """
    from .data.datasets import hours_cap_selection

    languages: dict[str, Any] = {}
    for spec in specs:
        for split in ("train", "validation"):
            cap = cfg.train.max_hours(split)
            if cap is None:
                continue
            selection = hours_cap_selection(spec, split, cap, cfg.train.cv_train_source)
            languages.setdefault(spec.code, {})[split] = selection.to_record()
    return {
        "max_train_hours": cfg.train.max_train_hours,
        "max_val_hours": cfg.train.max_val_hours,
        "languages": languages,
    }


def metrics_record(result: EvalResult) -> dict[str, Any]:
    """One language's scored evaluation, for ``results.json``."""
    return {
        "wer": result.wer,
        "cer": result.cer,
        "n": result.n,
        "n_empty_refs": result.n_empty_refs,
    }


def _require_same_config(out: Path, cfg: ExperimentConfig, specs: list[LangSpec]) -> None:
    """Refuse to resume a run that was started under a different configuration.

    Reusing a phase 0 trained at one learning rate under experts trained at
    another produces a results file that describes neither. The check reads the
    config the earlier invocation wrote, so it is a comparison between what ran
    and what is about to, not between two files someone edited.
    """
    from .stages import finished_stages

    recorded_path = out / RESOLVED_CONFIG
    # Nothing finished means nothing will be reused, so there is nothing for a
    # changed setting to be inconsistent with: a job that died in its first
    # stage can be resubmitted under a corrected config.
    if not recorded_path.exists() or not finished_stages(out):
        return
    recorded = yaml.safe_load(recorded_path.read_text(encoding="utf-8")) or {}
    # Round-tripped through YAML so the comparison is between two things that
    # have both been through the same serializer.
    current = yaml.safe_load(yaml.safe_dump(resolved_config(cfg, specs)))
    differences = config_differences(recorded, current)
    if differences:
        raise SystemExit(
            f"[svb] cannot resume {out}: it was started under a different configuration "
            f"({', '.join(differences)}). Finish it with the settings in its "
            f"{RESOLVED_CONFIG}, or start it again without --resume."
        )


def _invocation(env_path: Path) -> dict[str, Any]:
    """What identifies this process in the stage ledger, read back from env.json."""
    meta = json.loads(env_path.read_text(encoding="utf-8"))
    return {key: meta.get(key) for key in ("timestamp_utc", "git_sha", "git_dirty")}


def cmd_run(args: argparse.Namespace) -> None:
    from .data.registry import get_heldout, get_preset
    from .stages import RunLockedError, run_lock

    # Only flags the caller actually passed become overrides, so an unset flag
    # cannot outrank the YAML with the parser's default.
    overrides = {
        key: value
        for key, value in (
            ("merge_strategy", args.merge_strategy),
            ("merge_head", args.merge_head),
        )
        if value is not None
    }
    cfg = load_config(
        args.arm, args.scale, args.seed, yaml_path=args.config, overrides=overrides or None
    )

    # Resolve the languages before anything is written. An unpopulated preset
    # raises, and raising after the run directory exists leaves behind the two
    # files a started run writes first — a directory indistinguishable from a
    # job that died in training, one per seed, on every scheduler slot the
    # matrix submitted.
    try:
        specs = get_preset(cfg.scale)
        heldout = get_heldout()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[svb] {exc}") from exc

    out = _run_dir(results_root(args.results_root), cfg.arm, cfg.scale, cfg.seed)
    # The lock comes before everything else that touches the directory. A second
    # job on the same cell has to find out it is the second before it has
    # replaced the first one's config, environment record or ledger.
    try:
        with run_lock(out):
            _execute_run(
                cfg, specs, heldout, out, args.device, resume=args.resume, restart=args.restart
            )
    except RunLockedError as exc:
        raise SystemExit(f"[svb] {exc}") from exc


def _execute_run(
    cfg: ExperimentConfig,
    specs: list[LangSpec],
    heldout: list[LangSpec],
    out: Path,
    device: str,
    resume: bool,
    restart: bool,
) -> None:
    """One (arm, scale, seed), with the run directory already held."""
    import torch

    from .btm.pipeline import (
        build_training_vocab,
        load_split,
        merge_experts,
        run_phase0,
        train_expert,
    )
    from .data.collate import make_ctc_collate
    from .data.datasets import load_language
    from .eval.evaluate import evaluate
    from .eval.transfer import transfer_one
    from .model.ctc_vocab import CtcVocab
    from .model.xeus_ctc import make_model
    from .stages import StageLedger, StageOutput, finished_stages
    from .train.trainer import train

    finished = finished_stages(out)
    if finished and not (resume or restart):
        # Hours of training are behind those stages. Starting over is a decision
        # and has to be said, not the default meaning of running a command twice.
        raise SystemExit(
            f"[svb] {out} already holds {len(finished)} finished stage(s) "
            f"({', '.join(finished)}). Pass --resume to continue from them, or --restart "
            "to discard them and start again."
        )
    if resume:
        _require_same_config(out, cfg, [*specs, *heldout])
    dump_config(cfg, out, [*specs, *heldout])
    env_path = dump_run_meta(
        out, policies=policies_for_specs([*specs, *heldout], cfg.text.override)
    )
    ledger = StageLedger(out, cfg.seed, resume=resume, invocation=_invocation(env_path))
    # results.json is what marks a seed as finished, and it is rewritten when
    # this invocation finishes. Left in place, one from an earlier attempt would
    # let a run that is halfway through being redone be aggregated as complete.
    (out / "results.json").unlink(missing_ok=True)
    # Covers what happens between stages. Each stage reseeds itself from the run
    # seed and its own name, so nothing it draws depends on this call.
    set_all_seeds(cfg.seed)

    vocab, evicted = build_training_vocab(specs, cfg.text, cfg.train)
    if resume and finished and (out / "vocab.json").exists():
        # The config check cannot see the data. If the corpus under $CV_ROOT has
        # changed since the finished stages ran, the vocabulary built now maps
        # ids to different characters than the one their checkpoints were
        # trained with — and at the same size that would load without a word.
        recorded = CtcVocab.load(out / "vocab.json")
        if recorded.id_to_char != vocab.id_to_char:
            raise SystemExit(
                f"[svb] cannot resume {out}: the training vocabulary built from the data now "
                f"({vocab.size} entries) is not the one its finished stages were trained with "
                f"({recorded.size} entries). The corpora have changed since; start the run "
                "again with --restart."
            )
    vocab.save(out / "vocab.json")
    write_text_stats(out / "text_stats.json", specs, heldout, cfg.text, vocab, evicted, cfg.train)
    results = run_manifest(cfg, specs, heldout)
    results["data"] = hours_cap_records(cfg, [*specs, *heldout])
    # Two collates: training truncates long audio and drops the transcripts that
    # no longer fit, evaluation does neither — a truncated test utterance scored
    # against its full reference is a fabricated error rate.
    train_collate = make_ctc_collate(
        vocab, max_audio_samples=cfg.train.max_audio_samples, drop_overlong=True, drop_empty=True
    )
    eval_collate = make_ctc_collate(vocab)

    def trained(result: TrainResult) -> StageOutput:
        """A finished training stage: its record, and the checkpoint it names.

        The checkpoint path is stored as the trainer returned it, relative to
        the run, and later stages read it back from the record — not from a
        path re-derived here, which would be a second source of truth for it.
        """
        record = {**training_record(result), "checkpoint": os.path.relpath(result.checkpoint, out)}
        return StageOutput(result=record, artifacts=[result.checkpoint])

    def scored(model: XeusCTC, spec: LangSpec) -> StageOutput:
        sidecar = _predictions_path(out, spec.code)
        result = evaluate(
            model,
            load_language(spec, "test", None),
            vocab,
            eval_collate,
            device,
            cfg.optim.batch_size,
            save_predictions=sidecar,
            spec=spec,
        )
        return StageOutput(result=metrics_record(result), artifacts=[sidecar])

    results["training"] = {}
    if cfg.uses_btm:
        phase0 = ledger.run(
            "phase0", lambda: trained(run_phase0(cfg, specs, vocab, out / "phase0", device))
        )
        results["training"]["phase0"] = phase0
        phase0_ckpt = out / phase0["checkpoint"]

        experts: dict[str, Path] = {}
        for spec in specs:

            def expert(spec: LangSpec) -> StageOutput:
                return trained(train_expert(cfg, phase0_ckpt, spec, vocab, out / "experts", device))

            record = ledger.run(f"expert_{spec.code}", partial(expert, spec))
            results["training"][f"expert_{spec.code}"] = record
            experts[spec.code] = out / record["checkpoint"]

        def merge() -> StageOutput:
            path = merge_experts(
                experts,
                cfg.merge_strategy,
                out / "merged",
                base_ckpt=phase0_ckpt,
                seed=cfg.seed,
                merge_head=cfg.merge_head,
            )
            return StageOutput(result={"checkpoint": os.path.relpath(path, out)}, artifacts=[path])

        merged_path = out / ledger.run("merge", merge)["checkpoint"]

        # Evaluate the merged model in-distribution on every language's test
        # split. Built once and only if some evaluation still has to run: a
        # resume that has them all should not load the encoder to do nothing.
        merged_models: list[XeusCTC] = []

        def score_merged(spec: LangSpec) -> StageOutput:
            if not merged_models:
                model = make_model(cfg, vocab.size)
                model.load_state_dict(torch.load(merged_path, map_location=device))
                merged_models.append(model)
            return scored(merged_models[0], spec)

        for spec in specs:
            results["in_distribution"][spec.code] = ledger.run(
                f"eval_{spec.code}", partial(score_merged, spec)
            )
        merged_models.clear()
        transfer_init: Path | None = merged_path
    else:
        # Arm A: independent per-language fine-tune from the SSL encoder.
        for spec in specs:

            def finetune(spec: LangSpec) -> StageOutput:
                return trained(
                    train(
                        make_model(cfg, vocab.size),
                        cfg,
                        load_split(cfg, spec, "train"),
                        load_split(cfg, spec, "validation"),
                        train_collate,
                        cfg.train.finetune_epochs,
                        out / "finetune" / spec.code,
                        device,
                    )
                )

            record = ledger.run(f"finetune_{spec.code}", partial(finetune, spec))
            results["training"][f"finetune_{spec.code}"] = record

            def score_finetuned(spec: LangSpec, checkpoint: Path) -> StageOutput:
                model = make_model(cfg, vocab.size)
                model.load(checkpoint)
                return scored(model, spec)

            results["in_distribution"][spec.code] = ledger.run(
                f"eval_{spec.code}", partial(score_finetuned, spec, out / record["checkpoint"])
            )
        transfer_init = None  # arm A transfers from the bare SSL encoder

    # Held-out transfer (all arms).
    results["transfer"] = {}
    for held in heldout:

        def transfer(held: LangSpec) -> StageOutput:
            transfer_dir = out / "transfer" / held.code
            transferred = transfer_one(cfg, transfer_init, vocab, held, transfer_dir, device)
            artifacts = [transfer_dir / "predictions.json"]
            training: dict[str, Any] = {}
            if transferred.training is not None:
                training = trained(transferred.training).result
                artifacts.append(transferred.training.checkpoint)
            # to_record carries the derived split's policy and test-set digest
            # alongside the metrics: the held-out corpora ship no partition, so
            # which utterances were scored is part of the number.
            return StageOutput(
                result={"metrics": transferred.to_record(), "training": training},
                artifacts=artifacts,
            )

        record = ledger.run(f"transfer_{held.code}", partial(transfer, held))
        results["transfer"][held.code] = record["metrics"]
        results["training"][f"transfer_{held.code}"] = record["training"]

    # Which stages ran in which invocation, what each cost in wall-clock time,
    # and the code version of every invocation that contributed.
    results.update(ledger.summary())
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[svb] wrote {out / 'results.json'}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    from .report.aggregate import aggregate_runs, load_runs, to_json, to_markdown

    agg = aggregate_runs(load_runs(results_root(args.results_root), args.arm, args.scale))
    if args.as_json:
        # Nothing else on stdout, so the command can be piped straight into a
        # parser without a filtering step that would have to know the layout.
        print(json.dumps(to_json(agg), ensure_ascii=False, indent=2))
        return
    print(to_markdown(agg), end="")


def cmd_analyze(args: argparse.Namespace) -> None:
    from .report.aggregate import load_runs
    from .report.analyze import render_metric_tables, require_same_policies

    root = results_root(args.results_root)
    runs = [r.path for r in load_runs(root, args.arm, args.scale) if _wanted(r.seed, args.seeds)]
    if not runs:
        raise FileNotFoundError(
            f"no runs for {args.arm}/{args.scale} with seed(s) {sorted(args.seeds)} under {root}"
        )

    comparisons: list[tuple[Path, Path]] = []
    if args.compare_to:
        # Pair within a seed. Two arms at the same seed scored the same test
        # split, which is what makes the test paired; across seeds it would not
        # be, and the sidecar reference check would reject it anyway.
        other = {r.seed: r.path for r in load_runs(root, args.compare_to, args.scale)}
        comparisons = [
            (run, other[seed])
            for run, seed in zip(runs, _seeds_of(runs), strict=True)
            if seed in other
        ]
        # Before any sidecar is read. The per-utterance reference check catches
        # a policy difference too, but only for languages both runs evaluated,
        # and it reports a string difference where this reports which language
        # and which two policies.
        for left, right in comparisons:
            require_same_policies(left, right)

    written = render_metric_tables(
        scale=args.scale,
        runs=runs,
        comparisons=comparisons,
        out_dir=Path(args.tables_dir),
        n_resamples=args.resamples,
    )
    for path in written:
        print(f"[svb] wrote {path}")


SCOPES = (*SCALES, "heldout", "all")


def specs_for_scope(scope: str) -> list[LangSpec]:
    """The languages one ``--scope`` names, each listed once.

    ``all`` sweeps every preset plus the held-out set. A preset that is still a
    placeholder raises on load, which is right for a run and wrong here: this
    command reports what is on disk, so an unpopulated scale is named and the
    sweep continues. English is in both the 3 and 16 presets, so duplicates are
    dropped — counting a language twice would double its hours in the total.
    """
    from .data.registry import get_heldout, get_preset

    collected: list[LangSpec] = []
    scales = SCALES if scope == "all" else ((scope,) if scope in SCALES else ())
    for scale in scales:
        try:
            collected += get_preset(scale)
        except (ValueError, FileNotFoundError) as exc:
            print(f"[svb] scale {scale}: {exc}")
    if scope in ("all", "heldout"):
        collected += get_heldout()

    seen: set[tuple[str, str]] = set()
    unique: list[LangSpec] = []
    for spec in collected:
        key = (spec.source, spec.code)
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def cmd_data_stats(args: argparse.Namespace) -> None:
    from .report.durations import collect_durations, write_tables

    rows = collect_durations(
        specs_for_scope(args.scope),
        min_train_hours=args.min_train_hours,
        train_source=args.cv_train_source,
    )
    for path in write_tables(rows, Path(args.tables_dir), min_train_hours=args.min_train_hours):
        print(f"[svb] wrote {path}")


def _seeds_of(runs: list[Path]) -> list[int]:
    return [int(run.name.removeprefix("seed")) for run in runs]


def _wanted(seed: int, chosen: list[int]) -> bool:
    """No ``--seed`` means every seed that has results."""
    return not chosen or seed in chosen


def _add_results_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-root",
        dest="results_root",
        default=None,
        help="where runs are written (default: $SVB_RESULTS_ROOT, else ./results)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="svb")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one (arm, scale, seed)")
    r.add_argument("--arm", required=True, choices=ARMS)
    r.add_argument("--scale", required=True, choices=SCALES)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML of settings; repeatable, each layered over the last "
        "(e.g. --config configs/base.yaml --config configs/studies/small.yaml)",
    )
    # Both opt-in. A run directory with finished stages is neither reused nor
    # discarded unless the caller says which: reusing by accident mixes two
    # attempts, and discarding by accident throws away GPU-days.
    again = r.add_mutually_exclusive_group()
    again.add_argument(
        "--resume",
        action="store_true",
        help="continue a run from the stages it already finished (see stages.json)",
    )
    again.add_argument(
        "--restart",
        action="store_true",
        help="discard the stages a run already finished and start it again",
    )
    r.add_argument(
        "--merge-strategy",
        dest="merge_strategy",
        default=None,
        choices=["average", "ties", "dare_ties"],
    )
    # Encoder-only merging is the ablation for how much of the merging penalty
    # lives in the CTC head, so it needs to be reachable without editing a YAML.
    # Default None, not True: the flag must be distinguishable from its own
    # default, or passing nothing would override a `merge_head: false` set in
    # the YAML with the parser's idea of the default.
    r.add_argument(
        "--merge-head",
        dest="merge_head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="merge the CTC head along with the encoder (default: yes)",
    )
    r.add_argument("--device", default="cuda")
    _add_results_root(r)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("aggregate", help="mean±std across seeds (WER, CER, primary)")
    # Same choices as `run`: unvalidated, a misspelled arm printed an empty
    # table, which reads like a run that has not happened rather than a typo.
    a.add_argument("--arm", required=True, choices=ARMS)
    a.add_argument("--scale", required=True, choices=SCALES)
    a.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit JSON on stdout instead of the Markdown table",
    )
    _add_results_root(a)
    a.set_defaults(func=cmd_aggregate)

    n = sub.add_parser(
        "analyze", help="utterance-level bootstrap CIs and paired tests, into tables/"
    )
    n.add_argument("--arm", required=True, choices=ARMS)
    n.add_argument("--scale", required=True, choices=SCALES)
    n.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        default=[],
        help="restrict to this seed; repeatable. Default: every seed with results.",
    )
    n.add_argument(
        "--compare-to",
        dest="compare_to",
        default=None,
        choices=ARMS,
        help="also run a paired permutation test against this arm, seed for seed",
    )
    n.add_argument("--tables-dir", dest="tables_dir", default="tables")
    n.add_argument(
        "--resamples",
        type=int,
        default=10_000,
        help="bootstrap and permutation draws (default: 10000)",
    )
    _add_results_root(n)
    n.set_defaults(func=cmd_analyze)

    d = sub.add_parser(
        "data-stats", help="per-language audio hours and utterance counts, into tables/"
    )
    d.add_argument("--scope", default="all", choices=SCOPES)
    d.add_argument("--tables-dir", dest="tables_dir", default="tables")
    d.add_argument(
        "--min-train-hours",
        dest="min_train_hours",
        type=float,
        default=0.0,
        help="flag languages with less training audio than this (default: no threshold)",
    )
    d.add_argument(
        "--cv-train-source",
        dest="cv_train_source",
        default=DEFAULT_TRAIN_SOURCE,
        choices=TRAIN_SOURCES,
        help="which Common Voice rows count as training audio (default: %(default)s)",
    )
    d.set_defaults(func=cmd_data_stats)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
