"""Command-line entry point: ``svb run|aggregate|analyze``.

``run`` executes one (arm, scale, seed) end to end and writes a single
``results.json`` plus ``resolved_config.yaml``, ``env.json``,
``text_stats.json`` and a per-language predictions sidecar.

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ExperimentConfig, TextConfig, dump_config, load_config
from .provenance import dump_run_meta
from .seeding import set_all_seeds
from .text.stats import TextStats, collect_text_stats

if TYPE_CHECKING:
    from .data.registry import LangSpec
    from .model.ctc_vocab import CtcVocab
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
    """
    from .data.datasets import load_texts
    from .model.ctc_vocab import expand_vocab

    languages: dict[str, Any] = {}
    for spec in [*specs, *heldout]:
        is_heldout = spec in heldout
        lang_vocab = vocab
        lang_evicted: dict[str, int] = {}
        if is_heldout:
            lang_vocab, _, lang_evicted = expand_vocab(
                vocab, load_texts(spec, "train"), min_char_count=text_cfg.min_char_count
            )
        splits: dict[str, Any] = {}
        for split in SPLITS:
            stats = collect_text_stats(load_texts(spec, split), text_cfg.policy, lang_vocab)
            if split == "train":
                stats.evicted_by_floor = lang_evicted
            _warn_on_unknown_characters(spec.code, split, stats)
            splits[split] = stats.to_dict()
        languages[spec.code] = {
            "heldout": is_heldout,
            "word_boundary": spec.word_boundary,
            "splits": splits,
        }

    payload = {
        "normalizer_version": text_cfg.policy.version,
        "policy_hash": text_cfg.policy.policy_hash(),
        "min_char_count": text_cfg.min_char_count,
        "training_vocab_evicted": evicted,
        "languages": languages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def cmd_run(args: argparse.Namespace) -> None:
    import torch

    from .btm.pipeline import (
        build_training_vocab,
        merge_experts,
        run_phase0,
        train_experts,
    )
    from .data.collate import make_ctc_collate
    from .data.datasets import load_language
    from .data.registry import get_heldout, get_preset
    from .eval.evaluate import evaluate
    from .eval.transfer import transfer_one
    from .model.xeus_ctc import make_model
    from .train.trainer import train

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
    device = args.device

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
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out)
    dump_run_meta(out, policy=cfg.text.policy)
    set_all_seeds(cfg.seed)

    vocab, evicted = build_training_vocab(specs, cfg.text)
    vocab.save(out / "vocab.json")
    write_text_stats(out / "text_stats.json", specs, heldout, cfg.text, vocab, evicted)
    results = run_manifest(cfg, specs, heldout)
    # Two collates: training truncates long audio and drops the transcripts that
    # no longer fit, evaluation does neither — a truncated test utterance scored
    # against its full reference is a fabricated error rate.
    train_collate = make_ctc_collate(
        vocab, max_audio_samples=cfg.train.max_audio_samples, drop_overlong=True, drop_empty=True
    )
    eval_collate = make_ctc_collate(vocab)

    results["training"] = {}
    if cfg.uses_btm:
        phase0_result = run_phase0(cfg, specs, vocab, out / "phase0", device)
        phase0 = phase0_result.checkpoint
        results["training"]["phase0"] = training_record(phase0_result)
        expert_results = train_experts(cfg, phase0, specs, vocab, out / "experts", device)
        for code, expert in expert_results.items():
            results["training"][f"expert_{code}"] = training_record(expert)
        experts = {code: expert.checkpoint for code, expert in expert_results.items()}
        merged_path = merge_experts(
            experts,
            cfg.merge_strategy,
            out / "merged",
            base_ckpt=phase0,
            seed=cfg.seed,
            merge_head=cfg.merge_head,
        )
        # Evaluate the merged model in-distribution on every language's test split.
        merged_model = make_model(cfg, vocab.size)
        merged_model.load_state_dict(torch.load(merged_path, map_location=device))
        for spec in specs:
            test_ds = load_language(spec, "test", None)
            r = evaluate(
                merged_model,
                test_ds,
                vocab,
                eval_collate,
                device,
                cfg.optim.batch_size,
                save_predictions=_predictions_path(out, spec.code),
                spec=spec,
            )
            results["in_distribution"][spec.code] = {
                "wer": r.wer,
                "cer": r.cer,
                "n": r.n,
                "n_empty_refs": r.n_empty_refs,
            }
        transfer_init: Path | None = merged_path
    else:
        # Arm A: independent per-language fine-tune from the SSL encoder.
        for spec in specs:
            model = make_model(cfg, vocab.size)
            lang_dir = out / "finetune" / spec.code
            tr = load_language(spec, "train", cfg.train.max_audio_samples)
            va = load_language(spec, "validation", cfg.train.max_audio_samples)
            res = train(
                model, cfg, tr, va, train_collate, cfg.train.finetune_epochs, lang_dir, device
            )
            results["training"][f"finetune_{spec.code}"] = training_record(res)
            model.load(res.checkpoint)
            test_ds = load_language(spec, "test", None)
            r = evaluate(
                model,
                test_ds,
                vocab,
                eval_collate,
                device,
                cfg.optim.batch_size,
                save_predictions=_predictions_path(out, spec.code),
                spec=spec,
            )
            results["in_distribution"][spec.code] = {
                "wer": r.wer,
                "cer": r.cer,
                "n": r.n,
                "n_empty_refs": r.n_empty_refs,
            }
        transfer_init = None  # arm A transfers from the bare SSL encoder

    # Held-out transfer (all arms).
    results["transfer"] = {}
    for held in heldout:
        transferred = transfer_one(
            cfg, transfer_init, vocab, held, out / "transfer" / held.code, device
        )
        # to_record carries the derived split's policy and test-set digest
        # alongside the metrics: the held-out corpora ship no partition, so
        # which utterances were scored is part of the number.
        results["transfer"][held.code] = transferred.to_record()

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
    from .report.analyze import render_metric_tables

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
        except ValueError as exc:
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

    rows = collect_durations(specs_for_scope(args.scope), min_train_hours=args.min_train_hours)
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
    r.add_argument("--config", default=None, help="optional base YAML")
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
    d.set_defaults(func=cmd_data_stats)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
