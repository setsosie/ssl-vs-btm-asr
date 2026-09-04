"""Command-line entry point: ``svb run|aggregate``.

``run`` executes one (arm, scale, seed) end to end and writes a single
``results.json`` plus ``resolved_config.yaml`` and ``env.json``. ``aggregate``
reads every seed's results for an (arm, scale) and prints mean ± std.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import dump_config, load_config
from .provenance import dump_run_meta
from .seeding import set_all_seeds

RESULTS_ROOT = Path("results")


def _run_dir(arm: str, scale: str, seed: int) -> Path:
    return RESULTS_ROOT / arm / scale / f"seed{seed}"


def _predictions_path(out: Path, code: str) -> Path:
    """Per-utterance sidecar for one language's evaluation.

    Bootstrap CIs and paired-permutation tests resample utterances, so they
    need these pairs on disk; a corpus-level WER cannot be resampled. Held-out
    transfer already wrote one, which left the in-distribution results — the
    ones behind the merging findings — with no way to attach an interval.
    """
    return out / "predictions" / f"{code}.json"


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

    cfg = load_config(
        args.arm,
        args.scale,
        args.seed,
        yaml_path=args.config,
        overrides={"merge_strategy": args.merge_strategy} if args.merge_strategy else None,
    )
    device = args.device
    out = _run_dir(cfg.arm, cfg.scale, cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out)
    dump_run_meta(out)
    set_all_seeds(cfg.seed)

    specs = get_preset(cfg.scale)
    vocab = build_training_vocab(specs)
    vocab.save(out / "vocab.json")
    # Two collates: training truncates long audio and drops the transcripts that
    # no longer fit, evaluation does neither — a truncated test utterance scored
    # against its full reference is a fabricated error rate.
    train_collate = make_ctc_collate(
        vocab, max_audio_samples=cfg.train.max_audio_samples, drop_overlong=True
    )
    eval_collate = make_ctc_collate(vocab)
    results: dict = {"arm": cfg.arm, "scale": cfg.scale, "seed": cfg.seed, "in_distribution": {}}

    if cfg.uses_btm:
        phase0 = run_phase0(cfg, specs, vocab, out / "phase0", device)
        experts = train_experts(cfg, phase0, specs, vocab, out / "experts", device)
        merged_path = merge_experts(experts, cfg.merge_strategy, out / "merged", base_ckpt=phase0)
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
            )
            results["in_distribution"][spec.code] = {"wer": r.wer, "cer": r.cer, "n": r.n}
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
            )
            results["in_distribution"][spec.code] = {"wer": r.wer, "cer": r.cer, "n": r.n}
        transfer_init = None  # arm A transfers from the bare SSL encoder

    # Held-out transfer (all arms).
    results["transfer"] = {}
    for held in get_heldout():
        r = transfer_one(cfg, transfer_init, vocab, held, out / "transfer" / held.code, device)
        results["transfer"][held.code] = {"wer": r.wer, "cer": r.cer, "n": r.n}

    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[svb] wrote {out / 'results.json'}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    from .stats.analysis import aggregate_seeds

    rows: dict[str, list[float]] = {}
    base = RESULTS_ROOT / args.arm / args.scale
    seeds = sorted(p for p in base.glob("seed*") if (p / "results.json").exists())
    for p in seeds:
        data = json.loads((p / "results.json").read_text())
        for section in ("in_distribution", "transfer"):
            for lang, m in data.get(section, {}).items():
                rows.setdefault(f"{section}/{lang}", []).append(m["wer"])
    print(f"# {args.arm} scale={args.scale}  ({len(seeds)} seeds)")
    print(f"{'metric':32s} {'mean':>8s} {'std':>7s}  n")
    for k in sorted(rows):
        agg = aggregate_seeds(rows[k])
        print(f"{k:32s} {agg.mean:8.2f} {agg.std:7.2f}  {agg.n_seeds}")


def main() -> None:
    p = argparse.ArgumentParser(prog="svb")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one (arm, scale, seed)")
    r.add_argument("--arm", required=True, choices=["A_ssl", "B_btm_ssl", "C_btm_scratch"])
    r.add_argument("--scale", required=True, choices=["3", "16", "64"])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--config", default=None, help="optional base YAML")
    r.add_argument(
        "--merge-strategy",
        dest="merge_strategy",
        default=None,
        choices=["average", "ties", "dare_ties"],
    )
    r.add_argument("--device", default="cuda")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("aggregate", help="mean±std across seeds")
    a.add_argument("--arm", required=True)
    a.add_argument("--scale", required=True)
    a.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
