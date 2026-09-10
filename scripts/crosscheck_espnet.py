#!/usr/bin/env python3
"""Check the standalone XEUS encoder against the reference ESPnet implementation.

`src/svb/model/xeus_standalone.py` re-expresses the XEUS encoder in PyTorch so
the published checkpoint loads without ESPnet. A reimplementation is a
correctness risk in its own right — an earlier version of this one had
E-Branchformer forward deviations from the reference — and the only thing that
retires that risk is loading the same checkpoint into both and comparing the
features they produce.

That cannot happen inside this project's environment. The reference XEUS load
needs a fork of ESPnet pinned to `numpy<1.24`, which has no wheel for Python 3.12
and whose source build fails there because `distutils` left the standard
library. So this is a standalone script, run from a separate Python 3.10
environment; see `scripts/espnet-crosscheck/requirements.txt` and
`docs/espnet-crosscheck.md`. It imports nothing from the installed `svb`
package — the standalone encoder is loaded from its source file — and ESPnet is
imported lazily, so this file imports cleanly on 3.12 with no fork present.

Two cases are run, not one. A single utterance answers "is the architecture
right". A padded batch of two answers a question the single case cannot: the
encoder's two time-axis convolutions do not mask padded frames, matching the
reference the published weights were pretrained under, so an utterance's
encoding depends slightly on what shares its batch. Per-utterance deltas on the
padded batch say whether any disagreement concentrates in the padded utterance,
which is where a masking divergence would show up.

Usage:

    python scripts/crosscheck_espnet.py --checkpoint /path/to/xeus.pth \\
        --device cuda --out results/crosscheck/report.json

Exits 0 when both cases agree within `--tol`, 1 when they do not, 2 when the
reference could not be loaded at all.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

#: Sample rate the encoder expects.
SAMPLE_RATE = 16_000
#: Fixed seed for the synthetic audio, so two invocations compare the same input.
WAVEFORM_SEED = 0
DEFAULT_TOL = 1e-3
RECIPE = "scripts/espnet-crosscheck"


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    """One batch of audio fed to both encoders."""

    name: str
    waveforms: torch.Tensor  # (batch, samples)
    lengths: torch.Tensor  # (batch,) true sample counts


def default_cases(seed: int = WAVEFORM_SEED) -> list[Case]:
    """A single one-second utterance, and a padded batch of two.

    The second case pads a 7000-sample utterance out to 16000 beside a full-length
    one. Both encoders see identical tensors, so any difference in how they treat
    the padded region is attributable to the implementations rather than the data.
    """
    generator = torch.Generator().manual_seed(seed)
    single = torch.randn(1, SAMPLE_RATE, generator=generator)
    pair = torch.randn(2, SAMPLE_RATE, generator=generator)
    return [
        Case("single", single, torch.tensor([SAMPLE_RATE])),
        Case("padded_batch", pair, torch.tensor([SAMPLE_RATE, 7_000])),
    ]


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass
class CaseResult:
    """What the two encoders did on one case."""

    name: str
    batch_size: int
    reference_shape: list[int]
    standalone_shape: list[int]
    shapes_agree: bool
    #: Largest absolute difference over *valid* frames, or None when the shapes
    #: disagree and no elementwise comparison is defined.
    max_abs_delta: float | None
    #: The same, per utterance. A batch-wide maximum cannot say which utterance
    #: it came from, which is exactly what the padded case is asking.
    per_utterance_max_abs_delta: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "reference_shape": self.reference_shape,
            "standalone_shape": self.standalone_shape,
            "shapes_agree": self.shapes_agree,
            "max_abs_delta": self.max_abs_delta,
            "per_utterance_max_abs_delta": self.per_utterance_max_abs_delta,
        }


def _valid_frames(lengths: torch.Tensor | None, index: int, frames: int) -> int:
    """How many output frames belong to utterance ``index``."""
    if lengths is None:
        return frames
    return min(int(lengths[index]), frames)


def compare(reference: Any, standalone: Any, cases: list[Case]) -> list[CaseResult]:
    """Run both encoders over every case and measure where they differ.

    Only frames inside an utterance's own output length are compared. Frames past
    it are padding in both encoders and are sliced off before the CTC head ever
    sees them, so a difference there is not one any reported number can carry.
    """
    results: list[CaseResult] = []

    for case in cases:
        with torch.no_grad():
            ref_feats, ref_lengths = reference.encode(
                case.waveforms, case.lengths, use_final_output=True
            )
            std_feats, _ = standalone.encode(case.waveforms, case.lengths, use_final_output=True)

        ref_shape = list(ref_feats.shape)
        std_shape = list(std_feats.shape)
        if ref_shape != std_shape:
            results.append(
                CaseResult(
                    name=case.name,
                    batch_size=int(case.waveforms.shape[0]),
                    reference_shape=ref_shape,
                    standalone_shape=std_shape,
                    shapes_agree=False,
                    max_abs_delta=None,
                    per_utterance_max_abs_delta=None,
                )
            )
            continue

        delta = (ref_feats.float() - std_feats.float()).abs()
        frames = delta.shape[1]
        per_utterance = [
            float(delta[i, : _valid_frames(ref_lengths, i, frames)].max())
            for i in range(delta.shape[0])
        ]
        results.append(
            CaseResult(
                name=case.name,
                batch_size=int(case.waveforms.shape[0]),
                reference_shape=ref_shape,
                standalone_shape=std_shape,
                shapes_agree=True,
                max_abs_delta=max(per_utterance),
                per_utterance_max_abs_delta=per_utterance,
            )
        )

    return results


def worst_delta(results: list[CaseResult]) -> float | None:
    """The largest delta across cases, or None if any case has no delta at all.

    None rather than 0.0 on a shape mismatch: an absent measurement must not read
    as perfect agreement.
    """
    deltas = [r.max_abs_delta for r in results]
    if any(d is None for d in deltas):
        return None
    return max([d for d in deltas if d is not None], default=0.0)


def passed(results: list[CaseResult], tol: float) -> bool:
    """Whether every case agreed on shape and stayed within ``tol``."""
    if not results or not all(r.shapes_agree for r in results):
        return False
    worst = worst_delta(results)
    return worst is not None and worst <= tol


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    """The checkout this script lives in — scripts/ is one level down."""
    return Path(__file__).resolve().parents[1]


def load_standalone_module(root: Path | None = None) -> Any:
    """Import ``xeus_standalone`` from source, without installing the package.

    The reference environment has ESPnet's dependency pins in it and cannot also
    hold this project's, so `svb` is not importable there. The encoder module
    imports nothing from its own package, which is what makes this possible.
    """
    root = root or repo_root()
    path = root / "src" / "svb" / "model" / "xeus_standalone.py"
    if not path.exists():
        raise RuntimeError(f"cannot find the standalone encoder at {path}")
    spec = importlib.util.spec_from_file_location("xeus_standalone", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference(checkpoint: str, device: str) -> Any:
    """Build the reference XEUS model through ESPnet's SSL task.

    Imported here rather than at module scope so this file stays importable in
    the project environment, where the fork cannot be installed at all.
    """
    try:
        # Unresolvable here by construction: the fork cannot be installed in this
        # project's environment, which is the whole reason this script exists.
        # Ignored locally rather than configured away, so the exception stays
        # attached to the one line that needs it.
        from espnet2.tasks.ssl import SSLTask  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"espnet is not installed; build the reference environment from {RECIPE}"
        ) from exc

    model, _ = SSLTask.build_model_from_file(None, checkpoint, device)
    return model.eval()


def load_standalone(checkpoint: str, device: str) -> Any:
    """Build the standalone encoder from the same checkpoint."""
    module = load_standalone_module()
    return module.load_xeus_from_checkpoint(checkpoint, device).eval()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str:
    """Installed version of a distribution, or "absent"."""
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except Exception:
        return "absent"


def build_report(
    results: list[CaseResult], checkpoint: Path | None, tol: float, device: str
) -> dict[str, Any]:
    """Everything needed to read this run's verdict later, without rerunning it.

    The checkpoint digest is the load-bearing field: "the two implementations
    agree" is a claim about a specific set of weights, and a report that does not
    pin which ones cannot be checked against a later run.
    """
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "tol": tol,
        "device": device,
        "passed": passed(results, tol),
        "worst_max_abs_delta": worst_delta(results),
        "checkpoint": (
            None
            if checkpoint is None
            else {
                "name": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
        ),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": _version("numpy"),
            "espnet": _version("espnet"),
        },
        "cases": [r.to_dict() for r in results],
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"checkpoint : {(report['checkpoint'] or {}).get('name', '—')}",
        f"device     : {report['device']}   tolerance: {report['tol']}",
        "",
    ]
    for case in report["cases"]:
        if not case["shapes_agree"]:
            lines.append(
                f"  {case['name']:<13} SHAPE MISMATCH "
                f"{case['reference_shape']} vs {case['standalone_shape']}"
            )
            continue
        per_utt = ", ".join(f"{d:.3e}" for d in case["per_utterance_max_abs_delta"])
        lines.append(
            f"  {case['name']:<13} max|Δ| = {case['max_abs_delta']:.3e}   per-utterance: [{per_utt}]"
        )
    verdict = "PASS" if report["passed"] else "FAIL"
    lines += ["", f"{verdict}  (worst max|Δ| = {report['worst_max_abs_delta']})"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crosscheck_espnet",
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument("--checkpoint", required=True, help="path to the published XEUS checkpoint")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"maximum tolerated absolute feature difference (default: {DEFAULT_TOL})",
    )
    parser.add_argument("--out", default=None, help="write the JSON report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        reference = load_reference(args.checkpoint, args.device)
        standalone = load_standalone(args.checkpoint, args.device)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = compare(reference, standalone, default_cases())
    checkpoint = Path(args.checkpoint)
    report = build_report(
        results,
        checkpoint=checkpoint if checkpoint.exists() else None,
        tol=args.tol,
        device=args.device,
    )

    print(format_summary(report))
    if args.out:
        written = write_report(report, Path(args.out))
        print(f"\nwrote {written}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
