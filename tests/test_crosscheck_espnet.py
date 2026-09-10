"""The ESPnet cross-check script, exercised on CPU with no ESPnet installed.

The real check needs a fork that cannot be installed in this environment, so
what is tested here is everything around the model load: how cases are built,
how two encoders are compared, how the tolerance decides the exit code, and what
lands in the report. Two fake encoders with a known offset stand in for the real
pair, which makes the expected deltas exact rather than approximate.
"""

from __future__ import annotations

import json
from types import ModuleType

import pytest
import torch


class FakeEncoder:
    """Deterministic stand-in with a constant, per-utterance feature value.

    ``encode`` mirrors the contract the real encoders share:
    ``(waveforms, wav_lengths, use_mask=..., use_final_output=...)`` returning
    ``(features, output_lengths)``. Because every frame of utterance *i* holds
    ``offsets[i]``, the difference between two of these is exactly the
    difference of their offsets — so a test can assert a delta rather than
    bracket it.
    """

    downsample = 320

    def __init__(self, offsets: float | list[float] = 0.0, hidden: int = 4) -> None:
        self.offsets = offsets
        self.hidden = hidden

    def _offset(self, index: int) -> float:
        if isinstance(self.offsets, list):
            return self.offsets[index]
        return self.offsets

    def encode(
        self,
        waveforms: torch.Tensor,
        wav_lengths: torch.Tensor,
        use_mask: bool = False,
        use_final_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(waveforms.shape[0])
        out_lengths = (wav_lengths // self.downsample).long()
        frames = int(out_lengths.max())
        feats = torch.zeros(batch, frames, self.hidden)
        for i in range(batch):
            feats[i] = self._offset(i)
        return feats, out_lengths


@pytest.fixture
def script(crosscheck_espnet: ModuleType) -> ModuleType:
    return crosscheck_espnet


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


def test_default_cases_are_a_single_utterance_and_a_padded_pair(script: ModuleType) -> None:
    cases = script.default_cases()

    assert [c.name for c in cases] == ["single", "padded_batch"]
    assert cases[0].waveforms.shape == (1, 16000)
    assert cases[0].lengths.tolist() == [16000]
    # The second utterance is shorter than the tensor it sits in, which is the
    # only way padding sensitivity can show up at all.
    assert cases[1].waveforms.shape == (2, 16000)
    assert cases[1].lengths.tolist() == [16000, 7000]


def test_default_cases_are_reproducible(script: ModuleType) -> None:
    assert torch.equal(script.default_cases()[0].waveforms, script.default_cases()[0].waveforms)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_identical_encoders_agree_exactly(script: ModuleType) -> None:
    results = script.compare(FakeEncoder(), FakeEncoder(), script.default_cases())

    assert [r.name for r in results] == ["single", "padded_batch"]
    assert all(r.shapes_agree for r in results)
    assert all(r.max_abs_delta == pytest.approx(0.0) for r in results)
    assert script.worst_delta(results) == pytest.approx(0.0)


def test_a_uniform_offset_shows_up_as_exactly_that_delta(script: ModuleType) -> None:
    results = script.compare(FakeEncoder(0.0), FakeEncoder(0.25), script.default_cases())

    assert all(r.max_abs_delta == pytest.approx(0.25) for r in results)


def test_per_utterance_deltas_isolate_the_padded_utterance(script: ModuleType) -> None:
    # The whole point of the padded case: an implementation that mishandles
    # padding disagrees on the short utterance and not on the long one. A single
    # batch-wide maximum would report 0.5 without saying which utterance it came
    # from.
    reference = FakeEncoder([0.0, 0.0])
    standalone = FakeEncoder([0.0, 0.5])

    padded = next(
        r
        for r in script.compare(reference, standalone, script.default_cases())
        if r.name == "padded_batch"
    )

    assert padded.per_utterance_max_abs_delta == pytest.approx([0.0, 0.5])
    assert padded.max_abs_delta == pytest.approx(0.5)


def test_only_valid_frames_are_compared(script: ModuleType) -> None:
    # Frames past an utterance's own length are padding in both encoders and are
    # never consumed downstream, so including them would report a disagreement
    # that no downstream number can see.
    class TailNoise(FakeEncoder):
        def encode(self, waveforms, wav_lengths, use_mask=False, use_final_output=True):
            feats, lengths = super().encode(waveforms, wav_lengths, use_mask, use_final_output)
            for i in range(feats.shape[0]):
                feats[i, int(lengths[i]) :] = 99.0  # pure padding region
            return feats, lengths

    results = script.compare(FakeEncoder([0.0, 0.0]), TailNoise([0.0, 0.0]), script.default_cases())
    padded = next(r for r in results if r.name == "padded_batch")

    assert padded.max_abs_delta == pytest.approx(0.0)


def test_a_shape_mismatch_is_reported_and_has_no_delta(script: ModuleType) -> None:
    results = script.compare(FakeEncoder(hidden=4), FakeEncoder(hidden=8), script.default_cases())

    assert all(not r.shapes_agree for r in results)
    assert all(r.max_abs_delta is None for r in results)
    # A missing number must not read as agreement.
    assert script.worst_delta(results) is None


# --------------------------------------------------------------------------- #
# Verdict and exit code
# --------------------------------------------------------------------------- #


def test_verdict_passes_below_the_tolerance_and_fails_above(script: ModuleType) -> None:
    close = script.compare(FakeEncoder(0.0), FakeEncoder(1e-6), script.default_cases())
    far = script.compare(FakeEncoder(0.0), FakeEncoder(1e-2), script.default_cases())

    assert script.passed(close, tol=1e-3) is True
    assert script.passed(far, tol=1e-3) is False


def test_a_shape_mismatch_never_passes(script: ModuleType) -> None:
    results = script.compare(FakeEncoder(hidden=4), FakeEncoder(hidden=8), script.default_cases())

    assert script.passed(results, tol=1e9) is False


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_report_records_the_versions_and_the_checkpoint_digest(script: ModuleType, tmp_path):
    ckpt = tmp_path / "xeus.pth"
    ckpt.write_bytes(b"not a real checkpoint")
    results = script.compare(FakeEncoder(0.0), FakeEncoder(0.5), script.default_cases())

    report = script.build_report(results, checkpoint=ckpt, tol=1e-3, device="cpu")

    assert report["passed"] is False
    assert report["tol"] == 1e-3
    assert report["checkpoint"]["sha256"] == script.sha256_file(ckpt)
    assert report["checkpoint"]["name"] == "xeus.pth"
    assert report["versions"]["torch"] == torch.__version__
    assert "numpy" in report["versions"]
    # ESPnet is not installed here, and the report says so rather than omitting it.
    assert report["versions"]["espnet"] == "absent"
    assert [c["name"] for c in report["cases"]] == ["single", "padded_batch"]
    assert report["cases"][1]["per_utterance_max_abs_delta"] == pytest.approx([0.5, 0.5])


def test_write_report_round_trips(script: ModuleType, tmp_path):
    results = script.compare(FakeEncoder(), FakeEncoder(), script.default_cases())
    report = script.build_report(results, checkpoint=None, tol=1e-3, device="cpu")
    out = tmp_path / "nested" / "report.json"

    script.write_report(report, out)

    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True


def test_report_handles_an_absent_checkpoint_path(script: ModuleType) -> None:
    report = script.build_report([], checkpoint=None, tol=1e-3, device="cpu")

    assert report["checkpoint"] is None


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_checkpoint_is_required(script: ModuleType) -> None:
    with pytest.raises(SystemExit):
        script.build_parser().parse_args([])


def test_parser_defaults(script: ModuleType) -> None:
    args = script.build_parser().parse_args(["--checkpoint", "/tmp/x.pth"])

    assert args.tol == 1e-3
    assert args.device == "cpu"
    assert args.out is None


def test_parser_accepts_the_documented_flags(script: ModuleType, tmp_path) -> None:
    args = script.build_parser().parse_args(
        [
            "--checkpoint",
            "/tmp/x.pth",
            "--device",
            "cuda",
            "--tol",
            "1e-4",
            "--out",
            str(tmp_path / "r.json"),
        ]
    )

    assert (args.device, args.tol) == ("cuda", 1e-4)
    assert args.out == str(tmp_path / "r.json")


# --------------------------------------------------------------------------- #
# Import hygiene
# --------------------------------------------------------------------------- #


def test_the_script_imports_without_espnet(script: ModuleType) -> None:
    # It has already been imported by the fixture, in an environment with no
    # espnet. That is the assertion: importing must not require the fork.
    import importlib.util

    assert importlib.util.find_spec("espnet2") is None
    assert hasattr(script, "load_reference")


def test_loading_the_reference_without_espnet_fails_with_one_line(script: ModuleType) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        script.load_reference("/tmp/nonexistent.pth", "cpu")

    message = str(excinfo.value)
    assert "\n" not in message
    # Points at the recipe rather than just naming the missing module.
    assert "scripts/espnet-crosscheck" in message


def test_main_reports_the_missing_fork_rather_than_traceback(script: ModuleType, capsys) -> None:
    code = script.main(["--checkpoint", "/tmp/nonexistent.pth"])

    assert code == 2
    assert "espnet" in capsys.readouterr().err.lower()
