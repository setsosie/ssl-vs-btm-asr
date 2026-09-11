"""Reports name the policy each language ran under, and refuse to pool across.

Two runs whose languages were normalized differently produce different reference
strings from the same corpus, so a difference between their error rates is a
difference in scoring convention as much as in the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svb.report.analyze import language_policies, require_same_policies


def _write(run_dir: Path, policies: dict[str, str]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "text_stats.json").write_text(
        json.dumps(
            {
                "languages": {
                    code: {"policy": name, "policy_hash": f"h-{name}", "word_boundary": True}
                    for code, name in policies.items()
                }
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_the_policy_each_language_ran_under_is_readable_from_the_run(tmp_path) -> None:
    run = _write(tmp_path / "a", {"en": "whisper-basic", "hi": "indic-vistaar"})

    assert language_policies(run) == {"en": "whisper-basic", "hi": "indic-vistaar"}


def test_two_runs_with_the_same_policies_may_be_compared(tmp_path) -> None:
    a = _write(tmp_path / "a", {"en": "whisper-basic", "hi": "indic-vistaar"})
    b = _write(tmp_path / "b", {"en": "whisper-basic", "hi": "indic-vistaar"})

    require_same_policies(a, b)  # does not raise


def test_a_language_scored_under_two_policies_refuses_the_comparison(tmp_path) -> None:
    a = _write(tmp_path / "a", {"en": "whisper-basic", "hi": "indic-vistaar"})
    b = _write(tmp_path / "b", {"en": "whisper-basic", "hi": "whisper-basic"})

    with pytest.raises(ValueError, match=r"hi: .*indic-vistaar.*whisper-basic"):
        require_same_policies(a, b)


def test_a_run_that_records_no_policies_is_not_treated_as_matching(tmp_path) -> None:
    """Silence is not agreement: a run predating the per-language record cannot
    be shown to have used the same rules."""
    a = _write(tmp_path / "a", {"en": "whisper-basic"})
    b = tmp_path / "b"
    b.mkdir()

    with pytest.raises(ValueError, match="records no per-language policies"):
        require_same_policies(a, b)


def test_a_macro_average_over_mixed_policies_says_so() -> None:
    """Averaging across different text rules is legitimate; presenting the
    result as one quantity is not."""
    from svb.report.aggregate import macro_policy_caption

    mixed = {"en": "whisper-basic", "hi": "indic-vistaar"}
    same = {"en": "whisper-basic", "hi": "whisper-basic"}

    assert "policies differ by language" in macro_policy_caption(mixed, ["en", "hi"])
    assert "indic-vistaar" in macro_policy_caption(mixed, ["en", "hi"])
    assert macro_policy_caption(same, ["en", "hi"]) == ""
    assert macro_policy_caption(mixed, ["en"]) == ""
