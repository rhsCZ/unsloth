# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The base-vs-head installer comparer has to be able to fail.

This is the lane that answers "did hardening break anything", and its whole value rests on one
property: that a real change produces a non-zero exit. A comparer whose normalisation has drifted
wide reports "no differences" forever and every hardening PR after it ships unverified, with a green
check saying the opposite.

So the failure direction is tested first here, and harder than the pass direction. The pass
direction matters too, but only because a lane that fails on every run gets disabled, which costs
the same coverage by a slower route.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "compare_installer_evidence.py"

sys.path.insert(0, str(SCRIPT.parent))
import compare_installer_evidence as cmp  # noqa: E402


BASELINE = "\n".join(
    [
        "  python         3.13.14 ready",
        "  uv             0.12.1 installed",
        "  studio         installed in 12.4s",
        "  shortcut       desktop and Start Menu",
        "  next           run: unsloth studio",
    ]
)

SHORTCUTS = [
    {
        "name": "Unsloth.lnk",
        "targetPath": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        "arguments": "-NoProfile -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File launch-studio.ps1",
        "workingDirectory": r"C:\Users\runneradmin\.unsloth\studio",
        "windowStyle": "1",
        "iconLocation": r"C:\Users\runneradmin\.unsloth\studio\unsloth.ico,0",
    }
]

ARTIFACTS = {
    "files": {
        "launch-studio.ps1": {"content": "Start-Process -WindowStyle Hidden pwsh\n"},
        "unsloth.cmd": {"content": "@echo off\n"},
    },
    "rewrittenOnSecondRun": [],
}


def _write(
    directory: Path,
    transcript: str = BASELINE,
    shortcuts = None,
    artifacts = None,
) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "transcript.txt").write_text(transcript, encoding = "utf-8")
    (directory / "shortcuts.json").write_text(
        json.dumps(SHORTCUTS if shortcuts is None else shortcuts), encoding = "utf-8"
    )
    (directory / "artifacts.json").write_text(
        json.dumps(ARTIFACTS if artifacts is None else artifacts), encoding = "utf-8"
    )
    return directory


def _run(base: Path, head: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            str(base),
            "--head",
            str(head),
            "--base-sha",
            "a" * 40,
            "--head-sha",
            "b" * 40,
        ],
        capture_output = True,
        text = True,
        timeout = 120,
    )


# ---------------------------------------------------------------------------
# It must be able to fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation, why",
    [
        ("shortcut       desktop only", "a line of user-visible output changed"),
        ("  shortcut       desktop and Start Menu ", "trailing space only, which must NOT fail"),
    ],
)
def test_a_changed_output_line_is_reported(tmp_path: Path, mutation: str, why: str) -> None:
    base = _write(tmp_path / "base")
    mutated = BASELINE.replace("  shortcut       desktop and Start Menu", mutation)
    head = _write(tmp_path / "head", transcript = mutated)
    result = _run(base, head)
    if "must NOT fail" in why:
        assert (
            result.returncode == 0
        ), f"trailing whitespace was treated as a change: {result.stdout}"
    else:
        assert result.returncode == 2, f"{why} was not reported: {result.stdout}\n{result.stderr}"


def test_a_lost_indent_is_reported(tmp_path: Path) -> None:
    """`step` pads its label to exactly 15 columns and the output lock pins the indent."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = BASELINE.replace("  python", "   python"))
    assert _run(base, head).returncode == 2


def test_a_relaxed_execution_policy_in_a_shortcut_is_reported(tmp_path: Path) -> None:
    """The single substitution this whole effort is about, and invisible in the transcript."""
    base = _write(tmp_path / "base")
    relaxed = [
        dict(SHORTCUTS[0], arguments = SHORTCUTS[0]["arguments"].replace("RemoteSigned", "Bypass"))
    ]
    head = _write(tmp_path / "head", shortcuts = relaxed)
    result = _run(base, head)
    assert result.returncode == 2
    assert "arguments" in result.stdout and "Bypass" in result.stdout


def test_a_shortcut_that_stopped_being_created_is_reported(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", shortcuts = [])
    assert _run(base, head).returncode in (2, 3)


def test_a_changed_generated_launcher_is_reported(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    changed = json.loads(json.dumps(ARTIFACTS))
    changed["files"]["launch-studio.ps1"]["content"] = "Start-Process pwsh\n"
    head = _write(tmp_path / "head", artifacts = changed)
    assert _run(base, head).returncode == 2


def test_a_second_run_that_rewrites_files_is_reported(tmp_path: Path) -> None:
    """Several changes here touch content-comparison paths, so idempotency is a real risk."""
    base = _write(tmp_path / "base")
    noisy = json.loads(json.dumps(ARTIFACTS))
    noisy["rewrittenOnSecondRun"] = ["launch-studio.ps1"]
    head = _write(tmp_path / "head", artifacts = noisy)
    result = _run(base, head)
    assert result.returncode == 2
    assert "second time" in result.stdout


# ---------------------------------------------------------------------------
# It must not fail on noise, or it gets disabled
# ---------------------------------------------------------------------------


def test_known_noise_does_not_fail(tmp_path: Path) -> None:
    noisy = "\n".join(
        [
            "  python         3.13.9 ready",
            "  uv             0.12.4 installed",
            "  studio         installed in 41.9s",
            "  shortcut       desktop and Start Menu",
            "  next           run: unsloth studio",
        ]
    )
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = noisy)
    result = _run(base, head)
    assert result.returncode == 0, f"noise failed the lane: {result.stdout}"


def test_version_drift_is_normalised_but_still_printed(tmp_path: Path) -> None:
    """Normalising something away without saying so is how a lane stops telling you anything."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = BASELINE.replace("0.12.1", "0.12.4"))
    result = _run(base, head)
    assert result.returncode == 0
    assert "version drift" in result.stdout and "0.12.4" in result.stdout


# ---------------------------------------------------------------------------
# VOID is not a pass
# ---------------------------------------------------------------------------


def test_missing_evidence_is_void_and_not_a_pass(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    result = _run(base, tmp_path / "absent")
    assert result.returncode == 3, "a side with no evidence must not read as agreement"
    assert "VOID" in result.stdout


def test_an_empty_transcript_is_void(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = "\n\n  \n")
    result = _run(base, head)
    assert result.returncode == 3
    assert "did not run" in result.stdout


def test_comparing_a_commit_with_itself_is_void() -> None:
    verdict = cmp.compare_directories(Path("/x"), Path("/y"), "c" * 40, "c" * 40)
    assert verdict.is_void
    assert verdict.exit_code() == 3
    assert any("itself" in reason for reason in verdict.void)


def test_two_empty_shortcut_manifests_are_void_not_agreement() -> None:
    verdict = cmp.Verdict()
    cmp.compare_shortcuts([], [], verdict)
    assert verdict.is_void


def test_void_and_different_have_distinct_exit_codes() -> None:
    """A workflow that treats every non-zero the same cannot tell "it changed" from "we did not
    look", and those need different reactions from whoever reads the run."""
    void = cmp.Verdict()
    void.void.append("x")
    different = cmp.Verdict()
    different.differences.append("y")
    assert void.exit_code() == 3
    assert different.exit_code() == 2
    assert cmp.Verdict().exit_code() == 0


# ---------------------------------------------------------------------------
# The comparer's own controls
# ---------------------------------------------------------------------------


def test_the_self_test_passes() -> None:
    """CI runs this before any real comparison, so a broken normaliser refuses to produce a
    verdict instead of producing a reassuring one."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"], capture_output = True, text = True, timeout = 120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_self_test_can_fail() -> None:
    """A control that cannot fail is decoration. Widen the normaliser until it swallows a real
    change and the self-test must notice."""
    original = cmp._NORMALISERS
    try:
        import re

        cmp._NORMALISERS = original + ((re.compile(r".*"), "", "everything"),)
        failures = cmp.self_test()
        assert failures, "a normaliser that erases every line still passed the controls"
        assert any("too loose" in f for f in failures)
    finally:
        cmp._NORMALISERS = original


def test_every_normaliser_records_why_it_exists() -> None:
    """A rule with no recorded cause is a rule nobody can argue with, and this list is exactly
    where a future 'just make the lane green' change would land."""
    for pattern, _replacement, why in cmp._NORMALISERS:
        assert why and len(why) > 8, f"the normaliser {pattern.pattern!r} has no stated reason"


def test_the_shortcut_fields_compared_include_the_launch_contract() -> None:
    for field in ("arguments", "targetPath", "windowStyle"):
        assert field in cmp._SHORTCUT_FIELDS


def test_a_single_shortcut_serialised_as_an_object_still_compares(tmp_path: Path) -> None:
    """ConvertTo-Json unwraps a one-element collection into a bare object.

    Observed while smoke-running the collector: with one shortcut found, shortcuts.json was an
    object rather than an array. Iterating that yields dictionary *keys*, so two different launch
    contracts would have compared as agreement for entirely the wrong reason.
    """
    base = _write(tmp_path / "base")
    (base / "shortcuts.json").write_text(json.dumps(SHORTCUTS[0]), encoding = "utf-8")
    head = _write(tmp_path / "head")
    relaxed = dict(
        SHORTCUTS[0], arguments = SHORTCUTS[0]["arguments"].replace("RemoteSigned", "Bypass")
    )
    (head / "shortcuts.json").write_text(json.dumps(relaxed), encoding = "utf-8")

    result = _run(base, head)
    assert (
        result.returncode == 2
    ), f"an unwrapped single shortcut was not compared as a shortcut: {result.stdout}"
    assert "Bypass" in result.stdout


def test_a_symmetric_collection_failure_is_void_not_a_pass(tmp_path: Path) -> None:
    """The hole this closes was live: two sides that both failed to read any shortcut reported
    "1 compared, every field equal" and exited zero.

    Collection failures are symmetric far more often than behaviour changes are, because they come
    from the host and both sides share it.
    """
    failed = [{"name": "<collection failed>", "error": "could not create WScript.Shell"}]
    base = _write(tmp_path / "base", shortcuts = failed)
    head = _write(tmp_path / "head", shortcuts = failed)
    result = _run(base, head)
    assert result.returncode == 3, f"a shared collection failure read as agreement: {result.stdout}"
    assert "prove nothing" in result.stdout
