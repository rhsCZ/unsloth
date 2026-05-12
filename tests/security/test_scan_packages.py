"""Regression tests for `scripts/scan_packages.py`.

The scanner's primary entry point (`download_packages`) reaches PyPI;
to keep the suite offline we exercise it via the module's public
in-process helpers (`scan_archive`) and assert against the binary
wheel / sdist fixtures committed under `tests/security/fixtures/`.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(REPO_ROOT))
from scripts import scan_packages as sp  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture sanity.
# ---------------------------------------------------------------------------


def test_fixture_files_exist():
    for name in ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz"):
        assert (FIXTURES / name).is_file(), name


def test_fixture_bytes_are_deterministic(tmp_path):
    """Re-running `_build.py` must produce byte-identical archives.

    The build helper sets every member's mtime/uid/gid/mode and emits
    members in sorted order. We rebuild into a temp dir and compare
    SHA-256 against the committed bytes.
    """
    # Snapshot committed hashes.
    expected: dict[str, str] = {}
    for name in ("malicious_wheel.whl", "clean_wheel.whl", "malicious_sdist.tar.gz"):
        expected[name] = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()

    # Rebuild into a sibling dir to avoid clobbering the committed files.
    rebuild_dir = tmp_path / "rebuild"
    rebuild_dir.mkdir()
    # The build helper writes to its own directory; copy + patch HERE.
    builder_src = (FIXTURES / "_build.py").read_text()
    rebuilt_helper = rebuild_dir / "_build.py"
    rebuilt_helper.write_text(builder_src)
    # Run with SOURCE_DATE_EPOCH=0 and HERE-override via a tiny shim.
    shim = rebuild_dir / "run.py"
    shim.write_text(
        "import sys, pathlib\n"
        f"sys.path.insert(0, {str(rebuild_dir)!r})\n"
        "import _build\n"
        f"_build.HERE = pathlib.Path({str(rebuild_dir)!r})\n"
        "_build.build_all()\n"
    )
    env = dict(os.environ, SOURCE_DATE_EPOCH = "0")
    proc = subprocess.run(
        [sys.executable, str(shim)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 30,
    )
    assert proc.returncode == 0, proc.stderr

    for name, want_sha in expected.items():
        got = hashlib.sha256((rebuild_dir / name).read_bytes()).hexdigest()
        assert got == want_sha, (
            f"rebuild of {name} produced different bytes:\n"
            f"  expected: {want_sha}\n"
            f"  actual:   {got}\n"
            "_build.py is non-deterministic; pin members tighter."
        )


# ---------------------------------------------------------------------------
# scan_archive() against the fixture wheel + sdist.
# ---------------------------------------------------------------------------


def _critical_or_high(findings) -> list:
    return [f for f in findings if f.severity in (sp.CRITICAL, sp.HIGH)]


def test_malicious_wheel_triggers_critical():
    findings = sp.scan_archive(
        str(FIXTURES / "malicious_wheel.whl"),
        "malicious_fixture",
    )
    assert findings, "no findings on malicious wheel; scanner regression"
    blockers = _critical_or_high(findings)
    assert blockers, f"no CRITICAL/HIGH findings: {[str(f) for f in findings]}"
    # At least one finding must reference setup.py.
    assert any("setup.py" in f.filename for f in blockers)


def test_malicious_sdist_triggers_critical():
    findings = sp.scan_archive(
        str(FIXTURES / "malicious_sdist.tar.gz"),
        "malicious_fixture",
    )
    blockers = _critical_or_high(findings)
    assert blockers, f"no CRITICAL/HIGH findings: {[str(f) for f in findings]}"
    assert any("setup.py" in f.filename for f in blockers)


def test_clean_wheel_no_findings():
    findings = sp.scan_archive(
        str(FIXTURES / "clean_wheel.whl"),
        "clean_fixture",
    )
    assert (
        findings == []
    ), f"unexpected findings on clean wheel: {[str(f) for f in findings]}"


# ---------------------------------------------------------------------------
# Fork 1 constants -- gated on availability.
# ---------------------------------------------------------------------------


_BLOCKED_AVAILABLE = hasattr(sp, "BLOCKED_PYPI_VERSIONS")
_MAY12_AVAILABLE = hasattr(sp, "RE_MAY12_IOC")


@pytest.mark.skipif(
    not _BLOCKED_AVAILABLE,
    reason = "Fork 1 (BLOCKED_PYPI_VERSIONS) not merged yet",
)
def test_blocked_pypi_versions_complete():
    table = sp.BLOCKED_PYPI_VERSIONS
    assert "guardrails-ai" in table
    assert "0.10.1" in table["guardrails-ai"]
    assert "mistralai" in table
    assert "2.4.6" in table["mistralai"]
    assert "lightning" in table
    assert {"2.6.2", "2.6.3"}.issubset(table["lightning"])


@pytest.mark.skipif(
    not _MAY12_AVAILABLE,
    reason = "Fork 1 (RE_MAY12_IOC) not merged yet",
)
def test_re_may12_ioc_catches_each_literal():
    expected_literals = [
        "git-tanstack.com",
        "/tmp/transformers.pyz",
        "transformers.pyz",
        "With Love TeamPCP",
        "We've been online over 2 hours",
    ]
    pattern: re.Pattern = sp.RE_MAY12_IOC
    for lit in expected_literals:
        assert pattern.search(lit), f"RE_MAY12_IOC missed literal {lit!r}"
    # Clean control: a plain string with none of the literals must not match.
    assert not pattern.search("import numpy as np")


@pytest.mark.skipif(
    not _MAY12_AVAILABLE,
    reason = "Fork 1 (RE_MAY12_IOC integration) not merged yet",
)
def test_may12_ioc_caught_by_scan_archive():
    """Once RE_MAY12_IOC is wired into check_py_file (per Fork 1's
    plan), the malicious wheel's setup.py must produce a finding
    that explicitly references the May-12 IOC string.
    """
    findings = sp.scan_archive(
        str(FIXTURES / "malicious_wheel.whl"),
        "malicious_fixture",
    )
    hit = any(
        "git-tanstack.com" in (f.evidence or "")
        or "transformers.pyz" in (f.evidence or "")
        or "may12" in (f.check or "").lower()
        for f in findings
    )
    assert hit, (
        "RE_MAY12_IOC integration missing; findings = "
        f"{[(f.severity, f.check, f.evidence[:80]) for f in findings]}"
    )
