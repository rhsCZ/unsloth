#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decide whether two installer runs behaved the same, given evidence collected on Windows.

This is the half of the base-vs-head functional lane that does not need Windows, so it lives in a
file with unit tests rather than inline in YAML. The Windows jobs run an installer and write
evidence; this compares two sets of it and produces a verdict.

The comparison is deliberately narrow: it answers "did the user-visible behaviour change", not "are
these runs identical". Two installs of the same commit are never byte-identical -- they differ in
timings, in temp directory names, in which mirror answered, in a `uv` patch release that shipped
between the two jobs. Every one of those is normalised away, and the normalisation rules are the
interesting part of this file: each one is a specific observed source of noise, and widening one to
silence a failure is how a lane like this stops being able to fail.

Three things are compared, because they fail independently:

- **transcript**: every line the installer printed, normalised. A hardening change must not alter
  what a user reads.
- **shortcuts**: the `.lnk` properties read back through the shell. This is where a change to the
  launch transport shows up, and it is invisible in the transcript.
- **artifacts**: a manifest of the files the installer wrote, with hashes for the ones whose content
  is a contract (`launch-studio.ps1`, `unsloth.cmd`).

`VOID` is a first-class outcome and not a pass. If the two sides are the same commit, or the
evidence is missing, or an installer did not finish, there is nothing to compare and saying "no
differences" would be a lie of exactly the kind this lane exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Each entry is (pattern, replacement, why). The `why` is not decoration: the next person to widen
# one of these needs to know what it was for, and a rule with no recorded cause is a rule nobody can
# argue with.
_NORMALISERS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b\d+\.\d+s\b"), "<duration>", "elapsed times, printed by every step"),
    (re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%"), "<percent>", "download progress"),
    (
        re.compile(r"\b\d+(?:\.\d+)?\s?(?:[KMGT]i?B|bytes)\b", re.I),
        "<size>",
        "download sizes, which differ with a CDN or a patch release",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"), "<timestamp>", "timestamps"),
    (
        re.compile(r"\b[0-9a-f]{40}\b|\b[0-9A-F]{64}\b|\b[0-9a-f]{64}\b"),
        "<hash>",
        "commit SHAs and file digests: the two sides are different commits by construction",
    ),
    (re.compile(r"unsloth-[A-Za-z0-9_]{6,}"), "unsloth-<temp>", "mkstemp-style temp names"),
    (re.compile(r"\\Temp\\[A-Za-z0-9._-]{6,}"), r"\\Temp\\<temp>", "Windows temp directory names"),
    (re.compile(r"\b(pid|PID)[= ]\d+"), r"\1=<pid>", "process ids"),
    (
        re.compile(r"127\.0\.0\.1:\d+|localhost:\d+"),
        "127.0.0.1:<port>",
        "the port Studio bound, which is chosen from what is free",
    ),
    (
        re.compile(r"\x1b\[[0-9;?]*[A-Za-z]"),
        "",
        "ANSI sequences, in case a run was not redirected after all",
    ),
    (re.compile(r"[\r\x08]"), "", "carriage returns and backspaces from progress redraws"),
)

# Volatile only because the two jobs ran minutes apart. A version drift is not a behaviour change,
# but it IS worth printing, so these are normalised and separately reported.
_VERSION_PATTERN = re.compile(
    r"\b(uv|python|Python|CPython|git|cmake|torch|node|npm)[\s/-]+v?(\d+\.\d+(?:\.\d+)?)",
)


def collect_versions(text: str) -> dict[str, set[str]]:
    """What each side reported installing, reported rather than compared.

    A `uv` patch release that shipped between the two jobs is not a behaviour change and must not
    fail the lane. But a *deliberate* pin bump looks identical after normalisation, so the versions
    are printed side by side: silently normalising something away and never mentioning it is how a
    lane loses the ability to tell you anything.
    """
    found: dict[str, set[str]] = {}
    for match in _VERSION_PATTERN.finditer(text):
        found.setdefault(match.group(1).lower(), set()).add(match.group(2))
    return found


def normalise_line(line: str) -> str:
    out = line
    for pattern, replacement, _why in _NORMALISERS:
        out = pattern.sub(replacement, out)
    out = _VERSION_PATTERN.sub(lambda m: f"{m.group(1)}/<version>", out)
    # Trailing whitespace only. Leading whitespace is load-bearing: `step` pads its label to exactly
    # 15 columns and REQUIRED_OUTPUT pins the indent, so stripping the left side would hide the one
    # regression class most likely to slip through a prose review.
    return out.rstrip()


def normalise_transcript(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = normalise_line(raw)
        if not line.strip():
            continue
        # Runner-injected noise. These carry the workflow's own group names and the side's SHA, so
        # they differ between sides for reasons that have nothing to do with the installer.
        if line.lstrip().startswith(
            (
                "##[group]",
                "##[endgroup]",
                "::group::",
                "::endgroup::",
                "##[debug]",
                "Run ",
                "shell: ",
                "env:",
            )
        ):
            continue
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class Verdict:
    """The outcome, with every difference kept rather than only the first."""

    def __init__(self) -> None:
        self.void: list[str] = []
        self.differences: list[str] = []
        self.notes: list[str] = []

    @property
    def is_void(self) -> bool:
        return bool(self.void)

    @property
    def passed(self) -> bool:
        return not self.void and not self.differences

    def exit_code(self) -> int:
        # VOID and DIFFERENT are both non-zero, and deliberately distinct: 2 means "measured, and it
        # changed"; 3 means "could not measure", which must never read as a pass.
        if self.void:
            return 3
        if self.differences:
            return 2
        return 0


def _unified(
    base: list[str],
    head: list[str],
    label: str,
    limit: int = 60,
) -> list[str]:
    import difflib

    diff = list(
        difflib.unified_diff(
            base, head, fromfile = f"base/{label}", tofile = f"head/{label}", lineterm = "", n = 2
        )
    )
    if len(diff) > limit:
        omitted = len(diff) - limit
        diff = diff[:limit] + [f"... {omitted} more diff lines omitted; the artifact has all of it"]
    return diff


def compare_transcripts(base: str, head: str, verdict: Verdict) -> None:
    base_lines = normalise_transcript(base)
    head_lines = normalise_transcript(head)
    if not base_lines or not head_lines:
        verdict.void.append(
            "one side's transcript is empty after normalisation, so there is nothing to compare. "
            "An installer that printed nothing did not run."
        )
        return
    base_versions = collect_versions(base)
    head_versions = collect_versions(head)
    for tool in sorted(set(base_versions) | set(head_versions)):
        before = ",".join(sorted(base_versions.get(tool, {"-"})))
        after = ",".join(sorted(head_versions.get(tool, {"-"})))
        if before != after:
            verdict.notes.append(
                f"version drift (normalised away, not a failure): {tool} base={before} head={after}"
            )

    if base_lines == head_lines:
        verdict.notes.append(f"transcript: identical over {len(base_lines)} normalised lines")
        return
    verdict.differences.append(
        "the installer's user-visible output changed:\n"
        + "\n".join(_unified(base_lines, head_lines, "transcript"))
    )


def _shortcut_key(entry: dict) -> str:
    return str(entry.get("name") or entry.get("path") or "<unnamed>")


# Read back from the shell, and every one of them is a contract. Arguments especially: it carries
# -WindowStyle and -ExecutionPolicy, which is the pair this whole effort is about, and a change
# there is completely invisible in the transcript.
_SHORTCUT_FIELDS = ("targetPath", "arguments", "workingDirectory", "windowStyle", "iconLocation")


def _as_list(value) -> list[dict]:
    """ConvertTo-Json unwraps a one-element collection into a bare object.

    The collector forces an array, but this side must not depend on that: a schema that changes with
    the number of shortcuts found would make the single-shortcut case iterate dictionary *keys* and
    compare strings, which reports agreement for entirely the wrong reason.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return [item for item in value if isinstance(item, dict)]


def compare_shortcuts(base, head, verdict: Verdict) -> None:
    base, head = _as_list(base), _as_list(head)
    if not base and not head:
        verdict.void.append(
            "neither side reported any shortcut. The installer creates a desktop and a Start Menu "
            "entry, so zero on both sides means the evidence was not collected, not that they agree."
        )
        return
    # A collection error on both sides compares equal to itself. Observed while wiring this up: two
    # runs that both failed to read any shortcut reported "1 compared, every field equal" and exited
    # zero. Failures are symmetric far more often than behaviour changes are -- they usually come
    # from the host, which both sides share -- so the symmetry is no comfort at all.
    for side, entries in (("base", base), ("head", head)):
        for entry in entries:
            if entry.get("error"):
                verdict.void.append(
                    f"{side} could not read shortcut {_shortcut_key(entry)!r}: {entry['error']}. "
                    f"Two sides that both failed to collect evidence agree with each other and "
                    f"prove nothing."
                )
    if verdict.is_void:
        return

    base_map = {_shortcut_key(e): e for e in base}
    head_map = {_shortcut_key(e): e for e in head}

    for missing in sorted(set(base_map) - set(head_map)):
        verdict.differences.append(f"shortcut {missing!r} exists on base and not on head")
    for added in sorted(set(head_map) - set(base_map)):
        verdict.differences.append(f"shortcut {added!r} exists on head and not on base")

    for name in sorted(set(base_map) & set(head_map)):
        for field in _SHORTCUT_FIELDS:
            before = normalise_line(str(base_map[name].get(field, "")))
            after = normalise_line(str(head_map[name].get(field, "")))
            if before != after:
                verdict.differences.append(
                    f"shortcut {name!r} field {field!r} changed:\n  base: {before}\n  head: {after}"
                )
    if not verdict.differences:
        verdict.notes.append(f"shortcuts: {len(base_map)} compared, every field equal")


def compare_artifacts(base: dict, head: dict, verdict: Verdict) -> None:
    """The installed files. Paths on both sides, content only where content is a contract."""
    base_files = base.get("files") or {}
    head_files = head.get("files") or {}
    if not base_files and not head_files:
        verdict.void.append("neither side listed any installed file, so nothing was measured")
        return

    for missing in sorted(set(base_files) - set(head_files)):
        verdict.differences.append(f"base installed {missing!r} and head did not")
    for added in sorted(set(head_files) - set(base_files)):
        verdict.differences.append(f"head installed {added!r} and base did not")

    for name in sorted(set(base_files) & set(head_files)):
        before, after = base_files[name], head_files[name]
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        if "content" in before and "content" in after:
            b = normalise_transcript(before["content"])
            a = normalise_transcript(after["content"])
            if b != a:
                verdict.differences.append(
                    f"the generated {name} changed:\n" + "\n".join(_unified(b, a, name))
                )

    # Idempotency is reported by the Windows side, which is the only place it can be observed: it
    # runs the installer twice and records whether the second run rewrote anything.
    for side, data in (("base", base), ("head", head)):
        rewritten = data.get("rewrittenOnSecondRun")
        if rewritten is None:
            verdict.notes.append(f"{side}: no idempotency evidence was collected")
        elif rewritten:
            verdict.differences.append(
                f"{side}: running the installer a second time rewrote {sorted(rewritten)}. "
                f"A reinstall that changed nothing must write nothing."
            )
        else:
            verdict.notes.append(f"{side}: the second run rewrote nothing")


def _load(path: Path, verdict: Verdict, what: str):
    if not path.is_file():
        verdict.void.append(f"{what} is missing at {path}, so this side produced no evidence")
        return None
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding = "utf-8", errors = "replace"))
        return path.read_text(encoding = "utf-8", errors = "replace")
    except (OSError, ValueError) as exc:
        verdict.void.append(f"{what} at {path} could not be read: {exc}")
        return None


def compare_directories(
    base_dir: Path,
    head_dir: Path,
    base_sha: str = "",
    head_sha: str = "",
) -> Verdict:
    verdict = Verdict()

    if base_sha and head_sha and base_sha == head_sha:
        verdict.void.append(
            f"both sides are {base_sha[:12]}. Comparing a commit with itself cannot show that a "
            f"change preserved behaviour; it shows only that the lane is deterministic."
        )
        return verdict

    base_transcript = _load(base_dir / "transcript.txt", verdict, "the base transcript")
    head_transcript = _load(head_dir / "transcript.txt", verdict, "the head transcript")
    base_shortcuts = _load(base_dir / "shortcuts.json", verdict, "the base shortcut manifest")
    head_shortcuts = _load(head_dir / "shortcuts.json", verdict, "the head shortcut manifest")
    base_artifacts = _load(base_dir / "artifacts.json", verdict, "the base artifact manifest")
    head_artifacts = _load(head_dir / "artifacts.json", verdict, "the head artifact manifest")

    if verdict.is_void:
        return verdict

    compare_transcripts(base_transcript, head_transcript, verdict)
    compare_shortcuts(base_shortcuts or [], head_shortcuts or [], verdict)
    compare_artifacts(base_artifacts or {}, head_artifacts or {}, verdict)
    return verdict


# ---------------------------------------------------------------------------
# The positive control
# ---------------------------------------------------------------------------

# A differ that reports no differences looks exactly the same whether it is working or broken. So
# before the real comparison is trusted, it is handed a pair it MUST call different, and a pair it
# MUST call equal. Both directions matter: a differ that flags everything is as useless as one that
# flags nothing, it just fails more loudly.
_CONTROL_TRANSCRIPT = "\n".join(
    [
        "  python         3.13.14 ready",
        "  studio         installed in 12.4s",
        "  shortcut       desktop and Start Menu",
    ]
)


def self_test() -> list[str]:
    failures: list[str] = []

    noisy = "\n".join(
        [
            "  python         3.13.9 ready",
            "  studio         installed in 41.9s",
            "  shortcut       desktop and Start Menu",
        ]
    )
    v = Verdict()
    compare_transcripts(_CONTROL_TRANSCRIPT, noisy, v)
    if v.differences:
        failures.append(
            "the normaliser is too strict: a version drift and a timing difference were reported "
            "as a behaviour change, which would make this lane fail on every run and get disabled. "
            f"Reported: {v.differences}"
        )

    changed = _CONTROL_TRANSCRIPT.replace("desktop and Start Menu", "desktop only")
    v = Verdict()
    compare_transcripts(_CONTROL_TRANSCRIPT, changed, v)
    if not v.differences:
        failures.append(
            "the normaliser is too loose: a changed output line was NOT reported. Every 'no "
            "differences' verdict this lane has ever produced would be worthless."
        )

    indented = _CONTROL_TRANSCRIPT.replace("  python", "   python")
    v = Verdict()
    compare_transcripts(_CONTROL_TRANSCRIPT, indented, v)
    if not v.differences:
        failures.append(
            "an indentation change was not reported. `step` pads its label to exactly 15 columns "
            "and the output lock pins the indent, so a lost space is a real regression."
        )

    v = Verdict()
    compare_shortcuts(
        [
            {
                "name": "Unsloth.lnk",
                "arguments": "-NoProfile -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File x",
            }
        ],
        [
            {
                "name": "Unsloth.lnk",
                "arguments": "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File x",
            }
        ],
        v,
    )
    if not v.differences:
        failures.append(
            "a shortcut whose execution policy changed from RemoteSigned to Bypass was NOT "
            "reported. That is the single substitution this entire effort is about."
        )

    v = Verdict()
    compare_shortcuts([], [], v)
    if not v.is_void:
        failures.append(
            "two empty shortcut manifests were treated as agreement rather than as VOID"
        )

    v = compare_directories(Path("/nonexistent/base"), Path("/nonexistent/head"), "aaa", "bbb")
    if not v.is_void or v.exit_code() != 3:
        failures.append("missing evidence did not produce VOID with exit code 3")

    return failures


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--base", type = Path, help = "directory holding the base side's evidence")
    parser.add_argument("--head", type = Path, help = "directory holding the head side's evidence")
    parser.add_argument("--base-sha", default = "")
    parser.add_argument("--head-sha", default = "")
    parser.add_argument(
        "--self-test", action = "store_true", help = "run the positive controls and exit"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        failures = self_test()
        for failure in failures:
            print(f"::error::self-test: {failure}")
        if failures:
            print(
                "::error::the comparer's own controls failed, so no verdict it produces can be "
                "trusted. Refusing to compare."
            )
            return 1
        print("self-test: the comparer reports real changes and ignores known noise")
        return 0

    if not args.base or not args.head:
        parser.error("--base and --head are required unless --self-test is given")

    verdict = compare_directories(args.base, args.head, args.base_sha, args.head_sha)

    for note in verdict.notes:
        print(f"  {note}")

    if verdict.is_void:
        print()
        for reason in verdict.void:
            print(f"::error::VOID: {reason}")
        print(
            "::error::VOID is not a pass. Nothing was compared, so nothing was shown to be "
            "unchanged."
        )
        return verdict.exit_code()

    if verdict.differences:
        print()
        for difference in verdict.differences:
            print(f"::error::{difference}")
        print(
            f"::error::{len(verdict.differences)} behaviour difference(s) between "
            f"{args.base_sha[:12] or 'base'} and {args.head_sha[:12] or 'head'}. A hardening "
            f"change must not alter what the installer does or what a user sees."
        )
        return verdict.exit_code()

    print()
    print(
        f"PASS: {args.base_sha[:12] or 'base'} and {args.head_sha[:12] or 'head'} produced the "
        f"same user-visible output, the same shortcuts and the same installed files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
