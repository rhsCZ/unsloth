#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Edge-case suite for scripts/check_frontend_dep_removal.py.

Each case patches a copy of studio/frontend/package.json to remove (or
move) a specific dependency, invokes the checker against the real
working tree's lockfile, and asserts the verdict matches expectations.

Run:
  python tests/studio/test_frontend_dep_removal.py

Exits 0 iff every case behaves as expected.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HEAD_PKG = REPO / "studio/frontend/package.json"
HEAD_LOCK = REPO / "studio/frontend/package-lock.json"
SCRIPT = REPO / "scripts/check_frontend_dep_removal.py"


@dataclass
class Case:
    id: str
    desc: str
    remove: list[str]
    expected_status: str  # "PASS" | "FAIL"
    expected_failures: list[str]
    move_to_dev: list[str] | None = None  # rare: deps moved, not removed


CASES: list[Case] = [
    Case("C1",  "removing next-themes breaks 2 src imports",
         ["next-themes"], "FAIL", ["next-themes"]),
    Case("C2",  "removing @xyflow/react breaks recipe-studio src imports "
                "(no other declared dep pulls @xyflow/react)",
         ["@xyflow/react"], "FAIL", ["@xyflow/react"]),
    Case("C3",  "removing katex is safe: streamdown/math, mermaid, "
                "rehype-katex all keep it at top level",
         ["katex"], "PASS", []),
    Case("C4",  "removing clsx is safe: streamdown keeps it",
         ["clsx"], "PASS", []),
    Case("C5",  "removing react is safe: peer of countless packages",
         ["react"], "PASS", []),
    Case("C6",  "removing @radix-ui/react-slot is safe: pulled by "
                "radix-ui umbrella + @assistant-ui/react",
         ["@radix-ui/react-slot"], "PASS", []),
    Case("C7",  "removing zustand is safe: @assistant-ui/react keeps "
                "top-level zustand@5.x (nested xyflow 4.x is irrelevant "
                "to src imports)",
         ["zustand"], "PASS", []),
    Case("C8",  "multi-remove with mixed safety: next-themes + "
                "@huggingface/hub + dexie all unsafe",
         ["next-themes", "@huggingface/hub", "dexie"], "FAIL",
         ["next-themes", "@huggingface/hub", "dexie"]),
    Case("C9",  "removing @huggingface/hub breaks 5+ src imports",
         ["@huggingface/hub"], "FAIL", ["@huggingface/hub"]),
    Case("C10", "removing tailwind-merge is safe: streamdown keeps it",
         ["tailwind-merge"], "PASS", []),
    Case("C11", "removing a non-existent name is a no-op",
         ["__never_existed_in_pkg__"], "PASS", []),
    Case("C12", "moving @hugeicons/react from deps to devDeps is NOT a "
                "removal (still declared)",
         [], "PASS", [], move_to_dev=["@hugeicons/react"]),
    Case("C13", "removing @huggingface/hub AND @xyflow/react together: both "
                "are root-only deps with no other parents, so both should FAIL",
         ["@huggingface/hub", "@xyflow/react"], "FAIL",
         ["@huggingface/hub", "@xyflow/react"]),
    Case("C14", "removing dexie breaks src imports (no other declared "
                "dep needs it)",
         ["dexie"], "FAIL", ["dexie"]),
    Case("C15", "removing motion (used in 20+ src imports including "
                "framer-motion-style animations); no transitive parent",
         ["motion"], "FAIL", ["motion"]),
    Case("C16", "removing canvas-confetti (imported in confetti.tsx); "
                "no transitive parent",
         ["canvas-confetti"], "FAIL", ["canvas-confetti"]),
    Case("C17", "removing recharts (imported in chart.tsx); no transitive "
                "parent",
         ["recharts"], "FAIL", ["recharts"]),
    Case("C18", "removing js-yaml is safe: @eslint/eslintrc keeps it "
                "(triggers @types/js-yaml orphan warning, non-fatal)",
         ["js-yaml"], "PASS", []),
    Case("C19", "removing node-forge (imported in providers-api.ts); "
                "no transitive parent",
         ["node-forge"], "FAIL", ["node-forge"]),
    Case("C20", "removing @tauri-apps/api is safe: all 5 @tauri-apps "
                "plugins declare it as a direct dep",
         ["@tauri-apps/api"], "PASS", []),
    Case("C21", "removing mammoth (imported in runtime-provider.tsx); "
                "no transitive parent",
         ["mammoth"], "FAIL", ["mammoth"]),
    Case("C22", "removing unpdf (imported in runtime-provider.tsx); "
                "no transitive parent",
         ["unpdf"], "FAIL", ["unpdf"]),
    Case("C23", "removing remark-gfm is safe: streamdown declares it "
                "as a direct dep",
         ["remark-gfm"], "PASS", []),
    Case("C24", "removing date-fns is safe: react-day-picker and "
                "@base-ui/react both declare it as a direct dep",
         ["date-fns"], "PASS", []),
]


def synth_head(head_pkg: dict, case: Case) -> dict:
    out = json.loads(json.dumps(head_pkg))
    for name in case.remove:
        for field in ("dependencies", "devDependencies",
                      "peerDependencies", "optionalDependencies"):
            (out.get(field) or {}).pop(name, None)
    if case.move_to_dev:
        for name in case.move_to_dev:
            v = (out.get("dependencies") or {}).pop(name, None)
            if v is not None:
                out.setdefault("devDependencies", {})[name] = v
    return out


def run_case(case: Case, head_pkg: dict) -> tuple[bool, str]:
    synth = synth_head(head_pkg, case)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(synth, f, indent=2)
        synth_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--base-pkg", str(HEAD_PKG),
             "--head-pkg", synth_path,
             "--head-lock", str(HEAD_LOCK)],
            capture_output=True, text=True,
        )
    finally:
        os.unlink(synth_path)

    actual_status = {0: "PASS", 1: "FAIL"}.get(proc.returncode, f"RC{proc.returncode}")
    failure_pkgs: list[str] = []
    in_summary = False
    for line in proc.stdout.splitlines():
        if "FAIL:" in line and "removed package" in line:
            in_summary = True
            continue
        if in_summary and line.strip().startswith("- "):
            failure_pkgs.append(line.strip()[2:])

    ok = (
        actual_status == case.expected_status
        and set(failure_pkgs) == set(case.expected_failures)
    )
    return ok, (
        f"expected: status={case.expected_status} fails={sorted(case.expected_failures)}\n"
        f"actual:   status={actual_status} fails={sorted(failure_pkgs)}\n"
        f"--- stdout (first 30 lines) ---\n"
        + "\n".join(proc.stdout.splitlines()[:30])
    )


def main() -> int:
    head_pkg = json.loads(HEAD_PKG.read_text())
    print(f"Running {len(CASES)} edge cases against {SCRIPT.relative_to(REPO)}")
    print()
    results: list[tuple[Case, bool, str]] = []
    for c in CASES:
        ok, detail = run_case(c, head_pkg)
        results.append((c, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {c.id}: {c.desc}")
        if not ok:
            for line in detail.splitlines():
                print(f"      {line}")
    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"{passed}/{total} edge cases pass")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
