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
    Case(
        "C1",
        "removing next-themes breaks 2 src imports",
        ["next-themes"],
        "FAIL",
        ["next-themes"],
    ),
    Case(
        "C2",
        "removing @xyflow/react breaks recipe-studio src imports "
        "(no other declared dep pulls @xyflow/react)",
        ["@xyflow/react"],
        "FAIL",
        ["@xyflow/react"],
    ),
    Case(
        "C3",
        "removing katex is safe: streamdown/math, mermaid, "
        "rehype-katex all keep it at top level",
        ["katex"],
        "PASS",
        [],
    ),
    Case("C4", "removing clsx is safe: streamdown keeps it", ["clsx"], "PASS", []),
    Case(
        "C5",
        "removing react is safe: peer of countless packages",
        ["react"],
        "PASS",
        [],
    ),
    Case(
        "C6",
        "removing @radix-ui/react-slot is safe: pulled by "
        "radix-ui umbrella + @assistant-ui/react",
        ["@radix-ui/react-slot"],
        "PASS",
        [],
    ),
    Case(
        "C7",
        "removing zustand is safe: @assistant-ui/react keeps "
        "top-level zustand@5.x (nested xyflow 4.x is irrelevant "
        "to src imports)",
        ["zustand"],
        "PASS",
        [],
    ),
    Case(
        "C8",
        "multi-remove with mixed safety: next-themes + "
        "@huggingface/hub + dexie all unsafe",
        ["next-themes", "@huggingface/hub", "dexie"],
        "FAIL",
        ["next-themes", "@huggingface/hub", "dexie"],
    ),
    Case(
        "C9",
        "removing @huggingface/hub breaks 5+ src imports",
        ["@huggingface/hub"],
        "FAIL",
        ["@huggingface/hub"],
    ),
    Case(
        "C10",
        "removing tailwind-merge is safe: streamdown keeps it",
        ["tailwind-merge"],
        "PASS",
        [],
    ),
    Case(
        "C11",
        "removing a non-existent name is a no-op",
        ["__never_existed_in_pkg__"],
        "PASS",
        [],
    ),
    Case(
        "C12",
        "moving @hugeicons/react from deps to devDeps is NOT a "
        "removal (still declared)",
        [],
        "PASS",
        [],
        move_to_dev = ["@hugeicons/react"],
    ),
    Case(
        "C13",
        "removing @huggingface/hub AND @xyflow/react together: both "
        "are root-only deps with no other parents, so both should FAIL",
        ["@huggingface/hub", "@xyflow/react"],
        "FAIL",
        ["@huggingface/hub", "@xyflow/react"],
    ),
    Case(
        "C14",
        "removing dexie breaks src imports (no other declared " "dep needs it)",
        ["dexie"],
        "FAIL",
        ["dexie"],
    ),
    Case(
        "C15",
        "removing motion (used in 20+ src imports including "
        "framer-motion-style animations); no transitive parent",
        ["motion"],
        "FAIL",
        ["motion"],
    ),
    Case(
        "C16",
        "removing canvas-confetti (imported in confetti.tsx); " "no transitive parent",
        ["canvas-confetti"],
        "FAIL",
        ["canvas-confetti"],
    ),
    Case(
        "C17",
        "removing recharts (imported in chart.tsx); no transitive " "parent",
        ["recharts"],
        "FAIL",
        ["recharts"],
    ),
    Case(
        "C18",
        "removing js-yaml is safe: @eslint/eslintrc keeps it "
        "(triggers @types/js-yaml orphan warning, non-fatal)",
        ["js-yaml"],
        "PASS",
        [],
    ),
    Case(
        "C19",
        "removing node-forge (imported in providers-api.ts); " "no transitive parent",
        ["node-forge"],
        "FAIL",
        ["node-forge"],
    ),
    Case(
        "C20",
        "removing @tauri-apps/api is safe: all 5 @tauri-apps "
        "plugins declare it as a direct dep",
        ["@tauri-apps/api"],
        "PASS",
        [],
    ),
    Case(
        "C21",
        "removing mammoth (imported in runtime-provider.tsx); " "no transitive parent",
        ["mammoth"],
        "FAIL",
        ["mammoth"],
    ),
    Case(
        "C22",
        "removing unpdf (imported in runtime-provider.tsx); " "no transitive parent",
        ["unpdf"],
        "FAIL",
        ["unpdf"],
    ),
    Case(
        "C23",
        "removing remark-gfm is safe: streamdown declares it " "as a direct dep",
        ["remark-gfm"],
        "PASS",
        [],
    ),
    Case(
        "C24",
        "removing date-fns is safe: react-day-picker and "
        "@base-ui/react both declare it as a direct dep",
        ["date-fns"],
        "PASS",
        [],
    ),
]


def synth_head(head_pkg: dict, case: Case) -> dict:
    out = json.loads(json.dumps(head_pkg))
    for name in case.remove:
        for field in (
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
        ):
            (out.get(field) or {}).pop(name, None)
    if case.move_to_dev:
        for name in case.move_to_dev:
            v = (out.get("dependencies") or {}).pop(name, None)
            if v is not None:
                out.setdefault("devDependencies", {})[name] = v
    return out


def run_case(case: Case, head_pkg: dict) -> tuple[bool, str]:
    synth = synth_head(head_pkg, case)
    with tempfile.NamedTemporaryFile("w", suffix = ".json", delete = False) as f:
        json.dump(synth, f, indent = 2)
        synth_path = f.name
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base-pkg",
                str(HEAD_PKG),
                "--head-pkg",
                synth_path,
                "--head-lock",
                str(HEAD_LOCK),
            ],
            capture_output = True,
            text = True,
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

    ok = actual_status == case.expected_status and set(failure_pkgs) == set(
        case.expected_failures
    )
    return ok, (
        f"expected: status={case.expected_status} fails={sorted(case.expected_failures)}\n"
        f"actual:   status={actual_status} fails={sorted(failure_pkgs)}\n"
        f"--- stdout (first 30 lines) ---\n" + "\n".join(proc.stdout.splitlines()[:30])
    )


# ---------------------------------------------------------------------------
# Classifier unit tests: feed hand-crafted snippets directly into classify()
# and assert the returned kind. Covers sneaky import shapes that an
# adversarial / careless dev might use to obscure a real usage.
# ---------------------------------------------------------------------------

# Import the script's classify() by file path so this test does not need
# the package to be installed.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_dep_check", str(SCRIPT))
_dep_check = _ilu.module_from_spec(_spec)
sys.modules["_dep_check"] = _dep_check  # required so @dataclass can resolve annotations
_spec.loader.exec_module(_dep_check)
classify = _dep_check.classify


@dataclass
class ClassifyCase:
    id: str
    desc: str
    pkg: str
    file: str
    content: str
    expected_kind: str | None  # None means "no detection"


CLASSIFY_CASES: list[ClassifyCase] = [
    # Bog-standard shapes
    ClassifyCase(
        "U01",
        "single-line static import",
        "next-themes",
        "src/x.tsx",
        'import { ThemeProvider } from "next-themes";',
        "static_import",
    ),
    ClassifyCase(
        "U02",
        "side-effect import",
        "katex",
        "src/x.tsx",
        'import "katex/dist/katex.min.css";',
        "side_effect_import",
    ),
    ClassifyCase(
        "U03",
        "dynamic import",
        "@tauri-apps/api",
        "src/x.tsx",
        'const { x } = await import("@tauri-apps/api/window");',
        "dynamic_import",
    ),
    ClassifyCase(
        "U04",
        "require()",
        "lodash",
        "src/x.js",
        'const _ = require("lodash");',
        "require",
    ),
    ClassifyCase(
        "U05",
        "CSS @import",
        "tailwindcss",
        "src/x.css",
        '@import "tailwindcss";',
        "css_import",
    ),
    # Sneaky shapes
    ClassifyCase(
        "U06",
        "multi-line static import",
        "next-themes",
        "src/x.tsx",
        'import {\n  ThemeProvider,\n  useTheme,\n} from "next-themes";',
        "static_import",
    ),
    ClassifyCase(
        "U07",
        "import type",
        "@huggingface/hub",
        "src/x.ts",
        'import type { PipelineType } from "@huggingface/hub";',
        "static_import",
    ),
    ClassifyCase(
        "U08",
        "export * from re-export",
        "@some-org/secrets",
        "src/x.ts",
        'export * from "@some-org/secrets";',
        "re_export",
    ),
    ClassifyCase(
        "U09",
        "export { x } from re-export",
        "lodash-es",
        "src/x.ts",
        'export { foo, bar } from "lodash-es";',
        "re_export",
    ),
    ClassifyCase(
        "U10",
        "export type ... from re-export",
        "@huggingface/hub",
        "src/x.ts",
        'export type { Foo } from "@huggingface/hub";',
        "re_export",
    ),
    ClassifyCase(
        "U11",
        "multi-line export from re-export",
        "@some/pkg",
        "src/x.ts",
        'export {\n  thing,\n  other,\n} from "@some/pkg";',
        "re_export",
    ),
    ClassifyCase(
        "U12",
        "JSDoc @import",
        "react",
        "src/x.ts",
        '/** @type {import("react").FC} */\nconst Foo = () => null;',
        "dynamic_import",
    ),
    ClassifyCase(
        "U13",
        "template literal package path",
        "@assistant-ui/react",
        "src/x.tsx",
        "const url = `@assistant-ui/react`;",
        "template_literal",
    ),
    ClassifyCase(
        "U14",
        "new URL import-meta",
        "monaco-editor",
        "src/x.ts",
        'new URL("monaco-editor/esm/vs/editor/editor.worker", import.meta.url);',
        "new_url",
    ),
    ClassifyCase(
        "U15",
        "tsc triple-slash type ref",
        "@types/some-pkg",
        "src/x.ts",
        '/// <reference types="@types/some-pkg" />',
        "tsc_triple_slash",
    ),
    ClassifyCase(
        "U16",
        "HTML script src",
        "alpinejs",
        "index.html",
        '<script src="/node_modules/alpinejs/dist/cdn.min.js"></script>',
        "html_script",
    ),
    ClassifyCase(
        "U17",
        "HTML link href",
        "alpinejs",
        "index.html",
        '<link rel="stylesheet" href="/node_modules/alpinejs/dist/style.css">',
        "html_link",
    ),
    ClassifyCase(
        "U18",
        "bare quoted string in tsconfig paths",
        "@huggingface/hub",
        "tsconfig.json",
        '"paths": { "hf": ["@huggingface/hub/*"] }',
        "string_literal",
    ),
    ClassifyCase(
        "U19",
        "vite alias key",
        "@dagrejs/dagre",
        "vite.config.ts",
        '"@dagrejs/dagre": path.resolve(__dirname, "./..."),',
        "string_literal",
    ),
    # False-positive guards (these should NOT detect)
    ClassifyCase(
        "U20",
        "different package with shared prefix",
        "foo",
        "src/x.ts",
        'import { x } from "foobar";',
        None,
    ),
    ClassifyCase(
        "U21",
        "package mentioned in plain comment text",
        "react",
        "src/x.ts",
        "// We migrated from react-router to tanstack-router",
        None,
    ),
    ClassifyCase(
        "U22",
        "package name as a URL path tail is NOT detected "
        "(boundary rule: pkg must be followed by quote or `/`)",
        "react",
        "src/x.ts",
        'const docs = "https://example.com/react";',
        None,
    ),
    ClassifyCase(
        "U23",
        "package name in Python file (ignored, "
        "Python can never import npm packages)",
        "playwright",
        "tests/x.py",
        'label: str = "playwright"',
        None,
    ),
    ClassifyCase(
        "U24",
        "exact-prefix collision: pkg 'lodash' and 'lodash-es'",
        "lodash",
        "src/x.ts",
        'import _ from "lodash-es";',
        None,
    ),
    ClassifyCase(
        "U25",
        "scoped pkg substring collision",
        "@radix-ui/react-label",
        "src/x.ts",
        'import x from "@radix-ui/react-label-extra";',
        None,
    ),
    ClassifyCase(
        "U26",
        "package only mentioned in a markdown link",
        "react",
        "README.md",
        "See [react](https://react.dev).",
        None,
    ),
    ClassifyCase(
        "U27",
        "side-effect import with subpath",
        "katex",
        "src/x.css",
        '@import "katex/dist/katex.min.css";',
        "css_import",
    ),
    ClassifyCase(
        "U28",
        "require.resolve",
        "lodash",
        "build/x.cjs",
        'const path = require.resolve("lodash/fp");',
        "require",
    ),
]


def run_classify_unit_tests() -> int:
    passed = 0
    for c in CLASSIFY_CASES:
        actual = classify(c.pkg, c.file, c.content)
        ok = actual == c.expected_kind
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {c.id}: {c.desc}")
        if not ok:
            print(f"      pkg={c.pkg!r} file={c.file!r}")
            print(f"      content={c.content!r}")
            print(f"      expected={c.expected_kind!r}, actual={actual!r}")
        if ok:
            passed += 1
    print()
    print(f"{passed}/{len(CLASSIFY_CASES)} classify-unit cases pass")
    return 0 if passed == len(CLASSIFY_CASES) else 1


# ---------------------------------------------------------------------------
# Adversarial end-to-end cases: drop a sneaky synthetic file into src/,
# run the checker, then clean up. Catches the case where pattern detection
# regresses for a real grep+classify pipeline (not just classify in isolation).
# ---------------------------------------------------------------------------

ADVERSARIAL_TMP_DIR = REPO / "studio/frontend/src/__dep_check_adversarial__"


@dataclass
class AdvCase:
    id: str
    desc: str
    filename: str
    content: str
    target_pkg: str
    expected_status: str
    expected_failures: list[str]


ADV_CASES: list[AdvCase] = [
    AdvCase(
        "A01",
        "multi-line import of removed pkg should FAIL",
        "adv01.ts",
        'import {\n  foo,\n  bar,\n} from "__adv_only_pkg_a__";\n',
        "__adv_only_pkg_a__",
        "FAIL",
        ["__adv_only_pkg_a__"],
    ),
    AdvCase(
        "A02",
        "export * from removed pkg should FAIL",
        "adv02.ts",
        'export * from "__adv_only_pkg_b__";\n',
        "__adv_only_pkg_b__",
        "FAIL",
        ["__adv_only_pkg_b__"],
    ),
    AdvCase(
        "A03",
        "export { x } from removed pkg should FAIL",
        "adv03.ts",
        'export { foo, bar } from "__adv_only_pkg_c__";\n',
        "__adv_only_pkg_c__",
        "FAIL",
        ["__adv_only_pkg_c__"],
    ),
    AdvCase(
        "A04",
        "export type ... from removed pkg should FAIL",
        "adv04.ts",
        'export type { Foo } from "__adv_only_pkg_d__";\n',
        "__adv_only_pkg_d__",
        "FAIL",
        ["__adv_only_pkg_d__"],
    ),
    AdvCase(
        "A05",
        "package with similar prefix should NOT trigger FAIL",
        "adv05.ts",
        # The file imports __adv_only_pkg_e_extra__, but we will try
        # to "remove" the shorter __adv_only_pkg_e__ name. The shorter
        # name has zero real usage, so removal must be safe.
        'import x from "__adv_only_pkg_e_extra__";\n',
        "__adv_only_pkg_e__",
        "PASS",
        [],
    ),
    AdvCase(
        "A06",
        "dynamic import of removed pkg should FAIL",
        "adv06.ts",
        'const m = await import("__adv_only_pkg_f__");\n',
        "__adv_only_pkg_f__",
        "FAIL",
        ["__adv_only_pkg_f__"],
    ),
    AdvCase(
        "A07",
        "new URL of removed pkg should FAIL",
        "adv07.ts",
        'const w = new URL("__adv_only_pkg_g__/worker.js", import.meta.url);\n',
        "__adv_only_pkg_g__",
        "FAIL",
        ["__adv_only_pkg_g__"],
    ),
    AdvCase(
        "A08",
        "string-concat dynamic import is unanalyzable (PASS)",
        "adv08.ts",
        'const m = await import("__adv_only_" + "pkg_h__");\n',
        "__adv_only_pkg_h__",
        "PASS",
        [],
    ),
    AdvCase(
        "A09",
        "package referenced only inside a JS comment "
        "is conservatively flagged via the string_literal fallback "
        "(this is acceptable -- err on the side of caution)",
        "adv09.ts",
        '// TODO: import x from "__adv_only_pkg_i__"\n',
        "__adv_only_pkg_i__",
        "FAIL",
        ["__adv_only_pkg_i__"],
    ),
    AdvCase(
        "A10",
        "package referenced only in a Python file should " "NOT trigger a JS FAIL",
        "adv10.py",
        'label = "__adv_only_pkg_j__"\n',
        "__adv_only_pkg_j__",
        "PASS",
        [],
    ),
    AdvCase(
        "A11",
        "package mentioned in a markdown doc file is "
        "ignored by JS-like-only string_literal",
        "adv11.md",
        "See [docs](https://example.com/__adv_only_pkg_k__).\n",
        "__adv_only_pkg_k__",
        "PASS",
        [],
    ),
    AdvCase(
        "A12",
        "JSDoc @import of removed pkg should FAIL",
        "adv12.ts",
        '/** @type {import("__adv_only_pkg_l__").Foo} */\n' "const x = null;\n",
        "__adv_only_pkg_l__",
        "FAIL",
        ["__adv_only_pkg_l__"],
    ),
]


def run_adversarial_cases() -> int:
    ADVERSARIAL_TMP_DIR.mkdir(parents = True, exist_ok = True)
    head_pkg = json.loads(HEAD_PKG.read_text())
    passed = 0
    for ac in ADV_CASES:
        # Drop the synthetic file.
        fpath = ADVERSARIAL_TMP_DIR / ac.filename
        try:
            fpath.write_text(ac.content)
            # Build a synthetic base that has the target pkg added; head
            # is the real head (without it). The script sees the pkg as
            # removed and scans the repo, which now includes our file.
            synth_base = json.loads(json.dumps(head_pkg))
            synth_base.setdefault("dependencies", {})[ac.target_pkg] = "^1.0.0"
            with tempfile.NamedTemporaryFile("w", suffix = ".json", delete = False) as f:
                json.dump(synth_base, f, indent = 2)
                base_path = f.name
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--base-pkg",
                        base_path,
                        "--head-pkg",
                        str(HEAD_PKG),
                        "--head-lock",
                        str(HEAD_LOCK),
                    ],
                    capture_output = True,
                    text = True,
                    cwd = str(REPO),
                )
            finally:
                os.unlink(base_path)
            actual_status = {0: "PASS", 1: "FAIL"}.get(
                proc.returncode, f"RC{proc.returncode}"
            )
            fails = []
            in_summary = False
            for line in proc.stdout.splitlines():
                if "FAIL:" in line and "removed package" in line:
                    in_summary = True
                    continue
                if in_summary and line.strip().startswith("- "):
                    fails.append(line.strip()[2:])
            ok = actual_status == ac.expected_status and set(fails) == set(
                ac.expected_failures
            )
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {ac.id}: {ac.desc}")
            if not ok:
                print(
                    f"      expected: status={ac.expected_status} fails={ac.expected_failures}"
                )
                print(f"      actual:   status={actual_status} fails={fails}")
                for ln in proc.stdout.splitlines()[:20]:
                    print(f"      {ln}")
            if ok:
                passed += 1
        finally:
            try:
                fpath.unlink()
            except FileNotFoundError:
                pass
    # Clean up the directory.
    try:
        ADVERSARIAL_TMP_DIR.rmdir()
    except OSError:
        pass
    print()
    print(f"{passed}/{len(ADV_CASES)} adversarial cases pass")
    return 0 if passed == len(ADV_CASES) else 1


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

    print()
    print(f"Running {len(CLASSIFY_CASES)} classify() unit cases")
    print()
    cls_rc = run_classify_unit_tests()

    print()
    print(f"Running {len(ADV_CASES)} adversarial end-to-end cases")
    print()
    adv_rc = run_adversarial_cases()

    if passed == total and cls_rc == 0 and adv_rc == 0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
