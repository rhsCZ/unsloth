#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Guard against breaking npm dependency removals in studio/frontend.

Diffs the current package.json against a git base, finds every package
that was removed, and confirms each is no longer referenced anywhere
in the repo. If a removed package is still imported and is not
transitively resolvable through the new lockfile, exits non-zero with
file:line citations.

Usage:
  python scripts/check_frontend_dep_removal.py
  python scripts/check_frontend_dep_removal.py --base origin/main
  python scripts/check_frontend_dep_removal.py --base HEAD~1
  python scripts/check_frontend_dep_removal.py --base-pkg PATH --base-lock PATH

Exit codes:
  0  every removed dep is safe (no source refs or still resolvable)
  1  at least one removed dep is referenced and not resolvable
  2  invocation error (bad args, missing file, git error)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_PKG = "studio/frontend/package.json"
FRONTEND_LOCK = "studio/frontend/package-lock.json"

DEP_FIELDS = (
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
)

# Sources where seeing a package name does NOT count as usage.
EXPECTED_NOISE_FILES = {
    "studio/frontend/package.json",
    "studio/frontend/package-lock.json",
    "studio/backend/core/data_recipe/oxc-validator/package.json",
    "studio/backend/core/data_recipe/oxc-validator/package-lock.json",
}

# Only quoted-string occurrences in these file types can be module specifiers.
JS_LIKE_EXT = re.compile(
    r"\.(ts|tsx|js|jsx|mjs|cjs|html|htm|css|scss|sass|json|jsonc)$"
)

GREP_INCLUDES = [
    "--include=*.ts",
    "--include=*.tsx",
    "--include=*.js",
    "--include=*.jsx",
    "--include=*.mjs",
    "--include=*.cjs",
    "--include=*.html",
    "--include=*.htm",
    "--include=*.css",
    "--include=*.scss",
    "--include=*.sass",
    "--include=*.json",
    "--include=*.jsonc",
    "--include=*.md",
    "--include=*.mdx",
    "--include=*.py",
    "--include=*.rs",
    "--include=*.toml",
    "--include=*.yml",
    "--include=*.yaml",
    "--include=*.sh",
    "--include=*.ps1",
    "--include=*.bat",
    "--include=Dockerfile*",
]
GREP_EXCLUDES = [
    "--exclude-dir=node_modules",
    "--exclude-dir=dist",
    "--exclude-dir=.git",
    "--exclude-dir=__pycache__",
    "--exclude-dir=target",
    "--exclude-dir=.next",
    "--exclude-dir=build",
    "--exclude-dir=.venv",
    "--exclude-dir=venv",
]

# A pip-installed playwright reference is the PyPI package, not npm.
PIP_PLAYWRIGHT = re.compile(
    r"(pip\s+install\s+['\"]?playwright"
    r"|python\s+-m\s+playwright"
    r"|from\s+playwright"
    r"|^\s*import\s+playwright)"
)


@dataclass
class Hit:
    file: str
    line: int
    kind: str
    snippet: str


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command, return stdout. On non-zero exit, return ''."""
    res = subprocess.run(
        cmd,
        cwd = cwd or REPO_ROOT,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    return res.stdout if res.returncode == 0 else ""


def read_pkg_at(base: str, path: str) -> dict:
    """Read JSON at `base:path` via git show. Empty dict if missing."""
    out = run(["git", "show", f"{base}:{path}"])
    if not out.strip():
        return {}
    return json.loads(out)


def read_pkg_file(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def all_decl_names(pkg: dict) -> set[str]:
    names: set[str] = set()
    for field in DEP_FIELDS:
        names.update((pkg.get(field) or {}).keys())
    return names


def lock_resolvable(lock: dict, name: str) -> list[str]:
    """Return lockfile paths that still install `name` (transitive ok).
    Naive: any path matching the name. May give false positives on a stale
    lockfile. Prefer `reachable_from_head` for sync-aware analysis.
    """
    pkgs = lock.get("packages", {})
    hits = []
    for path in pkgs:
        if path == f"node_modules/{name}" or path.endswith(f"/node_modules/{name}"):
            hits.append(path)
    return hits


def _resolve_install_path(parent_path: str, name: str, pkgs: dict) -> str | None:
    """Walk up the nested node_modules chain from `parent_path` to find
    where `name` actually resolves. Mirrors Node module resolution.
    """
    parts = parent_path.split("/node_modules/")
    for i in range(len(parts), 0, -1):
        prefix = "/node_modules/".join(parts[:i])
        trial = (prefix + "/node_modules/" if prefix else "node_modules/") + name
        if trial in pkgs:
            return trial
    if f"node_modules/{name}" in pkgs:
        return f"node_modules/{name}"
    return None


def _deps_of(meta: dict) -> dict:
    out = {}
    for field in ("dependencies", "peerDependencies", "optionalDependencies"):
        out.update(meta.get(field) or {})
    return out


def reachable_from_head(head_pkg: dict, lock: dict) -> set[str]:
    """BFS the lockfile dep graph starting from `head_pkg`'s top-level
    declared deps. Returns the set of lockfile install paths that survive.
    Stale lockfile entries (orphaned by the new package.json) are excluded.
    """
    pkgs = lock.get("packages", {})
    if not pkgs:
        return set()
    roots = all_decl_names(head_pkg)
    seen: set[str] = set()
    frontier: list[str] = []
    for name in roots:
        p = _resolve_install_path("", name, pkgs)
        if p:
            frontier.append(p)
    while frontier:
        path = frontier.pop()
        if path in seen:
            continue
        seen.add(path)
        meta = pkgs.get(path, {})
        for dep_name in _deps_of(meta):
            p = _resolve_install_path(path, dep_name, pkgs)
            if p and p not in seen:
                frontier.append(p)
    return seen


def classify(pkg: str, file: str, content: str) -> str | None:
    """Return why `content` references `pkg`, or None.

    `content` may span multiple lines (for multi-line imports/exports);
    each pattern uses re.DOTALL where it matters. The bare-spec
    regexes use a word-boundary check on the package name so that
    `foobar` does not match `foo`.
    """
    esc = re.escape(pkg)
    # Subpath gate: after the package name, the next char must be either
    # the closing quote, `/`, or end-of-string. Prevents foo matching foobar.
    sub = r"(?:/[^'\"`]*)?"

    flags_dotall = re.DOTALL | re.MULTILINE

    # CSS @import is checked first so it does not collide with the
    # side-effect-import regex below.
    if re.search(rf"@import\s+['\"]{esc}{sub}['\"]", content):
        return "css_import"
    # Static imports: handle multi-line `import { ... } from "pkg"` by
    # allowing arbitrary content (newlines included) between `import`
    # and `from`. The non-greedy match plus the required `from` keeps
    # this scoped to a single statement.
    if re.search(
        rf"(?<!@)\bimport\b[^;'\"]*?\bfrom\s+['\"]{esc}{sub}['\"]",
        content,
        flags_dotall,
    ):
        return "static_import"
    # Side-effect import: `import "pkg"` (no `from`). The negative
    # lookbehind rules out CSS `@import` lines.
    if re.search(rf"(?<!@)\bimport\s+['\"]{esc}{sub}['\"]", content):
        return "side_effect_import"
    # Dynamic import: `import("pkg")` and `await import("pkg")`.
    if re.search(rf"\bimport\(\s*['\"]{esc}{sub}['\"]\s*\)", content):
        return "dynamic_import"
    # require / require.resolve
    if re.search(rf"\brequire(?:\.resolve)?\(\s*['\"]{esc}{sub}['\"]\s*\)", content):
        return "require"
    # Re-exports: `export * from "pkg"`, `export { x } from "pkg"`,
    # `export type { Foo } from "pkg"`. Multi-line supported.
    if re.search(
        rf"\bexport\b[^;'\"]*?\bfrom\s+['\"]{esc}{sub}['\"]", content, flags_dotall
    ):
        return "re_export"
    # HTML script / link
    if re.search(rf"<script[^>]*src\s*=\s*['\"][^'\"]*/{esc}", content):
        return "html_script"
    if re.search(rf"<link[^>]*href\s*=\s*['\"][^'\"]*/{esc}", content):
        return "html_link"
    # TypeScript triple-slash
    if re.search(rf"///\s*<reference\s+types\s*=\s*['\"]{esc}{sub}['\"]", content):
        return "tsc_triple_slash"
    # new URL("pkg/...", import.meta.url)
    if re.search(rf"\bnew\s+URL\(\s*['\"]{esc}{sub}['\"]", content):
        return "new_url"
    # CSS url(...) referencing the package
    if re.search(rf"\burl\(\s*['\"]?[^)'\"]*/{esc}/", content):
        return "css_url"
    # Template literal containing the package as the leading specifier
    if re.search(rf"`{esc}{sub}`", content):
        return "template_literal"
    # JSDoc / TS @import comment: `@import("pkg")`
    if re.search(rf"@import\(\s*['\"]{esc}{sub}['\"]\s*\)", content):
        return "jsdoc_import"
    # Bare quoted-string fallback (config plugin lists, vite aliases,
    # tsconfig paths, biome config plugin arrays, shadcn registries).
    if file in EXPECTED_NOISE_FILES:
        return None
    if not JS_LIKE_EXT.search(file):
        return None
    # Boundary: pkg must be followed by `'`, `"`, or `/` to avoid
    # matching `foo` inside `foobar`.
    if re.search(rf"['\"]{esc}(?:['\"]|/)", content):
        return "string_literal"
    return None


def lockfile_root_sync(head_pkg: dict, head_lock: dict) -> list[str]:
    """Return a list of warnings if package-lock.json's <root> dep map
    disagrees with package.json (i.e., npm install was not re-run).
    """
    warnings = []
    if not head_lock:
        return warnings
    root = head_lock.get("packages", {}).get("", {})
    lock_decl = {
        **(root.get("dependencies") or {}),
        **(root.get("devDependencies") or {}),
        **(root.get("peerDependencies") or {}),
        **(root.get("optionalDependencies") or {}),
    }
    pkg_decl = {}
    for f in DEP_FIELDS:
        pkg_decl.update(head_pkg.get(f) or {})
    only_in_lock = set(lock_decl) - set(pkg_decl)
    only_in_pkg = set(pkg_decl) - set(lock_decl)
    if only_in_lock:
        warnings.append(
            f"lockfile <root> lists deps not in package.json (lockfile stale): {sorted(only_in_lock)}"
        )
    if only_in_pkg:
        warnings.append(
            f"package.json declares deps not in lockfile <root> (run npm install): {sorted(only_in_pkg)}"
        )
    return warnings


def types_orphan_warnings(head_pkg: dict) -> list[str]:
    """Flag @types/<X> deps where <X> is no longer declared anywhere
    in package.json. Removing X without also dropping @types/X leaves
    dangling type packages.
    """
    decl = set()
    for f in DEP_FIELDS:
        decl.update((head_pkg.get(f) or {}).keys())
    warnings = []
    for name in decl:
        if not name.startswith("@types/"):
            continue
        # @types/foo provides types for `foo`
        # @types/foo-bar provides types for `foo-bar`
        # @types/scope__pkg provides types for `@scope/pkg`
        target = name[len("@types/") :]
        if "__" in target:
            scope, sub = target.split("__", 1)
            target = f"@{scope}/{sub}"
        if target == "node":
            continue  # Node.js types are always implicit
        if target not in decl:
            warnings.append(
                f"@types/{target.replace('@', '').replace('/', '__')} present but '{target}' is not declared"
            )
    return warnings


def find_imports_without_decl(head_pkg: dict) -> list[tuple[str, int, str]]:
    """Reverse check: find bare-specifier imports in studio/frontend/src
    that don't correspond to any declared package.json dep. Catches the
    case where someone adds an import but forgets the dep declaration.
    Returns (file, line, spec) tuples.
    """
    decl = set()
    for f in DEP_FIELDS:
        decl.update((head_pkg.get(f) or {}).keys())
    # Also: anything tsconfig path-aliases (just '@/...' here) is internal
    pattern = r"(?:from|import)\s+['\"]([^'\"./][^'\"]*)['\"]"
    args = [
        "grep",
        "-rnE",
        pattern,
        "--include=*.ts",
        "--include=*.tsx",
        "--include=*.js",
        "--include=*.jsx",
        "studio/frontend/src",
    ]
    out = run(args)
    missing = []
    for line in out.splitlines():
        m = re.match(r"^(?:\./)?([^:]+):(\d+):(.*)$", line)
        if not m:
            continue
        file, ln, content = m.group(1), int(m.group(2)), m.group(3)
        for spec_match in re.finditer(pattern, content):
            spec = spec_match.group(1)
            # Resolve to package name (strip subpath)
            if spec.startswith("@"):
                parts = spec.split("/", 2)
                pkg_name = "/".join(parts[:2]) if len(parts) >= 2 else spec
            else:
                pkg_name = spec.split("/", 1)[0]
            if pkg_name in decl:
                continue
            # Internal aliases like '@/foo' or starts with builtin names
            if pkg_name == "@":
                continue
            if pkg_name in {
                "node:fs",
                "node:path",
                "fs",
                "path",
                "url",
                "stream",
                "crypto",
                "buffer",
                "util",
                "events",
                "child_process",
            }:
                continue
            missing.append((file, ln, spec))
    return missing


def grep_repo(pat: str) -> list[tuple[str, int, str]]:
    args = ["grep", "-rnE", pat] + GREP_INCLUDES + GREP_EXCLUDES + ["."]
    out = run(args)
    rows = []
    for line in out.splitlines():
        m = re.match(r"^(\./)?([^:]+):(\d+):(.*)$", line)
        if m:
            rows.append((m.group(2), int(m.group(3)), m.group(4)))
    return rows


_file_lines_cache: dict[str, list[str]] = {}


def _read_file(path: str) -> list[str]:
    if path not in _file_lines_cache:
        try:
            _file_lines_cache[path] = (
                Path(path).read_text(errors = "replace").splitlines()
            )
        except (OSError, UnicodeDecodeError):
            _file_lines_cache[path] = []
    return _file_lines_cache[path]


def find_usage(pkg: str) -> list[Hit]:
    """Return real usages of `pkg`. Filters pip-playwright separately.

    For each filename returned by grep, also feed a multi-line window
    around the matching line into classify() so multi-line imports
    (`import {\n a\n} from "pkg"`) get picked up.
    """
    rows = grep_repo(re.escape(pkg))
    hits = []
    seen_keys: set[tuple[str, str]] = set()
    for file, lineno, content in rows:
        if pkg == "playwright" and PIP_PLAYWRIGHT.search(content):
            continue
        # Try the single-line classify first.
        kind = classify(pkg, file, content)
        if not kind:
            # Multi-line window: 4 lines above + the line itself + 4 below.
            lines = _read_file(file)
            lo = max(0, lineno - 5)
            hi = min(len(lines), lineno + 4)
            window = "\n".join(lines[lo:hi])
            kind = classify(pkg, file, window)
        if kind:
            key = (file, kind)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append(Hit(file, lineno, kind, content[:160]))
    return hits


def main() -> int:
    p = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--base",
        default = "origin/main",
        help = "git ref to diff against (default: origin/main). "
        "Examples: HEAD~1, main, a-tag, a-sha.",
    )
    p.add_argument(
        "--base-pkg", help = "optional override: read base package.json from this path"
    )
    p.add_argument(
        "--base-lock", help = "optional override: read base lockfile from this path"
    )
    p.add_argument(
        "--head-pkg",
        default = str(REPO_ROOT / FRONTEND_PKG),
        help = "head package.json path (default: working tree)",
    )
    p.add_argument(
        "--head-lock",
        default = str(REPO_ROOT / FRONTEND_LOCK),
        help = "head lockfile path (default: working tree)",
    )
    p.add_argument("--verbose", action = "store_true")
    p.add_argument(
        "--strict",
        action = "store_true",
        help = "Also fail on hygiene warnings (lockfile sync, "
        "@types orphans, imports without declared dep).",
    )
    args = p.parse_args()

    if args.base_pkg:
        base_pkg = read_pkg_file(Path(args.base_pkg))
    else:
        base_pkg = read_pkg_at(args.base, FRONTEND_PKG)
    head_pkg = read_pkg_file(Path(args.head_pkg))
    if not base_pkg:
        print(
            f"ERROR: could not read base package.json at {args.base}:{FRONTEND_PKG}",
            file = sys.stderr,
        )
        return 2
    if not head_pkg:
        print(
            f"ERROR: could not read head package.json at {args.head_pkg}",
            file = sys.stderr,
        )
        return 2

    if args.base_lock:
        head_lock = read_pkg_file(Path(args.head_lock))
    else:
        head_lock = read_pkg_file(Path(args.head_lock))

    base_names = all_decl_names(base_pkg)
    head_names = all_decl_names(head_pkg)
    removed = sorted(base_names - head_names)

    if not removed:
        print("[OK] no dependencies removed from studio/frontend/package.json")
        return 0

    print(
        f"Checking {len(removed)} removed package(s) from studio/frontend/package.json"
    )
    print(f"Base: {args.base}    Head: working tree")
    print()

    reachable_paths = reachable_from_head(head_pkg, head_lock) if head_lock else set()

    def reachable_install_paths(name: str) -> tuple[str | None, list[str]]:
        """Return (top_level_path, nested_paths). top_level is what bare
        `import "name"` from src/ actually resolves to; nested copies are
        only visible inside the parent package that nested them.
        """
        top = f"node_modules/{name}"
        top_path = top if top in reachable_paths else None
        nested = sorted(
            p
            for p in reachable_paths
            if p != top and p.endswith(f"/node_modules/{name}")
        )
        return top_path, nested

    failures: list[tuple[str, list[Hit]]] = []
    for name in removed:
        hits = find_usage(name)
        top, nested = reachable_install_paths(name)
        importable_top_level = top is not None
        # Source imports of bare specifier `name` resolve ONLY to top-level
        # node_modules/<name>. Nested copies under another package are
        # invisible to src/ files.
        if hits and not importable_top_level:
            status = "FAIL"
        elif hits and importable_top_level:
            status = "OK-via-transitive"
        else:
            status = "OK"
        print(f"  [{status}] {name}")
        if top:
            print(f"    reachable (top-level): {top}")
        if nested:
            print(
                f"    reachable (nested, NOT importable from src/): {nested[0]}"
                + (f" (+{len(nested)-1} more)" if len(nested) > 1 else "")
            )
        if hits:
            for h in hits[:5]:
                print(f"    [{h.kind}] {h.file}:{h.line}  {h.snippet}")
        if status == "FAIL":
            failures.append((name, hits))
        if args.verbose and not hits and not (top or nested):
            print("    no references, not reachable -- clean removal")

    print()

    # Additional hygiene checks (warnings unless --strict).
    sync_warns = lockfile_root_sync(head_pkg, head_lock)
    if sync_warns:
        print("Lockfile sync warnings:")
        for w in sync_warns:
            print(f"  - {w}")
        print()
    types_warns = types_orphan_warnings(head_pkg)
    if types_warns:
        print("@types orphan warnings:")
        for w in types_warns:
            print(f"  - {w}")
        print()
    missing_imports = find_imports_without_decl(head_pkg)
    if missing_imports:
        print(f"Imports without a matching package.json dep ({len(missing_imports)}):")
        for file, ln, spec in missing_imports[:20]:
            print(f"  - {file}:{ln}  imports '{spec}'")
        print()

    strict_failures = bool(failures) or (
        args.strict and (sync_warns or types_warns or missing_imports)
    )

    if failures:
        print(
            f"FAIL: {len(failures)} removed package(s) still referenced and not resolvable"
        )
        for name, _ in failures:
            print(f"  - {name}")
        return 1
    if strict_failures:
        print("FAIL (--strict): one or more hygiene warnings present")
        return 1

    print("PASS: all removed packages are safe to drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
