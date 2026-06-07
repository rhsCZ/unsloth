#!/usr/bin/env python3
"""Ensure keyword arguments use spaces around '=', prune redundant pass statements,
drop the blank line after a short indented import block."""

from __future__ import annotations

import ast
import argparse
import io
import os
import sys
import tempfile
import tokenize
from collections import defaultdict
from pathlib import Path


def _atomic_write_text(path: Path, data: str, encoding: str) -> None:
    """Write ``data`` to ``path`` atomically.

    Stages a tmp file in the same directory (so it's on the same
    filesystem as the destination), fsyncs, then `os.replace`s into
    place. A crash mid-write therefore leaves either the previous
    content or the fully new content -- never a truncated source file.
    """
    dirpath = str(path.parent) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".kwargs_fix.", dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def enforce_spacing(text: str) -> tuple[str, bool]:
    """Return updated text with keyword '=' padded by spaces, plus change flag."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return text, False

    offsets: dict[int, int] = defaultdict(int)
    changed = False

    reader = io.StringIO(text).readline
    for token in tokenize.generate_tokens(reader):
        if token.type != tokenize.OP or token.string != "=":
            continue

        line_index = token.start[0] - 1
        col = token.start[1] + offsets[line_index]

        if line_index < 0 or line_index >= len(lines):
            continue

        line = lines[line_index]
        if col >= len(line) or line[col] != "=":
            continue

        line_changed = False

        # Insert a space before '=' when missing and not preceded by whitespace.
        if col > 0 and line[col - 1] not in {" ", "\t"}:
            line = f"{line[:col]} {line[col:]}"
            offsets[line_index] += 1
            col += 1
            line_changed = True
            changed = True

        # Insert a space after '=' when missing and not followed by whitespace or newline.
        next_index = col + 1
        if next_index < len(line) and line[next_index] not in {" ", "\t", "\n", "\r"}:
            line = f"{line[:next_index]} {line[next_index:]}"
            offsets[line_index] += 1
            line_changed = True
            changed = True

        if line_changed:
            lines[line_index] = line

    if not changed:
        return text, False

    return "".join(lines), True


def remove_redundant_passes(text: str) -> tuple[str, bool]:
    """Drop pass statements that share a block with other executable code."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    redundant: list[ast.Pass] = []

    def visit(node: ast.AST) -> None:
        for attr in ("body", "orelse", "finalbody"):
            value = getattr(node, attr, None)
            if not isinstance(value, list) or len(value) <= 1:
                continue
            for stmt in value:
                if isinstance(stmt, ast.Pass):
                    redundant.append(stmt)
            for stmt in value:
                if isinstance(stmt, ast.AST):
                    visit(stmt)
        handlers = getattr(node, "handlers", None)
        if handlers:
            for handler in handlers:
                visit(handler)

    visit(tree)

    if not redundant:
        return text, False

    lines = text.splitlines(keepends=True)
    changed = False

    for node in sorted(
        redundant, key=lambda item: (item.lineno, item.col_offset), reverse=True
    ):
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        if start >= len(lines):
            continue
        changed = True
        if start == end:
            line = lines[start]
            col_start = node.col_offset
            col_end = node.end_col_offset or (col_start + 4)
            segment = line[:col_start] + line[col_end:]
            lines[start] = segment if segment.strip() else ""
            continue

        # Defensive fall-back for unexpected multi-line 'pass'.
        prefix = lines[start][: node.col_offset]
        lines[start] = prefix if prefix.strip() else ""
        for idx in range(start + 1, end):
            lines[idx] = ""
        suffix = lines[end][(node.end_col_offset or 0) :]
        lines[end] = suffix

    # Normalise to ensure lines end with newlines except at EOF.
    result_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            continue
        if index < len(lines) - 1 and not line.endswith("\n"):
            result_lines.append(f"{line}\n")
        else:
            result_lines.append(line)

    return "".join(result_lines), changed


def remove_blank_after_short_import(text: str) -> tuple[str, bool]:
    """Drop blank line(s) after an import block in a *small* nested suite.

    Inside an indented suite of <= 3 statements (function/try/if/with/etc., never
    module level), when a run of consecutive ``import`` / ``from ... import``
    statements is directly followed -- across one or more blank lines and nothing
    else -- by another statement in the same suite, remove those blank lines so
    the import sits next to the code that uses it. A comment in the gap blocks the
    rule. Removing blank lines never changes the AST, so this is always
    semantics-preserving.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    lines = text.splitlines(keepends=True)
    import_types = (ast.Import, ast.ImportFrom)
    drop: set[int] = set()  # 1-based physical line numbers to delete

    def suites_of(node: ast.AST) -> list[list[ast.stmt]]:
        if isinstance(node, ast.Module):
            return []  # module-level import spacing is left alone
        out: list[list[ast.stmt]] = []
        for attr in ("body", "orelse", "finalbody"):
            val = getattr(node, attr, None)
            if isinstance(val, list) and val and all(isinstance(s, ast.stmt) for s in val):
                out.append(val)
        return out

    for node in ast.walk(tree):
        for suite in suites_of(node):
            if len(suite) > 3:  # only small blocks
                continue
            i = 0
            while i < len(suite):
                if not isinstance(suite[i], import_types):
                    i += 1
                    continue
                j = i
                while j + 1 < len(suite) and isinstance(suite[j + 1], import_types):
                    j += 1
                if j + 1 < len(suite):  # an import block followed by another statement
                    last_imp, nxt = suite[j], suite[j + 1]
                    gap = range((last_imp.end_lineno or last_imp.lineno) + 1, nxt.lineno)
                    nums = [n for n in gap if 1 <= n <= len(lines)]
                    if nums and all(lines[n - 1].strip() == "" for n in nums):
                        drop.update(nums)
                i = j + 1

    if not drop:
        return text, False
    kept = [ln for idx, ln in enumerate(lines, start=1) if idx not in drop]
    return "".join(kept), True


def process_file(path: Path) -> bool:
    try:
        with tokenize.open(path) as handle:
            original = handle.read()
            encoding = handle.encoding
    except (OSError, SyntaxError) as exc:  # SyntaxError from tokenize on invalid python
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
        return False

    updated, changed = enforce_spacing(original)
    updated, blanked = remove_blank_after_short_import(updated)
    updated, removed = remove_redundant_passes(updated)
    if changed or blanked or removed:
        _atomic_write_text(path, updated, encoding)
        return True
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Python files to fix")
    args = parser.parse_args(argv)

    touched: list[Path] = []
    self_path = Path(__file__).resolve()

    for entry in args.files:
        path = Path(entry)
        # Skip modifying this script to avoid self-edit loops.
        if path.resolve() == self_path:
            continue
        if not path.exists() or path.is_dir():
            continue
        if process_file(path):
            touched.append(path)

    if touched:
        for path in touched:
            print(f"Adjusted kwarg spacing in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
