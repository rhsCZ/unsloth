"""Tests for scripts/enforce_kwargs_spacing.py.

Focus on remove_blank_after_short_import (the blank-line-after-short-import-block
rule): it must fire on small nested import blocks, leave everything else alone,
never change the AST, and be idempotent. A couple of enforce_spacing checks pin
the existing kwarg-spacing behavior.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from enforce_kwargs_spacing import (  # noqa: E402
    enforce_spacing,
    remove_blank_after_short_import,
)


# (name, source) pairs where the blank after the import block MUST be removed.
_MUST_CHANGE = {
    "try_except_import": (
        "def f():\n"
        "    try:\n"
        "        import torch\n"
        "\n"
        "        return torch.inference_mode\n"
        "    except Exception:\n"
        "        from contextlib import nullcontext\n"
        "\n"
        "        return nullcontext\n"
    ),
    "if_from_import": (
        "def g():\n"
        "    if cond:\n"
        "        from . import locators\n"
        "\n"
        "        regions = locators.regions()\n"
    ),
    "multiple_consecutive_imports": (
        "def f():\n    import a\n    import b\n\n    return a, b\n"
    ),
    "type_checking_block": (
        "def f():\n"
        "    if TYPE_CHECKING:\n"
        "        import x\n"
        "\n"
        "        y = x\n"
        "        return y\n"
    ),
    "with_block": (
        "def f():\n    with ctx():\n        import a\n\n        return a.run()\n"
    ),
}

# Sources that MUST be left byte-for-byte unchanged.
_MUST_NOT_CHANGE = {
    "module_level": 'import os\n\nVALUE = os.environ.get("V")\n',
    "large_suite": (
        "def f():\n"
        "    import a\n"
        "\n"
        "    x = a.load()\n"
        "    y = transform(x)\n"
        "    return y\n"
    ),
    "comment_between": (
        "def f():\n    import a\n\n    # keep separated\n    return a.value\n"
    ),
    "import_is_last_stmt": "def f():\n    if cond:\n        import a\n\n",
    "no_blank_already": "def f():\n    import a\n    return a\n",
}


@pytest.mark.parametrize("name", sorted(_MUST_CHANGE))
def test_blank_removed_for_small_import_block(name):
    src = _MUST_CHANGE[name]
    out, changed = remove_blank_after_short_import(src)
    assert changed is True
    assert out != src
    # The import line and the following statement are now adjacent (no blank between).
    assert "\n\n" not in out or out.count("\n\n") < src.count("\n\n")
    # Semantics preserved and idempotent.
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    out2, changed2 = remove_blank_after_short_import(out)
    assert out2 == out and changed2 is False


@pytest.mark.parametrize("name", sorted(_MUST_NOT_CHANGE))
def test_blank_preserved_when_not_applicable(name):
    src = _MUST_NOT_CHANGE[name]
    out, changed = remove_blank_after_short_import(src)
    assert changed is False
    assert out == src


def test_exact_output_try_block():
    src = (
        "def f():\n"
        "    try:\n"
        "        import torch\n"
        "\n"
        "        return torch.inference_mode\n"
        "    except Exception:\n"
        "        from contextlib import nullcontext\n"
        "\n"
        "        return nullcontext\n"
    )
    expected = (
        "def f():\n"
        "    try:\n"
        "        import torch\n"
        "        return torch.inference_mode\n"
        "    except Exception:\n"
        "        from contextlib import nullcontext\n"
        "        return nullcontext\n"
    )
    out, changed = remove_blank_after_short_import(src)
    assert changed is True
    assert out == expected


def test_syntax_error_is_left_alone():
    src = "def f(:\n    import a\n\n    return a\n"
    out, changed = remove_blank_after_short_import(src)
    assert changed is False
    assert out == src


def test_enforce_spacing_pads_kwargs():
    src = "f(a=1, b = 2)\n"
    out, changed = enforce_spacing(src)
    assert changed is True
    assert "a = 1" in out and "b = 2" in out


def test_enforce_spacing_noop_when_already_spaced():
    src = "f(a = 1, b = 2)\n"
    out, changed = enforce_spacing(src)
    assert changed is False
    assert out == src
