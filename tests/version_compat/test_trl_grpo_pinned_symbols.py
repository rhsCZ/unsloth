# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across all TRL PyPI minor versions
unsloth + unsloth-zoo target. Catches API drift like:

  - trl 0.18 split DataCollatorForPreference into trl.trainer.dpo_trainer
    (was trl.trainer.utils). unsloth.models.rl_replacements:318 imports
    the post-split path; if a new TRL release moves it again, the
    GRPOTrainer.compile cell crashes with ImportError.
  - trl 0.20 introduced trl.experimental.openenv as a *gated* module;
    unsloth.models.rl_replacements:1765-1770 catches ImportError, but
    the gate must remain importable when present.
  - trl 0.22 introduced trl.generation.vllm_generation for the
    server-mode fast_inference path; unsloth.models.rl_replacements
    :1846-1848 catches ImportError, but the module must exist on
    versions where unsloth-zoo's vllm_utils dispatches to it.
  - trl unwrap_model_for_generation moved from trl.models to
    trl.models.utils across releases (unsloth/models/rl.py:152-155
    handles both with try/except).
  - trl GRPOTrainer / GRPOConfig must remain top-level exports for
    `from trl import GRPOTrainer` to work in user code, which is what
    `_patch_trl_rl_trainers("grpo_trainer")` discovers.

Strategy: for each tracked TRL tag, fetch the relevant source files
straight from github.com/huggingface/trl (no pip install required) and
assert that every symbol unsloth/unsloth-zoo's RL surface depends on
is present.

Versioning policy: cover the supported window declared in
pyproject.toml (`trl>=0.18.2,!=0.19.0,<=0.24.0`) PLUS several recent
releases ABOVE the cap, so we get early warning when TRL ships
something incompatible and the maintainer can extend the cap or add a
patch BEFORE a user hits it.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# Supported window: 0.18.2 -> 0.24.0 (excluding 0.19.0).
# Above-cap canaries: 0.25, 0.27, 0.29, 1.0, 1.3 (most recent stable at
# the time of writing). `main` is the bleeding edge. Add a row when a
# new minor lands; remove a row only when a release is unsupported
# AND we have a tracking issue.
TRL_TAGS = [
    "v0.18.2",
    "v0.20.0",
    "v0.21.0",
    "v0.22.2",
    "v0.23.0",
    "v0.24.0",  # current pyproject cap
    # Above-cap canaries:
    "v0.25.1",
    "v0.27.2",
    "v0.29.1",
    "v1.0.0",
    "v1.3.0",
    "main",
]


# -------------------------------------------------------------------------
# HARD-import top-level: from trl import X must keep working for these.
# unsloth/trainer.py + unsloth/models/rl.py rebind these by name.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_top_level_grpo_sft(tag: str):
    """`from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig`
    must keep resolving at the package root."""
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None, f"trl/__init__.py missing in {tag}"
    for name in ("GRPOTrainer", "GRPOConfig", "SFTTrainer", "SFTConfig"):
        assert name in src, (
            f"{tag}: `from trl import {name}` will fail; "
            f"unsloth/trainer.py + unsloth/models/rl.py rely on this re-export"
        )


# -------------------------------------------------------------------------
# trl.trainer.grpo_trainer.GRPOTrainer -- the canonical class. unsloth's
# RL patcher discovers it via `eval(f"trl.trainer.{trainer_file}.{name}")`
# in unsloth/models/rl.py:548-594.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_grpo_trainer_class_canonical_path(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None, (
        f"{tag}: trl/trainer/grpo_trainer.py missing — "
        f"unsloth.models.rl._patch_trl_rl_trainers('grpo_trainer') breaks"
    )
    assert has_def(src, "GRPOTrainer", "class"), (
        f"{tag}: trl.trainer.grpo_trainer.GRPOTrainer not defined as a class"
    )


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_grpo_config_class_canonical_path(tag: str):
    """unsloth/models/rl.py:579-618 looks for the *Config sibling of the
    Trainer class via heuristic discovery; the canonical one is in
    grpo_config.py."""
    candidates = ["trl/trainer/grpo_config.py", "trl/trainer/grpo_trainer.py"]
    hit = first_match("huggingface/trl", tag, candidates)
    assert hit is not None, f"{tag}: neither grpo_config.py nor grpo_trainer.py found"
    _, src = hit
    assert has_def(src, "GRPOConfig", "class"), (
        f"{tag}: GRPOConfig class missing in {[p for p, _ in [hit]]}; "
        f"unsloth's *Config heuristic in models/rl.py:579-618 will fail"
    )


# -------------------------------------------------------------------------
# DataCollatorForPreference: unsloth.models.rl_replacements:318 hard-imports
# from trl.trainer.dpo_trainer. Some old TRL versions had it in
# trl.trainer.utils; modern ones moved to trl.trainer.dpo_trainer.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_data_collator_for_preference_resolvable(tag: str):
    """Either the new path (trl.trainer.dpo_trainer) or the old path
    (trl.trainer.utils) must define DataCollatorForPreference. unsloth's
    string-emitted import in rl_replacements.py:318 uses dpo_trainer;
    if neither path resolves, we have a gap."""
    new_path = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    old_path = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    have = []
    if new_path is not None and "DataCollatorForPreference" in new_path:
        have.append("trl.trainer.dpo_trainer")
    if old_path is not None and "DataCollatorForPreference" in old_path:
        have.append("trl.trainer.utils")
    assert have, (
        f"{tag}: DataCollatorForPreference defined in NEITHER "
        f"trl/trainer/dpo_trainer.py NOR trl/trainer/utils.py — "
        f"unsloth/models/rl_replacements.py:318 will ImportError on real install"
    )


# -------------------------------------------------------------------------
# trl.trainer.utils.pad: emitted into the GRPO compile cell as
# _unsloth_trl_pad (rl_replacements.py:326).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_trainer_utils_pad(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    if src is None:
        # Some TRL versions split utils into a package; check the
        # alternative location.
        src = fetch_text("huggingface/trl", tag, "trl/trainer/utils/__init__.py")
    assert src is not None, f"{tag}: trl/trainer/utils[.py|/__init__.py] both missing"
    assert has_def(src, "pad", "func") or "def pad(" in src, (
        f"{tag}: trl.trainer.utils.pad missing — "
        f"unsloth/models/rl_replacements.py:326 emits `from trl.trainer.utils "
        f"import pad as _unsloth_trl_pad` into the GRPO compile cell"
    )


# -------------------------------------------------------------------------
# trl.models.unwrap_model_for_generation -- moved between submodules
# across releases. unsloth/models/rl.py:152-155 handles both paths.
# Assert at least one resolves on every tag.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_unwrap_model_for_generation_either_path(tag: str):
    candidates = [
        "trl/models/utils.py",
        "trl/models/__init__.py",
        "trl/extras/profiling.py",  # newer TRL versions hide it here
    ]
    found = False
    for path in candidates:
        src = fetch_text("huggingface/trl", tag, path)
        if src is None:
            continue
        if (
            "def unwrap_model_for_generation" in src
            or "unwrap_model_for_generation" in src
        ):
            found = True
            break
    assert found, (
        f"{tag}: trl.unwrap_model_for_generation not in any known path "
        f"({candidates}); unsloth/models/rl.py:152-155 will ImportError"
    )


# -------------------------------------------------------------------------
# trl.experimental.openenv: gated import (rl_replacements.py:1765-1770
# wraps in try/except). When present, must export the symbols unsloth
# patches.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_experimental_openenv_gated(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/experimental/openenv/__init__.py")
    if src is None:
        # OK: feature not in this release; unsloth's try/except handles it.
        pytest.skip(f"{tag}: trl.experimental.openenv not present (OK)")
    # Module exists -> at minimum, `utils` submodule must be importable
    # because unsloth patches via `import trl.experimental.openenv.utils`.
    utils_src = fetch_text(
        "huggingface/trl", tag, "trl/experimental/openenv/utils.py"
    )
    assert utils_src is not None, (
        f"{tag}: trl.experimental.openenv exists but utils.py missing; "
        f"unsloth/models/rl_replacements.py:1765 imports openenv.utils explicitly"
    )


# -------------------------------------------------------------------------
# trl.generation.vllm_generation: gated import for the fast_inference
# server mode (rl_replacements.py:1846-1848). When present, must define
# at least one symbol unsloth patches against.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_generation_vllm_generation_gated(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/generation/vllm_generation.py")
    if src is None:
        # OK: pre-server-mode TRL. unsloth's try/except handles absence.
        pytest.skip(f"{tag}: trl.generation.vllm_generation not present (OK)")
    # If present, at least one of these classes/funcs must be there;
    # unsloth-zoo dispatches via getattr() but the module being empty
    # means our patch will silently no-op rather than crash.
    needs_some = ["VLLMClient", "vllm_generate", "VLLM_AVAILABLE", "VLLMServer"]
    has_some = any(name in src for name in needs_some)
    assert has_some, (
        f"{tag}: trl.generation.vllm_generation exists but none of "
        f"{needs_some} present; unsloth-zoo's dispatch in vllm_utils "
        f"will silently no-op the server path"
    )


# -------------------------------------------------------------------------
# Sanity: TRL's __version__ string is parseable. unsloth/models/rl.py:63
# does `from trl import __version__ as trl_version_raw` and string-
# matches on it.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_version_parseable(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None
    # Recognised mechanisms:
    #   1. literal `__version__ = "x.y.z"`
    #   2. `from .version import __version__`
    #   3. `__version__ = version("trl")` from `importlib.metadata`
    #      (bare `version` symbol must be imported on a line above)
    has_literal = bool(re.search(r'^__version__\s*=\s*["\']', src, re.MULTILINE))
    has_subimport = bool(
        re.search(r"^from\s+\.version\s+import\s+__version__", src, re.MULTILINE)
    )
    # Importlib metadata path: any line `from importlib.metadata import ... version ...`
    # plus a `__version__ = version(` assignment somewhere below.
    has_metadata = bool(
        re.search(
            r"^from\s+importlib\.metadata\s+import\s+(?:[\w,\s]+,\s*)?version",
            src,
            re.MULTILINE,
        )
        and re.search(r'^\s*__version__\s*=\s*version\s*\(', src, re.MULTILINE)
    )
    assert has_literal or has_subimport or has_metadata, (
        f"{tag}: trl.__version__ not exported via any known mechanism; "
        f"unsloth/models/rl.py:63 will AttributeError"
    )
