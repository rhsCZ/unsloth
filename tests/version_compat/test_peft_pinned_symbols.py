# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across PEFT PyPI minor versions
unsloth + unsloth-zoo target. Catches API drift like:

  - peft 0.18 finalised the LoraConfig public surface (+ MoE-aware
    target_modules); unsloth uses target_modules + r + lora_alpha +
    lora_dropout + bias.
  - peft 0.19 introduced the LoraConfig.target_parameters extension;
    unsloth-zoo's MoE LoRA extractor in saving_utils.py reads it via
    getattr() so missing on older versions is OK but the attribute
    shape must remain stable on >= 0.19.
  - peft.tuners.lora package layout: LoraLayer / LoraConfig / Linear4bit
    re-exports must keep working under both `from peft import X` and
    `from peft.tuners.lora import X`.

Strategy: for each tracked PEFT tag, fetch source from
github.com/huggingface/peft (no pip install needed) and assert that
every symbol unsloth + unsloth-zoo's PEFT touchpoints depend on is
present.

Versioning policy: cover the supported window declared in
unsloth/pyproject.toml (`peft>=0.18.0,!=0.11.0`) plus `main`. The
`!=0.11.0` exclusion is for the historical broken release; we don't
test against it.
"""

from __future__ import annotations

import pytest

from tests.version_compat._fetch import fetch_text, has_def


# pyproject pin: peft>=0.18.0. Test the floor + each minor since.
# `main` catches breakage before a release lands.
PEFT_TAGS = [
    "v0.18.0",
    "v0.18.1",
    "v0.19.0",
    "v0.19.1",
    "main",
]


# -------------------------------------------------------------------------
# Top-level public re-exports. unsloth/models/sentence_transformer.py:1948
# does `from peft import LoraConfig, get_peft_model as peft_get_peft_model`.
# unsloth_zoo's saving_utils + lora extractors hit `peft.PeftModel`.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_top_level_exports(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/__init__.py")
    assert src is not None, f"{tag}: src/peft/__init__.py missing"
    needed = (
        "LoraConfig",
        "get_peft_model",
        "PeftModel",
    )
    missing = [n for n in needed if n not in src]
    assert not missing, (
        f"{tag}: peft top-level missing {missing}; "
        f"unsloth.models.sentence_transformer:1948 + unsloth-zoo saving_utils "
        f"will ImportError"
    )


# -------------------------------------------------------------------------
# LoraConfig at the canonical sub-module path: peft.tuners.lora.LoraConfig
# (or peft.tuners.lora.config.LoraConfig). unsloth-zoo's LoraConfig
# normaliser inspects it via getattr() and dataclass field
# introspection.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_config_class(tag: str):
    candidates = [
        "src/peft/tuners/lora/config.py",
        "src/peft/tuners/lora/__init__.py",
        "src/peft/tuners/lora.py",
    ]
    found_in = []
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is not None and has_def(src, "LoraConfig", "class"):
            found_in.append(p)
    assert found_in, f"{tag}: peft.tuners.lora.LoraConfig not in any of {candidates}"


# -------------------------------------------------------------------------
# get_peft_model: top-level helper used by sentence_transformer.py:2043.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_get_peft_model_function(tag: str):
    """`def get_peft_model(...)` may live in mapping.py (older
    layout) or mapping_func.py (peft 0.18+ split). Either is fine."""
    candidates = [
        "src/peft/mapping.py",
        "src/peft/mapping_func.py",
        "src/peft/__init__.py",
        "src/peft/peft_model.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is not None and has_def(src, "get_peft_model", "func"):
            return
    pytest.fail(f"{tag}: def get_peft_model(...) not found in any of {candidates}")


# -------------------------------------------------------------------------
# LoraLayer base class: unsloth-zoo's MoE LoRA extractor walks subclasses
# of peft.tuners.lora.LoraLayer to find quantised LoRA modules. If the
# class is renamed or moved, the walk silently returns 0 modules (the
# pytest tests mentioned in the audit report exercise exactly this).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_layer_class(tag: str):
    candidates = [
        "src/peft/tuners/lora/layer.py",
        "src/peft/tuners/lora/__init__.py",
        "src/peft/tuners/lora.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is not None and has_def(src, "LoraLayer", "class"):
            return
    pytest.fail(
        f"{tag}: class LoraLayer not in any of {candidates} — "
        f"unsloth-zoo MoE LoRA extractor relies on isinstance checks "
        f"against this class"
    )


# -------------------------------------------------------------------------
# bnb-aware LoRA: peft.tuners.lora.bnb is the integration point with
# bitsandbytes. unsloth + unsloth-zoo dispatch to this when the user
# loads a 4-bit base. Missing this module -> 4bit LoRA silently falls
# back to fp16 LoRA (silently bigger memory footprint).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_bnb_integration(tag: str):
    candidates = [
        "src/peft/tuners/lora/bnb.py",
        "src/peft/tuners/lora/_bnb.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is None:
            continue
        # The Linear4bit subclass naming is the contract -- either name
        # is fine, but at least one bnb-flavoured Linear must exist.
        has_4bit = any(
            cls in src
            for cls in (
                "class Linear4bit",
                "class Linear8bitLt",
                "class _Linear4bit",
                "class _Linear8bitLt",
            )
        )
        if has_4bit:
            return
    pytest.fail(
        f"{tag}: peft.tuners.lora.bnb missing or no Linear4bit/Linear8bitLt "
        f"class found; unsloth's 4-bit LoRA path silently degrades to fp16"
    )
