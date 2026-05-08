# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across bitsandbytes PyPI minor versions
unsloth + unsloth-zoo target. Catches API drift like:

  - bnb 0.46.0 release was broken (in pyproject.toml as `!=0.46.0`).
    Don't test against it.
  - bnb 0.48.0 release was broken (also `!=0.48.0`). Same.
  - bnb 0.45 series introduced fp4 + nf4 paged optimisers; unsloth-zoo
    expects bnb.functional.dequantize_4bit + bnb.nn.Linear4bit /
    Params4bit to remain stable from this point onward.
  - vLLM bitsandbytes-loader patches in unsloth_zoo/vllm_utils.py:
    apply_bnb_4bit (line 237), is_layer_skipped_bnb (line 281),
    BitsAndBytesLinearMethod._apply_4bit_weight (line 282) — these
    live in vllm.* but they call into bnb's public surface.

Strategy: GitHub raw fetch + symbol grep. CPU-only, no install.
"""

from __future__ import annotations

import pytest

from tests.version_compat._fetch import fetch_text, has_def, first_match


# pyproject pin: bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0
# Test floor + each safe minor since.
BNB_TAGS = [
    "0.45.5",
    "0.47.0",  # skip 0.46.0 (broken)
    "0.49.2",  # skip 0.48.0 (broken)
    "main",
]


# -------------------------------------------------------------------------
# bnb.functional: dequantize_4bit / quantize_4bit are the public 4-bit
# surface unsloth's compiled kernels and unsloth-zoo's vllm_utils
# bnb-loader patches all call into.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_functional_4bit(tag: str):
    candidates = [
        "bitsandbytes/functional.py",
        "bitsandbytes/functional/__init__.py",
    ]
    hit = first_match("bitsandbytes-foundation/bitsandbytes", tag, candidates)
    assert (
        hit is not None
    ), f"{tag}: bitsandbytes/functional[.py|/__init__.py] both missing"
    _, src = hit
    needed = ("dequantize_4bit", "quantize_4bit")
    missing = [n for n in needed if not has_def(src, n, "func") and n not in src]
    assert not missing, (
        f"{tag}: bnb.functional missing {missing}; "
        f"unsloth-zoo dequant kernels rely on these"
    )


# -------------------------------------------------------------------------
# bnb.nn.Linear4bit / Params4bit: the two classes peft and unsloth
# isinstance-check against. Renaming either silently breaks 4-bit LoRA.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_nn_linear4bit_classes(tag: str):
    candidates = [
        "bitsandbytes/nn/modules.py",
        "bitsandbytes/nn/__init__.py",
    ]
    found_linear = False
    found_params = False
    for p in candidates:
        src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, p)
        if src is None:
            continue
        if has_def(src, "Linear4bit", "class") or "Linear4bit" in src:
            found_linear = True
        if has_def(src, "Params4bit", "class") or "Params4bit" in src:
            found_params = True
        if found_linear and found_params:
            return
    pytest.fail(
        f"{tag}: Linear4bit={found_linear} Params4bit={found_params} "
        f"in {candidates}; unsloth + peft 4-bit isinstance checks fail"
    )
