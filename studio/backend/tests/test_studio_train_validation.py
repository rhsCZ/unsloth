# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Training hyperparameter cap tests.

PR #5375 (security hardening) added field validators on
``TrainingStartRequest``; the loosening follow-up here raises four caps:

* ``max_seq_length``: 131_072 -> 2_000_000
* ``batch_size``:    1_024   -> 4_096
* ``lora_r``:        512     -> 16_384
* ``lora_alpha``:    1_024   -> 32_768

These tests pin the at-cap-accepts / over-cap-rejects boundaries so a
future tighten can't slip through silently and the bot review-comment
fixes don't get accidentally reverted.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from models.training import (
    _MAX_BATCH_SIZE,
    _MAX_LORA_ALPHA,
    _MAX_LORA_R,
    _MAX_SEQ_LENGTH,
)


def _check_field(field_name: str, value):
    """Call the validator for ``field_name`` directly through the
    cached schema. Faster than instantiating the whole
    TrainingStartRequest, and isolates the boundary we're testing."""
    from models.training import TrainingStartRequest

    schema_field = TrainingStartRequest.model_fields[field_name]
    # Walk the post-validators on the field; the @field_validator
    # decorator registers them on the model's validator chain.
    return TrainingStartRequest.__pydantic_validator__.validate_assignment(
        TrainingStartRequest.model_construct(),
        field_name,
        value,
    )


class TestSeqLengthCap:
    def test_at_cap_accepts(self):
        _check_field("max_seq_length", _MAX_SEQ_LENGTH)
        assert _MAX_SEQ_LENGTH == 2_000_000

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError) as exc:
            _check_field("max_seq_length", _MAX_SEQ_LENGTH + 1)
        assert "max_seq_length" in str(exc.value)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("max_seq_length", 0)


class TestBatchSizeCap:
    def test_at_cap_accepts(self):
        _check_field("batch_size", _MAX_BATCH_SIZE)
        assert _MAX_BATCH_SIZE == 4096

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("batch_size", _MAX_BATCH_SIZE + 1)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("batch_size", 0)


class TestLoraRCap:
    def test_at_cap_accepts(self):
        _check_field("lora_r", _MAX_LORA_R)
        assert _MAX_LORA_R == 16_384

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_r", _MAX_LORA_R + 1)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_r", 0)


class TestLoraAlphaCap:
    def test_at_cap_accepts(self):
        _check_field("lora_alpha", _MAX_LORA_ALPHA)
        assert _MAX_LORA_ALPHA == 32_768

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_alpha", _MAX_LORA_ALPHA + 1)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_alpha", 0)
