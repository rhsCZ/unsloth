# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native GGUF companion path validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.inference import _validate_native_gguf_companion


def _write_pair(tmp_path: Path, folder: str | None = None) -> tuple[Path, Path]:
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"model")
    parent = tmp_path if folder is None else tmp_path / folder
    parent.mkdir(parents = True, exist_ok = True)
    companion = parent / "mtp-model.gguf"
    companion.write_bytes(b"draft")
    return weight, companion


def test_native_companion_allows_model_directory(tmp_path):
    weight, companion = _write_pair(tmp_path)
    _validate_native_gguf_companion(str(companion), str(weight), "vision companion")


@pytest.mark.parametrize("folder", ["MTP", "mtp", "MtP"])
def test_native_mtp_companion_allows_mtp_directory(tmp_path, folder):
    weight, companion = _write_pair(tmp_path, folder)
    _validate_native_gguf_companion(
        str(companion), str(weight), "MTP drafter", allow_mtp_subdir = True
    )


def test_native_vision_companion_rejects_mtp_directory(tmp_path):
    weight, companion = _write_pair(tmp_path, "MTP")
    with pytest.raises(HTTPException, match = "must live next to"):
        _validate_native_gguf_companion(str(companion), str(weight), "vision companion")


@pytest.mark.parametrize("folder", ["other", "MTP/deeper", "mtp/deeper"])
def test_native_companion_rejects_arbitrary_nesting(tmp_path, folder):
    weight, companion = _write_pair(tmp_path, folder)
    with pytest.raises(HTTPException, match = "must live beside") as error:
        _validate_native_gguf_companion(
            str(companion), str(weight), "MTP drafter", allow_mtp_subdir = True
        )
    assert error.value.status_code == 400


def test_native_companion_rejects_file_symlink(tmp_path):
    weight, companion = _write_pair(tmp_path)
    link = tmp_path / "mtp-link.gguf"
    try:
        link.symlink_to(companion)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(HTTPException, match = "regular file"):
        _validate_native_gguf_companion(str(link), str(weight), "MTP drafter")


def test_native_companion_rejects_directory_symlink_escape(tmp_path):
    model_dir = tmp_path / "model"
    outside = tmp_path / "outside"
    model_dir.mkdir()
    outside.mkdir()
    weight = model_dir / "model.gguf"
    weight.write_bytes(b"model")
    companion = outside / "mtp-model.gguf"
    companion.write_bytes(b"draft")
    try:
        (model_dir / "MTP").symlink_to(outside, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(HTTPException, match = "must live beside"):
        _validate_native_gguf_companion(
            str(model_dir / "MTP" / companion.name),
            str(weight),
            "MTP drafter",
            allow_mtp_subdir = True,
        )


def test_native_companion_rejects_missing_file(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"model")
    with pytest.raises(HTTPException, match = "no longer accessible"):
        _validate_native_gguf_companion(str(tmp_path / "missing.gguf"), str(weight), "MTP drafter")


def test_native_companion_rejects_directory(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"model")
    companion = tmp_path / "mtp-model.gguf"
    companion.mkdir()
    with pytest.raises(HTTPException, match = "regular file"):
        _validate_native_gguf_companion(str(companion), str(weight), "MTP drafter")


def test_native_companion_rejects_missing_weight(tmp_path):
    companion = tmp_path / "mtp-model.gguf"
    companion.write_bytes(b"draft")
    with pytest.raises(HTTPException, match = "no longer accessible"):
        _validate_native_gguf_companion(
            str(companion), str(tmp_path / "missing.gguf"), "MTP drafter"
        )


def test_native_companion_none_is_noop():
    _validate_native_gguf_companion(None, None, "MTP drafter")
