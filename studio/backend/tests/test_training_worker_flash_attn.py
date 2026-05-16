# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import builtins
import subprocess
import sys
from unittest import mock

from core.training import worker


def _missing_flash_attn_import():
    real_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "flash_attn":
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def test_should_try_runtime_flash_attn_install_threshold_and_skip(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    assert worker._should_try_runtime_flash_attn_install(32767) is False
    assert worker._should_try_runtime_flash_attn_install(
        32768
    ) is sys.platform.startswith("linux")

    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    assert worker._should_try_runtime_flash_attn_install(32768) is False


def test_runtime_flash_attn_prefers_prebuilt_wheel(monkeypatch):
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
    monkeypatch.setattr(
        worker,
        "flash_attn_wheel_url",
        lambda env: "https://example.com/fa.whl",
    )
    monkeypatch.setattr(worker, "url_exists", lambda url: True)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(
        worker,
        "install_wheel",
        lambda *args, **kwargs: [("pip", subprocess.CompletedProcess(["pip"], 0, ""))],
    )

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    assert statuses == ["Installing prebuilt flash-attn wheel..."]


def test_runtime_flash_attn_falls_back_to_pypi(monkeypatch):
    calls: list[list[str]] = []
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
    monkeypatch.setattr(
        worker,
        "probe_torch_wheel_env",
        lambda timeout = 30: {
            "python_tag": "cp313",
            "torch_mm": "2.10",
            "cuda_major": "13",
            "cxx11abi": "TRUE",
            "platform_tag": "linux_x86_64",
        },
    )
    monkeypatch.setattr(
        worker,
        "flash_attn_wheel_url",
        lambda env: "https://example.com/fa.whl",
    )
    monkeypatch.setattr(worker, "url_exists", lambda url: False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(worker, "install_wheel", mock.Mock())

    def fake_run(cmd, stdout = None, stderr = None, text = None):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    assert statuses == ["Installing flash-attn from PyPI for long-context training..."]
    assert calls == [[sys.executable, "-m", "pip", "install", "flash-attn"]]


def test_runtime_flash_attn_skip_env_avoids_all_install_work(monkeypatch):
    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    monkeypatch.setattr(worker._sp, "run", mock.Mock())

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    worker._sp.run.assert_not_called()


def test_runtime_flash_attn_skips_on_blackwell(monkeypatch):
    statuses: list[str] = []
    install_mock = mock.Mock()

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(
        worker, "_should_try_runtime_flash_attn_install", lambda max_seq: True
    )
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 65536)

    install_mock.assert_not_called()
    assert len(statuses) == 1
    assert "Blackwell" in statuses[0]


def test_causal_conv1d_fast_path_preserves_wheel_first_install_args(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    install_mock.assert_called_once_with(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = worker._CAUSAL_CONV1D_PACKAGE_VERSION,
        filename_prefix = "causal_conv1d",
        release_tag = worker._CAUSAL_CONV1D_RELEASE_TAG,
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
    )


def test_causal_conv1d_fast_path_includes_qwen3_6_variants(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "unsloth/Qwen3.6-4B",
    )
    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "unsloth/Qwen3_6-4B",
    )

    assert install_mock.call_count == 2


def test_mamba_ssm_path_preserves_wheel_first_install_args(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_mamba_ssm(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    install_mock.assert_called_once_with(
        event_queue = [],
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        pypi_version = worker._MAMBA_SSM_PACKAGE_VERSION,
        filename_prefix = "mamba_ssm",
        release_tag = worker._MAMBA_SSM_RELEASE_TAG,
        release_base_url = "https://github.com/state-spaces/mamba/releases/download",
    )


def _force_missing_fla_imports(monkeypatch):
    """Make fla.modules / fla.ops.gated_delta_rule imports raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("fla.modules") or name.startswith("fla.ops"):
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_flash_linear_attention_installs_pinned_pair_for_qwen3_5(monkeypatch):
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_fla_imports(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()
    args = run_mock.call_args[0][0]
    assert f"flash-linear-attention=={worker._FLA_PACKAGE_VERSION}" in args
    assert f"fla-core=={worker._FLA_CORE_PACKAGE_VERSION}" in args
    assert "--no-deps" in args
    assert run_mock.call_args.kwargs["timeout"] == worker._TILELANG_INSTALL_TIMEOUT_S
    assert any("flash-linear-attention" in s for s in statuses)


def test_flash_linear_attention_skips_for_unrelated_models(monkeypatch):
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "meta-llama/Llama-3.2-1B-Instruct",
    )

    run_mock.assert_not_called()


def test_flash_linear_attention_skips_for_ssm_only_models(monkeypatch):
    # Nemotron-H / Falcon-H1 / Granite-H / LFM2 take the mamba_ssm path
    # and never call FLA's gated_delta_rule kernels.
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    for name in (
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "nvidia/Nemotron-H-8B-Base",
        "ibm-granite/granite-4.0-h-tiny",
        "LiquidAI/LFM2-1.2B-Instruct",
    ):
        worker._ensure_flash_linear_attention(event_queue = [], model_name = name)

    run_mock.assert_not_called()


def test_flash_linear_attention_matches_full_qwen3_family(monkeypatch):
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_fla_imports(monkeypatch)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    for name in (
        "unsloth/Qwen3.5-2B",
        "unsloth/Qwen3_5-MoE-A22B",
        "unsloth/Qwen3.6-4B",
        "unsloth/Qwen3_6-4B",
        "unsloth/Qwen3-Next-80B-A3B",
        "unsloth/Qwen3_Next-80B-A3B",
    ):
        worker._ensure_flash_linear_attention(event_queue = [], model_name = name)

    assert run_mock.call_count == 6


def test_flash_linear_attention_skipped_below_python_3_10(monkeypatch):
    # sys.version_info is a structseq, not constructible; substitute a
    # plain tuple so the `< _FLA_MIN_PYTHON` comparison still works.
    monkeypatch.setattr(worker.sys, "version_info", (3, 9, 0, "final", 0))
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_flash_linear_attention_skipped_via_env(monkeypatch):
    monkeypatch.setenv(worker._FLA_SKIP_ENV, "1")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_flash_linear_attention_skipped_below_torch_2_7(monkeypatch):
    monkeypatch.delenv(worker._FLA_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 5))
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()
    assert any("torch>=" in s for s in statuses)


def test_flash_linear_attention_install_includes_einops(monkeypatch):
    monkeypatch.delenv(worker._FLA_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 9))
    monkeypatch.setattr(worker, "_flash_linear_attention_importable", lambda: False)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    args = run_mock.call_args[0][0]
    assert "--no-deps" in args
    # einops is declared by fla-core; packaging and triton are pulled in
    # because fla/utils.py imports them at module load but neither is
    # declared in fla-core's METADATA (an upstream FLA gap).
    assert "einops" in args
    assert "packaging" in args
    assert "triton" in args
    assert f"flash-linear-attention=={worker._FLA_PACKAGE_VERSION}" in args
    assert f"fla-core=={worker._FLA_CORE_PACKAGE_VERSION}" in args


def test_flash_linear_attention_logs_post_install_import_failure(monkeypatch):
    """pip exits 0 but `import fla.modules` still fails (missing transitive)."""
    monkeypatch.delenv(worker._FLA_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 9))
    import_calls = {"count": 0}

    def fake_importable():
        import_calls["count"] += 1
        # First call (pre-install probe) -> False so we attempt install.
        # Second call (post-install verify) -> still False.
        return False

    monkeypatch.setattr(worker, "_flash_linear_attention_importable", fake_importable)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    assert import_calls["count"] == 2
    assert any("not importable" in s for s in statuses)


def test_tilelang_backend_skipped_on_unsupported_linux_arch(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "ppc64le")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_pins_only_binary(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    monkeypatch.setattr(worker, "_tilelang_importable", lambda: False)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)
    # Need to bypass the post-install probe too.
    probe_calls = {"count": 0}

    def fake_probe():
        probe_calls["count"] += 1
        # First probe (pre-install): False so install runs.
        # Second probe (post-install): True so success branch taken.
        return probe_calls["count"] > 1

    monkeypatch.setattr(worker, "_tilelang_importable", fake_probe)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    args = run_mock.call_args[0][0]
    assert "--only-binary=:all:" in args
    assert "--no-deps" not in args


def _force_missing_tilelang_imports(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("tilelang", "tvm_ffi"):
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_tilelang_backend_installs_pinned_pair_for_qwen3_5(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_tilelang_imports(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()
    args = run_mock.call_args[0][0]
    assert f"apache-tvm-ffi=={worker._APACHE_TVM_FFI_PACKAGE_VERSION}" in args
    assert f"tilelang=={worker._TILELANG_PACKAGE_VERSION}" in args
    assert run_mock.call_args.kwargs["timeout"] == worker._TILELANG_INSTALL_TIMEOUT_S
    assert any("TileLang backend" in s for s in statuses)


def test_tilelang_backend_reinstalls_when_tvm_ffi_is_broken(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: "0.1.11")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()
    args = run_mock.call_args[0][0]
    assert "--force-reinstall" in args
    # Reinstall must NOT strip deps; tilelang needs z3-solver/ml-dtypes
    # and friends at runtime.
    assert "--no-deps" not in args
    assert "--only-binary=:all:" in args


def test_tilelang_backend_skipped_below_python_3_10(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    # sys.version_info is a structseq, not constructible; substitute a
    # plain tuple so the `< _FLA_MIN_PYTHON` comparison still works.
    monkeypatch.setattr(worker.sys, "version_info", (3, 9, 0, "final", 0))
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_skipped_on_windows(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.sys, "platform", "win32")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_swallows_install_timeout(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    _force_missing_tilelang_imports(monkeypatch)

    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd = "pip", timeout = 1)

    monkeypatch.setattr(worker._sp, "run", raise_timeout)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    # Should not raise.
    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    assert any("timed out" in s.lower() for s in statuses)


def test_tilelang_backend_skipped_for_ssm_models(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    # Nemotron-H / Falcon-H1 / Granite-H take the mamba_ssm path, not FLA's
    # gated_delta_rule -> tilelang has no effect on them.
    for name in (
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "nvidia/Nemotron-H-8B-Base",
        "ibm-granite/granite-4.0-h-tiny",
        "meta-llama/Llama-3.2-1B-Instruct",
    ):
        worker._ensure_tilelang_backend(event_queue = [], model_name = name)

    run_mock.assert_not_called()


def test_tilelang_backend_skipped_via_env(monkeypatch):
    monkeypatch.setenv(worker._TILELANG_SKIP_ENV, "1")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_swallows_install_failure(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 1, stdout = "boom"))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_tilelang_imports(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    # Should not raise even when pip exits non-zero.
    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()
    assert any("failed" in s.lower() for s in statuses)


# ───────────────────────────────────────────────────────────────────
# Runtime hook on `is_flash_linear_attention_available` /
# `is_causal_conv1d_available`. These are the primary gate in
# normal operation; the substring tests above cover the
# UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 fallback.
# ───────────────────────────────────────────────────────────────────


class _FakeQueue(list):
    """List with `.put` so worker._send_status can send into it during tests."""

    def put(self, item):
        self.append(item)


def _make_fake_gate(initial_return: bool):
    """Build a callable that mimics transformers' lru_cache-decorated gates.

    Tracks call count and exposes a `cache_clear` attribute. The return
    value can be flipped to mimic install-then-True behaviour by setting
    `.next_return`.
    """

    class Gate:
        def __init__(self, initial: bool) -> None:
            self.next_return = initial
            self.call_count = 0
            self.cache_clear_count = 0

        def __call__(self) -> bool:
            self.call_count += 1
            return self.next_return

        def cache_clear(self) -> None:
            self.cache_clear_count += 1

    return Gate(initial_return)


def _patch_iu_gates(monkeypatch, fla_gate, conv_gate):
    """Drop fake gates onto transformers.utils.import_utils for the test."""
    from transformers.utils import import_utils as _iu

    monkeypatch.setattr(_iu, "is_flash_linear_attention_available", fla_gate)
    monkeypatch.setattr(_iu, "is_causal_conv1d_available", conv_gate)


def test_hook_installs_when_gate_returns_false(monkeypatch):
    fla_gate = _make_fake_gate(initial_return=False)
    conv_gate = _make_fake_gate(initial_return=False)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    fla_install = mock.Mock(side_effect=lambda eq: setattr(fla_gate, "next_return", True))
    tile_install = mock.Mock(side_effect=lambda eq: None)
    conv_install = mock.Mock(side_effect=lambda **kw: setattr(conv_gate, "next_return", True))

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", tile_install
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising=False)

    worker._install_fast_path_hooks(event_queue=_FakeQueue())

    from transformers.utils import import_utils as _iu

    # Both gates are now wrapped. Call them — the hook should drive the install.
    assert _iu.is_flash_linear_attention_available() is True
    fla_install.assert_called_once()
    tile_install.assert_called_once()
    assert _iu.is_causal_conv1d_available() is True
    conv_install.assert_called_once()


def test_hook_skips_install_when_gate_already_true(monkeypatch):
    fla_gate = _make_fake_gate(initial_return=True)
    conv_gate = _make_fake_gate(initial_return=True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    fla_install = mock.Mock()
    tile_install = mock.Mock()
    conv_install = mock.Mock()
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", tile_install
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising=False)

    worker._install_fast_path_hooks(event_queue=_FakeQueue())

    from transformers.utils import import_utils as _iu

    assert _iu.is_flash_linear_attention_available() is True
    assert _iu.is_causal_conv1d_available() is True
    fla_install.assert_not_called()
    tile_install.assert_not_called()
    conv_install.assert_not_called()


def test_hook_idempotent_on_repeat_call(monkeypatch):
    fla_gate = _make_fake_gate(initial_return=False)
    conv_gate = _make_fake_gate(initial_return=False)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    fla_install = mock.Mock(side_effect=lambda eq: setattr(fla_gate, "next_return", True))
    tile_install = mock.Mock()
    conv_install = mock.Mock(side_effect=lambda **kw: setattr(conv_gate, "next_return", True))
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", tile_install
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising=False)

    worker._install_fast_path_hooks(event_queue=_FakeQueue())

    from transformers.utils import import_utils as _iu

    # First call: hook fires.
    _iu.is_flash_linear_attention_available()
    # Subsequent calls: must not re-trigger the installer.
    _iu.is_flash_linear_attention_available()
    _iu.is_flash_linear_attention_available()
    assert fla_install.call_count == 1
    assert tile_install.call_count == 1


def test_hook_handles_install_failure_gracefully(monkeypatch):
    fla_gate = _make_fake_gate(initial_return=False)
    conv_gate = _make_fake_gate(initial_return=True)  # bypass to focus on FLA
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    def raising_install(eq):
        raise RuntimeError("pip failed to fetch wheel")

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", raising_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: None
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: None)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising=False)

    worker._install_fast_path_hooks(event_queue=_FakeQueue())

    from transformers.utils import import_utils as _iu

    # Must not raise; returns False so transformers falls back to torch loop.
    assert _iu.is_flash_linear_attention_available() is False


def test_hook_can_be_disabled_via_env(monkeypatch):
    fla_gate = _make_fake_gate(initial_return=False)
    conv_gate = _make_fake_gate(initial_return=False)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    fla_install = mock.Mock()
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setenv(worker._FAST_PATH_HOOKS_SKIP_ENV, "1")

    worker._install_fast_path_hooks(event_queue=_FakeQueue())

    from transformers.utils import import_utils as _iu

    # Hook should NOT have been installed; gates remain the fakes.
    assert _iu.is_flash_linear_attention_available is fla_gate
    assert _iu.is_causal_conv1d_available is conv_gate
    fla_install.assert_not_called()


def test_hook_clears_lru_cache_before_first_check(monkeypatch):
    fla_gate = _make_fake_gate(initial_return=True)
    conv_gate = _make_fake_gate(initial_return=True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", lambda eq: None
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: None
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: None)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising=False)

    worker._install_fast_path_hooks(event_queue=_FakeQueue())
    from transformers.utils import import_utils as _iu

    _iu.is_flash_linear_attention_available()
    # The wrapper called cache_clear at least once before delegating.
    assert fla_gate.cache_clear_count >= 1


def test_hook_rewrites_previously_imported_module_bindings(monkeypatch):
    """Modeling files bind `is_flash_linear_attention_available` locally
    via `from ... import is_X`. Reassigning the attribute on
    transformers.utils.import_utils alone does NOT reach those local
    bindings. The hook installer sweeps sys.modules and rebinds them.
    """
    fla_gate = _make_fake_gate(initial_return=False)
    conv_gate = _make_fake_gate(initial_return=True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    # Create a fake modeling module that did `from ... import is_flash_linear_attention_available`.
    fake_mod = sys.modules.setdefault(
        "_test_fake_modeling_qwen35", type(sys)("_test_fake_modeling_qwen35")
    )
    fake_mod.is_flash_linear_attention_available = fla_gate

    def fake_install(eq):
        fla_gate.next_return = True

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fake_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: None
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: None)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising=False)

    worker._install_fast_path_hooks(event_queue=_FakeQueue())

    # The fake module's local binding has been rewritten to the wrapper.
    assert fake_mod.is_flash_linear_attention_available is not fla_gate
    # Calling through the fake module's reference triggers the install.
    assert fake_mod.is_flash_linear_attention_available() is True

    del sys.modules["_test_fake_modeling_qwen35"]


def test_hook_skips_when_import_utils_unavailable(monkeypatch):
    """If transformers.utils.import_utils can't be imported, the hook
    installer must log and return cleanly rather than crash the worker."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "transformers.utils" or name == "transformers.utils.import_utils":
            raise ImportError("transformers missing in worker venv")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising=False)

    # Should not raise.
    worker._install_fast_path_hooks(event_queue=_FakeQueue())


def test_substring_fallback_unchanged_when_hook_skipped(monkeypatch):
    """With the hook disabled, the orchestration falls back to the
    substring path. Confirm _ensure_flash_linear_attention(model_name)
    still gates on model name as before."""
    install_mock = mock.Mock()
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", install_mock
    )
    monkeypatch.setenv(worker._FAST_PATH_HOOKS_SKIP_ENV, "1")

    # Qwen3.5 model triggers install.
    worker._ensure_flash_linear_attention(event_queue=[], model_name="unsloth/Qwen3.5-2B")
    assert install_mock.call_count == 1

    # Llama doesn't.
    worker._ensure_flash_linear_attention(event_queue=[], model_name="meta-llama/Llama-3.1-8B")
    assert install_mock.call_count == 1
