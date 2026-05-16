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
    assert "einops" in args
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
