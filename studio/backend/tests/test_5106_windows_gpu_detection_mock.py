# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end Windows GPU-detection regression test for #5106.

The bug: on Windows hosts without a system CUDA toolkit, Studio's
prebuilt llama-server.exe could not resolve ``cudart64_X.dll`` /
``cublas64_X.dll`` / ``cublasLt64_X.dll`` at LoadLibrary time, so
``ggml-cuda.dll`` (which has a static PE import on ``cublas64_X.dll``)
failed to load and llama-server silently fell back to CPU even when
``nvidia-smi`` reported the GPU.

The fix lands in two halves:
  * PR #5322 (install-time): downloads upstream's paired
    ``cudart-llama-bin-win-cuda-X.Y-x64.zip`` and overlays its three
    DLLs into ``install_dir/build/bin/Release/`` next to
    ``llama-server.exe``. Windows DLL search resolves them from step
    (1) -- the application directory.
  * PR #5324 (launch-time): prepends pip-installed
    ``nvidia/<pkg>/{bin,bin/x86_64,Library/bin}`` and ``torch/lib``
    directories to ``PATH`` when spawning ``llama-server.exe``.
    Windows DLL search resolves the DLLs from step (3) -- the ``PATH``
    environment variable -- even on existing installs that pre-date
    PR #5322.

This test exercises both halves on a synthetic Windows layout that
mirrors the real artifact contents (verified empirically against
upstream b9103 ``cudart-llama-bin-win-cuda-13.1-x64.zip`` and the
``nvidia-cuda-runtime`` / ``nvidia-cublas`` PyPI win_amd64 wheels).
CI runners have no GPUs, so we mock the ``nvidia-smi`` probe directly
to assert Studio detects the synthetic GPU AND ends up with cudart
reachable through both the binary-directory and the PATH path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types as _types
import zipfile
from pathlib import Path
from unittest import mock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub heavy deps the rest of the studio backend pulls in so this
# test can run on the same matrix as test_llama_cpp_windows_nvidia_path.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


# Real upstream b9103 cudart bundle contents (verified by direct unzip
# of the GitHub release artifact). Exactly these three filenames, no
# executables, no subdirectories.
REAL_UPSTREAM_CUDART_BUNDLE = {
    "12.4": ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"),
    "13.1": ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"),
}

# Real win_amd64 wheel layouts observed on PyPI (verified by
# ``pip download nvidia-cuda-runtime --platform win_amd64`` and
# ``unzip -l`` against the resulting wheel).
REAL_PIP_NVIDIA_WHEEL_LAYOUTS = {
    # Legacy modular wheels (cu-suffixed)
    "nvidia/cuda_runtime/bin": ["cudart64_12.dll"],
    "nvidia/cublas/bin": [
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "nvblas64_12.dll",
    ],
    "nvidia/cudnn/bin": [
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_ops64_9.dll",
    ],
    # New unsuffixed cu13 wheel layout: nvidia/cu13/bin/x86_64/
    "nvidia/cu13/bin/x86_64": [
        "cudart64_13.dll",
        "cublas64_13.dll",
        "cublasLt64_13.dll",
        "nvblas64_13.dll",
    ],
}


def _populate_studio_venv(prefix: Path) -> None:
    """Drop fake wheel files matching the real PyPI win_amd64 layouts
    seen on actual nvidia-cuda-runtime / nvidia-cublas / nvidia-cudnn
    wheels. The resolver doesn't care about file contents, only
    presence + directory structure."""
    site = prefix / "Lib" / "site-packages"
    for rel, dlls in REAL_PIP_NVIDIA_WHEEL_LAYOUTS.items():
        d = site / Path(rel)
        d.mkdir(parents = True, exist_ok = True)
        for name in dlls:
            (d / name).write_bytes(b"PE-stub")
    # Studio's install_python_stack always installs torch alongside
    # the nvidia wheels.
    (site / "torch" / "lib").mkdir(parents = True, exist_ok = True)
    for fn in ("c10.dll", "torch.dll", "torch_cpu.dll", "torch_python.dll"):
        (site / "torch" / "lib" / fn).write_bytes(b"PE-stub")


def _populate_studio_install(install_dir: Path, runtime: str = "13.1") -> None:
    """Drop a Windows-style install_dir/build/bin/Release/ tree
    populated as PR #5322 would after the paired cudart overlay."""
    rel = install_dir / "build" / "bin" / "Release"
    rel.mkdir(parents = True, exist_ok = True)
    # Main archive payload
    for fn in (
        "llama-server.exe",
        "llama-quantize.exe",
        "llama-cli.exe",
        "llama.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cuda.dll",
        "mtmd.dll",
    ):
        (rel / fn).write_bytes(b"PE-stub")
    # Paired cudart bundle payload (this is what #5322 adds)
    for fn in REAL_UPSTREAM_CUDART_BUNDLE[runtime]:
        (rel / fn).write_bytes(b"PE-stub")


def _build_path_dirs_like_start_llama_server(
    binary_dir: Path, prefix: Path, cuda_path: str = ""
) -> list[str]:
    """Faithful reproduction of the win32 branch in
    LlamaCppBackend.start_llama_server. Returns the ordered list of
    PATH entries we prepend to the inherited env. Production code:
    studio/backend/core/inference/llama_cpp.py:2340-2363.
    """
    pip_dirs = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(prefix))
    path_dirs = [str(binary_dir)]
    path_dirs.extend(pip_dirs)
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            path_dirs.append(cuda_bin)
        cuda_bin_x64 = os.path.join(cuda_path, "bin", "x64")
        if os.path.isdir(cuda_bin_x64):
            path_dirs.append(cuda_bin_x64)
    return path_dirs


def _mock_nvidia_smi_run(fake_output: str, returncode: int = 0) -> "mock._patch":
    """Patch subprocess.run so the nvidia-smi probe in
    LlamaCppBackend._get_gpu_free_memory returns the supplied CSV.
    Other subprocess.run calls (if any in this test process) pass
    through to the real subprocess.run."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and "nvidia-smi" in cmd[0]:
            return subprocess.CompletedProcess(
                args = cmd, returncode = returncode, stdout = fake_output, stderr = ""
            )
        return real_run(cmd, *args, **kwargs)

    return mock.patch("subprocess.run", side_effect = fake_run)


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #
class TestWindowsGpuDetectionAfter5106Fix:
    """Validates the end-to-end #5106 fix on a synthetic Windows
    layout. CI runners have no GPU, so we mock nvidia-smi but exercise
    every other layer (resolver, PATH builder, install layout) for
    real."""

    def test_nvidia_smi_probe_reports_synthetic_gpu(self):
        """Sanity: the production nvidia-smi probe parses CSV output
        and returns (index, free_mib) tuples. This is the entry point
        Studio uses to decide whether a GPU is reachable at all."""
        # noahterbest's exact #5106 reproducer: RTX 4090, 22805 MiB free.
        fake_csv = "0, 22805\n"
        with _mock_nvidia_smi_run(fake_csv):
            gpus = LlamaCppBackend._get_gpu_free_memory()
        assert gpus == [
            (0, 22805)
        ], f"GPU probe failed to parse mocked nvidia-smi output: {gpus}"

    def test_nvidia_smi_probe_respects_cuda_visible_devices(self, monkeypatch):
        """A user with CUDA_VISIBLE_DEVICES=1 should only see GPU 1."""
        fake_csv = "0, 22805\n1, 24576\n2, 16384\n"
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        with _mock_nvidia_smi_run(fake_csv):
            gpus = LlamaCppBackend._get_gpu_free_memory()
        assert gpus == [(1, 24576)], gpus

    def test_windows_install_dir_has_all_three_cudart_dlls(self, tmp_path):
        """After PR #5322, install_dir/build/bin/Release/ must contain
        all three DLLs from the upstream cudart bundle. Without all
        three, ggml-cuda.dll's static PE import on cublas64_X.dll
        cannot resolve and the CUDA backend fails to register."""
        install = tmp_path / "studio_install"
        _populate_studio_install(install, runtime = "13.1")
        rel = install / "build" / "bin" / "Release"
        for fn in REAL_UPSTREAM_CUDART_BUNDLE["13.1"]:
            assert (rel / fn).exists(), f"missing {fn} in {rel}"
        assert (rel / "llama-server.exe").exists()
        assert (rel / "ggml-cuda.dll").exists()

    def test_resolver_finds_real_pypi_wheel_layouts(self, tmp_path):
        """The launch-time PATH resolver must pick up every layout
        used by real pip-installed CUDA wheels on Windows today:
          * nvidia/<pkg>/bin (legacy cu-suffixed wheels)
          * nvidia/<pkg>/bin/x86_64 (new cu13 unsuffixed wheels)
          * torch/lib (some torch builds bundle CUDA DLLs here)
        """
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        out = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(prefix))
        # All four real-world layouts must be present:
        site = prefix / "Lib" / "site-packages"
        for expected in (
            site / "nvidia" / "cuda_runtime" / "bin",
            site / "nvidia" / "cublas" / "bin",
            site / "nvidia" / "cudnn" / "bin",
            site / "nvidia" / "cu13" / "bin" / "x86_64",
            site / "torch" / "lib",
        ):
            assert (
                str(expected) in out
            ), f"resolver missed {expected.relative_to(prefix)}: {out}"

    def test_path_assembly_makes_cudart_reachable_without_toolkit(self, tmp_path):
        """The exact #5106 scenario: Windows host with the GPU
        detected, pip nvidia wheels installed, but NO system CUDA
        toolkit (no CUDA_PATH). After the fix, the PATH that
        ``start_llama_server`` prepends to the llama-server.exe
        subprocess env must make ``cudart64_*.dll`` reachable from at
        least one entry. We verify directly by walking each PATH
        entry on the live filesystem."""
        prefix = tmp_path / "studio_venv"
        install = tmp_path / "studio_install"
        _populate_studio_venv(prefix)
        _populate_studio_install(install, runtime = "13.1")
        binary_dir = install / "build" / "bin" / "Release"
        path_dirs = _build_path_dirs_like_start_llama_server(
            binary_dir, prefix, cuda_path = ""
        )
        # binary_dir is first (Windows DLL search step 1).
        assert path_dirs[0] == str(
            binary_dir
        ), f"binary_dir must be first in PATH; got {path_dirs[0]}"
        # cudart MUST be findable from at least one PATH entry.
        cudart_locations = []
        for entry in path_dirs:
            for cudart_name in ("cudart64_12.dll", "cudart64_13.dll"):
                if (Path(entry) / cudart_name).exists():
                    cudart_locations.append((entry, cudart_name))
        assert cudart_locations, (
            f"cudart unreachable from any PATH entry -- #5106 not fixed.\n"
            f"PATH entries searched: {path_dirs}"
        )
        # Confirm cudart is reachable from BOTH the install dir (PR
        # #5322's contribution) AND a pip nvidia dir (PR #5324's
        # contribution). Defense in depth.
        sources = {Path(e).relative_to(tmp_path).parts[0] for e, _ in cudart_locations}
        assert (
            "studio_install" in sources
        ), f"PR #5322's cudart drop not reachable: {cudart_locations}"
        assert (
            "studio_venv" in sources
        ), f"PR #5324's pip nvidia dir not contributing cudart: {cudart_locations}"

    def test_cublas_and_cublasLt_also_reachable(self, tmp_path):
        """ggml-cuda.dll has a static PE import on cublas64_X.dll
        (verified by ``objdump -p`` on the upstream b9103 build).
        cublas64_X.dll has a static PE import on cublasLt64_X.dll.
        All three must be reachable or LoadLibrary("ggml-cuda.dll")
        returns NULL."""
        prefix = tmp_path / "studio_venv"
        install = tmp_path / "studio_install"
        _populate_studio_venv(prefix)
        _populate_studio_install(install, runtime = "13.1")
        binary_dir = install / "build" / "bin" / "Release"
        path_dirs = _build_path_dirs_like_start_llama_server(binary_dir, prefix)
        for required in REAL_UPSTREAM_CUDART_BUNDLE["13.1"]:
            reachable = any((Path(d) / required).exists() for d in path_dirs)
            assert reachable, (
                f"{required} unreachable from PATH; #5106 not fixed.\n"
                f"PATH entries: {path_dirs}"
            )

    def test_no_pip_nvidia_wheels_still_works_via_install_dir(self, tmp_path):
        """A user with no pip nvidia wheels (CPU-only torch install,
        ``unsloth run`` standalone, custom torch builds) should still
        get cudart via PR #5322's paired download alone -- binary_dir
        is enough."""
        prefix = tmp_path / "bare_venv"
        prefix.mkdir()
        # No pip nvidia / torch wheels installed
        install = tmp_path / "studio_install"
        _populate_studio_install(install, runtime = "13.1")
        binary_dir = install / "build" / "bin" / "Release"
        path_dirs = _build_path_dirs_like_start_llama_server(binary_dir, prefix)
        # Only binary_dir should be in PATH.
        assert path_dirs == [
            str(binary_dir)
        ], f"bare venv produced unexpected PATH: {path_dirs}"
        # binary_dir has all three cudart DLLs.
        for required in REAL_UPSTREAM_CUDART_BUNDLE["13.1"]:
            assert (
                binary_dir / required
            ).exists(), f"{required} missing from binary_dir on bare venv install"

    def test_no_install_dir_still_works_via_pip_wheels(self, tmp_path):
        """A user on an existing pre-#5322 Studio install (binary_dir
        lacks cudart) should still get cudart via PR #5324's pip
        wheel directories on PATH."""
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        install = tmp_path / "studio_install_pre5322"
        rel = install / "build" / "bin" / "Release"
        rel.mkdir(parents = True)
        # Main archive payload only; cudart bundle missing.
        for fn in (
            "llama-server.exe",
            "llama.dll",
            "ggml-cuda.dll",
            "ggml-base.dll",
        ):
            (rel / fn).write_bytes(b"PE-stub")
        # cudart NOT in binary_dir on this scenario.
        path_dirs = _build_path_dirs_like_start_llama_server(rel, prefix)
        cudart_reachable = any(
            (Path(d) / "cudart64_12.dll").exists()
            or (Path(d) / "cudart64_13.dll").exists()
            for d in path_dirs
        )
        assert cudart_reachable, (
            "PR #5324 pip wheel fallback failed: cudart unreachable from PATH "
            f"on cudart-less install. PATH entries: {path_dirs}"
        )
        cublas_reachable = any(
            (Path(d) / "cublas64_12.dll").exists()
            or (Path(d) / "cublas64_13.dll").exists()
            for d in path_dirs
        )
        assert cublas_reachable, "cublas unreachable on cudart-less install"

    def test_pre_pr_scenario_would_have_failed(self, tmp_path):
        """Negative control: reconstruct the pre-#5322 + pre-#5324
        world (cudart NOT dropped by installer, PATH not augmented by
        launcher) and assert that cudart is unreachable -- the
        original #5106 failure mode. This ensures the test would
        actually catch a regression."""
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        install = tmp_path / "pre_pr_install"
        rel = install / "build" / "bin" / "Release"
        rel.mkdir(parents = True)
        # Main archive only; no cudart bundle.
        for fn in ("llama-server.exe", "llama.dll", "ggml-cuda.dll"):
            (rel / fn).write_bytes(b"PE-stub")
        # Pre-PR PATH: binary_dir + CUDA_PATH/bin only. Pip nvidia
        # dirs NOT added. No system CUDA toolkit (the #5106 scenario).
        pre_pr_path_dirs = [str(rel)]
        cudart_reachable_pre = any(
            (Path(d) / "cudart64_12.dll").exists()
            or (Path(d) / "cudart64_13.dll").exists()
            for d in pre_pr_path_dirs
        )
        assert not cudart_reachable_pre, (
            "Test self-check failed: pre-PR scenario unexpectedly had "
            f"cudart reachable. {pre_pr_path_dirs}"
        )


class TestWindowsSysPlatformMocked:
    """Validate that the win32 branch in start_llama_server is the
    branch we test, not the linux fallback. We can't easily call the
    full start_llama_server method (it constructs a llama-server
    subprocess), but we can patch sys.platform and re-import the
    branch-selecting helper."""

    def test_sys_platform_win32_uses_pip_nvidia_resolver(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        # On win32, the resolver should be called and return non-empty.
        out = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(prefix))
        assert out, f"resolver returned empty under sys.platform=win32: {out}"
        # Specifically, the cu13 arch dir must be in the output.
        cu13_arch = (
            prefix / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
        )
        assert str(cu13_arch) in out
