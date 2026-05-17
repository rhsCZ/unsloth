# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the offline GGUF cache fallback path (#5505).

Three failure modes hit users when ``huggingface.co`` is unreachable
but the requested GGUF repo is fully cached locally:

* ``list_gguf_variants`` raised through ``HTTPException(500)`` so the
  variant dropdown sat empty.
* ``detect_gguf_model_remote`` returned ``None`` so a GGUF-only repo
  was misrouted into the transformers/Unsloth backend (on macOS this
  surfaced as a hardware error).
* ``_download_gguf`` fell back to a synthetic ``{repo}-{variant}.gguf``
  name that did not exist in cache when the in-repo filename did not
  echo the repo name (e.g. ``unsloth/Qwen3.6-27B-MTP-GGUF`` ships
  ``Qwen3.6-27B-UD-Q4_K_XL.gguf`` with no ``MTP`` token).

Two follow-up regressions covered here:

* P1 #1: the cache-side variant filter must match the snapshot-relative
  path, not just the basename, so subdir layouts like
  ``BF16/foo.gguf`` are findable.
* P1 #2: the DNS auto-detect must scope ``HF_HUB_OFFLINE`` to one load
  via try/finally so a transient resolver hiccup cannot lock the
  long-lived ``LlamaCppBackend`` singleton offline forever.

No GPU, no network, no subprocess. Linux, macOS, Windows compatible.
"""

from __future__ import annotations

import os
import socket
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub heavy/unavailable external deps before importing the modules
# under test (same pattern as other studio backend tests).
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# Prefer real httpx if installed (CI always installs it). Falling back to
# a stub when httpx is genuinely missing keeps the test runnable in
# trimmed-down dev environments. The stub set has to track every attr
# huggingface_hub touches at import time, so when in doubt, install httpx.
try:
    import httpx  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
        "HTTPStatusError",
    ):
        setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Request = type("Request", (), {})

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


from huggingface_hub import constants as hf_constants

from core.inference.llama_cpp import _hf_offline_if_dns_dead, _probe_dns_dead
from utils.models.model_config import (
    _detect_gguf_from_hf_cache,
    _iter_hf_cache_snapshots,
    _list_gguf_variants_from_hf_cache,
    detect_gguf_model_remote,
    list_gguf_variants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_cache(
    root: Path,
    repo_id: str,
    files: dict[str, int],
    *,
    snapshot_sha: str = "a" * 40,
) -> Path:
    """Create ``$root/models--<repo>/snapshots/<sha>/<rel>`` for each entry."""
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    snap = repo_dir / "snapshots" / snapshot_sha
    snap.mkdir(parents = True, exist_ok = True)
    for rel, size in files.items():
        full = snap / rel
        full.parent.mkdir(parents = True, exist_ok = True)
        full.write_bytes(b"\0" * size)
    return snap


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """Point ``huggingface_hub.constants.HF_HUB_CACHE`` at a temp dir."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    return tmp_path


@pytest.fixture
def clean_offline_env(monkeypatch):
    """Strip ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` for the test."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


def _siblings(items: dict[str, int]):
    """Mock ``hf_model_info(...).siblings`` payload."""
    return _types.SimpleNamespace(
        siblings = [
            _types.SimpleNamespace(rfilename = name, size = size)
            for name, size in items.items()
        ],
    )


# ---------------------------------------------------------------------------
# _iter_hf_cache_snapshots
# ---------------------------------------------------------------------------


class TestIterHfCacheSnapshots:
    def test_returns_empty_when_cache_dir_missing(self, monkeypatch):
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", "/no/such/dir")
        assert list(_iter_hf_cache_snapshots("unsloth/foo")) == []

    def test_returns_empty_when_repo_not_cached(self, hf_cache):
        assert list(_iter_hf_cache_snapshots("unsloth/not-here")) == []

    def test_returns_empty_when_snapshots_dir_missing(self, hf_cache):
        # Repo dir exists but no snapshots/ inside.
        (hf_cache / "models--unsloth--bare").mkdir()
        assert list(_iter_hf_cache_snapshots("unsloth/bare")) == []

    def test_yields_newest_first(self, hf_cache):
        old = _build_cache(
            hf_cache, "unsloth/multi", {"x.gguf": 1}, snapshot_sha = "a" * 40
        )
        new = _build_cache(
            hf_cache, "unsloth/multi", {"y.gguf": 1}, snapshot_sha = "b" * 40
        )
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        out = list(_iter_hf_cache_snapshots("unsloth/multi"))
        assert [p.name for p in out] == ["b" * 40, "a" * 40]

    def test_repo_id_match_is_case_insensitive(self, hf_cache):
        _build_cache(hf_cache, "unsloth/Foo-GGUF", {"Foo-Q4_K_M.gguf": 1})
        # Lookup with a different casing of the org/name still resolves
        out = list(_iter_hf_cache_snapshots("UNSLOTH/foo-gguf"))
        assert len(out) == 1


# ---------------------------------------------------------------------------
# _list_gguf_variants_from_hf_cache / list_gguf_variants
# ---------------------------------------------------------------------------


class TestListGgufVariantsFromCache:
    def test_returns_variants_when_cached(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/Qwen3.5-4B-GGUF",
            {
                "Qwen3.5-4B-UD-Q4_K_XL.gguf": 100,
                "Qwen3.5-4B-Q2_K.gguf": 50,
            },
        )
        out = _list_gguf_variants_from_hf_cache("unsloth/Qwen3.5-4B-GGUF")
        assert out is not None
        variants, has_vision = out
        assert sorted(v.quant for v in variants) == ["Q2_K", "UD-Q4_K_XL"]
        assert has_vision is False

    def test_returns_none_when_not_cached(self, hf_cache):
        assert _list_gguf_variants_from_hf_cache("unsloth/absent") is None


class TestListGgufVariantsOffline:
    def test_offline_env_short_circuits_api(
        self, hf_cache, clean_offline_env, monkeypatch
    ):
        _build_cache(hf_cache, "unsloth/a", {"a-UD-Q4_K_XL.gguf": 1})
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise AssertionError("API must not be called when offline env set")

        with patch("huggingface_hub.model_info", boom):
            variants, _has = list_gguf_variants("unsloth/a")
        assert len(variants) == 1
        assert variants[0].quant == "UD-Q4_K_XL"

    def test_api_exception_falls_back_to_cache(
        self,
        hf_cache,
        clean_offline_env,
    ):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})

        def boom(*a, **k):
            raise OSError("network down")

        with patch("huggingface_hub.model_info", boom):
            variants, _has = list_gguf_variants("unsloth/a")
        assert len(variants) == 1
        assert variants[0].quant == "Q4_K_M"

    def test_api_exception_with_no_cache_reraises(self, hf_cache, clean_offline_env):
        def boom(*a, **k):
            raise OSError("network down")

        with patch("huggingface_hub.model_info", boom):
            with pytest.raises(OSError, match = "network down"):
                list_gguf_variants("unsloth/never-cached")

    def test_online_path_unaffected(self, hf_cache, clean_offline_env):
        # When the API succeeds, cache is not consulted.
        api_payload = _siblings({"a-UD-Q4_K_XL.gguf": 5, "a-Q2_K.gguf": 3})

        def hf_info(*a, **k):
            return api_payload

        with patch("huggingface_hub.model_info", hf_info):
            variants, _has = list_gguf_variants("unsloth/a")
        assert sorted(v.quant for v in variants) == ["Q2_K", "UD-Q4_K_XL"]


# ---------------------------------------------------------------------------
# _detect_gguf_from_hf_cache / detect_gguf_model_remote
# ---------------------------------------------------------------------------


class TestDetectGgufFromCache:
    def test_picks_best_quant(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/a",
            {"a-Q2_K.gguf": 1, "a-UD-Q4_K_XL.gguf": 1},
        )
        assert _detect_gguf_from_hf_cache("unsloth/a") == "a-UD-Q4_K_XL.gguf"

    def test_subdir_only_quant_resolves(self, hf_cache):
        """P1 #1 regression: ``BF16/foo.gguf`` (quant only in directory).
        Before the fix, the offline cache scan matched on basename and
        missed this layout, falling through to the synthetic
        ``{repo}-{variant}.gguf`` heuristic."""
        _build_cache(
            hf_cache,
            "unsloth/gpt-oss-20b-BF16",
            {"BF16/foo.gguf": 1},
        )
        out = _detect_gguf_from_hf_cache("unsloth/gpt-oss-20b-BF16")
        assert (
            out == "BF16/foo.gguf"
        ), f"subdir-only layout must resolve to relative path, got {out}"

    def test_returns_none_when_no_gguf(self, hf_cache):
        _build_cache(hf_cache, "unsloth/a", {"README.md": 10})
        assert _detect_gguf_from_hf_cache("unsloth/a") is None


class TestDetectGgufModelRemoteOffline:
    def test_offline_env_short_circuits_retries(
        self,
        hf_cache,
        clean_offline_env,
        monkeypatch,
    ):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise AssertionError("API must not be called when offline env set")

        with patch("huggingface_hub.model_info", boom):
            assert detect_gguf_model_remote("unsloth/a") == "a-Q4_K_M.gguf"

    def test_api_3x_failure_then_cache(self, hf_cache, clean_offline_env):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})

        def boom(*a, **k):
            raise OSError("hub down")

        # Patch time.sleep so the 1s/2s/4s backoff doesn't slow the test.
        with (
            patch("huggingface_hub.model_info", boom),
            patch("time.sleep", lambda *_: None),
        ):
            out = detect_gguf_model_remote("unsloth/a")
        assert out == "a-Q4_K_M.gguf"

    def test_repository_not_found_does_not_consult_cache(
        self,
        hf_cache,
        clean_offline_env,
    ):
        # Cache has a file but the API explicitly says repo is gone.
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})

        class RepositoryNotFoundError(Exception):
            pass

        def gone(*a, **k):
            raise RepositoryNotFoundError("404")

        with patch("huggingface_hub.model_info", gone):
            out = detect_gguf_model_remote("unsloth/a")
        # Early-return semantics preserved: 404 wins over a stale cache.
        assert out is None


# ---------------------------------------------------------------------------
# _probe_dns_dead / _hf_offline_if_dns_dead
# ---------------------------------------------------------------------------


class _DnsState:
    """Tiny helper that toggles ``socket.gethostbyname`` failure mode."""

    def __init__(self, monkeypatch):
        self._mp = monkeypatch
        self._real = socket.gethostbyname

    def fail(self):
        def _fail(*a, **k):
            raise socket.gaierror(-2, "Name or service not known")

        self._mp.setattr(socket, "gethostbyname", _fail)

    def ok(self):
        self._mp.setattr(socket, "gethostbyname", lambda *a, **k: "127.0.0.1")

    def restore(self):
        self._mp.setattr(socket, "gethostbyname", self._real)


@pytest.fixture
def dns(monkeypatch):
    return _DnsState(monkeypatch)


class TestProbeDnsDead:
    def test_returns_false_on_success(self, dns):
        dns.ok()
        assert _probe_dns_dead() is False

    def test_returns_true_on_failure(self, dns):
        dns.fail()
        assert _probe_dns_dead() is True

    def test_restores_prior_socket_timeout(self, dns):
        dns.ok()
        socket.setdefaulttimeout(7.5)
        try:
            _probe_dns_dead()
            assert socket.getdefaulttimeout() == 7.5
        finally:
            socket.setdefaulttimeout(None)


class TestHfOfflineIfDnsDead:
    def test_dns_fail_sets_env_inside_block_only(self, dns, clean_offline_env):
        dns.fail()
        assert "HF_HUB_OFFLINE" not in os.environ
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is True
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        # P1 #2: env must be restored after the block
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ

    def test_dns_ok_is_noop(self, dns, clean_offline_env):
        dns.ok()
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is False
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_dns_recovers_between_calls(self, dns, clean_offline_env):
        # First call: DNS dead -> env set inside, cleared on exit.
        dns.fail()
        with _hf_offline_if_dns_dead():
            pass
        assert "HF_HUB_OFFLINE" not in os.environ
        # Second call: DNS healthy -> no env mutation.
        dns.ok()
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is False
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_user_set_hf_hub_offline_is_preserved(
        self,
        dns,
        clean_offline_env,
        monkeypatch,
    ):
        # User explicitly set offline before launching Studio.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is False
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        # Helper must not pop a variable it did not set.
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_user_set_transformers_offline_is_preserved(
        self,
        dns,
        clean_offline_env,
        monkeypatch,
    ):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_dns_dead():
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        # HF_HUB_OFFLINE was set by helper -> removed.
        assert "HF_HUB_OFFLINE" not in os.environ
        # TRANSFORMERS_OFFLINE pre-existed -> preserved.
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_exception_inside_block_still_restores_env(
        self,
        dns,
        clean_offline_env,
    ):
        dns.fail()
        with pytest.raises(RuntimeError, match = "boom"):
            with _hf_offline_if_dns_dead():
                raise RuntimeError("boom")
        # Cleanup must happen on exception as well.
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ
