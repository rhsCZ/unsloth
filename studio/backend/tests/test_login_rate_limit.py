# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the per-(ip, username) login rate limiter.

Covers:
  - bucket key composition is (client-ip, username.lower())
  - X-Forwarded-For is honoured only when UNSLOTH_STUDIO_TRUST_FORWARDED is set
  - 429 detail body does NOT leak the client IP
  - One username failing does not lock out a different user from the same IP
  - One IP failing does not lock out the same user from a different IP
"""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture(autouse = True)
def _reset_buckets():
    """Clear the in-memory bucket dict between tests."""
    from routes import auth as auth_routes
    auth_routes._LOGIN_BUCKETS.clear()
    yield
    auth_routes._LOGIN_BUCKETS.clear()


@pytest.fixture
def env_no_proxy(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)


@pytest.fixture
def env_trust_proxy(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")


class _FakeRequest:
    def __init__(self, client_host = "127.0.0.1", headers = None):
        from starlette.datastructures import Headers
        self.client = type("Client", (), {"host": client_host})()
        self.headers = Headers(headers or {})


# ---------- _client_ip ----------

class TestClientIp:
    def test_uses_request_client_host_by_default(self, env_no_proxy):
        from routes.auth import _client_ip
        assert _client_ip(_FakeRequest("203.0.113.5")) == "203.0.113.5"

    def test_ignores_xff_when_trust_off(self, env_no_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest(
            "127.0.0.1",
            {"x-forwarded-for": "198.51.100.7, 10.0.0.1"},
        )
        # The proxy header could be spoofed; without the opt-in we
        # only trust the direct connection.
        assert _client_ip(req) == "127.0.0.1"

    def test_honours_first_xff_when_trust_on(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest(
            "127.0.0.1",
            {"x-forwarded-for": "198.51.100.7, 10.0.0.1"},
        )
        assert _client_ip(req) == "198.51.100.7"

    def test_falls_back_to_client_host_when_xff_missing(self, env_trust_proxy):
        from routes.auth import _client_ip
        assert _client_ip(_FakeRequest("203.0.113.9")) == "203.0.113.9"

    def test_honours_forwarded_header_when_trust_on(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest(
            "127.0.0.1",
            {"forwarded": 'for="198.51.100.42";proto=https'},
        )
        assert _client_ip(req) == "198.51.100.42"

    def test_unknown_when_no_client(self, env_no_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest()
        req.client = None
        assert _client_ip(req) == "_unknown"


# ---------- bucket compose / blocking ----------

class TestBucketKeyAndBlocking:
    def test_record_per_user_isolates_other_users(self, env_no_proxy):
        from routes.auth import (
            _bucket_key, _record_login_failure, _login_blocked, _LOGIN_MAX_FAILS,
        )
        req = _FakeRequest("203.0.113.1")
        for _ in range(_LOGIN_MAX_FAILS):
            _record_login_failure(_bucket_key(req, "alice"))
        assert _login_blocked(_bucket_key(req, "alice")) > 0
        # bob's account from the same IP is unaffected by alice's typos.
        assert _login_blocked(_bucket_key(req, "bob")) == 0

    def test_record_per_ip_isolates_other_ips(self, env_no_proxy):
        from routes.auth import (
            _bucket_key, _record_login_failure, _login_blocked, _LOGIN_MAX_FAILS,
        )
        req_a = _FakeRequest("203.0.113.1")
        req_b = _FakeRequest("203.0.113.2")
        for _ in range(_LOGIN_MAX_FAILS):
            _record_login_failure(_bucket_key(req_a, "alice"))
        assert _login_blocked(_bucket_key(req_a, "alice")) > 0
        # Same username, different IP, not blocked.
        assert _login_blocked(_bucket_key(req_b, "alice")) == 0

    def test_username_lowercased_in_key(self, env_no_proxy):
        from routes.auth import _bucket_key
        req = _FakeRequest("203.0.113.1")
        assert _bucket_key(req, "Alice") == _bucket_key(req, "alice")
        assert _bucket_key(req, "ALICE") == _bucket_key(req, "alice")


# ---------- /login 429 body ----------

class TestLogin429Body:
    @pytest.fixture
    def login_client(self, tmp_path, monkeypatch):
        from auth import storage
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routes.auth import router as auth_router
        import secrets as _secrets

        monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
        monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
        monkeypatch.setattr(storage, "_bootstrap_password", None)
        storage.create_initial_user(
            username = storage.DEFAULT_ADMIN_USERNAME,
            password = "human-password-123",
            jwt_secret = _secrets.token_urlsafe(64),
            must_change_password = False,
        )

        app = FastAPI()
        app.include_router(auth_router, prefix = "/api/auth")
        return TestClient(app)

    def test_429_detail_does_not_leak_ip(self, env_no_proxy, login_client):
        from routes.auth import _LOGIN_MAX_FAILS

        # Drive 6 failures from the same client IP / username.
        for _ in range(_LOGIN_MAX_FAILS):
            r = login_client.post(
                "/api/auth/login",
                json = {"username": "unsloth", "password": "wrong"},
            )
            assert r.status_code == 401
        r = login_client.post(
            "/api/auth/login",
            json = {"username": "unsloth", "password": "wrong"},
        )
        assert r.status_code == 429
        detail = r.json()["detail"]
        # The 429 body must not interpolate the source IP.
        assert "127.0.0.1" not in detail
        assert "Too many" in detail
        # Retry-After header is still set for clients.
        assert "Retry-After" in r.headers
