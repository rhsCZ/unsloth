# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the sandboxed Python AST policy in
``studio/backend/core/inference/tools.py``.

Covers:

* The metadata-host denylist (cloud IMDS / link-local) was unchanged
  by this batch but the wrapper message shortened to "Blocked:
  cloud-metadata host" so the LLM can recover.
* The new trusted-host allowlist accepts Wikipedia (any language
  subdomain), Google, HuggingFace, GitHub raw, arXiv, StackOverflow,
  MDN, pypi, ReadTheDocs, etc. and rejects unlisted public hosts with
  a short LLM-readable "Blocked: host not in sandbox allowlist;
  use an allowed informational source".
* File upload shapes (``files=...``, ``data=open(...)``, ``data=b"..."``)
  on requests / httpx / urllib are blocked with
  "Blocked: file upload disallowed in sandbox" regardless of host;
  HfApi().upload_file / upload_folder / upload_large_folder / create_commit
  are blocked by method name.
* Dynamic hosts (computed URLs) remain unblocked at AST time (documented
  limit of static analysis).
* Edge cases that an earlier audit-style validator stumbled on:
  trailing dot, userinfo @, explicit port, IPv6 brackets.
"""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _check_code_safety


def _ok(code: str):
    assert _check_code_safety(code) is None, code


def _blocked(code: str, *, expect_phrase: str):
    msg = _check_code_safety(code)
    assert msg is not None, code
    assert expect_phrase in msg, (expect_phrase, msg)


# =====================================================================
# Metadata host denylist (short message)
# =====================================================================


class TestMetadataHostDenylist:
    def test_aws_imds_literal_blocked(self):
        _blocked(
            'import requests; requests.get("http://169.254.169.254/latest/meta-data/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_gcp_metadata_dns_blocked(self):
        _blocked(
            'import requests; requests.get("http://metadata.google.internal/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_alibaba_ecs_literal_blocked(self):
        _blocked(
            'import socket; s=socket.socket(); s.connect(("100.100.100.200", 80))',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_ipv6_imds_literal_blocked(self):
        _blocked(
            'import urllib.request; urllib.request.urlopen("http://[fd00:ec2::254]/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_metadata_link_local_prefix_blocked(self):
        # Any 169.254.x.x literal -- covers AWS IMDSv2 and ECS task metadata.
        _blocked(
            'import requests; requests.get("http://169.254.170.2/v3/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )


# =====================================================================
# Trusted-host allowlist
# =====================================================================


class TestTrustedHostAllowlist:
    @pytest.mark.parametrize(
        "url",
        [
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "https://fr.wikipedia.org/wiki/Python_(langage)",
            "https://www.google.com/search?q=foo",
            "https://duckduckgo.com/?q=foo",
            "https://huggingface.co/unsloth",
            "https://cdn-lfs.huggingface.co/repos/abc/def/file.bin",
            "https://raw.githubusercontent.com/foo/bar/main/README.md",
            "https://api.github.com/repos/foo/bar",
            "https://arxiv.org/abs/2401.12345",
            "https://export.arxiv.org/abs/2401.12345",
            "https://stackoverflow.com/questions/12345",
            "https://math.stackexchange.com/questions/12345",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "https://docs.python.org/3/library/asyncio.html",
            "https://pypi.org/project/requests/",
            "https://files.pythonhosted.org/packages/foo/bar.whl",
            "https://www.bbc.com/news",
            "https://api.weather.gov/points/40,-90",
            "https://numpy.org/doc/stable/",
            "https://pytorch.org/docs/stable/index.html",
        ],
    )
    def test_trusted_host_passes(self, url):
        _ok(f'import requests; requests.get({url!r})')

    def test_wikipedia_subdomain_passes(self):
        _ok('import urllib.request; urllib.request.urlopen("https://m.en.wikipedia.org/wiki/Foo")')

    def test_hf_co_short_form_passes(self):
        _ok('import requests; requests.get("https://hf.co/unsloth/Qwen3.5-4B-GGUF")')

    def test_github_io_pages_pass(self):
        _ok('import requests; requests.get("https://unslothai.github.io/")')


# =====================================================================
# Untrusted-host block (the new positive-allowlist enforcement)
# =====================================================================


class TestUntrustedHostBlock:
    def test_example_com_blocked(self):
        _blocked(
            'import requests; requests.get("https://example.com/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_random_blog_blocked(self):
        _blocked(
            'import urllib.request; urllib.request.urlopen("https://random-blog-host.example/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_socket_connect_random_host_blocked(self):
        _blocked(
            'import socket; s=socket.socket(); s.connect(("evil.example", 80))',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_dynamic_url_not_statically_blocked(self):
        # Documented limit of AST analysis: dynamic URL is allowed at
        # parse time. Defense-in-depth is the bash blocklist plus the
        # metadata-host check which still fires on .connect((LITERAL,...)).
        _ok(
            'import requests; url = "https://example.com/"; requests.get(url)'
        )


# =====================================================================
# Host normalization edge cases
# =====================================================================


class TestHostNormalization:
    def test_trailing_dot_treated_same(self):
        # Trailing dot is the DNS-absolute form; should normalise out.
        _ok('import requests; requests.get("https://wikipedia.org./")')

    def test_explicit_port_does_not_unblock_or_misblock(self):
        # Trusted host with explicit :443 must still pass the allowlist.
        _ok('import requests; requests.get("https://en.wikipedia.org:443/wiki/Foo")')
        # Untrusted host with port must still block.
        _blocked(
            'import requests; requests.get("https://example.com:8080/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_userinfo_at_does_not_smuggle_metadata_host(self):
        # ``https://trusted@169.254.169.254/`` must NOT pass as trusted.
        _blocked(
            'import requests; requests.get("https://wikipedia.org@169.254.169.254/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_uppercase_host_normalised(self):
        _ok('import requests; requests.get("https://EN.WIKIPEDIA.ORG/wiki/Foo")')


# =====================================================================
# Upload denylist
# =====================================================================


class TestUploadDenylist:
    def test_requests_post_files_blocked(self):
        _blocked(
            (
                'import requests\n'
                'requests.post("https://huggingface.co/api/repos/upload", '
                'files={"f": open("x.bin", "rb")})'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_requests_put_data_bytes_blocked(self):
        _blocked(
            (
                'import requests\n'
                'requests.put("https://huggingface.co/api/repos/upload", '
                'data=b"\\x00\\x01\\x02")'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_requests_post_data_open_handle_blocked(self):
        _blocked(
            (
                'import requests\n'
                'requests.post("https://huggingface.co/api/repos/upload", '
                'data=open("x.bin", "rb"))'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_httpx_post_files_blocked(self):
        _blocked(
            (
                'import httpx\n'
                'httpx.post("https://huggingface.co/api/repos/upload", '
                'files={"f": open("x.bin", "rb")})'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_hf_api_upload_file_blocked(self):
        _blocked(
            (
                'from huggingface_hub import HfApi\n'
                'HfApi().upload_file(path_or_fileobj="x.bin", '
                'path_in_repo="x.bin", repo_id="foo/bar")'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_hf_module_upload_folder_blocked(self):
        _blocked(
            (
                'import huggingface_hub\n'
                'huggingface_hub.upload_folder(folder_path="./", repo_id="foo/bar")'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_hf_create_commit_method_blocked(self):
        _blocked(
            (
                'import huggingface_hub\n'
                'api = huggingface_hub.HfApi()\n'
                'api.create_commit(repo_id="foo/bar", operations=[])'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_plain_post_json_not_blocked(self):
        # POST with json= to a trusted host is fine -- it is not an
        # upload shape.
        _ok(
            'import requests\n'
            'requests.post("https://api.weather.gov/lookup", json={"k": "v"})'
        )


# =====================================================================
# Sandbox CPU rlimit default (Task 7)
# =====================================================================


class TestSandboxCpuRlimitDefault:
    """Pin the env-variable default so a future tweak that lowers it
    back below 600 s without explicit user opt-in is caught."""

    def test_default_cpu_s_is_600(self):
        src = (_BACKEND_ROOT / "core" / "inference" / "tools.py").read_text()
        assert 'UNSLOTH_STUDIO_SANDBOX_CPU_S", "600"' in src, (
            "Sandbox CPU default must be 600 s. If you intentionally "
            "lowered it, update this test."
        )

    def test_clone_newnet_removed(self):
        src = (_BACKEND_ROOT / "core" / "inference" / "tools.py").read_text()
        # The unshare(CLONE_NEWNET) line is gone; the explanatory comment
        # remains so future maintainers know why.
        assert "_libc.unshare(0x40000000)" not in src
        assert "CLONE_NEWNET" in src  # comment retained


# =====================================================================
# Body cap default (Task 3)
# =====================================================================


class TestMaxBodyDefault:
    def test_default_is_500_mb(self):
        src = (_BACKEND_ROOT / "main.py").read_text()
        assert 'UNSLOTH_STUDIO_MAX_BODY_MB", "500"' in src, (
            "Body cap default must be 500 MB. Update this test if "
            "intentionally changed."
        )
