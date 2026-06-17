# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Translate llama-server's Prometheus /metrics into a periodic, vLLM-style
engine-stats log line (throughput, requests in flight, KV cache usage).

llama-server already computes these (it needs `--metrics`); this lifts them
into Studio's structured log so the terminal shows serving health, not just
per-request access lines. Emitted only while there is activity.
"""

import os
import re
import threading
import time
import urllib.request

# Prometheus body lines: "llamacpp:<name> <value>" (skip "#" HELP/TYPE lines).
_METRIC_RE = re.compile(r"^llamacpp:(\w+)\s+([0-9.eE+-]+)", re.MULTILINE)
_OFF = {"0", "false", "no", "off"}


class LlamaServerStatsLogger:
    """Daemon poller that logs vLLM-style engine stats from llama-server.

    Keeps retrying through transient scrape failures; the backend stops it via
    stop() on unload/reload, so a brief /metrics stall does not silence stats.
    """

    def __init__(
        self,
        base_url,
        logger,
        interval_s = 10.0,
    ):
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._log = logger
        self._interval = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target = self._run, name = "llama-stats", daemon = True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _scrape(self):
        try:
            with urllib.request.urlopen(self._url, timeout = 3) as r:
                if r.status != 200:
                    return None
                body = r.read().decode("utf-8", "replace")
        except Exception:
            return None
        out = {}
        for k, v in _METRIC_RE.findall(body):
            try:  # a malformed value must not kill the daemon thread
                out[k] = float(v)
            except ValueError:
                continue
        return out

    def _run(self):
        misses = 0
        prev = None  # (monotonic_t, n_decode_total, prompt_tokens_total)
        while not self._stop.wait(self._interval):
            m = self._scrape()
            if not m:
                misses += 1
                if misses == 3:  # transient stall (load/GC); keep polling.
                    self._log.debug("engine_stats: /metrics scrape failing, still retrying")
                continue  # real shutdown is driven by stop() from _kill_process
            misses = 0
            # tokens_predicted_total only commits at request end; n_decode_total
            # increments live, so derive throughput from its delta over the tick.
            now, decoded, prompt = (
                time.monotonic(),
                m.get("n_decode_total", 0.0),
                m.get("prompt_tokens_total", 0.0),
            )
            gen_tps = prompt_tps = 0.0
            if prev is not None and now > prev[0]:
                dt = now - prev[0]
                gen_tps = max(0.0, (decoded - prev[1]) / dt)
                prompt_tps = max(0.0, (prompt - prev[2]) / dt)
            prev = (now, decoded, prompt)
            running, waiting = (
                int(m.get("requests_processing", 0)),
                int(m.get("requests_deferred", 0)),
            )
            if running or waiting or gen_tps or prompt_tps:  # skip idle ticks
                self._log.info(
                    "engine_stats",
                    gen_tok_s = round(gen_tps, 1),
                    prompt_tok_s = round(prompt_tps, 1),
                    running = running,
                    waiting = waiting,
                    kv_cache_pct = round(m.get("kv_cache_usage_ratio", 0.0) * 100, 1),
                )


def maybe_start_stats_logger(base_url, logger):
    """Start a stats logger unless UNSLOTH_STUDIO_ENGINE_STATS disables it."""
    if (os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS", "1") or "").strip().lower() in _OFF:
        return None
    try:
        interval = float(os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", "10"))
    except ValueError:
        interval = 10.0
    sl = LlamaServerStatsLogger(base_url, logger, interval)
    sl.start()
    return sl
