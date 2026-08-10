# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured logging configuration via structlog.

Environment-specific formats (JSON for prod, console for dev), ISO timestamps,
context-var integration, log-level filtering, and logger caching.
"""

import logging
import os
import sys
from typing import Optional

import structlog

from loggers.handlers import filter_sensitive_data


class _DropTorchDtypeDeprecation(logging.Filter):
    """Drop transformers' once-per-run "`torch_dtype` is deprecated" warning_once.
    It is emitted via logging (not warnings), so a warnings filter cannot catch it."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("torch_dtype" in msg and "deprecated" in msg)


def quiet_third_party_progress_bars() -> None:
    """Turn off the tqdm bars transformers / diffusers / huggingface_hub draw
    during an in-process model load.

    A bar is written with carriage returns to a terminal, so in Studio's log it
    lands as a burst of lines like

        Loading weights:   8%|>         | 30/398 [00:00<00:01, 277.16it/s][A

    and, because tqdm writes to a different stream than the structlog JSON
    writer with no line discipline between them, a bar can land mid-record:
    ``Loading pipeline components...:  20%|...|{"timestamp": ...}``. That line
    is no longer parseable JSON, so anything reading the log record-by-record
    loses the record.

    Nothing is lost by dropping them: download and load progress already reach
    the UI as real events (``hub_download_progress``, ``inference_load_progress``)
    and via /api/inference/{images,video}/load-progress. Only the bars go; the
    libraries' warnings and errors are untouched.

    The subprocess workers already do this by exporting
    HF_HUB_DISABLE_PROGRESS_BARS (hub/services/download_lifecycle.py,
    core/inference/stt_download_worker.py); the server process, which loads the
    RAG embedder at boot and every diffusers pipeline in-process, did not.

    Respects an explicit operator override: if HF_HUB_DISABLE_PROGRESS_BARS is
    already set, its value wins. Only modules that are ALREADY imported get the
    API call, so this never forces a heavy import at logging-setup time.
    """
    if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    elif os.environ["HF_HUB_DISABLE_PROGRESS_BARS"].strip().lower() in ("0", "false", ""):
        # Operator asked to keep them; leave every library alone.
        return

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001 — quieting logs must never break startup
        pass

    # transformers derives its own _tqdm_active from the hub flag at import time,
    # so a module imported BEFORE this ran still needs the explicit call.
    for _mod in ("transformers", "diffusers"):
        module = sys.modules.get(_mod)
        if module is None:
            continue
        try:
            module.utils.logging.disable_progress_bar()
        except Exception:  # noqa: BLE001
            pass


class LogConfig:
    """Structured logging configuration for the application."""

    @staticmethod
    def setup_logging(
        service_name: str = "unsloth-studio-backend", env: Optional[str] = None
    ) -> structlog.BoundLogger:
        """Configure structured logging for the application.
        Args:
            service_name: Name of the service for logging identification
            env: Environment (development/production), affects logging format
        """
        # Log level from environment; fall back to INFO if invalid.
        log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        # Non-ASCII on a non-UTF-8 stream raises UnicodeEncodeError (Windows,
        # LANG=C), so key off the stream, not the platform.
        for stream in (sys.stdout, sys.stderr):
            if getattr(stream, "encoding", "") and not str(stream.encoding).lower().replace(
                "-", ""
            ).startswith("utf8"):
                if hasattr(stream, "reconfigure"):
                    try:
                        stream.reconfigure(encoding = "utf-8", errors = "replace")
                    except Exception:
                        pass

        structlog.configure(
            processors = [
                # Ordered to control output field order.
                structlog.processors.TimeStamper(fmt = "iso"),  # timestamp first
                structlog.processors.add_log_level,  # level second
                structlog.contextvars.merge_contextvars,
                structlog.processors.format_exc_info,
                filter_sensitive_data,
                # Flatten the extra field into the main dict.
                lambda logger, method_name, event_dict: {
                    "timestamp": event_dict.get("timestamp"),
                    "level": event_dict.get("level"),
                    "event": event_dict.get("event"),
                    **(event_dict.get("extra", {})),  # Flatten extra into main dict
                    **{
                        k: v
                        for k, v in event_dict.items()
                        if k not in ["timestamp", "level", "event", "extra"]
                    },
                },
                (
                    structlog.processors.JSONRenderer(sort_keys = False)  # Preserve order
                    if env == "production"
                    else structlog.dev.ConsoleRenderer()
                ),
            ],
            wrapper_class = structlog.make_filtering_bound_logger(log_level),
            logger_factory = structlog.PrintLoggerFactory(file = sys.stdout),
            cache_logger_on_first_use = True,
        )

        # Silence third-party tqdm bars; they carry no signal and corrupt JSON records.
        quiet_third_party_progress_bars()

        # Drop transformers' cosmetic "`torch_dtype` is deprecated" warning_once (see filter).
        _dtype_filter = _DropTorchDtypeDeprecation()
        for _name in (
            "transformers.configuration_utils",
            "transformers.modeling_utils",
            "transformers.pipelines.base",
        ):
            logging.getLogger(_name).addFilter(_dtype_filter)

        return structlog.get_logger(service_name)
