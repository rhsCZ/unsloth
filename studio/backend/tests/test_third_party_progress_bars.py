# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Quieting third-party tqdm bars must not take anything real with it.

The bars themselves carry no signal in a log with no terminal, but three things
ride along with them and have to survive: the export dialog's live Hub upload
progress, the "Applying chat template ... 42%" status the UI derives from the
datasets bar's counter, and an operator's explicit choice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from loggers import config as log_config  # noqa: E402

_HUB = "HF_HUB_DISABLE_PROGRESS_BARS"


def test_the_default_is_installed_and_marked(monkeypatch):
    monkeypatch.delenv(_HUB, raising = False)
    monkeypatch.delenv(log_config._PROGRESS_BARS_DEFAULTED, raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", raising = False)

    log_config.quiet_third_party_progress_bars()

    import os
    assert os.environ[_HUB] == "1"
    assert os.environ[log_config._PROGRESS_BARS_DEFAULTED] == "1"


def test_verbose_leaves_the_bars_alone(monkeypatch):
    # --verbose zeroes both access-log windows and promises everything back; the flag
    # is inherited by the workers, so setting it here would keep them quiet anyway.
    monkeypatch.delenv(_HUB, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", "0")
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", "0")

    log_config.quiet_third_party_progress_bars()

    import os
    assert _HUB not in os.environ


def test_hugging_face_false_spellings_are_honored(monkeypatch):
    # The Hub reads only 1/ON/YES/TRUE as true, so "off" and "no" ask to keep bars.
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", raising = False)
    for value in ("off", "no", "0", "false", ""):
        monkeypatch.setenv(_HUB, value)
        called = []
        monkeypatch.setattr(
            log_config, "_silence_datasets_bar_output", lambda: called.append(1)
        )
        log_config.quiet_third_party_progress_bars()
        assert called == [], value


def test_the_hub_is_not_imported_just_to_quiet_it():
    # A worker calls setup_logging BEFORE prepending its transformers sidecar to
    # sys.path; importing the Hub here would cache the base environment's copy.
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import os\n"
        "os.environ.pop('HF_HUB_DISABLE_PROGRESS_BARS', None)\n"
        "os.environ.pop('UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS', None)\n"
        "os.environ.pop('UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS', None)\n"
        "from loggers.config import quiet_third_party_progress_bars\n"
        "quiet_third_party_progress_bars()\n"
        "print('HUB_IMPORTED' if 'huggingface_hub' in sys.modules else 'HUB_ABSENT')\n"
    ) % str(_BACKEND)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, timeout = 300
    )
    assert "HUB_ABSENT" in out.stdout, out.stdout + out.stderr


def test_allow_progress_bars_only_undoes_our_own_default(monkeypatch):
    monkeypatch.setenv(_HUB, "1")
    monkeypatch.setenv(log_config._PROGRESS_BARS_DEFAULTED, "1")
    log_config.allow_progress_bars()
    import os
    assert _HUB not in os.environ

    # An operator who set it themselves keeps it.
    monkeypatch.setenv(_HUB, "1")
    monkeypatch.delenv(log_config._PROGRESS_BARS_DEFAULTED, raising = False)
    log_config.allow_progress_bars()
    assert os.environ[_HUB] == "1"


def test_the_export_worker_keeps_its_progress_bars():
    text = (_BACKEND / "core/export/worker.py").read_text(encoding = "utf-8")
    assert "allow_progress_bars()" in text
    assert "quiet_progress_bars = False" in text


def test_the_datasets_bar_keeps_counting_but_writes_nothing(capfd):
    # chat_templates.py polls tqdm._instances for the formatting status, and
    # datasets' own disable_progress_bar() forces tqdm(disable = True), which never
    # registers the bar at all.
    import datasets  # noqa: F401
    from datasets.utils.tqdm import tqdm as ds_bar
    from tqdm.auto import tqdm as base_tqdm

    log_config._silence_datasets_bar_output()
    bar = ds_bar(total = 10, desc = "Applying chat template")
    try:
        bar.update(4)
        instances = [b for b in list(getattr(base_tqdm, "_instances", set())) if b is bar]
        assert instances, "the bar must stay registered for the UI status poller"
        assert instances[0].n == 4
    finally:
        bar.close()
    captured = capfd.readouterr()
    assert "Applying chat template" not in captured.out + captured.err


def test_silencing_the_datasets_bar_twice_is_harmless():
    from datasets.utils.tqdm import tqdm as ds_bar

    log_config._silence_datasets_bar_output()
    first = ds_bar.__init__
    log_config._silence_datasets_bar_output()
    assert ds_bar.__init__ is first


def test_trainer_summary_metrics_are_republished():
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    assert "trainer summary" in text
    for key in ("train_samples_per_second", "train_steps_per_second", "total_flos"):
        assert key in text, key


def test_a_resumed_run_does_not_report_checkpoint_history_as_throughput():
    text = (_BACKEND / "core/training/training.py").read_text(encoding = "utf-8")
    assert "not self._progress_run_resumed" in text
    assert 'self._progress_run_resumed = bool(config.get("resume_from_checkpoint"))' in text
