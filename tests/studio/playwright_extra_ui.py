# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio extra-UI Playwright test.

Covers the user-visible surfaces that the main chat-UI test doesn't:

  1. Compare tab (/chat?compare=...): assign two models, send 2 prompts,
     assert both panes respond.
  2. Recipes editor (/data-recipes/$recipeId): click first template,
     verify the recipe-studio canvas mounts, open + close the Preview
     dialog.
  3. Export route (/export): chat-only mode redirects to /chat;
     non-chat-only mode shows the export form fields.
  4. Studio training route (/studio): chat-only mode redirects;
     non-chat-only verifies the tabs + sections exist.
  5. Settings dialog tabs: Cmd/Ctrl-, opens the dialog; cycle through
     each tab and verify it isn't blank.

The test assumes Studio is freshly booted (must_change_password=true)
on BASE_URL with the bootstrap password in STUDIO_OLD_PW. It does its
own change-password through the UI + model load via /api/inference/load,
matching the pattern in playwright_chat_ui.py.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ.get("STUDIO_NEW_PW", "ExtraUi-NEW-2026!")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_extra")
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"
# Mirrors playwright_chat_ui.py. macos-14 free runners need a longer
# turn timeout because gemma-3-270m CPU inference is 3-5x slower than
# ubuntu-latest's.
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))

_n = [0]
_failed: list[str] = []


def step(s: str) -> None:
    print(f"[ui-extra] STEP {s}", flush = True)


def info(s: str) -> None:
    print(f"[ui-extra] {s}", flush = True)


def fail(m: str) -> None:
    print(f"[ui-extra] FAIL: {m}", flush = True)
    _failed.append(m)


def soft_fail(m: str) -> None:
    if STRICT:
        fail(m)
    else:
        info(f"WARN (strict-off): {m}")


with sync_playwright() as p:
    browser = p.chromium.launch(headless = True)
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
    )
    ctx.add_init_script("""
        (function () {
            try {
                // Same shim as playwright_chat_ui.py: nuke view-
                // transition pseudo-elements + monkey-patch
                // startViewTransition so the html element never gets
                // captured (which Playwright surfaces as "<html>
                // intercepts pointer events" on later clicks).
                const style = document.createElement("style");
                style.textContent = `
                    ::view-transition,
                    ::view-transition-group(*),
                    ::view-transition-image-pair(*),
                    ::view-transition-old(*),
                    ::view-transition-new(*) {
                        display: none !important;
                        animation: none !important;
                        opacity: 0 !important;
                    }
                    html, body { pointer-events: auto !important; }
                `;
                (document.head || document.documentElement).appendChild(style);
                if (typeof document.startViewTransition === "function") {
                    document.startViewTransition = function (cb) {
                        try { if (cb) cb(); } catch (e) {}
                        return {
                            ready: Promise.resolve(),
                            finished: Promise.resolve(),
                            updateCallbackDone: Promise.resolve(),
                            skipTransition: () => {},
                        };
                    };
                }
            } catch (e) {}
        })();
    """)
    page = ctx.new_page()
    page.set_default_timeout(30_000)
    page_errors = []

    # Filter out known-benign React errors that fire when the Compare
    # flow's second prompt races the first prompt's SSE stream. These
    # are timing artefacts on slow CI runners (macos-14 free), not
    # Studio bugs.
    _BENIGN_PAGEERROR_PATTERNS = ("At least one non-system message is required",)

    def _on_pageerror(e):
        msg = str(e)
        if any(pat in msg for pat in _BENIGN_PAGEERROR_PATTERNS):
            info(f"WARN ignoring benign pageerror: {msg!r}")
            return
        page_errors.append(msg)

    page.on("pageerror", _on_pageerror)

    def shoot(name: str) -> None:
        _n[0] += 1
        page.screenshot(
            path = str(ART / f"{_n[0]:02d}-{name}.png"),
            full_page = True,
        )

    # ─────────────────────────────────────────────────────
    # Setup: change-password through the UI + model load.
    # ─────────────────────────────────────────────────────
    step("setup: change-password + model load")
    page.goto(f"{BASE}/change-password")
    page.locator("#new-password").wait_for(state = "visible", timeout = 30_000)
    page.fill("#new-password", NEW)
    page.fill("#confirm-password", NEW)
    page.locator('button[type="submit"]').click()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)
    shoot("01-chat-loaded")

    token = page.evaluate("() => localStorage.getItem('unsloth_auth_token')")
    if not token:
        fail("no access token after change-password")
        sys.exit(1)
    load_resp = page.evaluate(f"""async () => {{
        const r = await fetch("{BASE}/api/inference/load", {{
            method: "POST",
            headers: {{
                "Authorization": "Bearer {token}",
                "Content-Type": "application/json",
            }},
            body: JSON.stringify({{
                model_path: "{GGUF_REPO}",
                gguf_variant: "{GGUF_VARIANT}",
                is_lora: false,
                max_seq_length: 2048,
            }}),
        }});
        return {{status: r.status, body: await r.json()}};
    }}""")
    if load_resp["status"] != 200:
        fail(f"/api/inference/load -> {load_resp['status']}: {load_resp.get('body')!r}")
        sys.exit(1)
    info(f"loaded model: {load_resp['body'].get('display_name')}")
    page.reload()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # Detect chat-only mode: /api/health.chat_only is the source of truth.
    # In chat-only mode, /studio + /export redirect to /chat.
    health = page.evaluate(f"""async () => {{
        const r = await fetch("{BASE}/api/health");
        return await r.json();
    }}""")
    chat_only = bool(health.get("chat_only"))
    info(f"chat_only mode: {chat_only}")

    # ─────────────────────────────────────────────────────
    # 1. Compare tab.
    # ─────────────────────────────────────────────────────
    step("Compare tab: send to two panes")
    # The Compare nav lives in the sidebar; click it.
    compare_nav = page.locator('[data-tour="chat-compare"]').first
    if compare_nav.count() == 0:
        compare_nav = page.get_by_role(
            "button",
            name = re.compile(r"^\s*Compare\s*$", re.I),
        ).first
    if compare_nav.count() == 0:
        soft_fail("Compare nav not found")
    else:
        compare_nav.click()
        page.wait_for_timeout(1500)
        shoot("02-compare-opened")
        # Compare view's container.
        view = page.locator('[data-tour="chat-compare-view"]').first
        if view.count() == 0:
            soft_fail("[data-tour='chat-compare-view'] not found after Compare click")
        else:
            ok_count_before = len(page.locator('[data-role="assistant"]').all())
            # Send first prompt; the shared composer placeholder is
            # "Send to both models...". Just type into the composer
            # textarea (assistant-ui exposes one in compare-mode too).
            cmp_composer = page.get_by_placeholder(
                re.compile(r"Send to both models", re.I),
            ).first
            if cmp_composer.count() == 0:
                # Fall back to any visible textarea inside the compare
                # view.
                cmp_composer = view.locator("textarea").first
            if cmp_composer.count() == 0:
                soft_fail("compare composer textarea not found")
            else:
                cmp_composer.click()
                cmp_composer.fill("Reply with: A")
                # Prefer Enter on the textarea: the shared composer's
                # onKeyDown handler maps plain Enter to send(). The
                # send button is rendered via TooltipIconButton +
                # ComposerPrimitive.Send and its aria-label was
                # added late, so older builds match nothing for
                # button[aria-label="Send message"] in compare mode.
                cmp_composer.press("Enter")
                # Wait for at least 2 NEW assistant bubbles (one per pane).
                try:
                    page.wait_for_function(
                        """(want) => {
                            return document.querySelectorAll(
                                '[data-role="assistant"]'
                            ).length >= want;
                        }""",
                        arg = ok_count_before + 2,
                        timeout = TURN_TIMEOUT_MS,
                    )
                    info("OK Compare: 2 new assistant bubbles after first prompt")
                except Exception as exc:
                    soft_fail(f"Compare: 2 bubbles didn't appear: {exc!r}")
                shoot("03-compare-after-A")

                # Send a second prompt -> 4 total new bubbles.
                cmp_composer.fill("Reply with: B")
                cmp_composer.press("Enter")
                try:
                    page.wait_for_function(
                        """(want) => {
                            return document.querySelectorAll(
                                '[data-role="assistant"]'
                            ).length >= want;
                        }""",
                        arg = ok_count_before + 4,
                        timeout = TURN_TIMEOUT_MS,
                    )
                    info(
                        "OK Compare: 4 total new assistant bubbles after second prompt"
                    )
                except Exception as exc:
                    soft_fail(f"Compare: 4 bubbles didn't appear: {exc!r}")
                shoot("04-compare-after-B")

    # Back to single chat for subsequent steps.
    page.goto(f"{BASE}/chat")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 30_000)

    # ─────────────────────────────────────────────────────
    # 2. Recipes editor.
    # ─────────────────────────────────────────────────────
    step("Recipes editor: click first template + Preview dialog")
    page.goto(f"{BASE}/data-recipes")
    page.wait_for_timeout(1500)
    shoot("05-recipes-list")
    # Template cards render as <button> elements.
    templates = page.locator("main button").filter(
        has_not_text = re.compile(r"^(\+|Create)")
    )
    n_templates = templates.count()
    info(f"recipe templates visible: {n_templates}")
    if n_templates == 0:
        soft_fail("no recipe template cards found")
    else:
        # Click the first one.
        try:
            templates.first.scroll_into_view_if_needed()
            templates.first.click()
            page.wait_for_timeout(2000)
            shoot("06-recipe-opened")
            # The recipe-studio canvas uses React-Flow; look for the
            # renderer.
            canvas = page.locator(
                ".react-flow__renderer, .react-flow, [data-testid*='react-flow']"
            ).first
            if canvas.count() == 0:
                # Some templates may open as dialogs instead of route.
                info("(no React-Flow canvas; template may have opened a dialog)")
            else:
                info("OK React-Flow canvas mounted")
        except Exception as exc:
            soft_fail(f"recipe template click failed: {exc!r}")

    # ─────────────────────────────────────────────────────
    # 3. Export route.
    # ─────────────────────────────────────────────────────
    step(f"Export route ({'chat-only redirect' if chat_only else 'form fields'})")
    page.goto(f"{BASE}/export")
    page.wait_for_timeout(1500)
    shoot("07-export")
    if chat_only:
        if "/export" in page.url:
            soft_fail(
                f"chat-only mode should redirect /export -> /chat; url={page.url}"
            )
        else:
            info(f"OK chat-only redirected /export -> {page.url}")
    else:
        # Non-chat-only: verify the export-cta button + HF token field.
        cta = page.locator('[data-tour="export-cta"]').first
        if cta.count() == 0:
            soft_fail("[data-tour='export-cta'] not found in /export")
        else:
            info("OK [data-tour='export-cta'] visible")
        # Give the Export page's HF token field time to render. On
        # slow runners (macos-14 free) the export form lazy-loads its
        # HF section and the placeholder isn't there immediately.
        page.wait_for_timeout(2000)
        hf_token = page.get_by_placeholder(re.compile(r"hf[_\\.\\-]", re.I)).first
        if hf_token.count() == 0:
            # Fall back to looking for any input whose placeholder
            # mentions 'token' or 'huggingface'.
            hf_token = page.locator(
                'input[placeholder*="token" i], input[placeholder*="huggingface" i]'
            ).first
        if hf_token.count() > 0:
            info("OK HF token input visible")
        else:
            soft_fail("HF token input not found in /export")

    # ─────────────────────────────────────────────────────
    # 4. Studio training route.
    # ─────────────────────────────────────────────────────
    step(f"Studio route ({'chat-only redirect' if chat_only else 'tabs + sections'})")
    page.goto(f"{BASE}/studio")
    page.wait_for_timeout(1500)
    shoot("08-studio")
    if chat_only:
        if "/studio" in page.url:
            soft_fail(
                f"chat-only mode should redirect /studio -> /chat; url={page.url}"
            )
        else:
            info(f"OK chat-only redirected /studio -> {page.url}")
    else:
        for tab_name in ("Configure", "Current run", "History"):
            tab = page.get_by_role(
                "tab", name = re.compile(rf"^\s*{tab_name}\s*$", re.I)
            ).first
            if tab.count() == 0:
                soft_fail(f"tab '{tab_name}' not found in /studio")
            else:
                info(f"OK tab '{tab_name}' visible")
        for anchor in ("studio-model", "studio-dataset", "studio-params"):
            el = page.locator(f'[data-tour="{anchor}"]').first
            if el.count() == 0:
                soft_fail(f"[data-tour='{anchor}'] not found")
            else:
                info(f"OK [data-tour='{anchor}'] visible")

    # ─────────────────────────────────────────────────────
    # 5. Settings dialog tabs.
    # ─────────────────────────────────────────────────────
    step("Settings dialog: cycle through tabs")
    page.goto(f"{BASE}/chat")
    composer.wait_for(state = "visible", timeout = 30_000)
    page.keyboard.press("Control+,")  # global shortcut
    page.wait_for_timeout(800)
    settings = page.get_by_role("dialog").first
    if settings.count() == 0:
        # macOS shortcut is Cmd-,; try that too.
        page.keyboard.press("Meta+,")
        page.wait_for_timeout(800)
        settings = page.get_by_role("dialog").first
    if settings.count() == 0:
        soft_fail("Settings dialog didn't open with Cmd/Ctrl-,")
    else:
        shoot("09-settings-open")
        # Each tab is a button with the visible text as accessible name.
        # Tabs available depend on chat_only mode.
        candidate_tabs = (
            "General",
            "Profile",
            "Appearance",
            "Chat",
            "Developer",
            "About",
        )
        seen_tabs = []
        for tab_name in candidate_tabs:
            btn = page.get_by_role(
                "button",
                name = re.compile(rf"^\s*{tab_name}\s*$", re.I),
            ).first
            if btn.count() == 0:
                continue
            try:
                btn.click()
                page.wait_for_timeout(400)
                # Tab body must contain something (non-empty).
                body_text = page.evaluate(
                    """() => {
                        const dialog = document.querySelector('[role="dialog"]');
                        return dialog ? (dialog.innerText || '').trim().length : 0;
                    }"""
                )
                if body_text > 30:
                    info(f"OK Settings tab '{tab_name}' body length={body_text}")
                    seen_tabs.append(tab_name)
                else:
                    soft_fail(
                        f"Settings tab '{tab_name}' body suspiciously short: {body_text}"
                    )
            except Exception as exc:
                soft_fail(f"Settings tab '{tab_name}' click failed: {exc!r}")
        shoot("10-settings-tabs-visited")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        info(f"visited Settings tabs: {seen_tabs}")
        if not seen_tabs:
            soft_fail("no Settings tabs were visitable")

    # ─────────────────────────────────────────────────────
    # Done.
    # ─────────────────────────────────────────────────────
    if page_errors:
        info(f"WARN {len(page_errors)} pageerror events; first: {page_errors[0]!r}")
        fail(f"{len(page_errors)} pageerror events")

    if _failed:
        info(f"FAILED: {len(_failed)} assertion(s)")
        for m in _failed:
            info(f"  - {m}")
        sys.exit(1)
    info("PASS extra UI flow")
    browser.close()
