# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio chat composer IME + multilingual regression smoke.

Catches two classes of regression that backend / frontend CI both miss:

  A. Stuck IME composition (issue #5318 / PR #5327). Repeated
     compositionstart without a matching compositionend left the
     assistant-ui composer with isComposing=true forever, so the
     textarea silently dropped every keystroke -- including plain
     ASCII after switching the IME back to English. Reported by
     Japanese, Chinese, and Korean users.

  B. Multilingual paste round-trip. Any future Unicode regression
     in the controlled textarea / React state plumbing (NFC
     re-normalisation, UTF-16 surrogate splits, combining-mark
     reorderings) would silently mangle non-ASCII text without
     tripping any JS path the existing chat smoke exercises. We
     paste a fixed string in each of 31 scripts covering >90% of
     the world's population and require byte-for-byte readback.

Model-free on purpose -- the bug surface is the composer, not
inference. Wall clock ~60 s on a warm runner.

Env contract matches playwright_chat_ui.py:
  BASE_URL          base URL of an already-running Studio
  STUDIO_OLD_PW     bootstrap password from auth/.bootstrap_password
  STUDIO_NEW_PW     password we rotate to in the first step
  PW_ART_DIR        artifact dir for screenshots
  STUDIO_UI_STRICT  '1' = hard-fail on missing elements (CI default)

Run locally:
  BASE_URL=http://127.0.0.1:8888 \
  STUDIO_OLD_PW=$(cat ~/.unsloth/studio/auth/.bootstrap_password) \
  STUDIO_NEW_PW=ChangeMe-$$ \
  PW_ART_DIR=logs/playwright_ime \
  python tests/studio/playwright_chat_ime_i18n.py
"""

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    recover_or_replace_page,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_ime")
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"

# Wall-clock cap. Realistic run is 30-60 s on a warm runner; 5 min
# leaves headroom for a cold browser launch + first-paint on the
# mac/win runners if this test ever gets ported to them.
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_IME_WALL_TIMEOUT_S", "300"))


# Top languages by total speakers (L1 + L2), rounded down -- this
# list is deliberately broad rather than maximal: each entry is a
# distinct script / shaping rule and catches a different class of
# Unicode regression. Together these cover >90% of the world's
# population by language. Ordering is by speaker count (descending,
# rough) so the most-impactful regressions surface first in the log.
#
# `code` is BCP-47-ish (purely for log labels).
# `label` is the English display name.
# `text` is what we paste -- a short universal greeting + arithmetic
#   string so the line stays under one composer row and the readback
#   diff is easy to read in the CI log on miss.
I18N_SAMPLES = [
    ("en",     "English",                 "Hello, 1+1=2"),
    ("zh-CN",  "Chinese (Simplified)",    "你好，1+1=2"),
    ("es",     "Spanish",                 "Hola, 1+1=2"),
    ("hi",     "Hindi (Devanagari)",      "नमस्ते, 1+1=2"),
    ("ar",     "Arabic (RTL)",            "مرحبا، ١+١=٢"),
    ("bn",     "Bengali",                 "নমস্কার, ১+১=২"),
    ("pt",     "Portuguese",              "Olá, 1+1=2"),
    ("ru",     "Russian (Cyrillic)",      "Привет, 1+1=2"),
    ("ja",     "Japanese",                "こんにちは、1+1=2"),
    ("pa",     "Punjabi (Gurmukhi)",      "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, 1+1=2"),
    ("de",     "German",                  "Hallo, 1+1=2"),
    ("jv",     "Javanese",                "Halo, 1+1=2"),
    ("ko",     "Korean (Hangul)",         "안녕하세요, 1+1=2"),
    ("fr",     "French",                  "Bonjour, 1+1=2"),
    ("tr",     "Turkish",                 "Merhaba, 1+1=2"),
    ("vi",     "Vietnamese (diacritics)", "Xin chào, 1+1=2"),
    ("ur",     "Urdu (Arabic-Naskh)",     "ہیلو، 1+1=2"),
    ("ta",     "Tamil",                   "வணக்கம், 1+1=2"),
    ("te",     "Telugu",                  "నమస్తే, 1+1=2"),
    ("mr",     "Marathi (Devanagari)",    "नमस्कार, 1+1=2"),
    ("it",     "Italian",                 "Ciao, 1+1=2"),
    ("th",     "Thai",                    "สวัสดี, ๑+๑=๒"),
    ("pl",     "Polish",                  "Cześć, 1+1=2"),
    ("uk",     "Ukrainian (Cyrillic)",    "Привіт, 1+1=2"),
    ("fa",     "Persian (RTL)",           "سلام، ۱+۱=۲"),
    ("nl",     "Dutch",                   "Hallo, 1+1=2"),
    ("he",     "Hebrew (RTL)",            "שלום, 1+1=2"),
    ("el",     "Greek",                   "Γειά, 1+1=2"),
    ("id",     "Indonesian",              "Halo, 1+1=2"),
    ("sw",     "Swahili",                 "Habari, 1+1=2"),
    ("emoji",  "Emoji + ZWJ + flag",      "👋 🇺🇳 👨‍👩‍👧‍👦 1+1=2"),
]


_n = [0]


def step(s):
    print(f"[ime] STEP {s}", flush = True)


def info(s):
    print(f"[ime] {s}", flush = True)


def fail(m):
    raise AssertionError(f"[ime] FAIL: {m}")


def soft_fail(m):
    """Hard fail in STRICT mode, info-warn otherwise.

    Mirrors the contract used by `playwright_chat_ui.py`: in CI we
    set STUDIO_UI_STRICT=1 so a missing button / composer selector
    fails loudly; locally a partial install is warned and skipped.
    """
    if STRICT:
        fail(m)
    info(f"WARN (strict-off): {m}")


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S,
        label = "ime",
        info = info,
    )
    wait_for_health(BASE, timeout = 30.0, info = info)
    browser = p.chromium.launch(
        headless = True,
        args = chromium_launch_args(),
    )
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
    )
    install_view_transition_killer(ctx)
    page = ctx.new_page()
    page.set_default_timeout(60_000)

    page_errors: list[str] = []
    page.on("pageerror", lambda e: page_errors.append(str(e)))
    console_errors: list[str] = []

    def _on_console(m):
        if m.type != "error":
            return
        try:
            console_errors.append(m.text)
        except Exception:
            return

    page.on("console", _on_console)

    def shoot(name):
        _n[0] += 1
        try:
            page.screenshot(
                path = str(ART / f"{_n[0]:02d}-{name}.png"),
                full_page = True,
                timeout = 90_000,
                animations = "disabled",
            )
        except Exception as _shoot_err:
            info(f"WARN: screenshot {name} failed: {_shoot_err}")

    # ─────────────────────────────────────────────────────
    # 1. Bootstrap auth via the UI's /change-password form so
    #    the auth state matches what a real first-run user sees.
    #    We mirror the retry-on-rerender pattern from
    #    playwright_chat_ui.py because the same React form-detach
    #    races happen here too.
    # ─────────────────────────────────────────────────────
    step("change-password through UI (Setup your account)")
    form_err: Exception | None = None
    for _form_attempt in range(3):
        try:
            page.goto(
                f"{BASE}/change-password",
                wait_until = "domcontentloaded",
                timeout = 60_000,
            )
            try:
                page.wait_for_load_state("networkidle", timeout = 30_000)
            except Exception:
                pass
            pw_field = page.locator("#new-password")
            pw_field.wait_for(state = "visible", timeout = 60_000)
            pw_field.fill(NEW, timeout = 60_000)
            page.fill("#confirm-password", NEW, timeout = 60_000)
            shoot("01-change-password-filled")
            with page.expect_response(
                lambda r: "/api/auth/change-password" in r.url and r.request.method == "POST",
                timeout = 30_000,
            ) as resp_info:
                page.locator('button[type="submit"]').click()
            resp = resp_info.value
            if resp.status >= 400:
                raise AssertionError(
                    f"change-password POST returned {resp.status}"
                )
            form_err = None
            break
        except Exception as e:
            form_err = e
            info(
                f"change-password attempt {_form_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            if _form_attempt < 2:
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[ime]   recovery: {m}", flush = True),
                )
    if form_err is not None:
        raise form_err

    # ─────────────────────────────────────────────────────
    # 2. Composer textarea is the test surface. Wait for it
    #    to mount after the post-auth redirect into the chat
    #    shell. We do NOT load a GGUF: the IME / paste bugs we
    #    are guarding against live in the composer's React
    #    state, not the inference path.
    # ─────────────────────────────────────────────────────
    step("wait for composer to mount")
    try:
        page.wait_for_load_state("networkidle", timeout = 30_000)
    except Exception:
        pass
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)
    composer.click()
    shoot("02-composer-focused")

    # Static guard: the composer must opt in to Unicode bidi auto-detection
    # so RTL scripts (Arabic / Hebrew / Persian / Urdu) flow right-to-left
    # without forcing the whole UI into RTL.
    dir_attr = composer.evaluate("(el) => el.getAttribute('dir')")
    if dir_attr != "auto":
        soft_fail(
            f'composer is missing dir="auto" (got {dir_attr!r}); RTL '
            'languages will render LTR.'
        )
    else:
        info('composer dir="auto" present')

    def read_value() -> str:
        return composer.evaluate("(el) => el.value")

    def set_value_via_setter(s: str) -> str:
        """Set textarea.value through React's monkey-patched setter
        AND dispatch the input event so assistant-ui's controlled
        textarea picks the new value up. Mirrors a real paste -- a
        plain `.value = s` would be silently overwritten on the next
        React render. Returns the post-dispatch readback."""
        return composer.evaluate(
            """(el, v) => {
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                setter.call(el, v);
                el.dispatchEvent(new Event('input', {bubbles:true}));
                return el.value;
            }""",
            s,
        )

    def clear() -> None:
        set_value_via_setter("")

    # ─────────────────────────────────────────────────────
    # 3. Baseline: ASCII keyboard typing works at all. If this
    #    fails, every later test is meaningless -- bail fast.
    # ─────────────────────────────────────────────────────
    step("baseline ASCII keyboard typing")
    clear()
    composer.click()
    for ch in "hello world":
        page.keyboard.type(ch)
    got = read_value()
    if got != "hello world":
        fail(f"ASCII typing readback {got!r} != 'hello world'")
    info("baseline ASCII OK")
    shoot("03-baseline-ascii")
    clear()

    # ─────────────────────────────────────────────────────
    # 4. Multilingual paste round-trip across 30+ scripts.
    #    Each entry must readback byte-for-byte after a single
    #    nativeInputValueSetter + input-event dispatch.
    # ─────────────────────────────────────────────────────
    step(f"multilingual paste round-trip ({len(I18N_SAMPLES)} samples)")
    paste_failures: list[tuple[str, str, str, str]] = []
    for code, label, text in I18N_SAMPLES:
        got = set_value_via_setter(text)
        if got != text:
            paste_failures.append((code, label, text, got))
            info(f"  {code:>6} ({label}): FAIL -- got {got!r}")
        else:
            info(f"  {code:>6} ({label}): OK")
        clear()
    if paste_failures:
        shoot("04-paste-failures")
        lines = [
            f"  {code} ({label}): want={want!r} got={got!r}"
            for code, label, want, got in paste_failures
        ]
        fail(
            f"{len(paste_failures)}/{len(I18N_SAMPLES)} languages failed paste round-trip:\n"
            + "\n".join(lines)
        )
    info(f"all {len(I18N_SAMPLES)} multilingual paste samples OK")
    shoot("04-paste-all-ok")

    # ─────────────────────────────────────────────────────
    # 5. Normal IME composition sequence: compose 你好 the way a
    #    healthy Chinese IME would -- compositionstart ->
    #    compositionupdate(你) -> compositionupdate(你好) ->
    #    compositionend -> insertFromComposition. The composer
    #    must end up with the committed text in its value.
    # ─────────────────────────────────────────────────────
    step("normal IME composition (compose 你好)")
    clear()
    composer.click()
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
            el.dispatchEvent(new CompositionEvent('compositionupdate', {bubbles:true, data:'你'}));
            el.dispatchEvent(new CompositionEvent('compositionupdate', {bubbles:true, data:'你好'}));
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(el, el.value + '你好');
            el.dispatchEvent(new InputEvent('input', {
                bubbles:true, inputType:'insertCompositionText',
                data:'你好', isComposing:true,
            }));
            el.dispatchEvent(new CompositionEvent('compositionend', {bubbles:true, data:'你好'}));
            el.dispatchEvent(new InputEvent('input', {
                bubbles:true, inputType:'insertFromComposition', data:'你好',
            }));
        }"""
    )
    got = read_value()
    if "你好" not in got:
        shoot("05-normal-composition-FAIL")
        fail(f"normal composition readback {got!r} missing '你好'")
    info(f"normal composition OK: ta.value={got!r}")
    shoot("05-normal-composition")
    clear()

    # ─────────────────────────────────────────────────────
    # 6. Stuck IME composition: this is the issue #5318 repro.
    #    Two compositionstart events fire with NO compositionend
    #    in between (the pattern observed from Japanese / Chinese
    #    / Korean IMEs on Safari + Arc + Chrome / Windows + macOS
    #    in the issue reports). On the BROKEN build the React
    #    composer sticks at isComposing=true and silently drops
    #    every subsequent input event -- even plain ASCII typed
    #    after switching the IME back to English. PR #5327 added
    #    a Studio-owned composition handler that clears stale
    #    isComposing state on blur / paste / non-composing input,
    #    so the field must recover.
    # ─────────────────────────────────────────────────────
    step("BUG REPRO: stuck IME composition recovery (issue #5318)")
    clear()
    composer.click()
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
            // Duplicate compositionstart with NO matching compositionend.
            // This is exactly the event sequence observed from the IMEs
            // in issue #5318 (kei-yamazaki / langxiaopiao030 / PapyrusNotes).
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
        }"""
    )
    # Now try to commit 'abc' as if the user switched back to an English
    # IME and started typing. On the broken build, the React composer
    # drops these events because isComposing is still true.
    stuck_value = composer.evaluate(
        """(el) => {
            el.focus();
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(el, 'abc');
            el.dispatchEvent(new InputEvent('input', {
                bubbles:true, inputType:'insertText', data:'abc',
            }));
            return el.value;
        }"""
    )
    # Add one more keystroke to also exercise the keyboard path -- if the
    # textarea unblocked via the input dispatch but the React-state path
    # is still wedged, this will surface as the textarea showing 'abc'
    # while the React-rendered value lags.
    page.keyboard.type("d")
    after_key = read_value()
    info(f"after_input='abc' readback={stuck_value!r}; after_key='d' readback={after_key!r}")
    shoot("06-stuck-composition-recovery")
    if "abc" not in stuck_value:
        fail(
            "stuck-composition repro: textarea never received the post-composition "
            f"input -- readback {stuck_value!r}. This is the regression from "
            "issue #5318 / before PR #5327."
        )
    if "abcd" not in after_key:
        fail(
            "stuck-composition repro: textarea received 'abc' via dispatched input "
            "but the follow-up keyboard 'd' was dropped -- readback "
            f"{after_key!r}. React state likely still stuck in isComposing=true."
        )
    # Also probe the Send button: if isComposing is still true, the
    # ComposerAction stays disabled (see PR #5327: ComposerAction({
    # disabled: disabled || isComposing })). Existence + enabled state
    # is the cheapest cross-check on React's view of composition state.
    send_btn = page.locator('button[aria-label="Send message"]')
    if send_btn.count() == 0:
        soft_fail("Send button not found after stuck-composition recovery")
    else:
        is_disabled = send_btn.evaluate("(el) => el.disabled === true")
        if is_disabled:
            soft_fail(
                "Send button still disabled after stuck-composition recovery -- "
                "React isComposing state likely never cleared"
            )
        else:
            info("Send button correctly enabled after stuck-composition recovery")
    info("stuck-composition recovery PASS")
    clear()

    # ─────────────────────────────────────────────────────
    # 7. Final state + console.error summary. We do not require
    #    zero console errors -- the change-password redirect
    #    typically emits one or two 401 noise lines while the
    #    page transitions -- but we surface the count so a
    #    regression that produces a new error class is
    #    debuggable from the CI log directly.
    # ─────────────────────────────────────────────────────
    shoot("07-final")
    info(f"page_errors={len(page_errors)} console_errors={len(console_errors)}")
    if page_errors:
        info(f"first page error: {page_errors[0][:200]!r}")
    if console_errors:
        info(f"first console error: {console_errors[0][:200]!r}")

    info(
        f"DONE: ascii=OK paste={len(I18N_SAMPLES)}/{len(I18N_SAMPLES)} "
        f"normal_composition=OK stuck_recovery=OK"
    )
    browser.close()
