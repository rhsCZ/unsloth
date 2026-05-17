// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * sonner `toast` wrapper that defaults `dismissible: false`.
 *
 * sonner's swipe-to-dismiss captures the pointer on mousedown, which
 * blocks browser text selection inside the toast body. Setting
 * `dismissible: false` skips that capture so users can highlight and
 * copy error / log messages. Auto-dismiss, close buttons, and action
 * buttons still work. Callers can opt back in with `dismissible: true`.
 *
 * Drop-in replacement: `import { toast } from "@/lib/toast"`.
 */

import { toast as sonnerToast, type ExternalToast } from "sonner";

type AnyFn = (...args: unknown[]) => unknown;

function withDismissibleFalse<F extends AnyFn>(fn: F): F {
  return ((...args: unknown[]) => {
    // Toast methods are `(message, options?)`. Decide by arity so
    // React-element messages are not mistaken for options.
    if (args.length <= 1) {
      args.push({ dismissible: false } satisfies ExternalToast);
    } else {
      const lastIdx = args.length - 1;
      const last = args[lastIdx];
      if (last && typeof last === "object" && !Array.isArray(last)) {
        const opts = last as ExternalToast;
        if (!("dismissible" in opts)) {
          args[lastIdx] = { dismissible: false, ...opts };
        }
      }
    }
    return fn(...args);
  }) as F;
}

const wrappedCallable = withDismissibleFalse(
  sonnerToast as unknown as AnyFn,
) as typeof sonnerToast;

// `promise`, `dismiss`, `getHistory`, `getToasts` don't take per-toast
// options so they pass through unchanged.
export const toast: typeof sonnerToast = Object.assign(wrappedCallable, {
  success: withDismissibleFalse(sonnerToast.success.bind(sonnerToast) as AnyFn) as typeof sonnerToast.success,
  info: withDismissibleFalse(sonnerToast.info.bind(sonnerToast) as AnyFn) as typeof sonnerToast.info,
  warning: withDismissibleFalse(sonnerToast.warning.bind(sonnerToast) as AnyFn) as typeof sonnerToast.warning,
  error: withDismissibleFalse(sonnerToast.error.bind(sonnerToast) as AnyFn) as typeof sonnerToast.error,
  message: withDismissibleFalse(sonnerToast.message.bind(sonnerToast) as AnyFn) as typeof sonnerToast.message,
  loading: withDismissibleFalse(sonnerToast.loading.bind(sonnerToast) as AnyFn) as typeof sonnerToast.loading,
  custom: withDismissibleFalse(sonnerToast.custom.bind(sonnerToast) as AnyFn) as typeof sonnerToast.custom,
  promise: sonnerToast.promise.bind(sonnerToast) as typeof sonnerToast.promise,
  dismiss: sonnerToast.dismiss.bind(sonnerToast) as typeof sonnerToast.dismiss,
  getHistory: sonnerToast.getHistory.bind(sonnerToast) as typeof sonnerToast.getHistory,
  getToasts: sonnerToast.getToasts.bind(sonnerToast) as typeof sonnerToast.getToasts,
});

export type { ExternalToast } from "sonner";
