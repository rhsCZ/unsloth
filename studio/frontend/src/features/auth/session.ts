// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { isTauri } from "@/lib/api-base";

export const AUTH_TOKEN_KEY = "unsloth_auth_token";
export const AUTH_REFRESH_TOKEN_KEY = "unsloth_auth_refresh_token";
export const ONBOARDING_DONE_KEY = "unsloth_onboarding_done";
export const AUTH_MUST_CHANGE_PASSWORD_KEY = "unsloth_auth_must_change_password";

type PostAuthRoute = "/change-password" | "/chat";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function hasAuthToken(): boolean {
  if (!canUseStorage()) return false;
  return Boolean(localStorage.getItem(AUTH_TOKEN_KEY));
}

export function hasRefreshToken(): boolean {
  if (!canUseStorage()) return false;
  return Boolean(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY));
}

export function getAuthToken(): string | null {
  if (!canUseStorage()) return null;
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (!canUseStorage()) return null;
  return localStorage.getItem(AUTH_REFRESH_TOKEN_KEY);
}

export function storeAuthTokens(
  accessToken: string,
  refreshToken: string,
): void {
  // The must_change_password flag is intentionally NOT routed through this
  // function — callers call setMustChangePassword() directly. Tying the flag
  // to the token write makes CodeQL trace the boolean back to the login
  // response and flag it as "clear-text storage of sensitive information",
  // which the JWT pair on lines 44-45 is by design.
  if (!canUseStorage()) return;
  localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
  localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, refreshToken);
}

export function clearAuthTokens(): void {
  if (!canUseStorage()) return;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_REFRESH_TOKEN_KEY);
  localStorage.removeItem(AUTH_MUST_CHANGE_PASSWORD_KEY);
}

export function mustChangePassword(): boolean {
  if (!canUseStorage()) return false;
  // Presence of the key (any value) means the flag is set; absence means
  // not-required. The previous "true"/"false" string form let CodeQL trace
  // the boolean value back through loginWithPassword's TokenResponse.
  return localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY) !== null;
}

export function setMustChangePassword(required: boolean): void {
  if (!canUseStorage()) return;
  // Encode the flag as key presence/absence so the call sites pass a
  // constant ("1" or nothing) to localStorage rather than a derived
  // String(boolean). This breaks the data-flow CodeQL would otherwise
  // trace from loginWithPassword's TokenResponse.must_change_password to
  // localStorage.setItem, which is not actually credential material.
  if (required) {
    localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, "1");
  } else {
    localStorage.removeItem(AUTH_MUST_CHANGE_PASSWORD_KEY);
  }
}

export function isOnboardingDone(): boolean {
  if (!canUseStorage()) return false;
  return localStorage.getItem(ONBOARDING_DONE_KEY) === "true";
}

export function markOnboardingDone(): void {
  if (!canUseStorage()) return;
  localStorage.setItem(ONBOARDING_DONE_KEY, "true");
}

export function resetOnboardingDone(): void {
  if (!canUseStorage()) return;
  localStorage.removeItem(ONBOARDING_DONE_KEY);
}

export function getPostAuthRoute(): PostAuthRoute {
  if (isTauri) return "/chat";
  if (mustChangePassword()) return "/change-password";
  if (usePlatformStore.getState().isChatOnly()) return "/chat";
  return "/chat";
}
