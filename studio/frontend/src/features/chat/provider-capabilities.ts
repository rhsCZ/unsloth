// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Per-provider sampling parameter capability matrix.
 *
 * Values are derived from each provider's published chat-completion docs as of
 * 2026-05. They describe which of our UI knobs map cleanly onto the provider's
 * request body; the panel hides params a provider does not accept so users
 * cannot dial a value that gets silently dropped or rejected.
 *
 * "Local" models (anything that is not an external provider) are represented by
 * a null capability — every knob renders for them.
 */

export interface ProviderCapabilities {
  /** top-k token sampling (only Anthropic on the providers we ship). */
  topK: boolean;
  /** min-p token cutoff (no SaaS provider currently exposes this). */
  minP: boolean;
  /** Repetition penalty (no SaaS provider currently exposes this). */
  repetitionPenalty: boolean;
  /** OpenAI-style presence penalty. */
  presencePenalty: boolean;
}

const OPENAI_COMPAT_BASE: ProviderCapabilities = {
  topK: false,
  minP: false,
  repetitionPenalty: false,
  presencePenalty: true,
};

const ALL_SUPPORTED: ProviderCapabilities = {
  topK: true,
  minP: true,
  repetitionPenalty: true,
  presencePenalty: true,
};

const PROVIDER_CAPABILITIES: Record<string, ProviderCapabilities> = {
  openai: OPENAI_COMPAT_BASE,
  // Anthropic's Messages API accepts top_k but not presence/frequency penalty.
  anthropic: {
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  mistral: OPENAI_COMPAT_BASE,
  gemini: OPENAI_COMPAT_BASE,
  // DeepSeek deprecated presence/frequency penalty in their current docs.
  deepseek: {
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  kimi: OPENAI_COMPAT_BASE,
  qwen: OPENAI_COMPAT_BASE,
  huggingface: OPENAI_COMPAT_BASE,
  // OpenRouter silently drops params the target model does not support, so we
  // surface every knob and let the gateway handle the per-model fan-out.
  openrouter: ALL_SUPPORTED,
  // Custom providers are assumed OpenAI-compatible by the backend; users who
  // point at vLLM/Ollama backends often want top_k / min_p / repetition,
  // so be permissive.
  custom: ALL_SUPPORTED,
};

const DEFAULT_EXTERNAL_CAPABILITIES = OPENAI_COMPAT_BASE;

/**
 * Resolve the capability set for an external provider. Returns `null` for
 * a local model (i.e. when `providerType` is null/undefined), which callers
 * should treat as "every knob applies".
 */
export function getProviderCapabilities(
  providerType: string | null | undefined,
): ProviderCapabilities | null {
  if (!providerType) return null;
  return PROVIDER_CAPABILITIES[providerType] ?? DEFAULT_EXTERNAL_CAPABILITIES;
}
