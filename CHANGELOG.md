# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-04-22

### Added
- **Retries / backoff.** New `RetryPolicy` + `RetryAttempt` dataclasses and a `retry_call` helper in `llmstitch.retry`. `Agent` gains a `retry_policy: RetryPolicy | None = None` field; when set, `Agent.run` wraps each `provider.complete(...)` call with exponential backoff (with jitter) and honors `Retry-After` headers. Each adapter exposes `default_retryable()` returning its vendor's transient error classes (rate-limit / timeout / connection / 5xx) so users can wire up a sensible policy in one line: `RetryPolicy(retry_on=AnthropicAdapter.default_retryable(), max_attempts=3)`. Non-retryable exceptions pass through unchanged.
- **Token counting.** New `TokenCount` dataclass and `ProviderAdapter.count_tokens(...)` method (async). `AnthropicAdapter` implements it via `client.messages.count_tokens`; `GeminiAdapter` via `client.aio.models.count_tokens`. `Agent.count_tokens(prompt)` forwards with the agent's system prompt + registered tools. `OpenAIAdapter`, `GroqAdapter`, and `OpenRouterAdapter` inherit the base implementation that raises `NotImplementedError` — we don't estimate with third-party tokenizers that may disagree with the provider's own count.
- **Agent usage tracking.** New `UsageTally` dataclass and an `Agent.usage` field that accumulates across the agent's lifetime. Tracks tokens (`input_tokens`, `output_tokens`, `turns`, `total_tokens`), call activity (`api_calls`, `retries`), and exposes `record_call()` / `record_retry()` / `add()` / `reset()`. Fed by `Agent.run` (bumps `api_calls` per `provider.complete(...)` invocation and `retries` via an auto-wrapped `on_retry` that preserves the user's callback) and `Agent.run_stream` (bumps `api_calls` per stream; streams are not retried). On a successful run with usage-reporting providers, `api_calls == turns + retries`. Not safe for concurrent use on the same agent — same constraint as the run loop itself.
- **Pricing / cost helpers.** New `Pricing` dataclass (`input_per_mtok`, `output_per_mtok` — USD per 1M tokens, matching vendor rate cards) and `Cost` dataclass (`input_cost`, `output_cost`, `total` property). `UsageTally.cost(pricing)` returns a `Cost` breakdown; `Agent` gains a `pricing: Pricing` field (default `Pricing(1.00, 2.00)` — a placeholder, not any real model's rates) and an `Agent.cost()` method that prices the current usage against it. Pass `pricing=Pricing(...)` to the `Agent` constructor for real vendor rates. No built-in per-model rate table — prices change and we don't want a stale source of truth in the library.
- New public exports: `RetryPolicy`, `RetryAttempt`, `TokenCount`, `UsageTally`, `Pricing`, `Cost`.

### Changed
- **Internal:** extracted `_normalize_prompt`, `_provider_kwargs`, `_apply_response` helpers from `Agent.run` / `Agent.run_stream` to remove duplication. `run()` is now 20 lines and `run_stream()` is 29 lines (down from 41 and 44). `count_tokens()` picks up `_normalize_prompt` for free. No behavior change; all public APIs and the full test suite (95 tests) unchanged.

### Notes
- Retries apply to `Agent.run` / `provider.complete(...)` only. Streaming is not retried in v0.1.3: deltas may already have been yielded to the caller before an error surfaces, with no safe way to roll them back. `Agent.run_stream` remains unchanged.
- Gemini's `count_tokens` endpoint takes `contents` only — it does not accept tool declarations or a system instruction, so tokens those contribute are not reflected in the returned count. This is a vendor limitation.

## [0.1.2] - 2026-04-21

### Added
- `OpenRouterAdapter` (`llmstitch.providers.openrouter`) for the OpenRouter API. OpenRouter speaks the OpenAI Chat Completions wire format, so the adapter subclasses `OpenAIAdapter` and only overrides `__init__` to point the `openai` SDK at `https://openrouter.ai/api/v1`, read `OPENROUTER_API_KEY`, and pass optional `HTTP-Referer` / `X-Title` ranking headers. Enabled via the new `llmstitch[openrouter]` extra (reuses `openai>=1.50`; no new SDK dependency) and rolled into `llmstitch[all]`.
- **Streaming.** `ProviderAdapter.stream()` is now implemented on `AnthropicAdapter`, `OpenAIAdapter`, `GeminiAdapter` (and inherited by `GroqAdapter` / `OpenRouterAdapter`), yielding a sequence of `StreamEvent`s — `TextDelta`, `ToolUseStart` / `ToolUseDelta` / `ToolUseStop`, `MessageStop`, and a terminal `StreamDone(CompletionResponse)`. New `Agent.run_stream(prompt)` drives the model → tool → model loop incrementally, yielding events per model turn and running tools silently between turns.
- New public types exported from `llmstitch`: `TextDelta`, `ToolUseStart`, `ToolUseDelta`, `ToolUseStop`, `MessageStop`, `StreamDone`, `StreamEvent`.
- `ARCHITECTURE.md` — contributor-facing walkthrough of the request pipeline, adapter contract, streaming event model, and release flow.
- `examples/streaming.py`, `examples/providers_gallery.py`, `examples/parallel_tools.py`, `examples/async_and_timeout.py` — runnable end-to-end examples for the new streaming surface and pre-existing behaviors.

### Changed
- **Breaking (but safe):** `ProviderAdapter.stream` signature changed from `AsyncIterator[CompletionResponse]` to `AsyncIterator[StreamEvent]`. The prior signature only ever raised `NotImplementedError`, so no working caller can depend on it.
- CI workflow installs the `openrouter` extra alongside the other adapters.
- `tests/conftest.py` `FakeAdapter` now implements `stream()` — by default it auto-decomposes scripted `CompletionResponse`s into a plausible event sequence; pass `stream_scripts=` to script specific event orderings.

## [0.1.1] - 2026-04-21

### Added
- `GroqAdapter` (`llmstitch.providers.groq`) for the Groq Chat Completions API. Subclasses `OpenAIAdapter` since the wire format is identical; only the client construction differs. Enabled via the new `llmstitch[groq]` extra (depends on `groq>=0.9`) and rolled into `llmstitch[all]`.

### Changed
- CI workflow (`.github/workflows/ci.yml`) now runs on pull requests only (the `push: main` trigger was redundant with PR runs) and installs the `groq` extra alongside the other adapters.

## [0.1.0] - 2026-04-21

### Added
- Core types: `Message`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `ToolDefinition`, `CompletionResponse`.
- `@tool` decorator with automatic JSON-Schema generation from Python type hints (`Optional`, `Literal`, defaults, async).
- `ToolRegistry` with concurrent tool execution, order preservation, per-call timeout, and exception capture.
- `Skill` primitive with functional and class-style construction plus `.extend()` for prompt/tool merging.
- `Agent` run loop (`run` / `run_sync`) with configurable `max_iterations` and tool timeout.
- `ProviderAdapter` ABC and three adapters — `AnthropicAdapter`, `OpenAIAdapter`, `GeminiAdapter` — each with lazy SDK imports.
- `py.typed` marker for PEP 561.
- GitHub Actions CI (lint + mypy + pytest on Python 3.10–3.13) and release workflow (PyPI Trusted Publishing on tag).

### Known limitations
- `stream()` raises `NotImplementedError` on every adapter; streaming support lands in v0.2.0.
- No retry/backoff, token counting helpers, or structured-output helpers yet — see `plan.md` for the roadmap.
