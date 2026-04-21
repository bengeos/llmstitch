# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
