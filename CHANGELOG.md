# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
