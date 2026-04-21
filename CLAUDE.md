# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`llmstitch` is a Python library published to PyPI (distribution name == import name) that puts Anthropic, OpenAI, Gemini, and Groq behind one `Agent` run loop with a typed tool-calling layer. Current release is v0.1.0 alpha; `plan.md` was the original spec and has been removed from the working tree (visible in `git status`). The authoritative surface now is `src/llmstitch/__init__.py`, `README.md`, and `CHANGELOG.md`.

## Common commands

```bash
pip install -e ".[dev,anthropic,openai,gemini,groq]"   # dev install with all extras (mirrors CI)
pytest -v                                         # run tests
pytest -v --cov=llmstitch --cov-report=term-missing
pytest tests/test_tools.py::test_tool_schema_basic  # single test
ruff check src tests
ruff format --check src tests                     # CI enforces formatting too
mypy src                                          # strict mode; see pyproject.toml
python -m build                                   # wheel + sdist into dist/
twine check dist/*                                # validate README rendering before upload
pre-commit install                                # once, to enable ruff + mypy on commit
```

## Locked-in project decisions

- **Package / import name:** `llmstitch` (both must match).
- **Build backend:** Hatchling. All metadata in `pyproject.toml`; no `setup.py` / `setup.cfg`.
- **Layout:** `src/llmstitch/` — the `src/` layout is mandatory (prevents "works locally, breaks installed" bugs).
- **License:** MIT. Python `>=3.10`, tested on 3.10–3.13.
- **Provider SDKs are extras, never core deps.** `pip install llmstitch` installs a zero-dependency core; users opt in with `llmstitch[anthropic]`, `[openai]`, `[gemini]`, `[groq]`, or `[all]`. Every adapter imports its vendor SDK **lazily inside `__init__` or `complete`** so the core import stays light — do not move these imports to module top-level.
- **Typed package:** `src/llmstitch/py.typed` is a PEP 561 marker; keep it empty.
- **Versioning:** SemVer. `pyproject.toml` is the single source of truth; `__init__.py` reads back via `importlib.metadata.version("llmstitch")` with a `PackageNotFoundError` fallback to `"0.0.0+local"` for editable/dev layouts.
- **mypy is `strict = true`** with `ignore_missing_imports = true` — this is intentional: vendor SDKs are optional extras and CI must not hard-fail when one is absent. Don't flip that flag to get stricter errors.

## Architecture

Every request flows through the same pipeline: `Agent.run` → `ProviderAdapter.complete` → translate tool calls → `ToolRegistry.run` (parallel) → feed `ToolResultBlock`s back → repeat until the model stops calling tools or `max_iterations` is hit.

- **`types.py`** — provider-neutral dataclasses: `Message`, `Role`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `ToolDefinition`, `CompletionResponse`. These are the *only* types that cross the adapter boundary; never leak vendor SDK types into `agent.py` or `tools.py`.
- **`providers/base.py`** — `ProviderAdapter` ABC with `complete()` (implemented) and `stream()` (raises `NotImplementedError` on v0.1.0; lands in v0.2.0).
- **`providers/{anthropic,openai,gemini,groq}.py`** — each adapter translates `Message`/`ToolDefinition` to the vendor wire format and parses the response back into `ContentBlock`s. The Anthropic adapter strips `role="system"` messages because the Messages API takes `system` as a top-level param — other adapters may handle system prompts differently, so check the specific `translate_messages` before assuming parity. The Groq adapter subclasses `OpenAIAdapter` (wire format is identical) and only overrides `__init__` to construct an `AsyncGroq` client.
- **`tools.py`** — `@tool` decorator builds a JSON Schema from type hints (handles `Optional`, `Literal`, `list[...]`, `dict[str, V]`, defaults, async functions) plus a best-effort Google-style docstring parser for parameter descriptions. `ToolRegistry.run` dispatches calls via `asyncio.gather`, preserves input order, applies an optional per-call timeout, and captures any exception as a `ToolResultBlock(is_error=True)` — the model sees the error rather than the process crashing. `Skill` bundles a system prompt with tools and supports functional *or* class-style construction plus `.extend()` for prompt/tool merging.
- **`agent.py`** — `Agent.run` is the loop; `run_sync` is a thin `asyncio.run` wrapper. `MaxIterationsExceeded` is raised (not returned) when the loop runs out.
- **`__init__.py`** — curated public surface. Anything users should import lives in `__all__`; users should never need to reach into submodules *except* for the concrete adapters in `llmstitch.providers.<name>` (kept out of the top-level namespace so the lazy-import property is preserved).

## Testing rules

- **No live API calls in CI — ever.** Every adapter test uses the `FakeAdapter` in `tests/conftest.py`, which replays a scripted list of `CompletionResponse`s and records every `complete()` call for assertions. If real-provider integration tests are ever added, isolate them under `tests/integration/` and gate on an env var so they never run in CI.
- `pytest-asyncio` runs in `asyncio_mode = "auto"` (set in `pyproject.toml`), so async tests don't need the `@pytest.mark.asyncio` decorator.

## Release process

- **CI** (`.github/workflows/ci.yml`) runs ruff + ruff-format + mypy + pytest on Python 3.10–3.13 for every pull request. It installs `".[dev,anthropic,openai,gemini,groq]"` so adapter tests can import vendor SDKs if needed.
- **Release** (`.github/workflows/release.yml`) fires when a PR is merged into `main` and publishes via **PyPI Trusted Publishing** (OIDC, environment `pypi`) — there are no API tokens in the repo or GitHub secrets, and there should never be. The workflow reads `project.version` from `pyproject.toml` and publishes *only* if the matching `v<version>` git tag does not already exist; if it does, the publish job is skipped. After a successful PyPI upload, the workflow tags the merge commit and creates a GitHub Release (notes extracted from the `## [<version>]` section of `CHANGELOG.md`, with the built wheel + sdist attached as release assets).
- Normal flow: open a PR that bumps `version` in `pyproject.toml` and updates `CHANGELOG.md` (add a `## [<new-version>]` section — the release-notes extractor keys off that exact format) → merge to `main` → the workflow builds + publishes + tags + creates the GitHub Release. No manual `git tag`/`git push --tags`/`gh release create` step.
- Because the workflow needs to push a tag, create a release, and fetch an OIDC token, the `publish` job has `permissions: { id-token: write, contents: write }`. Don't drop either. The `gh release` step uses the default `GITHUB_TOKEN`; no personal access token is involved.
- The very first release uses Trusted Publishing's **pending publisher** feature — configure owner `bengeos`, repo `llmstitch`, workflow `release.yml`, environment `pypi` at `https://pypi.org/manage/account/publishing/` *before* the first merge, so PyPI accepts the OIDC request for a project that doesn't exist yet. No `twine upload` step is needed.

## Roadmap notes

v0.1.0 shipped Anthropic, OpenAI, and Gemini adapters (this diverges from the earlier plan, which intended Anthropic-only in 0.1.0 — don't re-gate the others). A Groq adapter landed post-0.1.0 as a thin `OpenAIAdapter` subclass. `stream()` raising `NotImplementedError` is the actual 0.1.0 limitation; streaming, retries/backoff, token counting helpers, and structured-output helpers are all still on the roadmap.
