# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is **pre-implementation**. The only file is `plan.md`, which specifies a Python library to be built and published to PyPI as **`llmstitch`** (distribution name == import name). The prototype code referenced by the plan (named `llmkit`) lives elsewhere and has not yet been copied in. Read `plan.md` before any structural work — it is the authoritative spec for layout, versioning, CI, and release.

## Locked-in decisions (from plan.md)

- **Package / import name:** `llmstitch` (both must match).
- **Build backend:** Hatchling. All metadata in `pyproject.toml`; no `setup.py` / `setup.cfg`.
- **Layout:** `src/llmstitch/` — the `src/` layout is mandatory, not optional (prevents "works locally, breaks installed" bugs).
- **License:** MIT.
- **Python support:** `>=3.10`, tested against 3.10–3.13.
- **Provider SDKs are extras, not core deps:** `pip install llmstitch` installs a tiny core; users opt in via `llmstitch[anthropic]`, `[openai]`, `[gemini]`, or `[all]`. Provider imports inside adapters must be lazy so the core stays light.
- **Typed package:** ship an empty `src/llmstitch/py.typed` marker (PEP 561).
- **Versioning:** SemVer. Start at `0.1.0`; single source of truth is `pyproject.toml`, with `__init__.py` reading back via `importlib.metadata.version("llmstitch")`. Consider `hatch-vcs` later.

## Intended architecture

The library stitches multiple LLM providers behind one interface. Key modules once code is imported:

- `types.py` — `Message`, `Role`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `ToolDefinition`, `CompletionResponse`. Provider-neutral.
- `providers/` — each provider in its own file (`anthropic.py`, `openai.py`, `gemini.py`) behind a `ProviderAdapter` ABC in `base.py`. Lazy-import the vendor SDK inside the adapter.
- `tools.py` — `@tool` decorator, `Tool`, `ToolRegistry`, `Skill`. Decorator generates JSON schema from type hints (handles defaults, `Optional`, async).
- `agent.py` — `Agent` class and the run loop that drives provider → tool calls → feedback → provider until termination or `max_iterations`.
- `__init__.py` — curated public API: `Agent`, `tool`, `Tool`, `Skill`, `ToolRegistry`, and the `types` re-exports. Users should never need to import from submodules.

## Planned commands

Once `pyproject.toml` exists:

```bash
pip install -e ".[dev]"                           # dev install with test/lint tooling
pytest -v                                         # run tests
pytest -v --cov=llmstitch --cov-report=term-missing
pytest tests/test_tools.py::test_name             # single test
ruff check src tests
mypy src
python -m build                                   # build wheel + sdist into dist/
twine check dist/*                                # validate README rendering before upload
```

## Testing rules

- **No live API calls in CI** — ever. All provider adapter tests use a fake `ProviderAdapter`. If real-provider integration tests are added, isolate them under `tests/integration/` and gate on an env var so they never run in CI.
- `pytest-asyncio` with `asyncio_mode = "auto"` (already in the planned `pyproject.toml`).

## Release process

- CI (`.github/workflows/ci.yml`) runs lint + mypy + pytest on push/PR across Python 3.10–3.13.
- Release (`.github/workflows/release.yml`) fires on tags matching `v*` and publishes via **PyPI Trusted Publishing** (OIDC) — no API tokens in the repo or GitHub secrets.
- Release flow: bump version in `pyproject.toml` → update `CHANGELOG.md` → commit → `git tag v0.x.y` → `git push --tags`. GitHub Actions does the rest.
- The very first release must be published manually with `twine upload` to claim the name; Trusted Publishing is configured on the PyPI project page afterward.

## Phased roadmap

v0.1.0 ships Anthropic adapter only; other providers raise `NotImplementedError`. Streaming lands in v0.2.0, OpenAI in v0.3.0, Gemini in v0.4.0. Don't pull work forward across these boundaries unless asked — the plan is explicit about shipping a skeleton first.
