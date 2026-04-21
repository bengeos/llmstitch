# llmstitch — Project Plan

Turn the in-memory library we've been prototyping (`llmkit`) into a published Python package named **llmstitch** on PyPI, hosted on GitHub, with clean versioning, CI, and a release workflow that doesn't require running `twine` from your laptop at 2am.

---

## 1. Naming and availability

The working name is **llmstitch** — it conveys what the library does (stitching providers, tools, and skills together). Two names now matter, and they should match:

- **PyPI distribution name**: `llmstitch` (what users `pip install`).
- **Python import name**: `llmstitch` (what users `import`).

**Before anything else**, confirm both are free:

- PyPI: open `https://pypi.org/project/llmstitch/` in a browser. If you get a 404, the name is available.
- GitHub: check `https://github.com/<your-username>/llmstitch` is free.

If `llmstitch` is taken, reasonable fallbacks: `llm-stitch`, `stitchkit`, `stitch-llm`. Keep the import name and the project name identical — anything else is confusing for users.

Decisions already made this session:

| Choice | Value |
|---|---|
| Build backend | **Hatchling** (modern, lean, no setup.py) |
| License | **MIT** (permissive, most common for dev tools) |
| Package name | **llmstitch** |

---

## 2. Repository layout

Use the **`src/` layout**. It's not optional — it prevents a whole class of "it works locally but breaks when installed" bugs by forcing the package to be imported from its installed location, not from the working directory.

```
llmstitch/
├── src/
│   └── llmstitch/
│       ├── __init__.py              # re-exports the public API
│       ├── types.py                 # Message, ContentBlock, ToolDefinition, ...
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py              # ProviderAdapter ABC
│       │   ├── anthropic.py
│       │   ├── openai.py
│       │   └── gemini.py
│       ├── tools.py                 # @tool, Tool, ToolRegistry, Skill
│       ├── agent.py                 # Agent class and run loop
│       └── py.typed                 # empty marker file — signals PEP 561 typed package
├── tests/
│   ├── test_tools.py
│   ├── test_skills.py
│   ├── test_registry_concurrency.py
│   └── test_agent_loop.py           # with mocked adapters
├── examples/
│   ├── basic.py
│   └── skills_demo.py
├── docs/                            # optional; see §9
├── .github/
│   └── workflows/
│       ├── ci.yml                   # lint + test on push/PR
│       └── release.yml              # publish to PyPI on tag
├── pyproject.toml                   # all metadata lives here
├── README.md                        # project front page (shown on PyPI)
├── LICENSE                          # MIT text
├── CHANGELOG.md                     # human-readable release notes
├── .gitignore                       # Python + editor standard
└── .pre-commit-config.yaml          # optional but recommended
```

**Why `providers/` becomes a package:** each provider is meaningful on its own and they'll grow. Splitting them into files keeps import cost low (users who only want Anthropic don't load the OpenAI SDK) if you guard imports behind `TYPE_CHECKING` or lazy-load inside the adapter class, which we already do.

---

## 3. `pyproject.toml` — the single source of truth

One file replaces `setup.py`, `setup.cfg`, `MANIFEST.in`, and build config. Here's a complete working template you can drop in:

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "llmstitch"
version = "0.1.0"
description = "A provider-agnostic LLM toolkit with tool calling, skills, and parallel execution."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
  { name = "Your Name", email = "you@example.com" },
]
keywords = ["llm", "anthropic", "openai", "gemini", "tool-calling", "agents", "ai"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Typing :: Typed",
]

# Keep the core dependency list tiny. Provider SDKs are opt-in extras.
dependencies = []

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai    = ["openai>=1.50"]
gemini    = ["google-genai>=0.3"]
all       = ["llmstitch[anthropic,openai,gemini]"]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.23",
  "pytest-cov>=5",
  "ruff>=0.6",
  "mypy>=1.11",
  "pre-commit>=3",
]

[project.urls]
Homepage      = "https://github.com/<your-username>/llmstitch"
Repository    = "https://github.com/<your-username>/llmstitch"
Issues        = "https://github.com/<your-username>/llmstitch/issues"
Changelog     = "https://github.com/<your-username>/llmstitch/blob/main/CHANGELOG.md"

[tool.hatch.build.targets.wheel]
packages = ["src/llmstitch"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
strict = true
```

**Why extras matter:** users installing `pip install llmstitch` get a tiny core with no provider SDKs. `pip install "llmstitch[anthropic]"` adds the Anthropic client. `pip install "llmstitch[all]"` pulls everything. This keeps the install fast for people who only use one provider and avoids version conflicts between SDKs users don't need.

---

## 4. Rename `llmkit` → `llmstitch`

We've been using `llmkit` as the import name throughout the prototype. Before pushing to a repo:

1. Rename the package directory: `llmkit/` → `src/llmstitch/`.
2. Update every internal import: `from .types import ...` stays the same (they're relative); no changes needed inside the package.
3. Update `example.py` and the test scripts: `from llmkit.agent import Agent` → `from llmstitch.agent import Agent`.
4. Populate `src/llmstitch/__init__.py` with a curated public API so users can do `from llmstitch import Agent, tool, Skill` instead of digging through submodules.

Example `__init__.py`:

```python
"""llmstitch — provider-agnostic LLM toolkit."""
from .agent import Agent
from .tools import tool, Tool, Skill, ToolRegistry
from .types import (
    Message, Role,
    TextBlock, ToolUseBlock, ToolResultBlock,
    ToolDefinition, CompletionResponse,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "tool", "Tool", "Skill", "ToolRegistry",
    "Message", "Role",
    "TextBlock", "ToolUseBlock", "ToolResultBlock",
    "ToolDefinition", "CompletionResponse",
]
```

---

## 5. Required files beyond the code

### `README.md`

This is what PyPI shows on your project page. It's the closest thing the library has to a sales pitch. Structure:

1. One-sentence description.
2. Install command.
3. 30-second "hello world" — the shortest working example.
4. Feature list (provider swapping, tool calling, skills, parallel execution).
5. Link to full docs.
6. License line.

Keep it short. Users scroll. Long READMEs don't get read.

### `LICENSE`

Drop the exact MIT text in, replace `[year]` and `[fullname]`. GitHub's "Add file → Choose a license template" does this automatically when creating the repo.

### `CHANGELOG.md`

Follow [Keep a Changelog](https://keepachangelog.com) format. Every release gets a section. Release notes are your gift to your future self — "why did we bump to 0.3 again?" is answered here.

### `.gitignore`

Use GitHub's Python template as a starting point. Essentials: `__pycache__/`, `*.pyc`, `.venv/`, `dist/`, `build/`, `*.egg-info/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `.env`.

---

## 6. Versioning strategy

Use **Semantic Versioning** (`MAJOR.MINOR.PATCH`):

- `0.x.y` while the API can still change — break things freely, just bump the minor.
- Bump to `1.0.0` when you commit to API stability.
- After `1.x`: patch for bugfixes, minor for backward-compatible features, major for breaking changes.

**Version lives in two places** if you're not careful. Avoid that — pick one and have the other read it. Simplest approach with Hatch: version lives in `pyproject.toml`, and `__init__.py` reads it back via `importlib.metadata.version("llmstitch")`. Or use `hatch-vcs` to derive the version from git tags — no manual bumps, the tag *is* the version.

Recommended: start with `0.1.0` in `pyproject.toml`, hand-bump for the first few releases, move to `hatch-vcs` once you're past the novelty phase.

---

## 7. Testing

**Before every release**, the test suite must pass. Minimum coverage to aim for:

- `test_tools.py` — `@tool` schema generation for typed functions, including defaults, `Optional`, and async functions.
- `test_skills.py` — both Skill construction styles, `.extend()`, prompt merging.
- `test_registry_concurrency.py` — the test we already wrote (concurrent fan-out, serial phase, timeout, order preservation).
- `test_agent_loop.py` — mock the `ProviderAdapter` and assert the loop terminates, tool results feed back correctly, `max_iterations` is respected.

Run with:
```bash
pytest -v --cov=llmstitch --cov-report=term-missing
```

**No live API calls in CI.** Ever. They're slow, flaky, cost money, and expose keys. Every provider adapter test uses a fake adapter. If you want integration tests against real providers, put them in a separate `tests/integration/` folder, gate them on an environment variable, and run them manually.

---

## 8. CI — GitHub Actions

Two workflows:

### `.github/workflows/ci.yml` — runs on every push and PR

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: mypy src
      - run: pytest -v
```

### `.github/workflows/release.yml` — runs when you push a git tag like `v0.1.0`

```yaml
name: Release
on:
  push:
    tags: ["v*"]
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi                # PyPI "Trusted Publisher" environment
    permissions:
      id-token: write                # required for OIDC publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Notice there's no API token anywhere. That's **PyPI Trusted Publishing** — PyPI verifies the GitHub Actions job via OIDC rather than a long-lived token. Setup:

1. Create the project on PyPI the first time by publishing manually (see §10).
2. On the project page → "Publishing" → add a Trusted Publisher entry pointing at your GitHub repo + workflow filename + environment name.
3. Subsequent releases just work by pushing a tag.

---

## 9. Documentation

For a v0.1, a thorough README plus docstrings is enough. Upgrade paths when you outgrow it:

- **MkDocs + Material theme** — markdown-driven, fast to author, deploys to GitHub Pages with one workflow.
- **Sphinx** — the classic; better for large API references, has autodoc for pulling docstrings automatically. More setup.

Either way, put docs under `docs/` and wire them into a separate `docs.yml` workflow that deploys to GitHub Pages. Not a day-one concern.

---

## 10. The actual release process

### One-time setup

1. Create the GitHub repo. Push the code.
2. Register for a PyPI account at `https://pypi.org/account/register/`.
3. Enable 2FA on PyPI — required for publishing.
4. Register for TestPyPI (`https://test.pypi.org`) too — it's the staging environment.

### First release (manual, to claim the name)

```bash
# Build locally
pip install build twine
python -m build

# This creates dist/llmstitch-0.1.0-py3-none-any.whl and .tar.gz

# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Install from TestPyPI in a clean venv and verify it works
pip install --index-url https://test.pypi.org/simple/ llmstitch

# Real thing
twine upload dist/*
```

Once the project exists on PyPI, configure Trusted Publishing (§8) so you never need `twine upload` again.

### Every subsequent release

```bash
# 1. Bump version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit
git commit -am "Release 0.2.0"

# 4. Tag and push
git tag v0.2.0
git push && git push --tags

# 5. GitHub Actions builds and publishes automatically.
```

---

## 11. Pre-launch checklist

Tick these before pushing the first version:

- [ ] Package name available on PyPI **and** GitHub.
- [ ] `src/llmstitch/` layout in place; all imports updated.
- [ ] `pyproject.toml` complete with metadata, dependencies, extras, URLs.
- [ ] `LICENSE`, `README.md`, `CHANGELOG.md`, `.gitignore` all present.
- [ ] `py.typed` marker file included so type checkers pick up your annotations.
- [ ] Tests pass locally: `pytest -v`.
- [ ] Lints pass: `ruff check src tests`, `mypy src`.
- [ ] `python -m build` succeeds and produces both a wheel and sdist.
- [ ] `twine check dist/*` reports no errors (catches README rendering issues).
- [ ] Installed the built wheel in a fresh venv and imported it; the public API from `__init__.py` works.
- [ ] TestPyPI publish rehearsal done.
- [ ] GitHub Actions CI workflow green on `main`.
- [ ] PyPI Trusted Publisher configured.
- [ ] 2FA enabled on your PyPI account.

---

## 12. Phased rollout

Don't try to ship everything at once. Suggested cadence:

**v0.1.0 — skeleton release.** Types, Anthropic adapter complete, tool decorator, skills, agent loop, parallel execution. Streaming and other providers stubbed with clear `NotImplementedError`. Get the package on PyPI and the repo public.

**v0.2.0 — streaming.** Implement `Agent.stream()` and `AnthropicAdapter.stream()` end-to-end. Add streaming-specific tests.

**v0.3.0 — OpenAI adapter.** Full parity with Anthropic: complete + stream + tools.

**v0.4.0 — Gemini adapter.** Same.

**v0.5.0 — polish.** Retries with backoff, token counting helpers, structured output / JSON mode, better error types.

**v1.0.0 — API freeze.** Once the shape has survived real usage for a few months.

---

## 13. Things worth thinking about, but not yet

- **MCP server support.** Natural next step — the library already has a tool registry; an MCP adapter that registers remote tools as local ones would be a killer feature.
- **Observability.** Hooks for logging, tracing, token/cost accounting. Can be added non-breakingly.
- **A CLI.** `llmstitch chat` — nice to have, low priority. Ship the library first.
- **Conversation persistence.** `Message.to_dict()` / `from_dict()` so history can be serialized to JSON. Fifteen lines of code; add when a user asks.

---

## 14. Security

One non-negotiable: **never commit API keys, `.env` files, or test credentials**. Add `.env` to `.gitignore` on day one. Use `python-dotenv` in examples so users know the pattern.

Also: be aware of the recent `litellm` supply-chain incident — malicious versions were published after a maintainer's PyPI account was compromised. Once `llmstitch` has non-trivial adoption, enabling 2FA everywhere and using Trusted Publishing (no long-lived tokens to leak) is your main defense.

---

## Next step

Say the word and I'll generate the initial repo skeleton: `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, the CI workflows, and the renamed `src/llmstitch/` tree wired up with the code we already have. Then you can `git init`, push to GitHub, and the test-publish rehearsal on TestPyPI should be minutes away.