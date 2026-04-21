# Architecture

This document describes how `llmstitch` is put together — what the moving parts are, how they talk to each other, and which invariants the codebase depends on. It is the companion to `README.md` (user-facing, what the library does) and `CLAUDE.md` (contributor rules, what the codebase enforces). Source-of-truth surface: `src/llmstitch/__init__.py`.

---

## 1. Goals and constraints

`llmstitch` puts Anthropic, OpenAI, Gemini, Groq, and OpenRouter behind a single agent loop with a typed tool-calling layer. Five constraints shape every file in the repo:

1. **Zero-dependency core.** `pip install llmstitch` installs only the library — every vendor SDK is an optional extra (`[anthropic]`, `[openai]`, `[gemini]`, `[groq]`, `[openrouter]`, `[all]`). Adapters **must** import their SDK lazily inside `__init__` or `complete`; never at module top level.
2. **Provider-neutral types at the seam.** `types.py` defines the only dataclasses that cross the `ProviderAdapter` boundary. No vendor SDK type ever leaks into `agent.py` or `tools.py`.
3. **`src/` layout is mandatory.** Prevents "works in editable install, breaks from PyPI" class of bugs.
4. **Strict type-checking with missing-import tolerance.** `mypy --strict`, but `ignore_missing_imports = true` so CI doesn't hard-fail when an optional SDK isn't installed in the matrix.
5. **PyPI Trusted Publishing, no API tokens.** Releases go out via OIDC on PR merge. There is no secret in the repo or in GitHub Actions secrets, and there shouldn't be.

---

## 2. The request pipeline

Every request flows through the same six-stage pipeline, regardless of provider.

```
 caller                Agent.run                  ProviderAdapter                ToolRegistry.run
   │                       │                            │                               │
   │  prompt (str | list)  │                            │                               │
   │──────────────────────▶│                            │                               │
   │                       │  complete(model, msgs,     │                               │
   │                       │    system, tools, ...)     │                               │
   │                       │───────────────────────────▶│                               │
   │                       │                            │  translate_messages()         │
   │                       │                            │  translate_tools()            │
   │                       │                            │  SDK call (async)             │
   │                       │                            │  parse_response() ────┐       │
   │                       │  CompletionResponse        │                       │       │
   │                       │◀───────────────────────────│                       │       │
   │                       │                                                    │       │
   │                       │   if response has tool_uses:                       │       │
   │                       │     run(tool_uses, timeout=...) ──────────────────▶│       │
   │                       │                                                    │       │
   │                       │                               gather() in parallel ─┐      │
   │                       │                                per-call timeout     │      │
   │                       │                                exception → error    │      │
   │                       │     list[ToolResultBlock]                           │      │
   │                       │◀────────────────────────────────────────────────────┘      │
   │                       │                                                            │
   │                       │   append results as user-role message, loop                │
   │                       │   until no tool_uses or max_iterations hit                 │
   │  list[Message]        │                                                            │
   │◀──────────────────────│                                                            │
```

Concretely: `agent.py` orchestrates, `providers/<name>.py` translates, `tools.py` executes. Each stage is independently testable and the seams between them are dataclasses defined in `types.py`.

---

## 3. Module-by-module

### 3.1 `types.py` — the provider-neutral vocabulary

All cross-boundary types live here. Nothing downstream of this file should import vendor SDKs for *types*.

| Type | Shape | Notes |
|---|---|---|
| `Role` | `Literal["system", "user", "assistant", "tool"]` | — |
| `TextBlock` | `text: str` | Frozen, slots. |
| `ToolUseBlock` | `id: str`, `name: str`, `input: dict[str, Any]` | Model → tool. |
| `ToolResultBlock` | `tool_use_id: str`, `content: str`, `is_error: bool` | Tool → model. Errors round-trip as strings so the model can self-correct. |
| `ContentBlock` | `TextBlock \| ToolUseBlock \| ToolResultBlock` | Tagged union of the above. |
| `Message` | `role: Role`, `content: str \| list[ContentBlock]` | Mutable; helpers `text_blocks()`, `tool_uses()`. |
| `ToolDefinition` | `name`, `description`, `input_schema: dict` | JSON-Schema shape; adapters translate to vendor wire format. |
| `CompletionResponse` | `content: list[ContentBlock]`, `stop_reason: str`, `usage: dict \| None`, `raw: Any` | `raw` keeps the vendor response object for escape hatches — `repr=False` so logs stay readable. |

**Rule:** if you find yourself adding a vendor-specific field to `types.py`, you're probably leaking. Add it to an adapter instead.

### 3.2 `agent.py` — the run loop

One dataclass (`Agent`), one driver method (`run`), one sync wrapper (`run_sync`). The loop (`src/llmstitch/agent.py:27`) does exactly this:

1. Normalize the prompt into a `list[Message]`.
2. Call `provider.complete(...)` with the current history.
3. Append the assistant response to history.
4. If the response contains no `ToolUseBlock`s → return the history.
5. Otherwise run all tool calls via `ToolRegistry.run`, append the `ToolResultBlock`s as a new user-role message, and loop.
6. If `max_iterations` is exhausted without a text-only response, raise `MaxIterationsExceeded` (not a return value — callers should distinguish "loop terminated" from "model didn't stop").

The loop is deliberately dumb: no retries, no backoff. Streaming is a separate entry point (`Agent.run_stream`, added in v0.1.2) — see §7.3.

### 3.3 `tools.py` — decorator, schema generator, registry, skills

Four responsibilities in one file; they share enough helpers that splitting them would cost more than it saves.

**a) `@tool` decorator (`src/llmstitch/tools.py:152`).** Wraps a function into a `Tool` dataclass. Works bare (`@tool`) or with args (`@tool(name=..., description=...)`). The function's type hints become a JSON Schema via `build_schema`; the docstring becomes `description` (summary line) and parameter descriptions (Google-style `Args:` block).

**b) Schema generation (`_type_to_schema` at `src/llmstitch/tools.py:35`).** Handles:
- Primitives (`str`, `int`, `float`, `bool`)
- `Optional[X]` / `X | None` — unwraps and marks the param non-required
- `Literal["a", "b"]` / `Literal[1, 2]` — emits `{type, enum}`
- `list[X]`, `tuple[X, ...]`, `set[X]`, `frozenset[X]` — emits `{type: array, items: ...}`
- `dict[str, V]` — emits `{type: object, additionalProperties: ...}`
- Defaults → non-required; no default and not Optional → required
- Anything else → open `{}` schema (we don't guess)

Async functions are detected via `inspect.iscoroutinefunction` and dispatched natively; sync functions run in a thread (`asyncio.to_thread`) so a blocking tool can't freeze the event loop.

**c) `ToolRegistry.run` (`src/llmstitch/tools.py:203`).** The concurrency contract:
- **Parallel**: all calls in a single turn run under one `asyncio.gather`.
- **Order-preserving**: output list index matches input list index — critical because the model correlates results to calls by position, not only by id.
- **Per-call timeout** (`asyncio.wait_for`): a slow tool doesn't stall the batch; it becomes a `ToolResultBlock(is_error=True, content="Tool 'X' timed out after Ns")`.
- **Exception capture**: any exception becomes a `ToolResultBlock(is_error=True, content=f"{ExcType}: {message}")`. The model sees the error rather than the process crashing, so it can decide to retry, apologize, or give up.
- **Unknown tool name** → same error shape. The registry never raises from `run`.

**d) `Skill` (`src/llmstitch/tools.py:253`).** A named bundle of `(system_prompt, tools)`. Two construction styles:

```python
# Functional
skill = Skill(name="wx", system_prompt="...", tools=[get_weather])

# Class-style — pick up defaults from class attributes
class Weather(Skill):
    name = "wx"
    system_prompt = "..."
    tools = [get_weather]
```

`.extend(other)` returns a new skill with prompts concatenated (`\n\n`), tools merged by `name` (later wins), and names joined with `+`. `.into_registry()` materializes the tools into a `ToolRegistry`.

### 3.4 `providers/base.py` — the adapter contract

`ProviderAdapter` is an ABC with two methods:

```python
async def complete(*, model, messages, system, tools, max_tokens, **kwargs) -> CompletionResponse
async def stream(*, model, messages, system, tools, max_tokens, **kwargs) -> AsyncIterator[StreamEvent]
```

`complete()` is abstract; `stream()` has a default implementation that raises `NotImplementedError` so adapters can opt in. All five in-tree adapters implement both. The `yield` after the raise in the base stub exists only to satisfy the `AsyncIterator` return type for mypy; it's `# pragma: no cover`.

### 3.5 `providers/*.py` — per-vendor adapters

Every adapter has the same shape, split into four concerns:

1. **`__init__`** — construct the async client. Vendor SDK imported lazily here.
2. **`translate_messages`** — provider-neutral `list[Message]` → vendor wire format.
3. **`translate_tools`** — `list[ToolDefinition]` → vendor tool-declaration format.
4. **`parse_response`** — vendor response object → `CompletionResponse`.

`translate_*` and `parse_response` are `@staticmethod`s. That is not incidental — making them static means the translation tests in `tests/test_provider_translation.py` can exercise them with plain dicts and `SimpleNamespace` mocks without ever constructing a real client. **No live API calls in CI, ever.**

Current roster and notable quirks:

| Adapter | SDK | System prompt | Tool call id | Notes |
|---|---|---|---|---|
| `AnthropicAdapter` | `anthropic>=0.40` | Top-level `system=` param; `role="system"` messages are **dropped** in `translate_messages` | Vendor-provided | Messages API shape. |
| `OpenAIAdapter` | `openai>=1.50` | Prepended as `{"role": "system"}`; `role="system"` messages in history are preserved | Vendor-provided (`call_*`) | Chat Completions shape. Tool results become `{role: "tool", tool_call_id, content}` messages. |
| `GeminiAdapter` | `google-genai>=0.3` | `config.system_instruction` | Synthetic `f"{name}_{idx}"` — Gemini's function_response matches by `name`, not id | Role remap: `assistant` → `model`. |
| `GroqAdapter` | `groq>=0.9` | Same as OpenAI | Same as OpenAI | **Subclass of `OpenAIAdapter`.** Wire format identical; only the client constructor differs. |
| `OpenRouterAdapter` | `openai>=1.50` (reused) | Same as OpenAI | Same as OpenAI | **Subclass of `OpenAIAdapter`.** Uses the `openai` SDK with `base_url="https://openrouter.ai/api/v1"`. Optional `http_referer`/`x_title` kwargs become `HTTP-Referer`/`X-Title` headers for OpenRouter app-ranking. |

The Groq and OpenRouter adapters illustrate the extension story: if a provider speaks OpenAI's wire format, subclass `OpenAIAdapter` and override only what's different (typically just the client constructor). Don't reimplement translation.

### 3.6 `__init__.py` — the curated public surface

Everything users should need is re-exported here (`Agent`, `tool`, `Tool`, `Skill`, `ToolRegistry`, the `types.py` dataclasses, `MaxIterationsExceeded`, `__version__`). **Adapters are deliberately *not* re-exported at the top level** — users import `llmstitch.providers.anthropic` explicitly. This keeps the lazy-import guarantee intact: `import llmstitch` never triggers any vendor SDK.

`__version__` is read via `importlib.metadata.version("llmstitch")` with a `"0.0.0+local"` fallback for editable layouts where package metadata may be absent.

---

## 4. Packaging architecture

Locked-in decisions (and the "why" for each):

- **Package name == import name: `llmstitch`.** Predictable; no friction.
- **Build backend: Hatchling.** All metadata in `pyproject.toml`; no `setup.py` / `setup.cfg`.
- **`src/` layout.** Mandatory. Prevents accidental imports from the repo root masking installed-package bugs.
- **Extras map:**
  - `[anthropic]` → `anthropic>=0.40`
  - `[openai]` → `openai>=1.50`
  - `[gemini]` → `google-genai>=0.3`
  - `[groq]` → `groq>=0.9`
  - `[openrouter]` → `openai>=1.50` (reused)
  - `[all]` → all of the above via `llmstitch[anthropic,openai,gemini,groq,openrouter]`
  - `[dev]` → `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`
- **PEP 561 typed package.** `src/llmstitch/py.typed` is an empty marker. Keep it empty.
- **Version source of truth: `pyproject.toml:project.version`.** `__init__.py` reads it back via `importlib.metadata`. Don't hard-code a second copy.
- **`mypy --strict` with `ignore_missing_imports = true`.** Strict on our code; tolerant of optional SDKs being absent in the CI matrix. Don't flip that flag — CI will turn red for users who happen not to install one of the extras.

---

## 5. Testing architecture

- **`tests/conftest.py`** provides `FakeAdapter`, a scripted `ProviderAdapter` that replays a list of `CompletionResponse`s and records every `complete()` call for assertions. It is how every agent-loop test simulates a real model.
- **`tests/test_provider_translation.py`** exercises each adapter's static translation methods with plain dicts / `SimpleNamespace` mocks. These tests never construct a real client, so no API key is required and no network call is possible.
- **`pytest-asyncio` is in `asyncio_mode = "auto"`** (`pyproject.toml`), so async tests don't need `@pytest.mark.asyncio`.
- **No live API calls in CI, ever.** If real-provider integration tests are ever added, isolate them under `tests/integration/` and gate on an environment variable.

---

## 6. Release architecture

Two workflows under `.github/workflows/`:

- **`ci.yml`** — runs on pull request only. Installs `[dev,anthropic,openai,gemini,groq,openrouter]` so every adapter's translation tests can import its SDK. Runs `ruff check`, `ruff format --check`, `mypy src`, and `pytest --cov` across Python 3.10–3.13.
- **`release.yml`** — fires when a PR merges into `main`. Single source of truth for versioning: it reads `project.version` from `pyproject.toml`.

Release flow:

```
PR merges to main
    │
    ▼
 release.yml reads project.version from pyproject.toml  (say: 0.1.2)
    │
    ▼
 Does tag v0.1.2 already exist?
    │                         ┌─── yes ──▶ skip publish (idempotent re-run safety)
    │
    └─── no ──▶ python -m build
                  │
                  ▼
               PyPI Trusted Publishing (OIDC, environment `pypi`)
                  │
                  ▼
               git tag v0.1.2 on the merge commit
                  │
                  ▼
               gh release create (notes extracted from CHANGELOG.md
                                  `## [0.1.2]` section; wheel + sdist attached)
```

Key invariants:
- **No PyPI tokens anywhere.** Not in the repo, not in Actions secrets. Trusted Publishing + OIDC only.
- **Permissions**: `publish` job needs `id-token: write` (for OIDC) and `contents: write` (to push the tag and create the release). Don't drop either.
- **GitHub Release uses the default `GITHUB_TOKEN`** — no personal access token.
- **First-release bootstrap** was done via PyPI's "pending publisher" flow: owner `bengeos`, repo `llmstitch`, workflow `release.yml`, environment `pypi` registered at `https://pypi.org/manage/account/publishing/` *before* the first merge. After v0.1.0, this is a no-op for subsequent releases.
- **CHANGELOG format is load-bearing.** The release-notes extractor keys off `## [<version>]` headings verbatim. Keep the format exact.

Normal contributor flow for a release:
1. Open a PR that bumps `version` in `pyproject.toml` and moves the `[Unreleased]` section in `CHANGELOG.md` under `## [<new-version>] - <YYYY-MM-DD>`.
2. Merge to `main`.
3. The workflow does the rest. **No manual `git tag`, `git push --tags`, `twine upload`, or `gh release create` step.**

---

## 7. Extension points

### 7.1 Adding a new provider

Two paths depending on the wire format.

**If the provider is OpenAI-compatible** (OpenRouter, Groq, Together, Anyscale, Fireworks, most compat gateways): subclass `OpenAIAdapter` and override only `__init__`. Example (`src/llmstitch/providers/openrouter.py`):

```python
class OpenRouterAdapter(OpenAIAdapter):
    def __init__(self, api_key: str | None = None, *, base_url: str = ..., **kw) -> None:
        import openai  # lazy
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
            **kw,
        )
```

Nothing else needs to change — `translate_messages`, `translate_tools`, and `parse_response` are inherited.

**If the provider has a novel wire format** (Anthropic, Gemini): subclass `ProviderAdapter` directly and implement all four concerns: `__init__`, `translate_messages`, `translate_tools`, `parse_response`. Follow the Anthropic or Gemini adapter as a template. Then:

1. Add a static-method test per concern in `tests/test_provider_translation.py`.
2. Add an extra in `pyproject.toml` (`[myprovider] = ["my-sdk>=X"]`) and include it in `[all]`.
3. Add the extra to the CI install line in `.github/workflows/ci.yml`.
4. Mention it in `README.md` (install table, feature bullet) and `CHANGELOG.md` under `[Unreleased]`.

### 7.2 Adding a tool

```python
from llmstitch import tool

@tool
def get_weather(city: str, units: Literal["c", "f"] = "f") -> str:
    """Return a canned weather report.

    Args:
        city: Full city name, e.g. "Tokyo".
        units: Temperature unit — "c" or "f".
    """
    ...
```

The decorator produces a `Tool` object. Register with `agent.tools.register(get_weather)` (single) or `register_many([...])` (bulk), or wrap a collection as a `Skill` for reuse across agents. Async tools (`async def`) are dispatched natively; sync tools run in a thread.

### 7.3 Streaming

The streaming surface is parallel to the `complete()` surface — same inputs, but the adapter yields a sequence of `StreamEvent`s instead of returning a single `CompletionResponse`.

**Event union** (in `types.py`):

| Event | Meaning |
|---|---|
| `TextDelta(text)` | Incremental text chunk — render directly in a UI. |
| `ToolUseStart(id, name)` | A new tool-use block is beginning. |
| `ToolUseDelta(id, partial_json)` | A partial JSON fragment of the tool's input. |
| `ToolUseStop(id)` | A tool-use block has finished streaming. |
| `MessageStop(stop_reason, usage)` | The model has finished this message. |
| `StreamDone(response)` | Terminal event carrying the fully-assembled `CompletionResponse`. |

**Contract**: adapters MUST emit exactly one terminal `StreamDone`. `Agent.run_stream` depends on this — it uses the `StreamDone.response` to drive the tool-execution loop without re-accumulating from deltas. Emitting `StreamDone` twice, or not at all, is a bug.

**Per-adapter quirks worth knowing:**
- **Anthropic**: uses `client.messages.stream()` context manager; event types are `content_block_start/delta/stop`, `message_delta`, `message_stop`. We call `stream.get_final_message()` at the end for canonical assembly (more reliable than hand-accumulating deltas).
- **OpenAI / Groq / OpenRouter**: `chat.completions.create(stream=True)` with `stream_options={"include_usage": True}` so the final chunk carries token counts. Tool calls stream interleaved by `index`; the first chunk for a new index carries `id` + `function.name`, later chunks carry `function.arguments` fragments only.
- **Gemini**: `client.aio.models.generate_content_stream(...)`. Text streams progressively, but function calls arrive as **complete** parts (Gemini does not stream function arguments) — so we emit `ToolUseStart` + `ToolUseDelta` (with the full JSON) + `ToolUseStop` back-to-back when a function_call part appears.

**`Agent.run_stream` loop** (`src/llmstitch/agent.py`):

```
for each model turn, up to max_iterations:
    start provider.stream(...)
    async for event:
        if isinstance(event, StreamDone):
            remember event.response
        yield event                       # forward every event to the caller

    append assistant message to history (from StreamDone.response)
    if no tool calls in this turn:
        return
    run tool calls (silent, no events emitted)
    append tool results to history
raise MaxIterationsExceeded
```

Tool results are never streamed — they're computed locally. The stream shows only what the model produces.

**Testing streaming**: `tests/conftest.py` `FakeAdapter.stream()` auto-decomposes scripted `CompletionResponse`s into a plausible event sequence, so existing test fixtures work unchanged for `run_stream` tests. Pass `stream_scripts=[[...events...], ...]` to override with specific event orderings. Per-adapter assembly tests in `tests/test_streaming.py` inject hand-rolled async iterators mimicking each vendor's SDK stream shape — no network and no real SDK objects.

### 7.4 Adding a new `ContentBlock` type

Rare — do this only if every provider has a first-class concept you can't express as text or tool use (e.g. image input). You'd need to:

1. Add a frozen dataclass in `types.py` and extend the `ContentBlock` union.
2. Teach every adapter's `translate_messages` and `parse_response` to emit / consume it. Vendors that don't support it should degrade gracefully (drop the block, or stringify it) rather than raising.
3. Add cross-adapter tests that assert the degradation behavior is consistent.

The cost is linear in the number of adapters — consider whether the feature can live in a single adapter's `**kwargs` passthrough first.

---

## 8. What is *not* in the codebase (yet)

Tracked on the roadmap, not yet implemented:

- **Retries / backoff** — the agent loop does not retry transient provider errors today.
- **Token-counting helpers** — no adapter-level `count_tokens` method.
- **Structured-output helpers** — no JSON-mode / schema-constrained-decoding wrapper.
- **MCP integration** — on the roadmap per the README.

Adding any of these is a v0.2.x (minor-bump) concern and likely requires extending `ProviderAdapter` — do that behind default implementations that degrade gracefully, so existing adapters don't all need to change at once.
