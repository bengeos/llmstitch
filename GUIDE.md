# llmstitch Guide

A practical guide to using `llmstitch` to build agentic applications. This document covers the API surface, common recipes, and a gallery of end-to-end agent patterns.

If you are looking for the architectural rationale, see `ARCHITECTURE.md`. If you want a punch list of what changed per release, see `CHANGELOG.md`. This file focuses on **how to use the library**.

---

## Table of contents

1. [Why llmstitch](#1-why-llmstitch)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Core concepts](#4-core-concepts)
5. [Recipes](#5-recipes)
6. [Agentic application patterns](#6-agentic-application-patterns)
7. [Best practices](#7-best-practices)
8. [API reference](#8-api-reference)

---

## 1. Why llmstitch

`llmstitch` puts Anthropic, OpenAI, Gemini, Groq, and OpenRouter behind one `Agent` run loop and a typed tool-calling layer.

- **One loop, five providers.** Swap providers without rewriting your tool code.
- **Typed tools from plain Python functions.** `@tool` builds the JSON Schema from type hints and docstrings.
- **Parallel tool execution.** `ToolRegistry.run` dispatches via `asyncio.gather` and preserves order.
- **Streaming end-to-end.** Every adapter implements `stream()` and yields a typed `StreamEvent` union.
- **Zero-dependency core.** Vendor SDKs are opt-in extras; imports stay lazy.
- **Usage, cost, and retries built in.** `UsageTally`, `Pricing`, `Cost`, `RetryPolicy` are first-class.

---

## 2. Installation

```bash
# Core (zero dependencies)
pip install llmstitch

# Opt into providers
pip install "llmstitch[anthropic]"
pip install "llmstitch[openai]"
pip install "llmstitch[gemini]"
pip install "llmstitch[groq]"
pip install "llmstitch[openrouter]"

# Everything
pip install "llmstitch[all]"
```

Python `>=3.10` (tested on 3.10–3.13).

Each adapter reads its credential from a standard env var by default:

| Provider | Env var |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

You can also pass `api_key="..."` to any adapter constructor.

---

## 3. Quick start

```python
import asyncio
from llmstitch import Agent, tool
from llmstitch.providers.anthropic import AnthropicAdapter


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


async def main() -> None:
    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-sonnet-4-6",
        system="You are a careful calculator.",
    )
    agent.tools.register(add)

    messages = await agent.run("What is 19 + 23?")
    print(messages[-1].text_blocks()[0].text)


asyncio.run(main())
```

Sync variant (for scripts / REPL):

```python
messages = agent.run_sync("What is 19 + 23?")
```

---

## 4. Core concepts

### 4.1 The run loop

Every call to `Agent.run()` executes the same pipeline:

```
Agent.run
   ↓
ProviderAdapter.complete
   ↓
translate tool calls (ToolUseBlock)
   ↓
ToolRegistry.run (parallel, with timeout)
   ↓
feed ToolResultBlock back
   ↓
(repeat until model stops calling tools, or max_iterations)
```

If the loop hits `max_iterations`, `Agent.run` raises `MaxIterationsExceeded`.

### 4.2 Messages and content blocks

`Message.content` is either a `str` (shorthand for a single text block) or a list of `ContentBlock`:

```python
from llmstitch import Message, TextBlock, ToolUseBlock, ToolResultBlock

Message(role="user", content="Hello")
Message(role="assistant", content=[
    TextBlock(text="Let me check."),
    ToolUseBlock(id="t1", name="lookup", input={"q": "weather"}),
])
Message(role="tool", content=[
    ToolResultBlock(tool_use_id="t1", content="72°F, sunny"),
])
```

Helpers on `Message`:

- `message.text_blocks() -> list[TextBlock]`
- `message.tool_uses() -> list[ToolUseBlock]`

### 4.3 Tools

Decorate any function (sync or async). Type hints build the JSON schema; the docstring summary becomes the tool description; `Args:` entries become per-parameter descriptions.

```python
from typing import Literal
from llmstitch import tool


@tool
async def search(query: str, limit: int = 5, source: Literal["web", "docs"] = "web") -> list[str]:
    """Search for documents.

    Args:
        query: What to look for.
        limit: Maximum hits to return.
        source: Which corpus to query.
    """
    ...
```

Supported type hints:

| Hint | JSON Schema |
| --- | --- |
| `int`, `float`, `str`, `bool` | primitive |
| `Optional[T]` / `T \| None` | optional |
| `Literal["a", "b"]` | `{"enum": [...]}` |
| `list[T]`, `tuple[T]`, `set[T]` | `{"type": "array", "items": ...}` |
| `dict[str, T]` | `{"type": "object", "additionalProperties": ...}` |
| defaults in the signature | marks param as not required |
| `async def` | auto-detected, awaited properly |

### 4.4 Skills

A `Skill` bundles a system prompt with a set of tools. Two construction styles are supported.

Functional:

```python
from llmstitch import Skill

weather_skill = Skill(
    name="weather",
    description="Answer weather questions.",
    system_prompt="You are a weather assistant. Always cite the station ID.",
    tools=[get_forecast, get_alerts],
)
```

Class-based:

```python
class WeatherSkill(Skill):
    name = "weather"
    description = "Answer weather questions."
    system_prompt = "You are a weather assistant. Always cite the station ID."
    tools = [get_forecast, get_alerts]
```

Compose with `.extend()`:

```python
combined = weather_skill.extend(flights_skill)
# combined.system_prompt == weather_skill.system_prompt + "\n\n" + flights_skill.system_prompt
# combined.tools merged by name (later wins on conflict)

registry = combined.into_registry()
agent = Agent(provider=..., model=..., system=combined.system_prompt, tools=registry)
```

### 4.5 Providers

All adapters share the same `complete` / `stream` / `count_tokens` contract. The provider-specific wire format stays inside the adapter.

```python
from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.providers.openai import OpenAIAdapter
from llmstitch.providers.gemini import GeminiAdapter
from llmstitch.providers.groq import GroqAdapter
from llmstitch.providers.openrouter import OpenRouterAdapter

AnthropicAdapter(api_key=None, **client_kwargs)
OpenAIAdapter(api_key=None, **client_kwargs)
GeminiAdapter(api_key=None, **client_kwargs)
GroqAdapter(api_key=None, **client_kwargs)
OpenRouterAdapter(
    api_key=None,
    base_url="https://openrouter.ai/api/v1",
    http_referer=None,  # optional ranking header
    x_title=None,       # optional ranking header
    **client_kwargs,
)
```

`GroqAdapter` and `OpenRouterAdapter` subclass `OpenAIAdapter`; Groq uses the `groq` SDK client, OpenRouter uses the `openai` SDK pointed at OpenRouter's base URL.

---

## 5. Recipes

### 5.1 Streaming

```python
from llmstitch import TextDelta, ToolUseStart, StreamDone

async for event in agent.run_stream("Draft a short blog post about elephants."):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolUseStart):
        print(f"\n[calling {event.name}]")
    elif isinstance(event, StreamDone):
        # Terminal event; carries the fully-assembled CompletionResponse for this turn
        pass
```

Tool execution between turns is silent — you'll get one `StreamDone` per model turn, and the stream closes when the model produces no tool calls.

### 5.2 Retries with observability

```python
from llmstitch import Agent, RetryPolicy, RetryAttempt
from llmstitch.providers.anthropic import AnthropicAdapter


def log_retry(attempt: RetryAttempt) -> None:
    print(f"retry #{attempt.attempt} in {attempt.delay:.1f}s ({type(attempt.exc).__name__})")


provider = AnthropicAdapter()
policy = RetryPolicy(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    multiplier=2.0,
    jitter=0.2,
    retry_on=AnthropicAdapter.default_retryable(),
    respect_retry_after=True,
    on_retry=log_retry,
)
agent = Agent(provider=provider, model="claude-sonnet-4-6", retry_policy=policy)
```

Retries apply to `Agent.run()` only. `Agent.run_stream()` is not retried because deltas may already have been yielded.

### 5.3 Token counting

```python
count = await agent.count_tokens("How big is this prompt?")
print(count.input_tokens)
```

Supported natively on Anthropic and Gemini. OpenAI, Groq, and OpenRouter raise `NotImplementedError` in the current release — catch it and fall back to an estimator if you need cross-provider counts.

### 5.4 Cost tracking

```python
from llmstitch import Pricing

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),  # USD per 1M tokens
)

await agent.run("...")
await agent.run("...")

print(agent.usage.total_tokens, "tokens over", agent.usage.turns, "turns")
print(f"Cost: ${agent.cost().total:.4f}")
```

`agent.usage` is an `UsageTally`: counts tokens, turns, api_calls, and retries across the agent's lifetime. Call `agent.usage.reset()` to zero it.

### 5.5 Concurrent tool calls

Tools run in parallel automatically. If you want the model to issue multiple calls at once, tell it so in the system prompt:

```python
agent = Agent(
    provider=...,
    model=...,
    system=(
        "When multiple lookups are independent, issue them in a single turn "
        "so they run in parallel."
    ),
)
```

The registry handles order preservation and per-tool timeout (`tool_timeout=30.0` by default; `None` disables).

### 5.6 Tool error handling

Exceptions and timeouts don't crash the loop — they're captured as `ToolResultBlock(is_error=True, content="...")` and handed back to the model. This means **the model gets to see the error and can recover** (retry with different args, apologize, try another tool). You rarely need try/except inside tools.

### 5.7 Multi-provider swap — same code, five providers

```python
providers = {
    "anthropic": (AnthropicAdapter(), "claude-sonnet-4-6"),
    "openai":    (OpenAIAdapter(),    "gpt-4o-mini"),
    "gemini":    (GeminiAdapter(),    "gemini-2.5-flash"),
    "groq":      (GroqAdapter(),      "llama-3.3-70b-versatile"),
    "openrouter":(OpenRouterAdapter(),"anthropic/claude-sonnet-4.6"),
}

for label, (provider, model) in providers.items():
    agent = Agent(provider=provider, model=model)
    agent.tools.register(add)
    msgs = await agent.run("What is 19 + 23?")
    print(label, "→", msgs[-1].text_blocks()[0].text)
```

### 5.8 Observability with `EventBus`

An `EventBus` gives you a structured view of what the agent is doing — separate from the wire-level `StreamEvent`s emitted by `run_stream`. Pass one to the agent and subscribe with either a synchronous callback or an async iterator; omit it and emission is a compile-time no-op.

```python
from llmstitch import (
    Agent, EventBus, Event,
    AgentStarted, TurnStarted,
    ModelRequestSent, ModelResponseReceived,
    ToolExecutionStarted, ToolExecutionCompleted,
    UsageUpdated, AgentStopped,
    tool,
)
from llmstitch.providers.anthropic import AnthropicAdapter


def log_event(event: Event) -> None:
    if isinstance(event, AgentStarted):
        print(f"[start] model={event.model}")
    elif isinstance(event, TurnStarted):
        print(f"[turn {event.turn}] begin")
    elif isinstance(event, ToolExecutionCompleted):
        print(f"  {event.call.name} -> {event.result.content!r} "
              f"in {event.duration_s*1000:.1f}ms")
    elif isinstance(event, UsageUpdated) and event.delta is not None:
        print(f"  +{event.delta.get('input_tokens', 0)}in "
              f"/+{event.delta.get('output_tokens', 0)}out "
              f"(total {event.usage.total_tokens})")
    elif isinstance(event, AgentStopped):
        print(f"[stop] {event.stop_reason} after {event.turns} turns")


bus = EventBus()
unsubscribe = bus.subscribe(log_event)  # returns an unsubscribe closure

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    event_bus=bus,
)
agent.tools.register(get_weather)
await agent.run("What's the weather in Tokyo?")
```

Async / SSE-style consumption:

```python
async for event in bus.stream():
    # yields every event pushed to the bus until the next AgentStopped,
    # then terminates the iterator.
    ...
```

Event classes (all frozen dataclasses):

| Event | When it fires |
| --- | --- |
| `AgentStarted(prompt, model)` | Top of `run` / `run_stream` / `run_with_result` |
| `TurnStarted(turn)` | Start of each loop iteration, 1-indexed |
| `ModelRequestSent(turn, messages)` | Immediately before `provider.complete(...)` / `.stream(...)` |
| `ModelResponseReceived(turn, response)` | Assembled `CompletionResponse` per turn |
| `ToolExecutionStarted(turn, call)` | Before each tool call in the registry |
| `ToolExecutionCompleted(turn, call, result, duration_s)` | After each tool call, with wall-time |
| `UsageUpdated(turn, usage, delta)` | After each response is folded into the tally |
| `AgentStopped(stop_reason, turns, error=None)` | Terminal; `stop_reason ∈ {"complete", "max_iterations", "cost_ceiling", "error"}` |

A subscriber that raises is swallowed with a `RuntimeWarning` — observers cannot break the agent loop.

**Events are not interleaved into `run_stream`'s `StreamEvent` iterator.** Subscribe to the bus separately if you want both structured observability and live token deltas.

### 5.9 Non-raising runs with `AgentResult`

`Agent.run_with_result(prompt) -> AgentResult` packages every outcome — success, `MaxIterationsExceeded`, `CostCeilingExceeded`, or any vendor exception — into one structured result. Nothing propagates.

```python
from llmstitch import Agent, AgentResult, CostCeilingExceeded

result: AgentResult = await agent.run_with_result("Draft a short reply.")

match result.stop_reason:
    case "complete":
        print(result.text)
    case "max_iterations":
        print("loop did not terminate; partial history in result.messages")
    case "cost_ceiling":
        err: CostCeilingExceeded = result.error  # type: ignore[assignment]
        print(f"hit ceiling ${err.ceiling} after ${err.spent}")
    case "error":
        print(f"agent crashed: {type(result.error).__name__}: {result.error}")
```

`AgentResult` fields: `messages`, `text` (last assistant text), `stop_reason`, `turns`, `usage`, `cost`, `error`.

For the streaming variant:

```python
from llmstitch import AgentResultEvent, StreamEvent, TextDelta

async for event in agent.run_stream_with_result(prompt):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, AgentResultEvent):
        # Terminal event — always the last one yielded.
        print(f"\nstop_reason: {event.result.stop_reason}")
```

`run_stream_with_result` yields the same `StreamEvent`s as `run_stream` and then exactly one terminal `AgentResultEvent` carrying the `AgentResult`.

### 5.10 Enforcing a cost ceiling

Set `cost_ceiling` (USD) on the agent to halt the loop when accumulated spend crosses the budget. The check runs **after** each provider response is folded into `UsageTally` and **outside** the retry wrapper, so retries don't double-charge.

```python
from llmstitch import Agent, Pricing
from llmstitch.providers.anthropic import AnthropicAdapter

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),  # paste from vendor rate card
    cost_ceiling=0.50,  # USD
)
```

When the ceiling is crossed:

- `Agent.run()` raises `CostCeilingExceeded(spent, ceiling)`.
- `Agent.run_with_result()` returns `AgentResult(stop_reason="cost_ceiling", error=CostCeilingExceeded(...), messages=[...partial history...])`.
- `Agent.run_stream_with_result()` yields any pending deltas, an `AgentStopped(stop_reason="cost_ceiling")` event on the bus, and a terminal `AgentResultEvent`.

Because the default `Pricing(1.00, 2.00)` is a placeholder, **pass real rates** if you want the ceiling to mean real dollars. The library deliberately has no built-in per-model rate table — prices change and a stale one in the library would lie.

### 5.11 Tool concurrency flags and read-only subsets

Two flags live on `Tool` and on the `@tool` decorator:

- `is_concurrency_safe: bool = True` — tool is safe to run in parallel with its siblings. Read-only ops, pure computations, and idempotent RPCs fit here.
- `is_read_only: bool = False` — tool is safe for a planning phase that must not mutate state. `registry.read_only_subset()` returns a view containing only these tools.

```python
from llmstitch import tool


@tool(is_read_only=True)
async def list_files(path: str) -> list[str]:
    """List files under `path`. Safe to run anywhere."""
    ...


@tool(is_read_only=True)
async def read_file(path: str) -> str:
    """Read a file's contents."""
    ...


@tool(is_concurrency_safe=False)
async def write_file(path: str, content: str) -> str:
    """Write a file. Racy against concurrent writers — do not parallelize."""
    ...
```

`ToolRegistry.run` inspects the batch: if every resolved tool is concurrency-safe, it fans out with `asyncio.gather` (current behavior); otherwise the whole batch runs sequentially in input order. Unknown tools are treated as unsafe for the decision.

Planner / executor split:

```python
full = build_registry()                 # every tool
planner_tools = full.read_only_subset() # only is_read_only=True tools

planner = Agent(provider=..., model=..., tools=planner_tools,
                system="Propose a plan. You cannot mutate state.")
executor = Agent(provider=..., model=..., tools=full,
                 system="Execute the plan step by step.")

plan = (await planner.run(prompt))[-1].text_blocks()[0].text
final = await executor.run(f"Plan:\n{plan}")
```

The planner can't write even if a crafted prompt asks it to — the write tools aren't in its registry.

---

## 6. Agentic application patterns

This section is deliberately long: the point of `llmstitch` is to make it cheap to assemble real agentic applications, so the patterns are the product. Each example is self-contained and runnable once you drop in the provider + credentials.

### 6.1 Research assistant (web + summarize)

Give the model web-fetch and note-taking tools and let it iterate until it has enough to answer.

```python
import httpx
from llmstitch import Agent, tool
from llmstitch.providers.anthropic import AnthropicAdapter


NOTES: list[str] = []


@tool
async def web_fetch(url: str) -> str:
    """Fetch a URL and return the first 8000 chars of text.

    Args:
        url: Absolute URL to fetch.
    """
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        r = await client.get(url)
        return r.text[:8000]


@tool
def take_note(fact: str, source: str) -> str:
    """Record a fact and its source URL to the research log.

    Args:
        fact: One-sentence fact.
        source: URL the fact came from.
    """
    NOTES.append(f"- {fact}  (source: {source})")
    return f"noted ({len(NOTES)} total)"


@tool
def list_notes() -> list[str]:
    """Return the current research log."""
    return list(NOTES)


async def research(question: str) -> str:
    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-sonnet-4-6",
        system=(
            "You are a research assistant. For each question: "
            "1) fetch 2-3 relevant URLs, 2) record key facts via take_note with sources, "
            "3) stop when list_notes has enough to answer, 4) write a cited summary."
        ),
        max_iterations=15,
    )
    agent.tools.register_many([web_fetch, take_note, list_notes])
    msgs = await agent.run(question)
    return msgs[-1].text_blocks()[0].text
```

Key pattern: a shared mutable scratchpad (`NOTES`) that tools read and write. The model learns to treat it as working memory.

### 6.2 Code review agent

Point the agent at a diff and let it read the surrounding files before commenting.

```python
import subprocess
from pathlib import Path

from llmstitch import Agent, tool


@tool
def git_diff(base: str = "main") -> str:
    """Return the staged diff against a base branch.

    Args:
        base: Branch to diff against.
    """
    return subprocess.check_output(["git", "diff", f"{base}...HEAD"], text=True)


@tool
def read_file(path: str, start: int = 1, end: int = 400) -> str:
    """Read lines [start, end] from a file in the working tree.

    Args:
        path: Repo-relative path.
        start: 1-indexed first line.
        end: 1-indexed last line (inclusive).
    """
    lines = Path(path).read_text().splitlines()
    return "\n".join(f"{i+start}: {ln}" for i, ln in enumerate(lines[start-1:end]))


@tool
def list_tests_for(path: str) -> list[str]:
    """List test files that likely exercise the given source file.

    Args:
        path: Repo-relative path to a source file.
    """
    stem = Path(path).stem
    return [str(p) for p in Path("tests").rglob(f"*{stem}*.py")]


agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    system=(
        "You are a senior reviewer. For each hunk in the diff: read surrounding code, "
        "check related tests, and report concrete issues with file:line references. "
        "Skip stylistic nitpicks."
    ),
    max_iterations=25,
)
agent.tools.register_many([git_diff, read_file, list_tests_for])
review = (await agent.run("Review the current branch.")).pop().text_blocks()[0].text
```

Key pattern: read-only, idempotent tools for a review-like task. The agent can loop freely without risk of side effects.

### 6.3 Customer-support triage

Combine two skills — a lookup skill and an action skill — and route based on the user's message.

```python
from llmstitch import Agent, Skill, tool


@tool
def find_order(email: str, order_ref: str) -> dict:
    """Look up an order by customer email and reference.

    Args:
        email: Customer email.
        order_ref: Order reference like ORD-12345.
    """
    ...


@tool
def order_status(order_id: str) -> str:
    """Return the current shipping status of an order."""
    ...


lookup_skill = Skill(
    name="lookup",
    system_prompt="You can look up orders and shipments. Never apologize without data.",
    tools=[find_order, order_status],
)


@tool
def issue_refund(order_id: str, amount_cents: int, reason: str) -> str:
    """Issue a refund. Returns a refund ID on success.

    Args:
        order_id: The order to refund.
        amount_cents: Refund amount in cents.
        reason: Short reason string logged on the refund.
    """
    ...


@tool
def escalate_to_human(summary: str) -> str:
    """Create a ticket for a human agent and return the ticket ID.

    Args:
        summary: One-paragraph summary of the issue and what you've tried.
    """
    ...


action_skill = Skill(
    name="actions",
    system_prompt=(
        "You may issue refunds up to $50 without escalation. Anything larger or unclear: "
        "call escalate_to_human with a clear summary."
    ),
    tools=[issue_refund, escalate_to_human],
)

support = lookup_skill.extend(action_skill)

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    system=support.system_prompt,
    tools=support.into_registry(),
    max_iterations=10,
)
```

Key pattern: `.extend()` merges skills, including their system prompts. Policy lives in the system prompt of the skill that owns the sensitive tools, so you can ship skills independently and recombine them per product surface.

### 6.4 Data-analyst agent (read-only SQL)

Give the agent introspection tools plus a sandboxed query tool.

```python
import sqlite3
from llmstitch import Agent, tool


DB = sqlite3.connect("warehouse.db")


@tool
def list_tables() -> list[str]:
    """Return every table name in the warehouse."""
    return [r[0] for r in DB.execute("SELECT name FROM sqlite_master WHERE type='table'")]


@tool
def describe_table(name: str) -> list[dict]:
    """Return the column schema for a table.

    Args:
        name: Table name.
    """
    return [
        {"name": r[1], "type": r[2]}
        for r in DB.execute(f"PRAGMA table_info({name})")
    ]


@tool
def run_query(sql: str, limit: int = 50) -> list[dict]:
    """Run a read-only SQL query and return rows as dicts.

    Args:
        sql: A SELECT statement. Non-SELECT statements are rejected.
        limit: Max rows to return.
    """
    stripped = sql.strip().lower()
    if not stripped.startswith("select"):
        raise ValueError("Only SELECT statements are permitted.")
    cur = DB.execute(f"SELECT * FROM ({sql}) LIMIT {limit}")
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


analyst = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    system=(
        "You answer data questions against a SQLite warehouse. "
        "Always list tables first if unsure of the schema. "
        "Prefer one well-scoped query over many. Cite the query in your final answer."
    ),
    max_iterations=12,
)
analyst.tools.register_many([list_tables, describe_table, run_query])

answer = await analyst.run("Which 5 products had the biggest MoM revenue drop in March?")
```

Key pattern: **guardrails live in the tool**. `run_query` rejects non-SELECTs; the model can't bypass that even if the prompt says so. The schema check is also how the model learns to do its own exploration instead of guessing column names.

### 6.5 File-system assistant

A constrained file agent: list/read/search/write, with writes scoped to a workspace dir.

```python
from pathlib import Path
from llmstitch import Agent, tool

WORKSPACE = Path("/tmp/agent-workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)


def _safe(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if not str(p).startswith(str(WORKSPACE)):
        raise ValueError("path escapes workspace")
    return p


@tool
def list_dir(path: str = ".") -> list[str]:
    """List files and folders under a workspace-relative path."""
    return sorted(str(p.relative_to(WORKSPACE)) for p in _safe(path).iterdir())


@tool
def read_file(path: str) -> str:
    """Read a text file from the workspace."""
    return _safe(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write (overwrite) a text file in the workspace."""
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


@tool
def grep(pattern: str, path: str = ".") -> list[str]:
    """Substring-search files under path; returns file:line matches.

    Args:
        pattern: Substring to search for.
        path: Subtree to search.
    """
    hits: list[str] = []
    for f in _safe(path).rglob("*"):
        if not f.is_file():
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if pattern in line:
                hits.append(f"{f.relative_to(WORKSPACE)}:{i}: {line.strip()}")
    return hits[:100]


agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    system="You are a file assistant operating in a sandboxed workspace.",
)
agent.tools.register_many([list_dir, read_file, write_file, grep])
```

Key pattern: **`_safe(path)` is the security boundary**. Every path-taking tool goes through it, so the model can't escape the workspace even if the prompt tries to trick it.

### 6.6 Deep-research planner (plan → execute → verify)

A longer loop where the agent writes a plan, executes each step, and verifies before answering.

```python
PLAN: list[str] = []
RESULTS: dict[str, str] = {}


@tool
def set_plan(steps: list[str]) -> str:
    """Replace the current plan with a new list of steps."""
    PLAN.clear()
    PLAN.extend(steps)
    return f"plan has {len(PLAN)} steps"


@tool
def record_result(step: str, finding: str) -> str:
    """Record what you learned from executing a plan step."""
    RESULTS[step] = finding
    return "recorded"


@tool
def get_state() -> dict:
    """Return the current plan and recorded results."""
    return {"plan": list(PLAN), "results": dict(RESULTS)}


# ...plus any domain tools (web_fetch, run_query, etc.)

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    system=(
        "Work in three phases: (1) call set_plan with a numbered list of steps, "
        "(2) execute each step and call record_result after each, "
        "(3) read get_state and answer. Never skip phases."
    ),
    max_iterations=30,
)
agent.tools.register_many([set_plan, record_result, get_state, web_fetch])
```

Key pattern: the plan and results are **tool-mediated memory**, not hidden state. You can log / replay / resume by persisting `PLAN` and `RESULTS`.

### 6.7 Streaming chatbot with live tool indicators

```python
from llmstitch import TextDelta, ToolUseStart, ToolUseStop, StreamDone

history: list[Message] = []

while True:
    user = input("> ")
    if not user:
        break
    history.append(Message(role="user", content=user))

    print("assistant: ", end="", flush=True)
    async for event in agent.run_stream(history):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolUseStart):
            print(f"\n  [→ {event.name}]", flush=True)
        elif isinstance(event, ToolUseStop):
            print(f"  [✓ {event.id}]", flush=True)
        elif isinstance(event, StreamDone):
            history.append(Message(role="assistant", content=event.response.content))
    print()
```

Key pattern: feed the whole message history back in each turn. The streaming loop yields per-turn `StreamDone` events you can use to update history without re-accumulating deltas yourself.

### 6.8 Multi-agent orchestration (agent-as-tool)

One agent can drive another by wrapping the sub-agent's `run` in a `@tool`.

```python
from llmstitch import Agent, tool
from llmstitch.providers.anthropic import AnthropicAdapter


# Specialist: only knows how to search + summarize
specialist = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    system="You summarize search results crisply. Cite sources.",
)
specialist.tools.register_many([web_fetch, take_note])


@tool
async def research_subagent(question: str) -> str:
    """Delegate a focused research question to the research sub-agent.

    Args:
        question: Self-contained question. The sub-agent has no memory of prior calls.
    """
    msgs = await specialist.run(question)
    return msgs[-1].text_blocks()[0].text


# Orchestrator: plans and delegates
orchestrator = Agent(
    provider=AnthropicAdapter(),
    model="claude-opus-4-7",
    system=(
        "You break complex questions into sub-questions, delegate each to "
        "research_subagent, and synthesize a final answer."
    ),
    max_iterations=10,
)
orchestrator.tools.register(research_subagent)
```

Key pattern: the **boundary between orchestrator and worker is a tool call**. Prompts stay composable, each agent has a narrow system prompt, and you can swap providers/models per role (e.g. Opus orchestrator, Haiku workers for cost).

### 6.9 Cost-aware budget guard

Use `UsageTally` + `Pricing` to stop early if a session blows its budget.

```python
from llmstitch import Agent, Pricing


agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-sonnet-4-6",
    pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),
    max_iterations=50,
)

BUDGET_USD = 0.25
history: list[Message] = []

while True:
    msgs = await agent.run(history or "Start the research.")
    history = msgs
    if agent.cost().total >= BUDGET_USD:
        print(f"Hit budget: ${agent.cost().total:.3f}; stopping.")
        break
    if "FINAL ANSWER" in msgs[-1].text_blocks()[0].text:
        break
```

Key pattern: cost is an **observable** on the agent, not a hidden metric — gate your loop on it. For a stricter in-run budget, set `cost_ceiling=0.25` on the `Agent` itself (see §5.10) — that halts the current `run()` mid-loop instead of after it finishes.

### 6.10 Resilient production agent

Wire retries, a per-tool timeout, and a budget in one place.

```python
from llmstitch import Agent, Pricing, RetryPolicy
from llmstitch.providers.anthropic import AnthropicAdapter


provider = AnthropicAdapter()

agent = Agent(
    provider=provider,
    model="claude-sonnet-4-6",
    system="You are a production support agent.",
    max_iterations=20,
    tool_timeout=15.0,
    max_tokens=4096,
    retry_policy=RetryPolicy(
        max_attempts=4,
        initial_delay=1.0,
        max_delay=20.0,
        retry_on=provider.default_retryable(),
        respect_retry_after=True,
    ),
    pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),
)
```

`provider.default_retryable()` returns that vendor's transient error classes so you don't have to remember which exception types mean "retry me" vs "give up."

### 6.11 Nested agents: a tool that is itself an agent

Use case: **a customer-support assistant that answers "where is my order?" style questions by delegating the database lookup to a SQL-speaking sub-agent.**

The setup in §6.8 wrapped one agent's `run` in a `@tool` as a sketch. This example fleshes that out for a realistic production shape: two agents, two models, two system prompts, two tool registries, and explicit propagation of tokens and cost from the inner agent back to the outer one.

**Why nest at all?**

1. **Separation of voice.** The customer-facing agent must sound warm and careful; the SQL agent is allowed to be terse and technical. Mixing the two in one system prompt usually degrades both.
2. **Separation of context.** Each call to the tool spawns a fresh conversation for the inner agent. The outer loop never sees SQL errors, malformed queries, schema chatter, or the 12 exploratory `describe_table` calls the inner agent needed. The outer model's context window stays clean.
3. **Separation of cost.** The outer agent wants strong writing and judgment (Sonnet / Opus). The inner agent wants speed and cheap tokens for many short, iterative queries (Haiku / Groq Llama). You pick the right model per role instead of paying premium for both.
4. **Separation of trust.** The SQL tool is read-only and the SQL agent is sandboxed to it. The outer agent cannot execute SQL directly even if a crafted user prompt asks it to — there's no SQL tool on its registry.

```python
import sqlite3
from llmstitch import Agent, Pricing, tool
from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.providers.groq import GroqAdapter


DB = sqlite3.connect("commerce.db")


# ---------- Inner agent: SQL specialist ----------

@tool
def list_tables() -> list[str]:
    """Return every table name in the commerce DB."""
    return [r[0] for r in DB.execute("SELECT name FROM sqlite_master WHERE type='table'")]


@tool
def describe_table(name: str) -> list[dict]:
    """Return the column schema for a table."""
    return [{"name": r[1], "type": r[2]} for r in DB.execute(f"PRAGMA table_info({name})")]


@tool
def run_query(sql: str, limit: int = 20) -> list[dict]:
    """Run a read-only SELECT and return rows as dicts.

    Args:
        sql: A single SELECT statement.
        limit: Row cap.
    """
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT statements are permitted.")
    cur = DB.execute(f"SELECT * FROM ({sql}) LIMIT {limit}")
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def build_sql_agent() -> Agent:
    """Fresh SQL agent per call — no memory between lookups."""
    inner = Agent(
        provider=GroqAdapter(),                 # cheap + fast for many small queries
        model="llama-3.3-70b-versatile",
        system=(
            "You answer data questions against a SQLite commerce DB. "
            "Always list_tables and describe_table before your first query. "
            "Prefer one well-scoped SELECT. "
            "Reply with a single short paragraph of plain facts — no SQL, no markdown, "
            "no apology. If the data doesn't exist, say so plainly."
        ),
        max_iterations=8,
        tool_timeout=10.0,
        pricing=Pricing(input_per_mtok=0.59, output_per_mtok=0.79),
    )
    inner.tools.register_many([list_tables, describe_table, run_query])
    return inner


# ---------- Outer agent: customer-support voice ----------

# Tally the inner agent's costs on the outer agent, so the outer loop can
# enforce a single end-to-end budget.
INNER_USAGE_BUDGET_USD = 0.10


@tool
async def fetch_account_facts(customer_id: str, question: str) -> str:
    """Look up facts about a customer's account from the commerce database.

    Use this for any question that needs data (order status, shipping dates,
    refund history, payment methods). Pass a self-contained natural-language
    question — the lookup tool has no memory of previous calls.

    Args:
        customer_id: The customer's ID (always include it in `question` too).
        question: A specific, self-contained question.

    Returns:
        A short paragraph of plain facts, or a message explaining that the
        data isn't available.
    """
    sql_agent = build_sql_agent()
    try:
        msgs = await sql_agent.run(
            f"Customer ID: {customer_id}\nQuestion: {question}"
        )
    except Exception as exc:
        return f"lookup failed: {type(exc).__name__}: {exc}"

    # Propagate tokens + cost from inner → outer so the outer budget is honest.
    outer = fetch_account_facts.__agent__           # see wiring below
    outer.usage.input_tokens += sql_agent.usage.input_tokens
    outer.usage.output_tokens += sql_agent.usage.output_tokens
    outer.usage.api_calls += sql_agent.usage.api_calls

    if sql_agent.cost().total > INNER_USAGE_BUDGET_USD:
        return "lookup aborted: cost budget exceeded"

    return msgs[-1].text_blocks()[0].text


support = Agent(
    provider=AnthropicAdapter(),                    # warm, careful voice
    model="claude-sonnet-4-6",
    system=(
        "You are a customer-support assistant. For any question that needs data, "
        "call fetch_account_facts with the customer_id and a focused question. "
        "Then write a reply that is empathetic, concrete, and under 4 sentences. "
        "Never mention internal tools, databases, or that you looked something up."
    ),
    max_iterations=6,
    pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),
)
support.tools.register(fetch_account_facts)

# Tiny back-reference so the tool can reach the outer agent to fold usage back in.
fetch_account_facts.__agent__ = support            # type: ignore[attr-defined]


reply = await support.run(
    "Customer cust_842: they're asking why order ORD-1180 hasn't shipped yet."
)
print(reply[-1].text_blocks()[0].text)
print(f"\nTotal cost (outer + inner): ${support.cost().total:.4f}")
```

**What happens at runtime**

1. The support agent receives the user turn and decides to call `fetch_account_facts(customer_id="cust_842", question="Why hasn't order ORD-1180 shipped?")`.
2. `fetch_account_facts` spins up a **fresh** SQL agent with its own tool registry and its own system prompt. That agent does its own 3–8-turn loop: `list_tables` → `describe_table("orders")` → `run_query("SELECT … WHERE order_ref = …")` → final paragraph.
3. The SQL agent's usage is folded into `support.usage`. The outer loop only sees one `ToolResultBlock` containing the final paragraph.
4. The support agent composes a warm one-paragraph reply citing the facts, not the source.

**Design notes worth lifting into your own projects**

- **Build a new inner agent per call, not once at import time.** Agents are cheap to construct, and a fresh inner agent means a fresh conversation — no accumulated state from the last customer bleeds in.
- **Return a string from the tool, not a `CompletionResponse`.** The outer model wants a clean plain-text fact, not a content-block structure. The SQL agent's system prompt enforces that shape.
- **Keep the boundary typed.** The outer agent's tool signature (`customer_id: str, question: str`) is the entire public API of the subsystem. Everything inside is implementation detail.
- **Propagate usage upward** (the `outer.usage.input_tokens += ...` lines). Without this, `support.cost()` under-reports and budget guards lie.
- **Handle inner failures as tool errors.** The `try/except` turns an inner-agent crash into a returned string; the outer model sees "lookup failed: …" and can apologize or ask a clarifying question instead of propagating a 500.
- **You can nest more than two levels.** An SRE-style incident-response agent could have a `run_log_investigation` tool whose inner agent in turn exposes a `grep_service_logs` tool backed by yet another agent that knows each service's log format. Same pattern recurses cleanly — though each level adds latency, so keep the tree shallow in practice.

### 6.12 Production-shaped agent: observability, budget, and non-raising

This example ties the four v0.1.4 features together — `EventBus`, `run_with_result`, `cost_ceiling`, and the tool concurrency flags — into one production-shaped agent. Use it as a template when wiring an agent behind a service.

```python
import logging
import time

from llmstitch import (
    Agent, AgentResult, AgentStopped, CostCeilingExceeded, Event, EventBus,
    ModelResponseReceived, Pricing, RetryPolicy, ToolExecutionCompleted,
    TurnStarted, UsageUpdated, tool,
)
from llmstitch.providers.anthropic import AnthropicAdapter


log = logging.getLogger("research-agent")


# ---------- Tools ----------

@tool(is_read_only=True)
async def web_fetch(url: str) -> str:
    """Fetch a URL and return its text (capped)."""
    ...


@tool(is_read_only=True, is_concurrency_safe=True)
async def search(query: str) -> list[str]:
    """Return top search hits."""
    ...


@tool(is_concurrency_safe=False)
async def append_note(text: str) -> str:
    """Append a finding to the shared research log (not safe to run concurrently)."""
    ...


# ---------- Observability adapter ----------

class RunMetrics:
    """Converts bus events into structured logs + per-run summary stats."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.started_at = time.time()
        self.tool_timings: list[tuple[str, float]] = []

    def __call__(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            log.info("run=%s turn=%d start", self.run_id, event.turn)
        elif isinstance(event, ModelResponseReceived):
            log.info("run=%s turn=%d response_blocks=%d",
                     self.run_id, event.turn, len(event.response.content))
        elif isinstance(event, ToolExecutionCompleted):
            self.tool_timings.append((event.call.name, event.duration_s))
            if event.result.is_error:
                log.warning("run=%s tool=%s ERROR %s",
                            self.run_id, event.call.name, event.result.content)
        elif isinstance(event, UsageUpdated) and event.delta is not None:
            log.info("run=%s usage_total=%d", self.run_id, event.usage.total_tokens)
        elif isinstance(event, AgentStopped):
            log.info("run=%s stop=%s turns=%d elapsed=%.2fs",
                     self.run_id, event.stop_reason, event.turns,
                     time.time() - self.started_at)


# ---------- Agent ----------

def build_agent(run_id: str) -> tuple[Agent, RunMetrics]:
    bus = EventBus()
    metrics = RunMetrics(run_id)
    bus.subscribe(metrics)

    provider = AnthropicAdapter()
    agent = Agent(
        provider=provider,
        model="claude-sonnet-4-6",
        system=(
            "You are a research agent. Fetch, summarize, and cite. "
            "Use append_note after each new finding."
        ),
        max_iterations=20,
        tool_timeout=15.0,
        retry_policy=RetryPolicy(
            max_attempts=4,
            initial_delay=1.0,
            max_delay=20.0,
            retry_on=provider.default_retryable(),
            respect_retry_after=True,
        ),
        pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),
        cost_ceiling=0.75,  # hard budget for this run
        event_bus=bus,
    )
    agent.tools.register_many([web_fetch, search, append_note])
    return agent, metrics


# ---------- Entry point ----------

async def handle_request(run_id: str, prompt: str) -> dict:
    agent, metrics = build_agent(run_id)
    result: AgentResult = await agent.run_with_result(prompt)

    payload = {
        "run_id": run_id,
        "stop_reason": result.stop_reason,
        "turns": result.turns,
        "total_tokens": result.usage.total_tokens,
        "cost_usd": round(result.cost.total, 6) if result.cost else None,
        "tool_timings_ms": [
            {"tool": name, "ms": round(dur * 1000, 1)}
            for name, dur in metrics.tool_timings
        ],
        "text": result.text,
    }
    if isinstance(result.error, CostCeilingExceeded):
        payload["budget"] = {"spent": result.error.spent, "ceiling": result.error.ceiling}
    elif result.error is not None:
        payload["error"] = f"{type(result.error).__name__}: {result.error}"
    return payload
```

**What this gives you**

- **No leaked exceptions.** `run_with_result` folds every failure mode into `AgentResult`. The service layer just inspects `stop_reason`.
- **Structured logs.** `RunMetrics` is the single place where events become log lines. Swap it for a `StatsD` / OpenTelemetry emitter without touching the agent code.
- **Hard budget.** `cost_ceiling=0.75` means no single call can exceed 75 ¢. Retries don't double-charge because the ceiling is checked outside the retry wrapper.
- **Tool-level timings.** `ToolExecutionCompleted.duration_s` lets you attribute latency without wrapping each tool.
- **Concurrency safety.** `search` and `web_fetch` fan out; `append_note` forces the batch sequential — no racing writes on the shared log.
- **Retries on transient failures.** The `RetryPolicy` uses `provider.default_retryable()` and bumps `usage.retries` automatically.

---

## 7. Best practices

**Tools**

- **Tool docstrings matter.** The model picks tools using the description and parameter descriptions. Write them as if for another engineer who can't see the code.
- **Keep tools small and composable.** One verb, one noun. Multiple small tools beat one mega-tool — the registry runs them in parallel anyway.
- **Guardrails live in tools, not prompts.** Validate inputs, scope paths, reject non-SELECT SQL. Prompts drift; code doesn't.
- **Let tool errors propagate.** They become `ToolResultBlock(is_error=True)` and the model sees them — don't swallow them with try/except returning fake success.
- **Prefer async for I/O.** Tools share an event loop and run in parallel; a blocking `requests.get` will serialize them.

**Agents**

- **Set `max_iterations` deliberately.** 10 is fine for simple Q&A; raise it for research/planning loops, but set it — infinite loops are real.
- **Set `tool_timeout`.** Default is 30s; tighten for fast tools, or set `None` only when you trust the tool.
- **Write system prompts in the second person, imperative voice.** "You are X. For each request: 1) …, 2) …, 3) …" works better than a description.
- **Measure usage.** `agent.usage` + `agent.cost()` make it cheap to catch regressions.

**Providers**

- **Provider parity is not perfect.** OpenAI/Groq/OpenRouter don't have native `count_tokens` in this release. Anthropic strips `role="system"` messages (handled internally). Test any cross-provider claim against the adapter you're shipping.
- **Groq and OpenRouter are `OpenAIAdapter` subclasses.** If OpenAI wire-format changes break them, they likely break together.
- **OpenRouter ranking headers.** Pass `http_referer` and `x_title` if you want your app to show up on openrouter.ai leaderboards.

**Streaming**

- **`StreamDone` is the contract.** Adapters emit exactly one at the end of each model turn, carrying the full `CompletionResponse`. Rely on that instead of re-accumulating deltas.
- **Streaming isn't retried.** Deltas may already be on the wire; there's no safe rollback. Use non-streaming `run()` if you need retry semantics.

---

## 8. API reference

Quick index. See `src/llmstitch/__init__.py` for the exact `__all__`.

### Agent

```python
Agent(
    provider: ProviderAdapter,
    model: str,
    tools: ToolRegistry = ToolRegistry(),
    system: str | None = None,
    max_iterations: int = 10,
    tool_timeout: float | None = 30.0,
    max_tokens: int = 4096,
    retry_policy: RetryPolicy | None = None,
    pricing: Pricing = Pricing(1.00, 2.00),
    usage: UsageTally = UsageTally(),
    event_bus: EventBus | None = None,     # v0.1.4
    cost_ceiling: float | None = None,     # v0.1.4 — USD
)

async def run(prompt: str | list[Message]) -> list[Message]
async def run_stream(prompt: str | list[Message]) -> AsyncIterator[StreamEvent]
async def run_with_result(prompt: str | list[Message]) -> AgentResult                    # v0.1.4
async def run_stream_with_result(
    prompt: str | list[Message],
) -> AsyncIterator[StreamEvent | AgentResultEvent]                                        # v0.1.4
def run_sync(prompt: str | list[Message]) -> list[Message]
async def count_tokens(prompt: str | list[Message]) -> TokenCount
def cost() -> Cost
```

`run()` / `run_stream()` / `run_sync()` raise `MaxIterationsExceeded` and `CostCeilingExceeded`. The `*_with_result` variants never raise — they fold outcomes into `AgentResult`.

### Tools

```python
@tool
@tool(name=..., description=..., is_read_only=False, is_concurrency_safe=True)

Tool(name, description, input_schema, fn, is_async,
     is_read_only=False, is_concurrency_safe=True)    # v0.1.4 flags
  .definition() -> ToolDefinition
  async call(**kwargs) -> Any

ToolRegistry()
  .register(item)
  .register_many(items)
  .get(name) -> Tool
  .definitions() -> list[ToolDefinition]
  .read_only_subset() -> ToolRegistry              # v0.1.4 — filtered view
  async run(calls, *, timeout=None,
            on_start=None, on_complete=None) -> list[ToolResultBlock]

Skill(name=..., description=..., system_prompt=..., tools=[...])
  .extend(other) -> Skill
  .into_registry() -> ToolRegistry
```

`ToolRegistry.run` fans out with `asyncio.gather` only when every resolved tool in the batch has `is_concurrency_safe=True`; otherwise it runs sequentially preserving input order.

### Types

```python
Role = Literal["system", "user", "assistant", "tool"]
Message(role, content)
TextBlock(text)
ToolUseBlock(id, name, input)
ToolResultBlock(tool_use_id, content, is_error=False)
ToolDefinition(name, description, input_schema)
CompletionResponse(content, stop_reason, usage=None, raw=None)
  .tool_uses() -> list[ToolUseBlock]
  .text() -> str
```

### Streaming

```python
TextDelta(text)
ToolUseStart(id, name)
ToolUseDelta(id, partial_json)
ToolUseStop(id)
MessageStop(stop_reason, usage=None)
StreamDone(response: CompletionResponse)  # terminal, exactly one per turn
StreamEvent = TextDelta | ToolUseStart | ToolUseDelta | ToolUseStop | MessageStop | StreamDone
```

### Usage, pricing, retries

```python
TokenCount(input_tokens, output_tokens=None, details=None)
Pricing(input_per_mtok, output_per_mtok)
Cost(input_cost, output_cost); .total
UsageTally()
  .add(usage), .record_call(), .record_retry()
  .cost(pricing) -> Cost
  .reset()

RetryPolicy(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=30.0,
    multiplier=2.0,
    jitter=0.2,
    retry_on=(),
    respect_retry_after=True,
    on_retry=None,
)
RetryAttempt(attempt, delay, exc)
```

### Errors (v0.1.4)

```python
MaxIterationsExceeded            # re-exported from llmstitch.agent for BC
CostCeilingExceeded(spent, ceiling)
```

Both live in `llmstitch.errors`; both are re-exported from the top-level package.

### Observability (v0.1.4)

```python
EventBus()
  .subscribe(cb) -> unsubscribe_fn
  .emit(event)
  .stream() -> AsyncIterator[Event]       # SSE-style; ends after next AgentStopped

# Frozen event dataclasses:
AgentStarted(prompt, model)
TurnStarted(turn)
ModelRequestSent(turn, messages)
ModelResponseReceived(turn, response)
ToolExecutionStarted(turn, call)
ToolExecutionCompleted(turn, call, result, duration_s)
UsageUpdated(turn, usage, delta)
AgentStopped(stop_reason, turns, error=None)

StopReason = Literal["complete", "max_iterations", "cost_ceiling", "error"]
Event = AgentStarted | TurnStarted | ModelRequestSent | ModelResponseReceived
      | ToolExecutionStarted | ToolExecutionCompleted | UsageUpdated | AgentStopped
```

A subscriber that raises is swallowed with a `RuntimeWarning` — observers cannot break the agent loop. Events are **not** interleaved into `run_stream`'s `StreamEvent` iterator; subscribe to the bus separately to observe a streaming run.

### Result types (v0.1.4)

```python
AgentResult(messages, text, stop_reason, turns, usage, cost, error)
AgentResultEvent(result: AgentResult)    # terminal event from run_stream_with_result
```

### Providers

```python
AnthropicAdapter(api_key=None, **client_kwargs)
OpenAIAdapter(api_key=None, **client_kwargs)
GeminiAdapter(api_key=None, **client_kwargs)
GroqAdapter(api_key=None, **client_kwargs)                        # subclass of OpenAIAdapter
OpenRouterAdapter(api_key=None, base_url=..., http_referer=None, x_title=None, **client_kwargs)
```

Each adapter implements `complete`, `stream`, and `default_retryable()`. `count_tokens` is implemented on `AnthropicAdapter` and `GeminiAdapter`; the OpenAI-family adapters raise `NotImplementedError`.
