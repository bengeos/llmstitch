# llmstitch

A provider-agnostic LLM toolkit with tool calling, skills, and parallel execution.

Stitch together Anthropic, OpenAI, Gemini, Groq, and OpenRouter behind one `Agent` loop. Define tools with a decorator, compose behaviors as skills, and execute tool calls concurrently — all with a tiny, typed core.

## Install

```bash
pip install llmstitch[anthropic]       # just the Anthropic SDK
pip install llmstitch[openai]          # just the OpenAI SDK
pip install llmstitch[gemini]          # just the Gemini SDK
pip install llmstitch[groq]            # just the Groq SDK
pip install llmstitch[openrouter]      # OpenRouter (reuses the openai SDK)
pip install llmstitch[all]             # all five
```

The bare `pip install llmstitch` has zero runtime dependencies — provider SDKs are opt-in extras.

## 30-second example

```python
import asyncio
from llmstitch import Agent, tool
from llmstitch.providers.anthropic import AnthropicAdapter

@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"{city}: 72°F and sunny"

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-opus-4-7",
    system="You are a helpful weather assistant.",
)
agent.tools.register(get_weather)

messages = asyncio.run(agent.run("What's the weather in Tokyo?"))
print(messages[-1].content)
```

## Features

- **Provider-agnostic** — swap `AnthropicAdapter` for `OpenAIAdapter`, `GeminiAdapter`, `GroqAdapter`, or `OpenRouterAdapter` without touching your agent code.
- **Typed `@tool` decorator** — JSON Schema generated from type hints (`Optional`, `Literal`, defaults, async).
- **Parallel tool execution** — when a model returns multiple tool calls in one turn, they run concurrently.
- **Streaming** — `Agent.run_stream()` yields provider-neutral events (`TextDelta`, `ToolUseStart` / `Delta` / `Stop`, `MessageStop`, terminal `StreamDone`) and handles tool execution between turns.
- **Retries** — opt in with a `RetryPolicy`; exponential backoff with jitter, honors `Retry-After` headers, uses each adapter's own transient-error classes.
- **Token counting** — `Agent.count_tokens(prompt)` via native provider endpoints (Anthropic, Gemini).
- **Usage and cost** — `agent.usage` (a `UsageTally`) accumulates tokens, turns, API calls, and retries across a run; `agent.cost()` prices it against a `Pricing` rate card in USD.
- **Observability** — attach an `EventBus` and subscribe (sync callback or async iterator) to per-turn model / tool / usage / stop events. Zero overhead when unused.
- **Cost ceiling** — set `cost_ceiling=` (USD) and the run halts mid-loop if accumulated spend crosses it. Retries don't double-charge.
- **Non-raising `run_with_result()`** — structured `AgentResult` with `stop_reason ∈ {"complete", "max_iterations", "cost_ceiling", "error"}` so service code never has to catch.
- **Concurrency-aware tools** — `@tool(is_read_only=True, is_concurrency_safe=False)` annotates tools; mixed-safety batches run sequentially, all-safe batches parallelize with `asyncio.gather`.
- **Skills** — bundle a system prompt with a set of tools; compose with `.extend()`.
- **PEP 561 typed** — ships with `py.typed`, fully checked under `mypy --strict`.

## Streaming example

```python
import asyncio
from llmstitch import Agent, TextDelta, StreamDone
from llmstitch.providers.anthropic import AnthropicAdapter

async def main() -> None:
    agent = Agent(provider=AnthropicAdapter(), model="claude-opus-4-7")
    async for event in agent.run_stream("Tell me a haiku about streams."):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, StreamDone):
            print(f"\n[stop_reason={event.response.stop_reason}]")

asyncio.run(main())
```

## Retries

```python
from llmstitch import Agent, RetryPolicy
from llmstitch.providers.anthropic import AnthropicAdapter

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-opus-4-7",
    retry_policy=RetryPolicy(
        max_attempts=3,
        retry_on=AnthropicAdapter.default_retryable(),
    ),
)
```

Transient errors (rate limits, timeouts, connection drops, 5xx) are retried with exponential backoff + jitter; `Retry-After` headers raise the delay floor. Non-retryable exceptions pass through unchanged. Retries cover `Agent.run` (non-streaming) — `run_stream` is not retried in v0.1.3 because deltas may already have been yielded to the caller.

## Token counting

```python
count = await agent.count_tokens("How many tokens is this?")
print(count.input_tokens)
```

Available natively on `AnthropicAdapter` and `GeminiAdapter`. Other adapters raise `NotImplementedError` — llmstitch doesn't estimate with third-party tokenizers, since the counts can disagree with the provider's own.

## Usage and cost

```python
from llmstitch import Agent, Pricing
from llmstitch.providers.anthropic import AnthropicAdapter

agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-opus-4-7",
    pricing=Pricing(input_per_mtok=15.00, output_per_mtok=75.00),  # paste from vendor rate card
)

await agent.run("Summarize the Iliad in three sentences.")

print(agent.usage)         # UsageTally(input_tokens=..., output_tokens=..., turns=1, api_calls=1, retries=0)
print(agent.cost().total)  # USD
```

`agent.usage` accumulates across every `run` / `run_stream` on that agent — tokens (fed by adapters that report usage), `turns` (model responses folded in), `api_calls` (provider invocations), and `retries` (from the retry policy). Call `agent.usage.reset()` to zero the counters between logical sessions, or `usage.cost(other_pricing)` directly to price the same tally against a different rate card. The default `Pricing(1.00, 2.00)` is a placeholder — pass real vendor rates for accurate costs.

## Observability

```python
from llmstitch import (
    Agent, EventBus, Event,
    AgentStopped, ToolExecutionCompleted, UsageUpdated,
)
from llmstitch.providers.anthropic import AnthropicAdapter


def on_event(event: Event) -> None:
    if isinstance(event, ToolExecutionCompleted):
        print(f"{event.call.name} -> {event.result.content!r} in {event.duration_s*1000:.0f}ms")
    elif isinstance(event, UsageUpdated) and event.delta is not None:
        print(f"+{event.delta.get('input_tokens', 0)}in / "
              f"+{event.delta.get('output_tokens', 0)}out "
              f"(total {event.usage.total_tokens})")
    elif isinstance(event, AgentStopped):
        print(f"stop: {event.stop_reason} after {event.turns} turns")


bus = EventBus()
bus.subscribe(on_event)                          # also supports `async for event in bus.stream()`

agent = Agent(provider=AnthropicAdapter(), model="claude-opus-4-7", event_bus=bus)
```

`EventBus` emits frozen dataclasses for every phase of the run — `AgentStarted`, `TurnStarted`, `ModelRequestSent`, `ModelResponseReceived`, `ToolExecutionStarted`/`Completed`, `UsageUpdated`, `AgentStopped`. Subscriber exceptions are swallowed with a `RuntimeWarning` so observers cannot break the agent loop. Events flow through the bus only — they are **not** interleaved into `run_stream`'s `StreamEvent` iterator.

## Cost ceiling and non-raising runs

```python
from llmstitch import Agent, AgentResult, CostCeilingExceeded, Pricing
from llmstitch.providers.anthropic import AnthropicAdapter


agent = Agent(
    provider=AnthropicAdapter(),
    model="claude-opus-4-7",
    pricing=Pricing(input_per_mtok=15.00, output_per_mtok=75.00),
    cost_ceiling=0.50,                           # USD — run halts if spend crosses this
)

result: AgentResult = await agent.run_with_result("Draft a short reply.")

match result.stop_reason:
    case "complete":        print(result.text)
    case "cost_ceiling":    print(f"hit budget: {result.error}")
    case "max_iterations":  print("loop overran")
    case "error":           print(f"crashed: {type(result.error).__name__}")
```

`run_with_result()` never raises — it catches `MaxIterationsExceeded`, `CostCeilingExceeded`, and vendor errors into the returned `AgentResult` (with partial message history, usage, and cost). `run_stream_with_result()` is the streaming variant: same `StreamEvent`s as `run_stream`, then one terminal `AgentResultEvent`.

The `cost_ceiling` check runs after each response is folded into the usage tally and outside the retry wrapper, so retries don't double-charge. `Agent.run()` / `Agent.run_stream()` still raise `CostCeilingExceeded` if you prefer classical error handling.

## More examples

The [`examples/`](examples/) directory has runnable scripts for:

- [`basic.py`](examples/basic.py) — minimal agent with one tool.
- [`skills_demo.py`](examples/skills_demo.py) — composing two `Skill`s with `.extend()`.
- [`streaming.py`](examples/streaming.py) — `Agent.run_stream` with rich event handling.
- [`providers_gallery.py`](examples/providers_gallery.py) — the same agent against every provider.
- [`parallel_tools.py`](examples/parallel_tools.py) — parallel tool execution with order-preserving results.
- [`async_and_timeout.py`](examples/async_and_timeout.py) — async tools, per-call timeout, captured-exception semantics.
- [`retries.py`](examples/retries.py) — `RetryPolicy` with backoff, jitter, and an `on_retry` observability hook.
- [`token_counting.py`](examples/token_counting.py) — `Agent.count_tokens` on Anthropic + Gemini, with graceful fallback on adapters that don't support native counting.
- [`observability.py`](examples/observability.py) — `EventBus` with a structured-logging subscriber that covers every event type.
- [`cost_ceiling.py`](examples/cost_ceiling.py) — `cost_ceiling=` plus `run_with_result()`: one scenario completes, one scenario trips the ceiling; both inspect `result.stop_reason`.
- [`streaming_with_result.py`](examples/streaming_with_result.py) — `run_stream_with_result()`: live `TextDelta` rendering plus a terminal `AgentResultEvent`, with `EventBus` side-channel progress.
- [`tool_concurrency.py`](examples/tool_concurrency.py) — `is_read_only` / `is_concurrency_safe` flags, mixed-safety batches going sequential, all-safe batches going parallel, and a planner/executor split via `registry.read_only_subset()`.

## Guide

See [`GUIDE.md`](GUIDE.md) for a full walkthrough — core concepts, recipes, ten end-to-end agentic application patterns (research assistant, code review agent, support triage, SQL analyst, nested agents, production observability template), best practices, and a full API reference.

## Status

Alpha. MCP support and structured-output helpers are on the roadmap. See [CHANGELOG.md](CHANGELOG.md) for release history and [ARCHITECTURE.md](ARCHITECTURE.md) for a walkthrough of how the library is put together.

## License

MIT
