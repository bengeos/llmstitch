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

## Status

Alpha. MCP support and structured-output helpers are on the roadmap. See [CHANGELOG.md](CHANGELOG.md) for release history and [ARCHITECTURE.md](ARCHITECTURE.md) for a walkthrough of how the library is put together.

## License

MIT
