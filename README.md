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

## More examples

The [`examples/`](examples/) directory has runnable scripts for:

- [`basic.py`](examples/basic.py) — minimal agent with one tool.
- [`skills_demo.py`](examples/skills_demo.py) — composing two `Skill`s with `.extend()`.
- [`streaming.py`](examples/streaming.py) — `Agent.run_stream` with rich event handling.
- [`providers_gallery.py`](examples/providers_gallery.py) — the same agent against every provider.
- [`parallel_tools.py`](examples/parallel_tools.py) — parallel tool execution with order-preserving results.
- [`async_and_timeout.py`](examples/async_and_timeout.py) — async tools, per-call timeout, captured-exception semantics.

## Status

Alpha. Retries and MCP support are on the roadmap. See [CHANGELOG.md](CHANGELOG.md) for release history and [ARCHITECTURE.md](ARCHITECTURE.md) for a walkthrough of how the library is put together.

## License

MIT
