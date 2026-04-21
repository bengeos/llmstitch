# llmstitch

A provider-agnostic LLM toolkit with tool calling, skills, and parallel execution.

Stitch together Anthropic, OpenAI, Gemini, and Groq behind one `Agent` loop. Define tools with a decorator, compose behaviors as skills, and execute tool calls concurrently — all with a tiny, typed core.

## Install

```bash
pip install llmstitch[anthropic]       # just the Anthropic SDK
pip install llmstitch[openai]          # just the OpenAI SDK
pip install llmstitch[gemini]          # just the Gemini SDK
pip install llmstitch[groq]            # just the Groq SDK
pip install llmstitch[all]             # all four
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

- **Provider-agnostic** — swap `AnthropicAdapter` for `OpenAIAdapter`, `GeminiAdapter`, or `GroqAdapter` without touching your agent code.
- **Typed `@tool` decorator** — JSON Schema generated from type hints (`Optional`, `Literal`, defaults, async).
- **Parallel tool execution** — when a model returns multiple tool calls in one turn, they run concurrently.
- **Skills** — bundle a system prompt with a set of tools; compose with `.extend()`.
- **PEP 561 typed** — ships with `py.typed`, fully checked under `mypy --strict`.

## Status

v0.1.0 alpha. Streaming, retries, and MCP support are on the roadmap. See [CHANGELOG.md](CHANGELOG.md) and the [project plan](plan.md).

## License

MIT
