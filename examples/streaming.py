"""Streaming example: render text deltas live and print tool activity.

`Agent.run_stream` yields provider-neutral `StreamEvent`s — text deltas render
straight to stdout, tool-use events report what the model is calling, and a
terminal `StreamDone` per model turn carries the assembled `CompletionResponse`
so you can inspect stop reasons, token usage, or final structure.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/streaming.py
"""

from __future__ import annotations

import asyncio
import os

from llmstitch import (
    Agent,
    MessageStop,
    StreamDone,
    TextDelta,
    ToolUseDelta,
    ToolUseStart,
    ToolUseStop,
    tool,
)
from llmstitch.providers.anthropic import AnthropicAdapter


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"{city}: 72°F and sunny"


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-opus-4-7",
        system="You are a helpful weather assistant. Use the tool when asked about a city.",
    )
    agent.tools.register(get_weather)

    turn = 1
    prompt = "What's the weather in Tokyo? Then describe it poetically."
    async for event in agent.run_stream(prompt):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolUseStart):
            print(f"\n[tool:{event.name} started id={event.id}]", flush=True)
        elif isinstance(event, ToolUseDelta):
            # Arguments stream as JSON fragments — accumulate to parse later.
            print(f"[tool:{event.id} args += {event.partial_json!r}]", flush=True)
        elif isinstance(event, ToolUseStop):
            print(f"[tool:{event.id} stopped]", flush=True)
        elif isinstance(event, MessageStop):
            print(f"\n[turn {turn} stop_reason={event.stop_reason} usage={event.usage}]")
        elif isinstance(event, StreamDone):
            turn += 1


if __name__ == "__main__":
    asyncio.run(main())
