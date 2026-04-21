"""Minimal llmstitch example: one tool, the Anthropic adapter, a single run.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/basic.py
"""

from __future__ import annotations

import asyncio
import os

from llmstitch import Agent, tool
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

    history = await agent.run("What's the weather in Tokyo?")
    final = history[-1]
    if isinstance(final.content, list):
        from llmstitch.types import TextBlock

        print("".join(b.text for b in final.content if isinstance(b, TextBlock)))
    else:
        print(final.content)


if __name__ == "__main__":
    asyncio.run(main())
