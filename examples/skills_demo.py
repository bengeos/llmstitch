"""Skill composition example: two skills extended into one agent.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/skills_demo.py
"""

from __future__ import annotations

import asyncio
import os

from llmstitch import Agent, Skill, tool
from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.types import TextBlock


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


math_skill = Skill(
    name="math",
    description="Arithmetic helpers.",
    system_prompt="You can perform arithmetic using the `add` tool when asked.",
    tools=[add],
)

string_skill = Skill(
    name="strings",
    description="String helpers.",
    system_prompt="You can reverse strings using the `reverse` tool when asked.",
    tools=[reverse],
)


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    combined = math_skill.extend(string_skill)

    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-haiku-4-5-20251001",
        system=combined.system_prompt,
        tools=combined.into_registry(),
    )

    history = await agent.run("What is 17 + 25, and what does 'hello' look like reversed?")
    final = history[-1]
    if isinstance(final.content, list):
        print("".join(b.text for b in final.content if isinstance(b, TextBlock)))
    else:
        print(final.content)


if __name__ == "__main__":
    asyncio.run(main())
