"""Run the same Agent against every in-tree provider.

Demonstrates the provider-agnostic property: `AnthropicAdapter`,
`OpenAIAdapter`, `GeminiAdapter`, `GroqAdapter`, and `OpenRouterAdapter` all
satisfy `ProviderAdapter`, so only the constructor and model string change.

Each provider is gated on its API key env var — the example skips any adapter
whose credentials aren't present, so you can run it with whichever subset you
have keys for.

Usage:
    pip install "llmstitch[all]"
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... python examples/providers_gallery.py
"""

from __future__ import annotations

import asyncio
import os

from llmstitch import Agent, tool
from llmstitch.providers.base import ProviderAdapter


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def build_providers() -> list[tuple[str, ProviderAdapter, str]]:
    """(label, adapter, model) for each provider whose API key is present."""
    entries: list[tuple[str, ProviderAdapter, str]] = []

    if os.environ.get("ANTHROPIC_API_KEY"):
        from llmstitch.providers.anthropic import AnthropicAdapter

        entries.append(("anthropic", AnthropicAdapter(), "claude-opus-4-7"))

    if os.environ.get("OPENAI_API_KEY"):
        from llmstitch.providers.openai import OpenAIAdapter

        entries.append(("openai", OpenAIAdapter(), "gpt-4o"))

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        from llmstitch.providers.gemini import GeminiAdapter

        entries.append(("gemini", GeminiAdapter(), "gemini-1.5-pro"))

    if os.environ.get("GROQ_API_KEY"):
        from llmstitch.providers.groq import GroqAdapter

        entries.append(("groq", GroqAdapter(), "llama-3.1-70b-versatile"))

    if os.environ.get("OPENROUTER_API_KEY"):
        from llmstitch.providers.openrouter import OpenRouterAdapter

        # OpenRouter's ranking headers are optional — surfaces your app on
        # openrouter.ai's leaderboards when set.
        adapter = OpenRouterAdapter(
            http_referer="https://example.com",
            x_title="llmstitch-gallery",
        )
        entries.append(("openrouter", adapter, "anthropic/claude-3.5-sonnet"))

    return entries


async def ask_one(label: str, adapter: ProviderAdapter, model: str) -> None:
    agent = Agent(
        provider=adapter,
        model=model,
        system="Use the `add` tool to answer arithmetic.",
    )
    agent.tools.register(add)
    history = await agent.run("What is 17 + 25?")
    final_msg = history[-1]
    print(f"\n[{label}] → {final_msg.content!r}")


async def main() -> None:
    providers = build_providers()
    if not providers:
        raise SystemExit(
            "No provider API keys found. Set at least one of "
            "ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / "
            "GROQ_API_KEY / OPENROUTER_API_KEY."
        )
    # Run sequentially — parallel would hit rate limits and interleave output.
    for label, adapter, model in providers:
        try:
            await ask_one(label, adapter, model)
        except Exception as exc:  # noqa: BLE001 — demo script should keep going
            print(f"[{label}] failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
