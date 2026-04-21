"""Pre-flight token counting via native provider endpoints.

`Agent.count_tokens(prompt)` forwards to the adapter's `count_tokens`, which
hits the provider's native counting endpoint — no generation, no response
tokens, just "how many input tokens would this request cost?" Useful for
cost estimation, budget gating, and picking a model based on whether the
input fits its context window.

Coverage today (v0.1.3):

    Anthropic   native — client.messages.count_tokens
    Gemini      native — client.aio.models.count_tokens
                  (vendor limitation: counts `messages` only, not tools/system)
    OpenAI      NotImplementedError — llmstitch does not estimate with
    Groq        third-party tokenizers whose counts may disagree with the
    OpenRouter  provider's own.

Usage:
    pip install "llmstitch[anthropic,gemini,openai]"
    ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... python examples/token_counting.py
"""

from __future__ import annotations

import asyncio
import os

from llmstitch import Agent, TokenCount, tool
from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.providers.gemini import GeminiAdapter
from llmstitch.providers.openai import OpenAIAdapter


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"{city}: 72°F and sunny"


PROMPT = (
    "You're helping me budget a conversation. Explain in two short paragraphs "
    "what prompt caching is and why it matters for cost."
)


async def count_on(agent: Agent, label: str) -> TokenCount | None:
    """Print the count, or explain gracefully why the adapter can't count."""
    try:
        count = await agent.count_tokens(PROMPT)
    except NotImplementedError as exc:
        print(f"[{label}] count_tokens not supported: {exc}")
        return None
    print(f"[{label}] input_tokens={count.input_tokens}")
    return count


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY and GOOGLE_API_KEY")

    common = dict(
        system="You are a concise technical writer.",
    )

    # Same prompt, three providers — note how counts differ because each
    # provider uses its own tokenizer. The OpenAI call will raise
    # NotImplementedError, which we catch and report.
    anthropic = Agent(provider=AnthropicAdapter(), model="claude-opus-4-7", **common)
    anthropic.tools.register(get_weather)

    gemini = Agent(provider=GeminiAdapter(), model="gemini-2.0-flash", **common)
    gemini.tools.register(get_weather)

    openai = Agent(provider=OpenAIAdapter(), model="gpt-4o-mini", **common)
    openai.tools.register(get_weather)

    await count_on(anthropic, "anthropic")
    await count_on(gemini, "gemini")
    await count_on(openai, "openai")

    # Practical use: gate a run on a token budget.
    budget = 200
    count = await anthropic.count_tokens(PROMPT)
    if count.input_tokens > budget:
        print(f"\n[anthropic] skipping run — {count.input_tokens} > budget={budget}")
    else:
        print(f"\n[anthropic] within budget ({count.input_tokens} <= {budget}); running…")
        history = await anthropic.run(PROMPT)
        print(history[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
