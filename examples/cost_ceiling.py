"""Cost ceiling + `run_with_result`: cap spend and never raise.

`Agent.run_with_result(prompt)` returns an `AgentResult` regardless of
outcome — no exception escapes. Combined with `cost_ceiling=`, this makes
agent runs budget-safe: the loop halts mid-session if the running cost
crosses the ceiling, and the caller inspects `result.stop_reason`
(`"complete" | "max_iterations" | "cost_ceiling" | "error"`).

What this example shows:

1. **Real pricing.** Pass `pricing=Pricing(...)` so the ceiling compares
   against USD, not the placeholder default. There's no built-in per-model
   rate table — paste numbers from the vendor's pricing page.
2. **Cost-ceiling trip.** The ceiling is checked after each successful
   provider response is folded into `UsageTally`, *outside* the retry
   wrapper. Retries don't double-charge.
3. **Stop reasons.** Pattern-match on `result.stop_reason` to branch cleanly.
4. **Partial results.** On any non-"complete" path, `result.messages` still
   holds whatever was produced before the stop — inspect what the model
   managed to say before the budget ran out.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/cost_ceiling.py
"""

from __future__ import annotations

import asyncio
import os

from llmstitch import (
    Agent,
    AgentResult,
    CostCeilingExceeded,
    Pricing,
    tool,
)
from llmstitch.providers.anthropic import AnthropicAdapter

# Illustrative rates for Claude Haiku 4.5 — verify at https://www.anthropic.com/pricing.
HAIKU_4_5_PRICING = Pricing(input_per_mtok=1.00, output_per_mtok=5.00)


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"{city}: 72°F and sunny"


def render(result: AgentResult) -> None:
    print(f"stop_reason: {result.stop_reason}")
    print(f"turns: {result.turns}  api_calls: {result.usage.api_calls}")
    print(
        f"tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out "
        f"(total {result.usage.total_tokens})"
    )
    if result.cost is not None:
        print(
            f"cost: ${result.cost.total:.6f} "
            f"(in ${result.cost.input_cost:.6f} / out ${result.cost.output_cost:.6f})"
        )
    if isinstance(result.error, CostCeilingExceeded):
        print(f"ceiling: spent ${result.error.spent:.6f} > ${result.error.ceiling:.6f}")
    elif result.error is not None:
        print(f"error: {type(result.error).__name__}: {result.error}")
    if result.text:
        print(f"\nfinal text: {result.text[:200]}{'…' if len(result.text) > 200 else ''}")


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    # --- Path 1: generous ceiling, run completes normally. -----------------
    print("=== Scenario 1: $0.10 ceiling (should complete) ===\n")
    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-haiku-4-5-20251001",
        system="You are a helpful weather assistant.",
        pricing=HAIKU_4_5_PRICING,
        cost_ceiling=0.10,
    )
    agent.tools.register(get_weather)
    result = await agent.run_with_result("What's the weather in Tokyo?")
    render(result)

    # --- Path 2: very tight ceiling, ceiling trips. ------------------------
    # Set to a tiny budget so the very first turn's cost blows past it. The
    # assistant message is still captured in result.messages even though
    # the tool call never got to execute for a second turn.
    print("\n\n=== Scenario 2: $0.0000001 ceiling (should trip) ===\n")
    tight = Agent(
        provider=AnthropicAdapter(),
        model="claude-haiku-4-5-20251001",
        system="You are a helpful weather assistant.",
        pricing=HAIKU_4_5_PRICING,
        cost_ceiling=1e-7,
    )
    tight.tools.register(get_weather)
    result = await tight.run_with_result("Write a 300-word poem about Addis Ababa.")
    render(result)


if __name__ == "__main__":
    asyncio.run(main())
