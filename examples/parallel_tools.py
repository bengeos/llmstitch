"""Parallel tool execution with order-preserving results.

When a model returns multiple `ToolUseBlock`s in one turn, llmstitch fans them
out through `asyncio.gather` and collects the results in the original call
order — models correlate tool results to calls positionally, so preserving
order is part of the contract.

Here three "fetch" tools each sleep for a different duration. Wall-clock time
is ~max(durations), not sum(durations), proving the fanout. Output order
matches the call order the model emitted, independent of completion order.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/parallel_tools.py
"""

from __future__ import annotations

import asyncio
import os
import time

from llmstitch import Agent, tool
from llmstitch.providers.anthropic import AnthropicAdapter


@tool
async def fetch_city_population(city: str) -> str:
    """Return a canned population figure for a city (simulates a slow API)."""
    await asyncio.sleep(1.0)
    known = {"Tokyo": 13_960_000, "Paris": 2_100_000, "Cairo": 10_100_000}
    pop = known.get(city, 0)
    return f"{city}: {pop:,} people"


@tool
async def fetch_city_timezone(city: str) -> str:
    """Return a canned timezone for a city (simulates a slow API)."""
    await asyncio.sleep(1.0)
    known = {"Tokyo": "Asia/Tokyo", "Paris": "Europe/Paris", "Cairo": "Africa/Cairo"}
    return f"{city}: {known.get(city, 'UTC')}"


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-haiku-4-5-20251001",
        system=(
            "For each city, call both fetch_city_population AND fetch_city_timezone. "
            "Issue all tool calls in a single turn so they can run in parallel."
        ),
    )
    agent.tools.register(fetch_city_population)
    agent.tools.register(fetch_city_timezone)

    start = time.perf_counter()
    history = await agent.run("Give me the population and timezone for Tokyo, Paris, Addis Ababa, and Cairo.")
    elapsed = time.perf_counter() - start

    print(f"Total wall time: {elapsed:.2f}s (serial would be ~6s)")
    final = history[-1]
    print("Final answer:")                                                                                                                                   
    for block in final.text_blocks():                                                                                                                               
        print(block.text)


if __name__ == "__main__":
    asyncio.run(main())
