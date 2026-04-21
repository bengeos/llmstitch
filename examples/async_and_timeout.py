"""Async tools, per-call timeout, and captured-exception semantics.

Three things this example demonstrates:

1. **Async tools** — `@tool` detects `async def` and dispatches natively.
2. **Per-call timeout** — `Agent(tool_timeout=...)` wraps every tool execution
   in `asyncio.wait_for`. A slow tool doesn't stall the batch; it becomes a
   `ToolResultBlock(is_error=True, content="Tool 'X' timed out after Ns")`
   and the model sees the timeout message rather than the process hanging.
3. **Exception capture** — any exception raised inside a tool is serialized
   into a `ToolResultBlock(is_error=True)` with the exception type and message.
   The model gets a chance to recover (apologize, retry with different args,
   or give up gracefully) instead of the agent crashing.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/async_and_timeout.py
"""

from __future__ import annotations

import asyncio
import os

from llmstitch import Agent, tool
from llmstitch.providers.anthropic import AnthropicAdapter


@tool
async def fast_ping(host: str) -> str:
    """Ping a host and return its latency (fast: ~50ms)."""
    await asyncio.sleep(0.05)
    return f"{host}: 42ms"


@tool
async def slow_ping(host: str) -> str:
    """Ping a host that takes a long time (simulates a 10s hang)."""
    await asyncio.sleep(10.0)
    return f"{host}: eventually"


@tool
def buggy_lookup(record_id: int) -> str:
    """Look up a record (but this tool has a bug and raises for id=0)."""
    if record_id == 0:
        raise ValueError("record_id must be positive")
    return f"record {record_id}: ok"


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-opus-4-7",
        system=(
            "You have three tools: fast_ping, slow_ping, buggy_lookup. "
            "When tools return errors, explain clearly what went wrong."
        ),
        tool_timeout=2.0,  # slow_ping will exceed this and come back as an error.
    )
    agent.tools.register(fast_ping)
    agent.tools.register(slow_ping)
    agent.tools.register(buggy_lookup)

    history = await agent.run(
        "Please do three things: ping fast.example.com using fast_ping, "
        "ping slow.example.com using slow_ping, and look up record 0 using buggy_lookup. "
        "Summarize the results."
    )
    final = history[-1]
    print(final.content)


if __name__ == "__main__":
    asyncio.run(main())
