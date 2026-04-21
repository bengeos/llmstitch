"""`run_stream_with_result`: stream deltas, then receive an `AgentResult`.

Streaming variant of the non-raising API. Yields the same `StreamEvent`s as
`run_stream` — `TextDelta`, `ToolUseStart`/`ToolUseDelta`/`ToolUseStop`,
`MessageStop`, and a per-turn `StreamDone` — and then one terminal
`AgentResultEvent(result=AgentResult(...))` summarizing the whole run.

Events produced by `EventBus` are orthogonal: they do NOT interleave into
the `StreamEvent` iterator. Subscribe to the bus in parallel if you want
structured observability alongside the live UI updates.

What this example shows:

1. **Live text rendering.** `TextDelta` writes to stdout as tokens arrive.
2. **Per-turn `StreamDone`.** Carries the assembled `CompletionResponse`
   for the turn — useful for stop-reason and usage inspection before the
   loop continues.
3. **Terminal `AgentResultEvent`.** The last event yielded is always an
   `AgentResultEvent`, never a `StreamEvent`. The whole-run `AgentResult`
   it carries has `stop_reason`, final message history, usage, cost, and
   (on non-happy paths) the captured exception.
4. **EventBus side-channel.** A subscriber prints structured progress;
   the streaming iterator handles the render path.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/streaming_with_result.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from llmstitch import (
    Agent,
    AgentResultEvent,
    AgentStopped,
    Event,
    EventBus,
    MessageStop,
    Pricing,
    StreamDone,
    TextDelta,
    ToolExecutionCompleted,
    ToolUseStart,
    tool,
)
from llmstitch.providers.anthropic import AnthropicAdapter

HAIKU_4_5_PRICING = Pricing(input_per_mtok=1.00, output_per_mtok=5.00)


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"{city}: 72°F and sunny"


def log_event(event: Event) -> None:
    """Structured progress — runs alongside the text stream."""
    if isinstance(event, ToolExecutionCompleted):
        sys.stderr.write(
            f"\n[bus] tool {event.call.name} -> "
            f"{event.result.content!r} in {event.duration_s * 1000:.1f}ms\n"
        )
    elif isinstance(event, AgentStopped):
        sys.stderr.write(f"\n[bus] stop: {event.stop_reason} after {event.turns} turn(s)\n")


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    bus = EventBus()
    bus.subscribe(log_event)

    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-haiku-4-5-20251001",
        system="You are a helpful weather assistant. Use the tool when asked.",
        event_bus=bus,
        pricing=HAIKU_4_5_PRICING,
        cost_ceiling=0.10,  # generous — this run won't trip it
    )
    agent.tools.register(get_weather)

    turn = 1
    prompt = "What's the weather in Tokyo? Then describe it in a short haiku."
    print(f"[turn {turn}] ", end="", flush=True)

    async for event in agent.run_stream_with_result(prompt):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolUseStart):
            print(f"\n[tool call] {event.name}({event.id}) ", end="", flush=True)
        elif isinstance(event, MessageStop):
            print(f"\n[stop_reason] {event.stop_reason}")
        elif isinstance(event, StreamDone):
            turn += 1
            # Only print a fresh turn prefix if another turn is coming — we
            # don't know yet, so tolerate a trailing one on the very last turn.
            print(f"[turn {turn}] ", end="", flush=True)
        elif isinstance(event, AgentResultEvent):
            result = event.result
            print("\n\n--- AgentResult ---")
            print(f"stop_reason: {result.stop_reason}")
            print(f"turns: {result.turns}")
            print(f"tokens: {result.usage.input_tokens}in / {result.usage.output_tokens}out")
            if result.cost is not None:
                print(f"cost: ${result.cost.total:.6f}")
            if result.error is not None:
                print(f"error: {type(result.error).__name__}: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
