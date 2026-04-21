"""EventBus observability: subscribe to agent run events.

`EventBus` exposes a structured view of what the agent is doing — separate
from `run_stream`'s wire-level `StreamEvent`s. Subscribe with a callback
(sync) or `async for event in bus.stream()` (SSE-style).

What this example shows:

1. **Callback subscription.** `bus.subscribe(cb)` fires `cb` synchronously
   as each event is emitted. Perfect for logging, metrics, or a running
   counter. An unsubscribe closure is returned.
2. **Typed events.** Every event is a frozen dataclass — pattern-match on
   `isinstance(event, ...)` to pull turn numbers, messages, tool calls,
   results, or usage deltas.
3. **Zero overhead when unused.** Omit `event_bus=` and emission is a no-op.

Event sequence for a one-turn no-tool run:
    AgentStarted → TurnStarted → ModelRequestSent
                 → ModelResponseReceived → UsageUpdated → AgentStopped

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/observability.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from llmstitch import (
    Agent,
    AgentStarted,
    AgentStopped,
    Event,
    EventBus,
    ModelResponseReceived,
    TextBlock,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnStarted,
    UsageUpdated,
    tool,
)
from llmstitch.providers.anthropic import AnthropicAdapter


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"{city}: 72°F and sunny"


def make_logger() -> tuple[list[Event], Callable[[Event], None]]:
    """Return (collected_events, subscriber). The subscriber both prints and collects."""
    collected: list[Event] = []

    def _on_event(event: Event) -> None:
        collected.append(event)
        if isinstance(event, AgentStarted):
            print(f"[start] model={event.model}")
        elif isinstance(event, TurnStarted):
            print(f"[turn {event.turn}] begin")
        elif isinstance(event, ModelResponseReceived):
            n_blocks = len(event.response.content)
            print(f"[turn {event.turn}] response: {n_blocks} block(s)")
        elif isinstance(event, ToolExecutionStarted):
            print(f"[turn {event.turn}] tool {event.call.name}({event.call.input})")
        elif isinstance(event, ToolExecutionCompleted):
            print(
                f"[turn {event.turn}] tool {event.call.name} -> "
                f"{event.result.content!r} in {event.duration_s * 1000:.1f}ms"
            )
        elif isinstance(event, UsageUpdated):
            if event.delta is not None:
                print(
                    f"[turn {event.turn}] usage +{event.delta.get('input_tokens', 0)}in "
                    f"/+{event.delta.get('output_tokens', 0)}out "
                    f"(total: {event.usage.total_tokens})"
                )
        elif isinstance(event, AgentStopped):
            print(f"[stop] reason={event.stop_reason} turns={event.turns}")

    return collected, _on_event


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    bus = EventBus()
    collected, subscriber = make_logger()
    bus.subscribe(subscriber)

    agent = Agent(
        provider=AnthropicAdapter(),
        model="claude-haiku-4-5-20251001",
        system="You are a helpful weather assistant. Use the tool when asked about a city.",
        event_bus=bus,
    )
    agent.tools.register(get_weather)

    history = await agent.run("What's the weather in Tokyo?")
    final = history[-1]
    if isinstance(final.content, list):
        text = "".join(b.text for b in final.content if isinstance(b, TextBlock))
    else:
        text = final.content
    print(f"\nFinal reply: {text}")
    print(f"Collected {len(collected)} events")


if __name__ == "__main__":
    asyncio.run(main())
