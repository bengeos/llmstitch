"""Tests for the EventBus and agent event emission."""

from __future__ import annotations

import asyncio
import warnings

import pytest

from llmstitch import (
    Agent,
    AgentResultEvent,
    AgentStarted,
    AgentStopped,
    Event,
    EventBus,
    ModelRequestSent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnStarted,
    UsageUpdated,
    tool,
)
from llmstitch.types import (
    CompletionResponse,
    StreamDone,
    TextBlock,
    ToolUseBlock,
)

from .conftest import FakeAdapter


@tool
def add(a: int, b: int) -> int:
    return a + b


# ---------- EventBus unit behavior ---------- #


def test_event_bus_subscribe_receives_events() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)

    event = TurnStarted(turn=1)
    bus.emit(event)
    assert collected == [event]


def test_event_bus_unsubscribe() -> None:
    bus = EventBus()
    collected: list[Event] = []
    unsub = bus.subscribe(collected.append)
    unsub()
    bus.emit(TurnStarted(turn=1))
    assert collected == []


def test_event_bus_multiple_subscribers() -> None:
    bus = EventBus()
    a: list[Event] = []
    b: list[Event] = []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    evt = TurnStarted(turn=2)
    bus.emit(evt)
    assert a == [evt] and b == [evt]


async def test_event_bus_stream_iterates_events() -> None:
    bus = EventBus()
    stream = bus.stream()
    bus.emit(TurnStarted(turn=1))
    bus.emit(AgentStopped(stop_reason="complete", turns=1))

    received = [evt async for evt in stream]
    assert [type(e).__name__ for e in received] == ["TurnStarted", "AgentStopped"]


async def test_event_bus_stream_and_subscribe_coexist() -> None:
    bus = EventBus()
    callback_events: list[Event] = []
    bus.subscribe(callback_events.append)
    stream = bus.stream()
    bus.emit(TurnStarted(turn=1))
    bus.emit(AgentStopped(stop_reason="complete", turns=1))

    streamed = [evt async for evt in stream]
    assert len(callback_events) == 2
    assert len(streamed) == 2


def test_event_bus_subscriber_exception_does_not_break_agent() -> None:
    bus = EventBus()

    def boom(_: Event) -> None:
        raise RuntimeError("subscriber crash")

    ok_collected: list[Event] = []
    bus.subscribe(boom)
    bus.subscribe(ok_collected.append)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bus.emit(TurnStarted(turn=1))

    assert len(ok_collected) == 1
    assert any("subscriber" in str(warning.message).lower() for warning in w)


async def test_multiple_runs_share_bus_stream_ok() -> None:
    """Each run terminates its own stream; a fresh stream() works on the next run."""
    bus = EventBus()
    provider = FakeAdapter(
        [
            CompletionResponse(content=[TextBlock(text="one")], stop_reason="end_turn"),
            CompletionResponse(content=[TextBlock(text="two")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, model="m", event_bus=bus)

    s1 = bus.stream()
    await agent.run("a")
    events1 = [e async for e in s1]
    assert isinstance(events1[-1], AgentStopped)
    assert events1[-1].stop_reason == "complete"

    s2 = bus.stream()
    await agent.run("b")
    events2 = [e async for e in s2]
    assert isinstance(events2[-1], AgentStopped)
    assert events2[-1].stop_reason == "complete"


# ---------- Agent emission ---------- #


async def test_agent_emits_full_event_sequence_no_tools() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="hi")],
                stop_reason="end_turn",
                usage={"input_tokens": 3, "output_tokens": 1},
            )
        ]
    )
    agent = Agent(provider=provider, model="m", event_bus=bus)

    await agent.run("hello")

    types = [type(e).__name__ for e in collected]
    assert types == [
        "AgentStarted",
        "TurnStarted",
        "ModelRequestSent",
        "ModelResponseReceived",
        "UsageUpdated",
        "AgentStopped",
    ]
    stopped = collected[-1]
    assert isinstance(stopped, AgentStopped)
    assert stopped.stop_reason == "complete"


async def test_agent_emits_full_event_sequence_with_tool_roundtrip() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[ToolUseBlock(id="t1", name="add", input={"a": 2, "b": 3})],
                stop_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 2},
            ),
            CompletionResponse(
                content=[TextBlock(text="5")],
                stop_reason="end_turn",
                usage={"input_tokens": 4, "output_tokens": 1},
            ),
        ]
    )
    agent = Agent(provider=provider, model="m", event_bus=bus)
    agent.tools.register(add)

    await agent.run("what is 2+3?")

    # Two turns: each has TurnStarted, ModelRequestSent, ModelResponseReceived, UsageUpdated.
    # Turn 1 has one tool call → ToolExecutionStarted + ToolExecutionCompleted between
    # UsageUpdated(turn=1) and TurnStarted(turn=2).
    type_names = [type(e).__name__ for e in collected]
    assert type_names[0] == "AgentStarted"
    assert type_names[-1] == "AgentStopped"
    assert "ToolExecutionStarted" in type_names
    assert "ToolExecutionCompleted" in type_names

    started_events = [e for e in collected if isinstance(e, ToolExecutionStarted)]
    completed_events = [e for e in collected if isinstance(e, ToolExecutionCompleted)]
    assert len(started_events) == 1
    assert len(completed_events) == 1
    assert started_events[0].call.id == "t1"
    assert completed_events[0].result.content == "5"
    assert completed_events[0].duration_s >= 0


async def test_agent_emits_usage_updated_with_live_tally_and_delta() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="ok")],
                stop_reason="end_turn",
                usage={"input_tokens": 7, "output_tokens": 2},
            )
        ]
    )
    agent = Agent(provider=provider, model="m", event_bus=bus)

    await agent.run("x")

    usage_events = [e for e in collected if isinstance(e, UsageUpdated)]
    assert len(usage_events) == 1
    evt = usage_events[0]
    assert evt.turn == 1
    assert evt.delta == {"input_tokens": 7, "output_tokens": 2}
    assert evt.usage.input_tokens == 7
    assert evt.usage.output_tokens == 2


async def test_agent_stopped_max_iterations() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)
    looping = CompletionResponse(
        content=[ToolUseBlock(id="u", name="add", input={"a": 1, "b": 1})],
        stop_reason="tool_use",
    )
    provider = FakeAdapter([looping, looping, looping])
    agent = Agent(provider=provider, model="m", max_iterations=2, event_bus=bus)
    agent.tools.register(add)

    from llmstitch import MaxIterationsExceeded

    with pytest.raises(MaxIterationsExceeded):
        await agent.run("loop")

    stopped = [e for e in collected if isinstance(e, AgentStopped)]
    assert len(stopped) == 1
    assert stopped[0].stop_reason == "max_iterations"


async def test_agent_stopped_error_when_vendor_raises() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)

    from llmstitch.providers.base import ProviderAdapter
    from llmstitch.types import Message, ToolDefinition

    class _Boom(ProviderAdapter):
        async def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            system: str | None = None,
            tools: list[ToolDefinition] | None = None,
            max_tokens: int = 4096,
            **kwargs: object,
        ) -> CompletionResponse:
            raise RuntimeError("vendor down")

    agent = Agent(provider=_Boom(), model="m", event_bus=bus)

    with pytest.raises(RuntimeError):
        await agent.run("x")

    stopped = [e for e in collected if isinstance(e, AgentStopped)]
    assert len(stopped) == 1
    assert stopped[0].stop_reason == "error"
    assert isinstance(stopped[0].error, RuntimeError)


async def test_agent_started_event_carries_prompt_and_model() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)
    provider = FakeAdapter(
        [CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")]
    )
    agent = Agent(provider=provider, model="m-1", event_bus=bus)

    await agent.run("hello there")

    started = collected[0]
    assert isinstance(started, AgentStarted)
    assert started.model == "m-1"
    assert started.prompt == "hello there"


async def test_model_request_sent_snapshots_messages() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)
    provider = FakeAdapter(
        [CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")]
    )
    agent = Agent(provider=provider, model="m", event_bus=bus)

    await agent.run("hi")

    requests = [e for e in collected if isinstance(e, ModelRequestSent)]
    assert len(requests) == 1
    # Snapshot should have the user prompt.
    assert requests[0].messages[0].role == "user"


async def test_event_bus_none_is_no_op() -> None:
    provider = FakeAdapter(
        [CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")]
    )
    agent = Agent(provider=provider, model="m")  # no event_bus
    history = await agent.run("hi")
    assert len(history) == 2


async def test_run_stream_emits_events_alongside_stream_events() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)
    final = CompletionResponse(
        content=[TextBlock(text="streamed")],
        stop_reason="end_turn",
        usage={"input_tokens": 5, "output_tokens": 2},
    )
    provider = FakeAdapter([final], stream_scripts=[[StreamDone(response=final)]])
    agent = Agent(provider=provider, model="m", event_bus=bus)

    async for _ in agent.run_stream("hi"):
        pass

    # Events flow through the bus; StreamEvents flow through run_stream's yield.
    type_names = [type(e).__name__ for e in collected]
    assert "AgentStarted" in type_names
    assert "ModelResponseReceived" in type_names
    assert "AgentStopped" in type_names


async def test_tool_execution_events_carry_result_and_duration() -> None:
    bus = EventBus()
    collected: list[Event] = []
    bus.subscribe(collected.append)

    @tool
    async def slow_echo(msg: str) -> str:
        await asyncio.sleep(0.02)
        return msg

    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[ToolUseBlock(id="s", name="slow_echo", input={"msg": "hi"})],
                stop_reason="tool_use",
            ),
            CompletionResponse(
                content=[TextBlock(text="done")],
                stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(provider=provider, model="m", event_bus=bus)
    agent.tools.register(slow_echo)

    await agent.run("x")

    completed = [e for e in collected if isinstance(e, ToolExecutionCompleted)]
    assert len(completed) == 1
    assert completed[0].result.content == "hi"
    assert completed[0].duration_s >= 0.02


# Re-export helpers to keep pyflakes quiet if fixtures are unused.
_ = AgentResultEvent
