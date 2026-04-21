"""Tests for `run_with_result`, `run_stream_with_result`, and the cost ceiling."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from llmstitch import (
    Agent,
    AgentResult,
    AgentResultEvent,
    CostCeilingExceeded,
    MaxIterationsExceeded,
    Pricing,
    RetryPolicy,
    tool,
)
from llmstitch.providers.base import ProviderAdapter
from llmstitch.types import (
    CompletionResponse,
    Message,
    StreamDone,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
)

from .conftest import FakeAdapter


@tool
def say(msg: str) -> str:
    return msg


# ---------- run_with_result ---------- #


async def test_run_with_result_complete() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="hello world")],
                stop_reason="end_turn",
                usage={"input_tokens": 10, "output_tokens": 3},
            )
        ]
    )
    agent = Agent(provider=provider, model="m")

    result = await agent.run_with_result("hi")

    assert isinstance(result, AgentResult)
    assert result.stop_reason == "complete"
    assert result.error is None
    assert result.text == "hello world"
    assert result.turns == 1
    assert len(result.messages) == 2
    assert result.cost is not None
    assert result.cost.total > 0


async def test_run_with_result_max_iterations() -> None:
    loop_response = CompletionResponse(
        content=[ToolUseBlock(id="u", name="say", input={"msg": "again"})],
        stop_reason="tool_use",
    )
    provider = FakeAdapter([loop_response] * 5)
    agent = Agent(provider=provider, model="m", max_iterations=3)
    agent.tools.register(say)

    result = await agent.run_with_result("loop")

    assert result.stop_reason == "max_iterations"
    assert result.error is None
    # 3 turns * (assistant + tool result) + initial user = 7 messages at cutoff
    assert len(result.messages) >= 3


async def test_run_with_result_cost_ceiling() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="big bill")],
                stop_reason="end_turn",
                usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            )
        ]
    )
    agent = Agent(
        provider=provider,
        model="m",
        pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),
        cost_ceiling=1.00,
    )

    result = await agent.run_with_result("go")

    assert result.stop_reason == "cost_ceiling"
    assert isinstance(result.error, CostCeilingExceeded)
    assert result.error.spent > 1.00
    assert result.error.ceiling == 1.00


async def test_run_with_result_vendor_error() -> None:
    class _Boom(ProviderAdapter):
        async def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            system: str | None = None,
            tools: list[ToolDefinition] | None = None,
            max_tokens: int = 4096,
            **kwargs: Any,
        ) -> CompletionResponse:
            raise RuntimeError("vendor down")

    agent = Agent(provider=_Boom(), model="m")

    result = await agent.run_with_result("x")

    assert result.stop_reason == "error"
    assert isinstance(result.error, RuntimeError)
    assert str(result.error) == "vendor down"


async def test_run_stays_raising_for_max_iterations() -> None:
    """Existing run() still raises — backwards-compat."""
    loop_response = CompletionResponse(
        content=[ToolUseBlock(id="u", name="say", input={"msg": "again"})],
        stop_reason="tool_use",
    )
    provider = FakeAdapter([loop_response] * 5)
    agent = Agent(provider=provider, model="m", max_iterations=2)
    agent.tools.register(say)

    with pytest.raises(MaxIterationsExceeded):
        await agent.run("loop")


async def test_run_stays_raising_for_cost_ceiling() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="done")],
                stop_reason="end_turn",
                usage={"input_tokens": 1_000_000, "output_tokens": 0},
            )
        ]
    )
    agent = Agent(
        provider=provider,
        model="m",
        pricing=Pricing(input_per_mtok=10.0, output_per_mtok=0.0),
        cost_ceiling=1.00,
    )

    with pytest.raises(CostCeilingExceeded) as exc_info:
        await agent.run("x")
    assert exc_info.value.ceiling == 1.00
    assert exc_info.value.spent == pytest.approx(10.0)


# ---------- Cost ceiling behavior ---------- #


async def test_cost_ceiling_none_disables_check() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="ok")],
                stop_reason="end_turn",
                usage={"input_tokens": 10_000_000, "output_tokens": 10_000_000},
            )
        ]
    )
    agent = Agent(provider=provider, model="m")  # no ceiling
    await agent.run("x")
    assert agent.cost().total > 0


async def test_cost_ceiling_fires_after_usage_add_not_before_provider_call() -> None:
    """The first turn's usage must already be folded in before the ceiling trips."""
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[ToolUseBlock(id="u", name="say", input={"msg": "on"})],
                stop_reason="tool_use",
                usage={"input_tokens": 1_000_000, "output_tokens": 0},
            ),
            # This second turn should never be reached.
            CompletionResponse(
                content=[TextBlock(text="never")],
                stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(
        provider=provider,
        model="m",
        pricing=Pricing(input_per_mtok=10.0, output_per_mtok=0.0),
        cost_ceiling=1.00,
    )
    agent.tools.register(say)

    result = await agent.run_with_result("go")
    assert result.stop_reason == "cost_ceiling"
    # Only the first provider.complete was called.
    assert len(provider.calls) == 1


class _FlakyAdapter(ProviderAdapter):
    def __init__(self, failures: int, success: CompletionResponse) -> None:
        self.failures = failures
        self.success = success
        self.calls = 0

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"transient #{self.calls}")
        return self.success


async def test_cost_ceiling_not_double_counted_across_retries() -> None:
    """Failed attempts don't count toward the ceiling — usage only adds on success."""
    success = CompletionResponse(
        content=[TextBlock(text="ok")],
        stop_reason="end_turn",
        usage={"input_tokens": 50_000, "output_tokens": 0},
    )
    provider = _FlakyAdapter(failures=2, success=success)
    agent = Agent(
        provider=provider,
        model="m",
        pricing=Pricing(input_per_mtok=10.0, output_per_mtok=0.0),
        cost_ceiling=1.00,  # 50k tokens * $10/M = $0.50 — under ceiling
        retry_policy=RetryPolicy(
            max_attempts=5,
            initial_delay=0.0,
            jitter=0.0,
            retry_on=(RuntimeError,),
        ),
    )

    result = await agent.run_with_result("x")
    assert result.stop_reason == "complete"
    assert agent.cost().total == pytest.approx(0.50)


# ---------- run_stream_with_result ---------- #


async def test_run_stream_with_result_yields_events_then_result() -> None:
    final = CompletionResponse(
        content=[TextBlock(text="streamed")],
        stop_reason="end_turn",
        usage={"input_tokens": 5, "output_tokens": 2},
    )
    provider = FakeAdapter([final], stream_scripts=[[StreamDone(response=final)]])
    agent = Agent(provider=provider, model="m")

    collected: list[Any] = []
    async for event in agent.run_stream_with_result("hi"):
        collected.append(event)

    terminal = collected[-1]
    assert isinstance(terminal, AgentResultEvent)
    assert terminal.result.stop_reason == "complete"
    assert terminal.result.text == "streamed"
    assert terminal.result.error is None


async def test_run_stream_with_result_cost_ceiling() -> None:
    final = CompletionResponse(
        content=[TextBlock(text="costly")],
        stop_reason="end_turn",
        usage={"input_tokens": 1_000_000, "output_tokens": 0},
    )
    provider = FakeAdapter([final], stream_scripts=[[StreamDone(response=final)]])
    agent = Agent(
        provider=provider,
        model="m",
        pricing=Pricing(input_per_mtok=10.0, output_per_mtok=0.0),
        cost_ceiling=1.00,
    )

    collected: list[Any] = []
    async for event in agent.run_stream_with_result("hi"):
        collected.append(event)

    terminal = collected[-1]
    assert isinstance(terminal, AgentResultEvent)
    assert terminal.result.stop_reason == "cost_ceiling"
    assert isinstance(terminal.result.error, CostCeilingExceeded)


# ---------- AgentResult type hygiene ---------- #


def test_agent_result_is_frozen() -> None:
    r = AgentResult(
        messages=[],
        text="",
        stop_reason="complete",
        turns=0,
        usage=Agent(provider=FakeAdapter([]), model="m").usage,
        cost=None,
    )
    with pytest.raises(FrozenInstanceError):
        r.text = "mutate"  # type: ignore[misc]


async def test_agent_result_cost_always_populated_with_default_pricing() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="ok")],
                stop_reason="end_turn",
                usage={"input_tokens": 100, "output_tokens": 50},
            )
        ]
    )
    agent = Agent(provider=provider, model="m")  # default pricing
    result = await agent.run_with_result("x")
    assert result.cost is not None
    assert result.cost.total > 0
