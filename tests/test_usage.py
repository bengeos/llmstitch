"""Tests for `UsageTally` and its integration with `Agent`."""

from __future__ import annotations

from typing import Any

import pytest

from llmstitch import Agent, Cost, Pricing, RetryAttempt, RetryPolicy, UsageTally
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

# ---------- UsageTally unit behavior ---------- #


def test_usage_tally_starts_empty() -> None:
    tally = UsageTally()
    assert (tally.input_tokens, tally.output_tokens, tally.turns) == (0, 0, 0)
    assert tally.total_tokens == 0


def test_usage_tally_add_accumulates() -> None:
    tally = UsageTally()
    tally.add({"input_tokens": 10, "output_tokens": 3})
    tally.add({"input_tokens": 5, "output_tokens": 7})
    assert tally.input_tokens == 15
    assert tally.output_tokens == 10
    assert tally.total_tokens == 25
    assert tally.turns == 2


def test_usage_tally_add_ignores_none() -> None:
    tally = UsageTally()
    tally.add(None)
    assert tally.turns == 0 and tally.total_tokens == 0


def test_usage_tally_add_tolerates_missing_keys() -> None:
    tally = UsageTally()
    tally.add({})  # counts as a turn with zero tokens
    tally.add({"input_tokens": 4})  # output_tokens defaults to 0
    assert tally.input_tokens == 4
    assert tally.output_tokens == 0
    assert tally.turns == 2


def test_usage_tally_reset_zeroes_counters() -> None:
    tally = UsageTally(input_tokens=10, output_tokens=5, turns=2, api_calls=4, retries=2)
    tally.reset()
    assert (
        tally.input_tokens,
        tally.output_tokens,
        tally.turns,
        tally.api_calls,
        tally.retries,
    ) == (0, 0, 0, 0, 0)


def test_usage_tally_record_call_and_retry() -> None:
    tally = UsageTally()
    tally.record_call()
    tally.record_call()
    tally.record_retry()
    assert tally.api_calls == 2
    assert tally.retries == 1


# ---------- Agent integration ---------- #


async def test_agent_run_accumulates_usage_across_turns() -> None:
    # Two turns: first asks for a tool call, second produces final text.
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[ToolUseBlock(id="t1", name="noop", input={})],
                stop_reason="tool_use",
                usage={"input_tokens": 100, "output_tokens": 20},
            ),
            CompletionResponse(
                content=[TextBlock(text="done")],
                stop_reason="end_turn",
                usage={"input_tokens": 30, "output_tokens": 5},
            ),
        ]
    )
    agent = Agent(provider=provider, model="m")

    # Register a no-op tool so the first response's tool call resolves.
    async def noop() -> str:
        return ""

    agent.tools.register(noop)

    await agent.run("hi")

    assert agent.usage.input_tokens == 130
    assert agent.usage.output_tokens == 25
    assert agent.usage.total_tokens == 155
    assert agent.usage.turns == 2


async def test_agent_run_is_cumulative_across_calls() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="first")],
                stop_reason="end_turn",
                usage={"input_tokens": 10, "output_tokens": 2},
            ),
            CompletionResponse(
                content=[TextBlock(text="second")],
                stop_reason="end_turn",
                usage={"input_tokens": 20, "output_tokens": 4},
            ),
        ]
    )
    agent = Agent(provider=provider, model="m")

    await agent.run("1")
    assert agent.usage.total_tokens == 12
    await agent.run("2")
    assert agent.usage.total_tokens == 36
    assert agent.usage.turns == 2


async def test_agent_usage_reset_between_calls() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="one")],
                stop_reason="end_turn",
                usage={"input_tokens": 7, "output_tokens": 1},
            ),
            CompletionResponse(
                content=[TextBlock(text="two")],
                stop_reason="end_turn",
                usage={"input_tokens": 9, "output_tokens": 3},
            ),
        ]
    )
    agent = Agent(provider=provider, model="m")

    await agent.run("a")
    agent.usage.reset()
    assert agent.usage.total_tokens == 0
    await agent.run("b")
    assert agent.usage.input_tokens == 9
    assert agent.usage.output_tokens == 3
    assert agent.usage.turns == 1


async def test_agent_run_tolerates_missing_usage() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="no usage reported")],
                stop_reason="end_turn",
                usage=None,
            ),
        ]
    )
    agent = Agent(provider=provider, model="m")

    await agent.run("hi")
    assert agent.usage.turns == 0
    assert agent.usage.total_tokens == 0


async def test_agent_run_stream_accumulates_usage() -> None:
    final = CompletionResponse(
        content=[TextBlock(text="streamed")],
        stop_reason="end_turn",
        usage={"input_tokens": 42, "output_tokens": 8},
    )
    provider = FakeAdapter(
        [final],
        stream_scripts=[[StreamDone(response=final)]],
    )
    agent = Agent(provider=provider, model="m")

    async for _ in agent.run_stream("hi"):
        pass

    assert agent.usage.input_tokens == 42
    assert agent.usage.output_tokens == 8
    assert agent.usage.turns == 1


class _FlakyAdapter(ProviderAdapter):
    """Fails `failures` times with `exc_type`, then returns `success`."""

    def __init__(
        self,
        failures: int,
        success: CompletionResponse,
        *,
        exc_type: type[BaseException] = RuntimeError,
    ) -> None:
        self.failures = failures
        self.success = success
        self.exc_type = exc_type
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
        del model, messages, system, tools, max_tokens, kwargs
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc_type(f"transient #{self.calls}")
        return self.success


async def test_agent_run_records_api_calls_and_retries() -> None:
    provider = _FlakyAdapter(
        failures=2,
        success=CompletionResponse(
            content=[TextBlock(text="ok")],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 3},
        ),
    )
    agent = Agent(
        provider=provider,
        model="m",
        retry_policy=RetryPolicy(
            max_attempts=5,
            initial_delay=0.0,
            jitter=0.0,
            retry_on=(RuntimeError,),
        ),
    )

    await agent.run("hi")

    # 2 failures + 1 success = 3 API calls, 2 retries, 1 successful turn.
    assert agent.usage.api_calls == 3
    assert agent.usage.retries == 2
    assert agent.usage.turns == 1
    assert agent.usage.input_tokens == 10
    assert agent.usage.output_tokens == 3
    # Identity: api_calls == turns + retries on the success path.
    assert agent.usage.api_calls == agent.usage.turns + agent.usage.retries


async def test_agent_run_preserves_user_on_retry_callback() -> None:
    """Wrapping on_retry to count retries must not clobber the user's callback."""
    observed: list[RetryAttempt] = []
    provider = _FlakyAdapter(
        failures=1,
        success=CompletionResponse(
            content=[TextBlock(text="ok")],
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        ),
    )
    agent = Agent(
        provider=provider,
        model="m",
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay=0.0,
            jitter=0.0,
            retry_on=(RuntimeError,),
            on_retry=observed.append,
        ),
    )

    await agent.run("hi")

    assert agent.usage.retries == 1
    assert len(observed) == 1
    assert isinstance(observed[0], RetryAttempt)
    assert observed[0].attempt == 1


async def test_agent_run_records_api_calls_when_retries_exhausted() -> None:
    provider = _FlakyAdapter(
        failures=10,  # more than max_attempts — will exhaust
        success=CompletionResponse(
            content=[TextBlock(text="never returned")],
            stop_reason="end_turn",
        ),
    )
    agent = Agent(
        provider=provider,
        model="m",
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay=0.0,
            jitter=0.0,
            retry_on=(RuntimeError,),
        ),
    )

    with pytest.raises(RuntimeError):
        await agent.run("hi")

    # 3 attempts all failed: 3 api_calls, 2 retries fired (last failure doesn't
    # retry), 0 turns.
    assert agent.usage.api_calls == 3
    assert agent.usage.retries == 2
    assert agent.usage.turns == 0


async def test_agent_run_without_policy_records_api_calls_only() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="hi")],
                stop_reason="end_turn",
                usage={"input_tokens": 2, "output_tokens": 1},
            )
        ]
    )
    agent = Agent(provider=provider, model="m")  # no retry_policy

    await agent.run("hi")

    assert agent.usage.api_calls == 1
    assert agent.usage.retries == 0
    assert agent.usage.turns == 1


async def test_agent_run_stream_records_api_calls_no_retries() -> None:
    final = CompletionResponse(
        content=[TextBlock(text="streamed")],
        stop_reason="end_turn",
        usage={"input_tokens": 5, "output_tokens": 2},
    )
    provider = FakeAdapter(
        [final],
        stream_scripts=[[StreamDone(response=final)]],
    )
    agent = Agent(provider=provider, model="m")

    async for _ in agent.run_stream("hi"):
        pass

    assert agent.usage.api_calls == 1
    assert agent.usage.retries == 0
    assert agent.usage.turns == 1


# ---------- Pricing / Cost ---------- #


def test_cost_total_sums_components() -> None:
    cost = Cost(input_cost=0.003, output_cost=0.015)
    assert cost.total == pytest.approx(0.018)


def test_usage_tally_cost_scales_per_mtok() -> None:
    tally = UsageTally(input_tokens=1_000_000, output_tokens=500_000)
    pricing = Pricing(input_per_mtok=3.00, output_per_mtok=15.00)
    cost = tally.cost(pricing)
    # 1M input tokens at $3.00/M = $3.00; 500K output at $15.00/M = $7.50.
    assert cost.input_cost == pytest.approx(3.00)
    assert cost.output_cost == pytest.approx(7.50)
    assert cost.total == pytest.approx(10.50)


def test_usage_tally_cost_handles_small_token_counts() -> None:
    tally = UsageTally(input_tokens=1500, output_tokens=250)
    pricing = Pricing(input_per_mtok=3.00, output_per_mtok=15.00)
    cost = tally.cost(pricing)
    # 1500 * 3.00 / 1e6 = 0.0045; 250 * 15.00 / 1e6 = 0.00375.
    assert cost.input_cost == pytest.approx(0.0045)
    assert cost.output_cost == pytest.approx(0.00375)


def test_usage_tally_cost_with_zero_tokens_is_zero() -> None:
    pricing = Pricing(input_per_mtok=3.00, output_per_mtok=15.00)
    assert UsageTally().cost(pricing).total == 0.0


async def test_agent_cost_uses_configured_pricing() -> None:
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="done")],
                stop_reason="end_turn",
                usage={"input_tokens": 2_000, "output_tokens": 500},
            )
        ]
    )
    agent = Agent(
        provider=provider,
        model="m",
        pricing=Pricing(input_per_mtok=3.00, output_per_mtok=15.00),
    )

    await agent.run("hi")

    cost = agent.cost()
    # 2000 * 3.00 / 1M = 0.006; 500 * 15.00 / 1M = 0.0075.
    assert cost.input_cost == pytest.approx(0.006)
    assert cost.output_cost == pytest.approx(0.0075)
    assert cost.total == pytest.approx(0.0135)


async def test_agent_cost_uses_default_pricing_when_unset() -> None:
    """Without an explicit pricing=, the agent uses a placeholder default ($1/$2)."""
    provider = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="done")],
                stop_reason="end_turn",
                usage={"input_tokens": 1_000_000, "output_tokens": 500_000},
            )
        ]
    )
    agent = Agent(provider=provider, model="m")  # no pricing configured

    assert agent.pricing.input_per_mtok == pytest.approx(1.00)
    assert agent.pricing.output_per_mtok == pytest.approx(2.00)

    await agent.run("hi")
    cost = agent.cost()
    # 1M input at $1.00/M = $1.00; 500K output at $2.00/M = $1.00.
    assert cost.input_cost == pytest.approx(1.00)
    assert cost.output_cost == pytest.approx(1.00)
    assert cost.total == pytest.approx(2.00)


def test_agent_pricing_default_is_instance_scoped() -> None:
    """Two agents must not share the same Pricing instance via default_factory."""
    a = Agent(provider=FakeAdapter([]), model="m")
    b = Agent(provider=FakeAdapter([]), model="m")
    assert a.pricing == b.pricing
    # Pricing is frozen so they can safely be the same object, but the
    # default_factory produces a fresh one per instance — verify that.
    assert a.pricing is not b.pricing


async def test_agent_usage_is_instance_scoped() -> None:
    """Two agents maintain independent tallies (no accidental class-level sharing)."""
    provider_a = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="a")],
                stop_reason="end_turn",
                usage={"input_tokens": 5, "output_tokens": 1},
            )
        ]
    )
    provider_b = FakeAdapter(
        [
            CompletionResponse(
                content=[TextBlock(text="b")],
                stop_reason="end_turn",
                usage={"input_tokens": 11, "output_tokens": 2},
            )
        ]
    )
    agent_a = Agent(provider=provider_a, model="m")
    agent_b = Agent(provider=provider_b, model="m")

    await agent_a.run("x")
    await agent_b.run("y")

    assert agent_a.usage.total_tokens == 6
    assert agent_b.usage.total_tokens == 13
    assert agent_a.usage is not agent_b.usage
