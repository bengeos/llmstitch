"""Tests for retry policy / backoff helper and its integration with Agent."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llmstitch import Agent, RetryAttempt, RetryPolicy
from llmstitch.providers.base import ProviderAdapter
from llmstitch.retry import _compute_delay, _retry_after_seconds, retry_call
from llmstitch.types import (
    CompletionResponse,
    Message,
    TextBlock,
    ToolDefinition,
)


class _Transient(Exception):
    pass


class _Other(Exception):
    pass


# ---------- retry_call ---------- #


async def test_retry_call_returns_on_first_success() -> None:
    calls = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        return 42

    policy = RetryPolicy(max_attempts=3, retry_on=(_Transient,))
    assert await retry_call(policy, factory) == 42
    assert calls == 1


async def test_retry_call_retries_until_success() -> None:
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _Transient("still flaky")
        return "ok"

    policy = RetryPolicy(
        max_attempts=5,
        initial_delay=0.0,
        jitter=0.0,
        retry_on=(_Transient,),
    )
    assert await retry_call(policy, factory) == "ok"
    assert calls == 3


async def test_retry_call_reraises_after_max_attempts() -> None:
    calls = 0

    async def factory() -> None:
        nonlocal calls
        calls += 1
        raise _Transient(f"attempt {calls}")

    policy = RetryPolicy(max_attempts=3, initial_delay=0.0, jitter=0.0, retry_on=(_Transient,))
    with pytest.raises(_Transient):
        await retry_call(policy, factory)
    assert calls == 3


async def test_retry_call_passes_non_retryable_through() -> None:
    async def factory() -> None:
        raise _Other("nope")

    policy = RetryPolicy(max_attempts=3, retry_on=(_Transient,))
    with pytest.raises(_Other):
        await retry_call(policy, factory)


async def test_retry_call_noops_when_policy_is_none() -> None:
    async def factory() -> int:
        return 7

    assert await retry_call(None, factory) == 7


async def test_retry_call_invokes_on_retry_callback() -> None:
    observed: list[RetryAttempt] = []
    calls = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise _Transient("once")
        return 1

    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.0,
        jitter=0.0,
        retry_on=(_Transient,),
        on_retry=observed.append,
    )
    await retry_call(policy, factory)

    assert len(observed) == 1
    assert observed[0].attempt == 1
    assert isinstance(observed[0].exc, _Transient)
    assert observed[0].delay == 0.0


# ---------- delay / Retry-After ---------- #


def test_compute_delay_exponential_without_jitter() -> None:
    policy = RetryPolicy(
        initial_delay=0.5, multiplier=2.0, max_delay=10.0, jitter=0.0, retry_on=(_Transient,)
    )
    assert _compute_delay(policy, attempt=1, exc=_Transient()) == pytest.approx(0.5)
    assert _compute_delay(policy, attempt=2, exc=_Transient()) == pytest.approx(1.0)
    assert _compute_delay(policy, attempt=3, exc=_Transient()) == pytest.approx(2.0)


def test_compute_delay_respects_max_delay() -> None:
    policy = RetryPolicy(
        initial_delay=10.0, multiplier=10.0, max_delay=12.0, jitter=0.0, retry_on=(_Transient,)
    )
    # 10 * 10^4 would be huge; capped at max_delay.
    assert _compute_delay(policy, attempt=5, exc=_Transient()) == pytest.approx(12.0)


def test_compute_delay_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(
        initial_delay=1.0, multiplier=1.0, max_delay=10.0, jitter=0.2, retry_on=(_Transient,)
    )
    for _ in range(50):
        d = _compute_delay(policy, attempt=1, exc=_Transient())
        assert 0.8 <= d <= 1.2


def test_retry_after_seconds_reads_header() -> None:
    class _Headers(dict[str, str]):
        def get(self, key: str, default: str | None = None) -> str | None:  # type: ignore[override]
            return super().get(key, default)

    exc = _Transient("rate limited")
    exc.response = SimpleNamespace(headers=_Headers({"retry-after": "3.5"}))  # type: ignore[attr-defined]
    assert _retry_after_seconds(exc) == 3.5


def test_retry_after_seconds_reads_attribute_fallback() -> None:
    exc = _Transient("x")
    exc.retry_after = "2"  # type: ignore[attr-defined]
    assert _retry_after_seconds(exc) == 2.0


def test_retry_after_seconds_returns_none_when_missing() -> None:
    assert _retry_after_seconds(_Transient("plain")) is None


def test_compute_delay_honors_retry_after_floor() -> None:
    policy = RetryPolicy(initial_delay=0.1, jitter=0.0, max_delay=60.0, retry_on=(_Transient,))
    exc = _Transient("rl")
    exc.retry_after = 5.0  # type: ignore[attr-defined]
    # Backoff for attempt 1 is 0.1s, but Retry-After raises the floor to 5s.
    assert _compute_delay(policy, attempt=1, exc=exc) == pytest.approx(5.0)


# ---------- Agent integration ---------- #


class _FlakyAdapter(ProviderAdapter):
    """Adapter whose `complete` fails `failures` times before returning a canned response."""

    def __init__(self, failures: int, *, exc_type: type[BaseException] = _Transient) -> None:
        self.failures = failures
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
            raise self.exc_type(f"transient {self.calls}")
        return CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")


async def test_agent_run_retries_on_transient_and_succeeds() -> None:
    provider = _FlakyAdapter(failures=2)
    agent = Agent(
        provider=provider,
        model="m",
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay=0.0,
            jitter=0.0,
            retry_on=(_Transient,),
        ),
    )
    history = await agent.run("hi")

    assert provider.calls == 3  # 2 transient failures + 1 success
    assert history[-1].role == "assistant"


async def test_agent_run_without_policy_does_not_retry() -> None:
    provider = _FlakyAdapter(failures=1)
    agent = Agent(provider=provider, model="m")  # no retry_policy
    with pytest.raises(_Transient):
        await agent.run("hi")
    assert provider.calls == 1


async def test_agent_run_reraises_non_retryable() -> None:
    provider = _FlakyAdapter(failures=1, exc_type=_Other)
    agent = Agent(
        provider=provider,
        model="m",
        retry_policy=RetryPolicy(
            max_attempts=5,
            initial_delay=0.0,
            jitter=0.0,
            retry_on=(_Transient,),
        ),
    )
    with pytest.raises(_Other):
        await agent.run("hi")
    assert provider.calls == 1


# ---------- Adapter default_retryable ---------- #


def test_providers_expose_default_retryable() -> None:
    """Each adapter's `default_retryable()` lazy-imports its SDK and returns a tuple of types.

    We only verify non-emptiness and types — the exact set is provider-defined.
    """
    from llmstitch.providers.anthropic import AnthropicAdapter
    from llmstitch.providers.gemini import GeminiAdapter
    from llmstitch.providers.groq import GroqAdapter
    from llmstitch.providers.openai import OpenAIAdapter
    from llmstitch.providers.openrouter import OpenRouterAdapter

    for adapter_cls in (
        AnthropicAdapter,
        OpenAIAdapter,
        GeminiAdapter,
        GroqAdapter,
        OpenRouterAdapter,
    ):
        retryable = adapter_cls.default_retryable()
        assert isinstance(retryable, tuple)
        assert len(retryable) >= 1
        for cls in retryable:
            assert isinstance(cls, type)
            assert issubclass(cls, BaseException)
