"""Retry policy: backoff, jitter, and an observability callback.

`Agent.retry_policy` wraps `provider.complete(...)` with exponential backoff
plus jitter, honors `Retry-After` as a delay floor, and passes each retry
through an optional callback so you can log / meter the flakiness.

What this example shows:

1. **Default transient set.** `AdapterCls.default_retryable()` returns the
   vendor's own rate-limit / timeout / connection / 5xx exception classes —
   imported lazily inside the classmethod, so no SDK is touched at policy
   construction time.
2. **`on_retry` observability hook.** Receives a `RetryAttempt(attempt, delay,
   exc)` per retry — wire it to your logger, metrics system, or a simple
   counter like we do here.
3. **Non-retryable pass-through.** Exceptions not listed in `retry_on`
   (e.g. `AuthenticationError`) propagate immediately on the first attempt.

To make the retry path observable without depending on a real 429/5xx from
Anthropic, this example wraps `AnthropicAdapter` with a one-shot flaky
subclass that raises `APITimeoutError` on its first `complete()` call and
then delegates normally. Delete that wrapper to see the healthy-path
behavior (the policy is a no-op — zero retries, zero `log_retry` output).

Retries apply to `Agent.run` (non-streaming) only. `Agent.run_stream` is not
retried — once a `TextDelta` has been yielded, the caller may already have
rendered it, and there is no safe rollback.

Usage:
    pip install "llmstitch[anthropic]"
    ANTHROPIC_API_KEY=... python examples/retries.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from llmstitch import Agent, Message, Pricing, RetryAttempt, RetryPolicy, TextBlock, tool
from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.types import CompletionResponse, ToolDefinition

# Illustrative rates for Claude Haiku 4.5 — verify current pricing at
# https://www.anthropic.com/pricing before relying on these numbers.
HAIKU_4_5_PRICING = Pricing(input_per_mtok=1.00, output_per_mtok=5.00)


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"{city}: 72°F and sunny"


def log_retry(attempt: RetryAttempt) -> None:
    """Observability hook — fires once per retry (not on the first attempt)."""
    exc_name = type(attempt.exc).__name__
    print(
        f"[retry] attempt={attempt.attempt} "
        f"sleeping={attempt.delay:.2f}s exc={exc_name}: {attempt.exc}"
    )


class FlakyAnthropicAdapter(AnthropicAdapter):
    """Raises `APITimeoutError` on the first `complete()` call, then delegates.

    Purely here so the retry demo is deterministic — it proves the policy
    fires without needing the live API to be genuinely unhealthy.
    """

    def __init__(self, fail_times: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._remaining_failures = fail_times

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
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            import anthropic
            import httpx

            raise anthropic.APITimeoutError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            )
        return await super().complete(
            model=model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            **kwargs,
        )


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    policy = RetryPolicy(
        max_attempts=4,
        initial_delay=0.5,
        max_delay=8.0,
        multiplier=2.0,
        jitter=0.2,  # ±20% jitter around the computed delay
        retry_on=AnthropicAdapter.default_retryable(),
        respect_retry_after=True,  # honor Retry-After as a floor
        on_retry=log_retry,
    )

    agent = Agent(
        provider=FlakyAnthropicAdapter(fail_times=1),
        model="claude-haiku-4-5-20251001",
        system="You are a helpful weather assistant. Use the tool when asked about a city.",
        retry_policy=policy,
        pricing=HAIKU_4_5_PRICING,
    )
    agent.tools.register(get_weather)

    history = await agent.run("What's the weather in Tokyo?")
    final = history[-1]
    if isinstance(final.content, list):
        print("".join(b.text for b in final.content if isinstance(b, TextBlock)))
    else:
        print(final.content)

    print(f"Total tokens used: {agent.usage.total_tokens}")
    print(f"Total input tokens: {agent.usage.input_tokens}")
    print(f"Total output tokens: {agent.usage.output_tokens}")

    cost = agent.cost()
    print(f"Cost: ${cost.total:.6f}  (in ${cost.input_cost:.6f} / out ${cost.output_cost:.6f})")


if __name__ == "__main__":
    asyncio.run(main())
