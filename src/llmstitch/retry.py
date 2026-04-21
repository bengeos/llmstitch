"""Retry policy and backoff helper for provider calls.

Scoped to `ProviderAdapter.complete()` in v0.1.3. Streaming is not retried —
deltas may already have been observed by the caller before an error surfaces,
and there is no safe way to roll those back.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryAttempt:
    """Per-attempt observability payload passed to `RetryPolicy.on_retry`."""

    attempt: int
    delay: float
    exc: BaseException


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff retry policy for provider calls.

    `retry_on` defaults to an empty tuple — an unconfigured policy retries
    nothing. Use `AdapterCls.default_retryable()` to get a sensible list
    of transient error classes for a given provider.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.2
    retry_on: tuple[type[BaseException], ...] = ()
    respect_retry_after: bool = True
    on_retry: Callable[[RetryAttempt], None] | None = None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a Retry-After hint from the exception, if any.

    Both `anthropic` and `openai` SDKs surface the header on the underlying
    response. Nothing here is provider-specific — we just probe a few shapes
    and fall back to `None` if we can't find a numeric value.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value: Any = None
    if headers is not None:
        # httpx.Headers supports both .get() and dict-like access.
        try:
            value = headers.get("retry-after")
        except Exception:
            value = None
    if value is None:
        value = getattr(exc, "retry_after", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_delay(policy: RetryPolicy, attempt: int, exc: BaseException) -> float:
    """Backoff for the given attempt (1-indexed). Honors Retry-After when present."""
    base = policy.initial_delay * (policy.multiplier ** (attempt - 1))
    base = min(base, policy.max_delay)
    if policy.jitter:
        spread = base * policy.jitter
        base = max(0.0, base + random.uniform(-spread, spread))
    if policy.respect_retry_after:
        hint = _retry_after_seconds(exc)
        if hint is not None:
            return min(policy.max_delay, max(base, hint))
    return base


async def retry_call(
    policy: RetryPolicy | None,
    factory: Callable[[], Awaitable[T]],
) -> T:
    """Invoke `factory()` with retry on exceptions in `policy.retry_on`.

    `factory` is a zero-arg coroutine *factory* (not a coroutine) so each
    attempt gets a fresh awaitable. When `policy` is `None` the call runs once
    with no wrapping — zero cost in the common path.
    """
    if policy is None or policy.max_attempts <= 1 or not policy.retry_on:
        return await factory()

    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await factory()
        except policy.retry_on as exc:
            last_exc = exc
            if attempt >= policy.max_attempts:
                raise
            delay = _compute_delay(policy, attempt, exc)
            if policy.on_retry is not None:
                policy.on_retry(RetryAttempt(attempt=attempt, delay=delay, exc=exc))
            await asyncio.sleep(delay)
    # Unreachable — the loop either returns or re-raises — but mypy wants a terminal.
    assert last_exc is not None
    raise last_exc
