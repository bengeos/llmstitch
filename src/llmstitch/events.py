"""Event bus and event dataclasses for agent-run observability.

Events flow through `EventBus` only — they are *not* interleaved into the
`AsyncIterator[StreamEvent]` returned by `Agent.run_stream`. Subscribe to a
bus (callback or async iterator) to observe a run without coupling to the
streaming wire format.

Emission is a tight synchronous loop; subscribers must be synchronous
callables. Async consumers should use `bus.stream()` which fans out through
an `asyncio.Queue`.
"""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal

from .types import (
    CompletionResponse,
    Message,
    ToolResultBlock,
    ToolUseBlock,
    UsageTally,
)

StopReason = Literal["complete", "max_iterations", "cost_ceiling", "error"]


@dataclass(frozen=True, slots=True)
class AgentStarted:
    prompt: str | list[Message]
    model: str


@dataclass(frozen=True, slots=True)
class TurnStarted:
    turn: int


@dataclass(frozen=True, slots=True)
class ModelRequestSent:
    turn: int
    messages: list[Message]


@dataclass(frozen=True, slots=True)
class ModelResponseReceived:
    turn: int
    response: CompletionResponse


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted:
    turn: int
    call: ToolUseBlock


@dataclass(frozen=True, slots=True)
class ToolExecutionCompleted:
    turn: int
    call: ToolUseBlock
    result: ToolResultBlock
    duration_s: float


@dataclass(frozen=True, slots=True)
class UsageUpdated:
    turn: int
    usage: UsageTally
    delta: dict[str, int] | None


@dataclass(frozen=True, slots=True)
class AgentStopped:
    stop_reason: StopReason
    turns: int
    error: Exception | None = None


Event = (
    AgentStarted
    | TurnStarted
    | ModelRequestSent
    | ModelResponseReceived
    | ToolExecutionStarted
    | ToolExecutionCompleted
    | UsageUpdated
    | AgentStopped
)


class _StreamEnd:
    """Sentinel pushed to streams after an AgentStopped to terminate iteration."""


_STREAM_END = _StreamEnd()


@dataclass(slots=True)
class EventBus:
    """Fan-out hub for agent run events.

    Supports two consumption styles in parallel:

    - `bus.subscribe(cb)`: synchronous callbacks fired in-band during emit
    - `bus.stream()`: async iterator that yields events through a queue

    A subscriber raising an exception is swallowed with a warning — an
    observer must not break the agent loop.
    """

    _subscribers: list[Callable[[Event], None]] = field(default_factory=list)
    _streams: list[asyncio.Queue[Event | _StreamEnd]] = field(default_factory=list)

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register a synchronous callback. Returns an unsubscribe function."""
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(callback)

        return _unsubscribe

    def emit(self, event: Event) -> None:
        """Fan `event` out to every subscriber and live stream."""
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001 — observers must not crash the agent
                warnings.warn(
                    f"EventBus subscriber raised {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for queue in list(self._streams):
            queue.put_nowait(event)
            if isinstance(event, AgentStopped):
                queue.put_nowait(_STREAM_END)

    def stream(self) -> AsyncIterator[Event]:
        """Return an async iterator of events. Terminates after the next `AgentStopped`."""
        queue: asyncio.Queue[Event | _StreamEnd] = asyncio.Queue()
        self._streams.append(queue)

        async def _iter() -> AsyncIterator[Event]:
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, _StreamEnd):
                        return
                    yield item
            finally:
                with contextlib.suppress(ValueError):
                    self._streams.remove(queue)

        return _iter()


__all__ = [
    "AgentStarted",
    "AgentStopped",
    "Event",
    "EventBus",
    "ModelRequestSent",
    "ModelResponseReceived",
    "StopReason",
    "ToolExecutionCompleted",
    "ToolExecutionStarted",
    "TurnStarted",
    "UsageUpdated",
]
