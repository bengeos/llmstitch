"""Non-raising agent result types.

`AgentResult` is the return shape of `Agent.run_with_result(...)`, which
catches `MaxIterationsExceeded`, `CostCeilingExceeded`, and vendor errors
and packages them into a structured result instead of propagating.

`AgentResultEvent` is the terminal event yielded by
`Agent.run_stream_with_result(...)` — it wraps an `AgentResult` and is
emitted after the final `StreamEvent` of the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .events import StopReason
from .types import Cost, Message, UsageTally


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Structured outcome of an `Agent.run_with_result(...)` invocation.

    `cost` is always populated in v0.1.4 (pricing defaults to a placeholder
    on `Agent`); the `| None` leaves room for a future `pricing=None` mode.
    """

    messages: list[Message]
    text: str
    stop_reason: StopReason
    turns: int
    usage: UsageTally
    cost: Cost | None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class AgentResultEvent:
    """Terminal event from `Agent.run_stream_with_result(...)`."""

    result: AgentResult


__all__ = ["AgentResult", "AgentResultEvent"]
