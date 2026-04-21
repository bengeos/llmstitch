"""Agent run loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import CostCeilingExceeded, MaxIterationsExceeded
from .events import (
    AgentStarted,
    AgentStopped,
    EventBus,
    ModelRequestSent,
    ModelResponseReceived,
    StopReason,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnStarted,
    UsageUpdated,
)
from .providers.base import ProviderAdapter
from .result import AgentResult, AgentResultEvent
from .retry import RetryAttempt, RetryPolicy, retry_call
from .tools import ToolRegistry
from .types import (
    CompletionResponse,
    Cost,
    Message,
    Pricing,
    StreamDone,
    StreamEvent,
    TextBlock,
    TokenCount,
    ToolResultBlock,
    ToolUseBlock,
    UsageTally,
)


@dataclass
class Agent:
    provider: ProviderAdapter
    model: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    system: str | None = None
    max_iterations: int = 10
    tool_timeout: float | None = 30.0
    max_tokens: int = 4096
    retry_policy: RetryPolicy | None = None
    pricing: Pricing = field(
        default_factory=lambda: Pricing(input_per_mtok=1.00, output_per_mtok=2.00)
    )
    usage: UsageTally = field(default_factory=UsageTally)
    event_bus: EventBus | None = None
    cost_ceiling: float | None = None

    async def run(self, prompt: str | list[Message]) -> list[Message]:
        """Drive the model → tool → model loop until the model stops calling tools.

        Returns the full message history (including the final assistant message).
        Raises `MaxIterationsExceeded` if the loop overruns, or
        `CostCeilingExceeded` if a configured ceiling is crossed.
        """
        messages = self._normalize_prompt(prompt)
        self._emit(AgentStarted(prompt=prompt, model=self.model))
        await self._run_loop(messages, prompt_for_event=prompt)
        return messages

    async def _run_loop(
        self,
        messages: list[Message],
        *,
        prompt_for_event: str | list[Message] | None = None,
    ) -> None:
        """Shared non-streaming loop body. Mutates `messages` in place."""
        del prompt_for_event  # reserved for future hooks; AgentStarted already emitted
        policy = self._instrumented_policy()

        async def _complete() -> CompletionResponse:
            self.usage.record_call()
            return await self.provider.complete(**self._provider_kwargs(messages))

        try:
            for turn in range(1, self.max_iterations + 1):
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                response = await retry_call(policy, _complete)
                self._emit(ModelResponseReceived(turn=turn, response=response))
                if not await self._apply_response(response, messages, turn=turn):
                    self._emit(AgentStopped(stop_reason="complete", turns=self.usage.turns))
                    return
        except BaseException as exc:
            self._emit_stopped_for(exc)
            raise

        self._emit(AgentStopped(stop_reason="max_iterations", turns=self.usage.turns))
        raise MaxIterationsExceeded(
            f"Agent exceeded max_iterations={self.max_iterations} without terminating"
        )

    async def run_stream(self, prompt: str | list[Message]) -> AsyncIterator[StreamEvent]:
        """Drive the model → tool → model loop, yielding events per model turn.

        Tool execution happens silently between turns (tool results are computed
        locally, not streamed). Each model turn emits one `StreamDone` event
        carrying that turn's assembled `CompletionResponse`; the stream ends
        when the model produces a turn with no tool calls.
        """
        messages = self._normalize_prompt(prompt)
        self._emit(AgentStarted(prompt=prompt, model=self.model))

        try:
            for turn in range(1, self.max_iterations + 1):
                final_response: CompletionResponse | None = None
                self.usage.record_call()
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                async for event in self.provider.stream(**self._provider_kwargs(messages)):
                    if isinstance(event, StreamDone):
                        final_response = event.response
                    yield event

                if final_response is None:
                    raise RuntimeError(
                        f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                    )
                self._emit(ModelResponseReceived(turn=turn, response=final_response))

                if not await self._apply_response(final_response, messages, turn=turn):
                    self._emit(AgentStopped(stop_reason="complete", turns=self.usage.turns))
                    return
        except BaseException as exc:
            self._emit_stopped_for(exc)
            raise

        self._emit(AgentStopped(stop_reason="max_iterations", turns=self.usage.turns))
        raise MaxIterationsExceeded(
            f"Agent exceeded max_iterations={self.max_iterations} without terminating"
        )

    def run_sync(self, prompt: str | list[Message]) -> list[Message]:
        """Synchronous wrapper around `run()` for scripts and REPL use."""
        return asyncio.run(self.run(prompt))

    async def run_with_result(self, prompt: str | list[Message]) -> AgentResult:
        """Non-raising variant of `run()` that packages outcomes into `AgentResult`.

        Catches `MaxIterationsExceeded`, `CostCeilingExceeded`, and any other
        exception raised by the provider or tool execution; never propagates.
        Always returns partial message history alongside a populated
        `stop_reason` and (when relevant) the captured `error`.
        """
        messages = self._normalize_prompt(prompt)
        self._emit(AgentStarted(prompt=prompt, model=self.model))
        stop_reason: StopReason = "complete"
        error: Exception | None = None
        try:
            await self._run_loop(messages)
        except MaxIterationsExceeded:
            stop_reason = "max_iterations"
        except CostCeilingExceeded as exc:
            stop_reason = "cost_ceiling"
            error = exc
        except Exception as exc:  # noqa: BLE001 — caller opted into non-raising
            stop_reason = "error"
            error = exc
        return self._build_result(messages, stop_reason=stop_reason, error=error)

    async def run_stream_with_result(
        self, prompt: str | list[Message]
    ) -> AsyncIterator[StreamEvent | AgentResultEvent]:
        """Non-raising streaming variant. Yields `StreamEvent`s, then `AgentResultEvent`.

        Tool results and message history are reconstructed from what the caller
        observes — same mechanism `run_stream` uses internally. Caller should
        iterate until the terminal `AgentResultEvent`.
        """
        messages: list[Message] = self._normalize_prompt(prompt)
        stop_reason: StopReason = "complete"
        error: Exception | None = None
        self._emit(AgentStarted(prompt=prompt, model=self.model))

        try:
            for turn in range(1, self.max_iterations + 1):
                final_response: CompletionResponse | None = None
                self.usage.record_call()
                self._emit(TurnStarted(turn=turn))
                self._emit(ModelRequestSent(turn=turn, messages=list(messages)))
                async for event in self.provider.stream(**self._provider_kwargs(messages)):
                    if isinstance(event, StreamDone):
                        final_response = event.response
                    yield event

                if final_response is None:
                    raise RuntimeError(
                        f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                    )
                self._emit(ModelResponseReceived(turn=turn, response=final_response))

                if not await self._apply_response(final_response, messages, turn=turn):
                    break
            else:
                stop_reason = "max_iterations"
        except MaxIterationsExceeded:
            stop_reason = "max_iterations"
        except CostCeilingExceeded as exc:
            stop_reason = "cost_ceiling"
            error = exc
        except Exception as exc:  # noqa: BLE001 — caller opted into non-raising
            stop_reason = "error"
            error = exc

        self._emit(AgentStopped(stop_reason=stop_reason, turns=self.usage.turns, error=error))
        yield AgentResultEvent(
            result=self._build_result(messages, stop_reason=stop_reason, error=error)
        )

    async def count_tokens(self, prompt: str | list[Message]) -> TokenCount:
        """Count input tokens for `prompt` as if `run(prompt)` were about to be called.

        Forwards to `provider.count_tokens`; raises `NotImplementedError` if
        the adapter has no native token-counting endpoint (true for OpenAI,
        Groq, and OpenRouter in v0.1.3).
        """
        messages = self._normalize_prompt(prompt)
        return await self.provider.count_tokens(
            model=self.model,
            messages=messages,
            system=self.system,
            tools=self.tools.definitions() or None,
        )

    def cost(self) -> Cost:
        """Compute the USD cost of everything tallied in `self.usage` so far.

        Prices against `self.pricing`. The default is a placeholder
        (`Pricing(input_per_mtok=1.00, output_per_mtok=2.00)`) — pass a
        real `pricing=Pricing(...)` to the agent when you want costs that
        reflect actual vendor rates. Call `self.usage.cost(some_pricing)`
        directly to price the same tally against multiple rate cards.
        """
        return self.usage.cost(self.pricing)

    # --- private helpers ---

    @staticmethod
    def _normalize_prompt(prompt: str | list[Message]) -> list[Message]:
        """Coerce a `str` prompt into `[user message]`; copy a list as-is."""
        if isinstance(prompt, str):
            return [Message(role="user", content=prompt)]
        return list(prompt)

    def _provider_kwargs(self, messages: list[Message]) -> dict[str, Any]:
        """Shared kwargs passed to `provider.complete(...)` and `provider.stream(...)`."""
        return {
            "model": self.model,
            "messages": messages,
            "system": self.system,
            "tools": self.tools.definitions() or None,
            "max_tokens": self.max_tokens,
        }

    async def _apply_response(
        self,
        response: CompletionResponse,
        messages: list[Message],
        *,
        turn: int,
    ) -> bool:
        """Fold one model turn into the conversation.

        Updates `usage`, appends the assistant message, enforces the cost
        ceiling, runs any tool calls, and appends tool results. Returns
        `True` if the loop should continue (the model called tools) or
        `False` if it should stop (the model produced a final response).
        """
        self.usage.add(response.usage)
        self._emit(UsageUpdated(turn=turn, usage=self.usage, delta=response.usage))
        self._check_cost_ceiling()
        messages.append(Message(role="assistant", content=list(response.content)))
        tool_uses = response.tool_uses()
        if not tool_uses:
            return False
        results = await self._run_tools(tool_uses, turn=turn)
        messages.append(Message(role="user", content=list(results)))
        return True

    async def _run_tools(
        self, tool_uses: list[ToolUseBlock], *, turn: int
    ) -> list[ToolResultBlock]:
        """Execute `tool_uses` through the registry, emitting per-tool events."""
        if self.event_bus is None:
            return await self.tools.run(tool_uses, timeout=self.tool_timeout)

        def _on_start(call: ToolUseBlock) -> None:
            self._emit(ToolExecutionStarted(turn=turn, call=call))

        def _on_complete(call: ToolUseBlock, result: ToolResultBlock, duration_s: float) -> None:
            self._emit(
                ToolExecutionCompleted(turn=turn, call=call, result=result, duration_s=duration_s)
            )

        return await self.tools.run(
            tool_uses,
            timeout=self.tool_timeout,
            on_start=_on_start,
            on_complete=_on_complete,
        )

    def _check_cost_ceiling(self) -> None:
        """Raise `CostCeilingExceeded` if running cost has crossed `self.cost_ceiling`."""
        if self.cost_ceiling is None:
            return
        spent = self.cost().total
        if spent > self.cost_ceiling:
            raise CostCeilingExceeded(spent=spent, ceiling=self.cost_ceiling)

    def _emit(
        self,
        event: AgentStarted
        | TurnStarted
        | ModelRequestSent
        | ModelResponseReceived
        | ToolExecutionStarted
        | ToolExecutionCompleted
        | UsageUpdated
        | AgentStopped,
    ) -> None:
        """Dispatch an event to the configured bus (no-op when none)."""
        if self.event_bus is not None:
            self.event_bus.emit(event)

    def _emit_stopped_for(self, exc: BaseException) -> None:
        """Emit the correct terminal `AgentStopped` for an exception about to propagate."""
        if isinstance(exc, MaxIterationsExceeded):
            self._emit(AgentStopped(stop_reason="max_iterations", turns=self.usage.turns))
        elif isinstance(exc, CostCeilingExceeded):
            self._emit(AgentStopped(stop_reason="cost_ceiling", turns=self.usage.turns, error=exc))
        elif isinstance(exc, Exception):
            self._emit(AgentStopped(stop_reason="error", turns=self.usage.turns, error=exc))

    def _build_result(
        self,
        messages: list[Message],
        *,
        stop_reason: StopReason,
        error: Exception | None,
    ) -> AgentResult:
        """Assemble an `AgentResult` from current agent state and the final history."""
        text = ""
        for msg in reversed(messages):
            if msg.role == "assistant":
                if isinstance(msg.content, str):
                    text = msg.content
                else:
                    text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
                break
        return AgentResult(
            messages=messages,
            text=text,
            stop_reason=stop_reason,
            turns=self.usage.turns,
            usage=self.usage,
            cost=self.cost(),
            error=error,
        )

    def _instrumented_policy(self) -> RetryPolicy | None:
        """Return `retry_policy` with `on_retry` wrapped so `usage.retries` ticks.

        Preserves the user's own `on_retry` callback (called after ours).
        Returns `None` when no policy is configured — `retry_call` then
        short-circuits with zero overhead.
        """
        policy = self.retry_policy
        if policy is None:
            return None
        user_cb = policy.on_retry

        def _count_and_forward(attempt: RetryAttempt) -> None:
            self.usage.record_retry()
            if user_cb is not None:
                user_cb(attempt)

        return replace(policy, on_retry=_count_and_forward)


# Re-export for backwards compatibility; canonical home is `llmstitch.errors`.
__all__ = ["Agent", "MaxIterationsExceeded"]
