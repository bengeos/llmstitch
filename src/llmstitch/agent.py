"""Agent run loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace

from .providers.base import ProviderAdapter
from .retry import RetryAttempt, RetryPolicy, retry_call
from .tools import ToolRegistry
from .types import (
    CompletionResponse,
    Cost,
    Message,
    Pricing,
    StreamDone,
    StreamEvent,
    TokenCount,
    ToolResultBlock,
    UsageTally,
)


class MaxIterationsExceeded(RuntimeError):
    """Raised when the agent loop hits `max_iterations` without producing a final response."""


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

    async def run(self, prompt: str | list[Message]) -> list[Message]:
        """Drive the model → tool → model loop until the model stops calling tools.

        Returns the full message history (including the final assistant message).
        """
        if isinstance(prompt, str):
            messages: list[Message] = [Message(role="user", content=prompt)]
        else:
            messages = list(prompt)

        policy = self._instrumented_policy()

        for _ in range(self.max_iterations):

            async def _complete() -> CompletionResponse:
                self.usage.record_call()
                return await self.provider.complete(
                    model=self.model,
                    messages=messages,
                    system=self.system,
                    tools=self.tools.definitions() or None,
                    max_tokens=self.max_tokens,
                )

            response = await retry_call(policy, _complete)
            self.usage.add(response.usage)
            messages.append(Message(role="assistant", content=list(response.content)))

            tool_uses = response.tool_uses()
            if not tool_uses:
                return messages

            results: list[ToolResultBlock] = await self.tools.run(
                tool_uses, timeout=self.tool_timeout
            )
            # Cast to the ContentBlock-compatible list expected by Message.
            messages.append(Message(role="user", content=list(results)))

        raise MaxIterationsExceeded(
            f"Agent exceeded max_iterations={self.max_iterations} without terminating"
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

    def cost(self) -> Cost:
        """Compute the USD cost of everything tallied in `self.usage` so far.

        Prices against `self.pricing`. The default is a placeholder
        (`Pricing(input_per_mtok=1.00, output_per_mtok=2.00)`) — pass a
        real `pricing=Pricing(...)` to the agent when you want costs that
        reflect actual vendor rates. Call `self.usage.cost(some_pricing)`
        directly to price the same tally against multiple rate cards.
        """
        return self.usage.cost(self.pricing)

    def run_sync(self, prompt: str | list[Message]) -> list[Message]:
        """Synchronous wrapper around `run()` for scripts and REPL use."""
        return asyncio.run(self.run(prompt))

    async def count_tokens(self, prompt: str | list[Message]) -> TokenCount:
        """Count input tokens for `prompt` as if `run(prompt)` were about to be called.

        Forwards to `provider.count_tokens`; raises `NotImplementedError` if
        the adapter has no native token-counting endpoint (true for OpenAI,
        Groq, and OpenRouter in v0.1.3).
        """
        messages = (
            [Message(role="user", content=prompt)] if isinstance(prompt, str) else list(prompt)
        )
        return await self.provider.count_tokens(
            model=self.model,
            messages=messages,
            system=self.system,
            tools=self.tools.definitions() or None,
        )

    async def run_stream(self, prompt: str | list[Message]) -> AsyncIterator[StreamEvent]:
        """Drive the model → tool → model loop, yielding events per model turn.

        Tool execution happens silently between turns (tool results are computed
        locally, not streamed). Each model turn emits one `StreamDone` event
        carrying that turn's assembled `CompletionResponse`; the stream ends
        when the model produces a turn with no tool calls.
        """
        if isinstance(prompt, str):
            messages: list[Message] = [Message(role="user", content=prompt)]
        else:
            messages = list(prompt)

        for _ in range(self.max_iterations):
            final_response: CompletionResponse | None = None
            self.usage.record_call()
            async for event in self.provider.stream(
                model=self.model,
                messages=messages,
                system=self.system,
                tools=self.tools.definitions() or None,
                max_tokens=self.max_tokens,
            ):
                if isinstance(event, StreamDone):
                    final_response = event.response
                yield event

            if final_response is None:
                raise RuntimeError(
                    f"{type(self.provider).__name__}.stream() ended without a StreamDone event"
                )

            self.usage.add(final_response.usage)
            messages.append(Message(role="assistant", content=list(final_response.content)))
            tool_uses = final_response.tool_uses()
            if not tool_uses:
                return

            results = await self.tools.run(tool_uses, timeout=self.tool_timeout)
            messages.append(Message(role="user", content=list(results)))

        raise MaxIterationsExceeded(
            f"Agent exceeded max_iterations={self.max_iterations} without terminating"
        )
