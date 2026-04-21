"""Agent run loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .providers.base import ProviderAdapter
from .tools import ToolRegistry
from .types import CompletionResponse, Message, StreamDone, StreamEvent, ToolResultBlock


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

    async def run(self, prompt: str | list[Message]) -> list[Message]:
        """Drive the model → tool → model loop until the model stops calling tools.

        Returns the full message history (including the final assistant message).
        """
        if isinstance(prompt, str):
            messages: list[Message] = [Message(role="user", content=prompt)]
        else:
            messages = list(prompt)

        for _ in range(self.max_iterations):
            response = await self.provider.complete(
                model=self.model,
                messages=messages,
                system=self.system,
                tools=self.tools.definitions() or None,
                max_tokens=self.max_tokens,
            )
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

    def run_sync(self, prompt: str | list[Message]) -> list[Message]:
        """Synchronous wrapper around `run()` for scripts and REPL use."""
        return asyncio.run(self.run(prompt))

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

            messages.append(Message(role="assistant", content=list(final_response.content)))
            tool_uses = final_response.tool_uses()
            if not tool_uses:
                return

            results = await self.tools.run(tool_uses, timeout=self.tool_timeout)
            messages.append(Message(role="user", content=list(results)))

        raise MaxIterationsExceeded(
            f"Agent exceeded max_iterations={self.max_iterations} without terminating"
        )
