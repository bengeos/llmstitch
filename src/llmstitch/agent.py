"""Agent run loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .providers.base import ProviderAdapter
from .tools import ToolRegistry
from .types import Message, ToolResultBlock


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
