"""Provider adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..types import CompletionResponse, Message, StreamEvent, ToolDefinition


class ProviderAdapter(ABC):
    """Abstract interface every provider implements."""

    @abstractmethod
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
        """Single-shot completion. Returns the model's response."""

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion as a sequence of `StreamEvent`s.

        Adapters must emit exactly one terminal `StreamDone` event carrying the
        fully-assembled `CompletionResponse`. Events in between — `TextDelta`,
        `ToolUseStart` / `ToolUseDelta` / `ToolUseStop`, `MessageStop` — arrive
        in the order the provider emits them and are safe to render directly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement streaming.")
        yield  # pragma: no cover — satisfies the AsyncIterator type
