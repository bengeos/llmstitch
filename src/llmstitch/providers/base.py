"""Provider adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..types import CompletionResponse, Message, ToolDefinition


class ProviderAdapter(ABC):
    """Abstract interface every provider implements.

    Streaming is intentionally not yet implemented for v0.1.0 — it lands in v0.2.0.
    """

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
    ) -> AsyncIterator[CompletionResponse]:
        raise NotImplementedError("Streaming lands in v0.2.0")
        yield  # pragma: no cover — satisfies the AsyncIterator type
