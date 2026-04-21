"""Provider adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..types import CompletionResponse, Message, StreamEvent, TokenCount, ToolDefinition


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

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> TokenCount:
        """Count input tokens for a pending request without generating.

        Returns a `TokenCount` with `output_tokens=None`. Adapters opt in by
        overriding this; the default raises `NotImplementedError` for
        providers without a native token-counting endpoint (llmstitch does
        not estimate, to avoid misleading counts that disagree with the
        model's own tokenizer).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement count_tokens. "
            "Native token counting is available on Anthropic and Gemini adapters."
        )

    @classmethod
    def default_retryable(cls) -> tuple[type[BaseException], ...]:
        """Return the vendor's transient error classes for use in a `RetryPolicy`.

        Default: empty tuple (no retries). Adapters override to list the
        exceptions raised by their SDK for rate limits, timeouts, and
        5xx errors. Vendor SDKs are imported lazily inside the override.
        """
        return ()
