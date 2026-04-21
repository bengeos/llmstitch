"""Shared test fixtures — notably the fake provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest

from llmstitch.providers.base import ProviderAdapter
from llmstitch.types import (
    CompletionResponse,
    Message,
    MessageStop,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolDefinition,
    ToolUseBlock,
    ToolUseDelta,
    ToolUseStart,
    ToolUseStop,
)


class FakeAdapter(ProviderAdapter):
    """Adapter that replays a scripted sequence of `CompletionResponse`s.

    Records every call to `complete()` / `stream()` so tests can assert on
    ordering and payloads. `stream()` auto-decomposes the scripted response
    into a plausible event sequence — callers who want specific event
    orderings can pass `stream_scripts` to override.
    """

    def __init__(
        self,
        responses: Iterable[CompletionResponse],
        *,
        stream_scripts: Iterable[list[StreamEvent]] | None = None,
    ) -> None:
        self._responses: list[CompletionResponse] = list(responses)
        self._stream_scripts: list[list[StreamEvent]] | None = (
            [list(s) for s in stream_scripts] if stream_scripts is not None else None
        )
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "method": "complete",
                "model": model,
                "messages": list(messages),
                "system": system,
                "tools": list(tools) if tools else None,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        if not self._responses:
            raise AssertionError("FakeAdapter out of scripted responses")
        return self._responses.pop(0)

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
        self.calls.append(
            {
                "method": "stream",
                "model": model,
                "messages": list(messages),
                "system": system,
                "tools": list(tools) if tools else None,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        if self._stream_scripts is not None:
            if not self._stream_scripts:
                raise AssertionError("FakeAdapter out of scripted stream sequences")
            for event in self._stream_scripts.pop(0):
                yield event
            return
        if not self._responses:
            raise AssertionError("FakeAdapter out of scripted responses")
        response = self._responses.pop(0)
        tool_ids: list[str] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                yield TextDelta(text=block.text)
            elif isinstance(block, ToolUseBlock):
                yield ToolUseStart(id=block.id, name=block.name)
                yield ToolUseDelta(id=block.id, partial_json=json.dumps(block.input))
                tool_ids.append(block.id)
        for tid in tool_ids:
            yield ToolUseStop(id=tid)
        yield MessageStop(stop_reason=response.stop_reason, usage=response.usage)
        yield StreamDone(response=response)


@pytest.fixture
def fake_adapter_factory() -> Any:
    def _factory(responses: Iterable[CompletionResponse]) -> FakeAdapter:
        return FakeAdapter(responses)

    return _factory
