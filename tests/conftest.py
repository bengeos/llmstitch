"""Shared test fixtures — notably the fake provider adapter."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from llmstitch.providers.base import ProviderAdapter
from llmstitch.types import CompletionResponse, Message, ToolDefinition


class FakeAdapter(ProviderAdapter):
    """Adapter that replays a scripted sequence of `CompletionResponse`s.

    Records every call to `complete()` so tests can assert on ordering and payloads.
    """

    def __init__(self, responses: Iterable[CompletionResponse]) -> None:
        self._responses: list[CompletionResponse] = list(responses)
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


@pytest.fixture
def fake_adapter_factory() -> Any:
    def _factory(responses: Iterable[CompletionResponse]) -> FakeAdapter:
        return FakeAdapter(responses)

    return _factory
