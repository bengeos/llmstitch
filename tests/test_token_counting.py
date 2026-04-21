"""Tests for `Agent.count_tokens` and per-adapter implementations.

Adapter tests inject stub clients so no SDK is actually constructed and no
network call is possible, mirroring the style of `tests/test_streaming.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llmstitch import Agent, TokenCount
from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.providers.gemini import GeminiAdapter
from llmstitch.providers.openai import OpenAIAdapter

# ---------- Base adapter default: NotImplementedError ---------- #


async def test_openai_count_tokens_raises_not_implemented() -> None:
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    agent = Agent(provider=adapter, model="gpt-test")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Anthropic and Gemini"):
        await agent.count_tokens("hi")


# ---------- Anthropic ---------- #


class _FakeAnthropicClient:
    def __init__(self, *, input_tokens: int) -> None:
        self.messages = SimpleNamespace(count_tokens=self._count)
        self.last_payload: dict[str, Any] | None = None
        self._input_tokens = input_tokens

    async def _count(self, **payload: Any) -> Any:
        self.last_payload = payload
        return SimpleNamespace(input_tokens=self._input_tokens)


async def test_anthropic_count_tokens_forwards_messages_and_system() -> None:
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    client = _FakeAnthropicClient(input_tokens=42)
    adapter._client = client  # type: ignore[attr-defined]

    agent = Agent(provider=adapter, model="claude-test", system="be kind")
    result = await agent.count_tokens("hello there")

    assert isinstance(result, TokenCount)
    assert result.input_tokens == 42
    assert result.output_tokens is None
    assert client.last_payload is not None
    assert client.last_payload["model"] == "claude-test"
    assert client.last_payload["system"] == "be kind"
    assert client.last_payload["messages"] == [{"role": "user", "content": "hello there"}]


async def test_anthropic_count_tokens_includes_tools() -> None:
    from llmstitch import tool

    @tool
    def ping(host: str) -> str:
        """Ping a host."""
        return host

    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    client = _FakeAnthropicClient(input_tokens=10)
    adapter._client = client  # type: ignore[attr-defined]

    agent = Agent(provider=adapter, model="claude-test")
    agent.tools.register(ping)
    await agent.count_tokens("x")

    assert client.last_payload is not None
    tools_payload = client.last_payload["tools"]
    assert tools_payload[0]["name"] == "ping"
    assert "input_schema" in tools_payload[0]


# ---------- Gemini ---------- #


class _FakeGeminiClient:
    def __init__(self, *, total_tokens: int) -> None:
        self.aio = SimpleNamespace(models=SimpleNamespace(count_tokens=self._count))
        self.last_kwargs: dict[str, Any] | None = None
        self._total = total_tokens

    async def _count(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return SimpleNamespace(total_tokens=self._total)


async def test_gemini_count_tokens_reads_total_tokens() -> None:
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    client = _FakeGeminiClient(total_tokens=17)
    adapter._client = client  # type: ignore[attr-defined]

    agent = Agent(provider=adapter, model="gemini-test")
    result = await agent.count_tokens("hi")

    assert result.input_tokens == 17
    assert result.output_tokens is None
    assert client.last_kwargs is not None
    assert client.last_kwargs["model"] == "gemini-test"
    # Translated to Gemini's contents shape.
    assert client.last_kwargs["contents"][0]["role"] == "user"
    assert client.last_kwargs["contents"][0]["parts"][0] == {"text": "hi"}


async def test_gemini_count_tokens_falls_back_to_total_token_count() -> None:
    """Older google-genai responses expose `total_token_count` instead of `total_tokens`."""

    class _LegacyClient:
        def __init__(self) -> None:
            self.aio = SimpleNamespace(models=SimpleNamespace(count_tokens=self._count))

        async def _count(self, **kwargs: Any) -> Any:
            del kwargs
            # No `total_tokens` attr — only the legacy name.
            return SimpleNamespace(total_token_count=25)

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter._client = _LegacyClient()  # type: ignore[attr-defined]
    agent = Agent(provider=adapter, model="gemini-test")

    result = await agent.count_tokens("legacy")
    assert result.input_tokens == 25
