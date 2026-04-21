"""Groq adapter.

Groq's Python SDK exposes an OpenAI-compatible Chat Completions surface, so the
message/tool translation and response parsing are identical to the OpenAI
adapter — only the client swap differs.
"""

from __future__ import annotations

import os
from typing import Any

from ..types import CompletionResponse, Message, ToolDefinition
from .openai import OpenAIAdapter


class GroqAdapter(OpenAIAdapter):
    """Adapter for Groq's Chat Completions API (the `groq` SDK, >= 0.9)."""

    _client: Any

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        import groq  # lazy

        self._client = groq.AsyncGroq(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            **client_kwargs,
        )

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
        translated = self.translate_messages(messages, system=system)
        payload: dict[str, Any] = {
            "model": model,
            "messages": translated,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self.translate_tools(tools)
        payload.update(kwargs)
        response = await self._client.chat.completions.create(**payload)
        return self.parse_response(response)
