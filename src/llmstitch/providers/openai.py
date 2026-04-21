"""OpenAI Chat Completions adapter."""

from __future__ import annotations

import json
from typing import Any

from ..types import (
    CompletionResponse,
    ContentBlock,
    Message,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from .base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI's Chat Completions API (the `openai` SDK, >= 1.50)."""

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        import openai  # lazy

        self._client = openai.AsyncOpenAI(api_key=api_key, **client_kwargs)

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

    @staticmethod
    def translate_messages(
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "system":
                content = msg.content if isinstance(msg.content, str) else ""
                out.append({"role": "system", "content": content})
                continue
            if isinstance(msg.content, str):
                out.append({"role": msg.role, "content": msg.content})
                continue
            # Tool results are user-side messages with role="tool" per the Chat Completions API.
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results and msg.role == "user":
                for tr in tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.content,
                        }
                    )
                remaining = [b for b in msg.content if not isinstance(b, ToolResultBlock)]
                if remaining:
                    text_parts = [b.text for b in remaining if isinstance(b, TextBlock)]
                    if text_parts:
                        out.append({"role": "user", "content": "\n".join(text_parts)})
                continue
            # Assistant messages can carry text + tool_use blocks.
            text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
            entry: dict[str, Any] = {"role": msg.role}
            if text_parts:
                entry["content"] = "\n".join(text_parts)
            else:
                entry["content"] = None
            if tool_uses:
                entry["tool_calls"] = [
                    {
                        "id": tu.id,
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.input),
                        },
                    }
                    for tu in tool_uses
                ]
            out.append(entry)
        return out

    @staticmethod
    def translate_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    @staticmethod
    def parse_response(response: Any) -> CompletionResponse:
        choice = response.choices[0]
        message = choice.message
        content: list[ContentBlock] = []
        text = getattr(message, "content", None)
        if text:
            content.append(TextBlock(text=text))
        for call in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments) if call.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": call.function.arguments}
            content.append(ToolUseBlock(id=call.id, name=call.function.name, input=args))
        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "input_tokens": getattr(raw_usage, "prompt_tokens", 0),
                "output_tokens": getattr(raw_usage, "completion_tokens", 0),
            }
        return CompletionResponse(
            content=content,
            stop_reason=choice.finish_reason or "stop",
            usage=usage,
            raw=response,
        )
