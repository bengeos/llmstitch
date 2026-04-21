"""Anthropic Messages API adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..types import (
    CompletionResponse,
    ContentBlock,
    Message,
    MessageStop,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    TokenCount,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
    ToolUseStart,
    ToolUseStop,
)
from .base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic's Messages API (the `anthropic` SDK, >= 0.40)."""

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        import anthropic  # lazy — core install must not require the SDK

        self._client = anthropic.AsyncAnthropic(api_key=api_key, **client_kwargs)

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
        translated = self.translate_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": translated,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self.translate_tools(tools)
        payload.update(kwargs)
        response = await self._client.messages.create(**payload)
        return self.parse_response(response)

    @staticmethod
    def translate_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                # System is passed via top-level `system=`; skip here.
                continue
            if isinstance(msg.content, str):
                out.append({"role": msg.role, "content": msg.content})
                continue
            blocks: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )
            out.append({"role": msg.role, "content": blocks})
        return out

    @staticmethod
    def translate_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    @staticmethod
    def parse_response(response: Any) -> CompletionResponse:
        content: list[ContentBlock] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content.append(TextBlock(text=block.text))
            elif btype == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "input_tokens": getattr(raw_usage, "input_tokens", 0),
                "output_tokens": getattr(raw_usage, "output_tokens", 0),
            }
        return CompletionResponse(
            content=content,
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            usage=usage,
            raw=response,
        )

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
        translated = self.translate_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": translated,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self.translate_tools(tools)
        payload.update(kwargs)

        # Track tool-use ids by content_block index so we can emit matching stops.
        tool_ids_by_idx: dict[int, str] = {}
        # Cast to Any at the SDK boundary — mirrors `parse_response(response: Any)`.
        # The SDK's tagged-union event types are narrower than our runtime checks need.
        stream_ctx: Any = self._client.messages.stream(**payload)
        async with stream_ctx as stream:
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "content_block_start":
                    block = event.content_block
                    btype = getattr(block, "type", "")
                    if btype == "tool_use":
                        tool_ids_by_idx[event.index] = block.id
                        yield ToolUseStart(id=block.id, name=block.name)
                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif dtype == "input_json_delta":
                        tool_id = tool_ids_by_idx.get(event.index, "")
                        yield ToolUseDelta(id=tool_id, partial_json=delta.partial_json)
                elif etype == "content_block_stop":
                    stop_id = tool_ids_by_idx.get(event.index)
                    if stop_id is not None:
                        yield ToolUseStop(id=stop_id)
            final_message: Any = await stream.get_final_message()

        content: list[ContentBlock] = []
        for block in final_message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content.append(TextBlock(text=block.text))
            elif btype == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
        usage = None
        raw_usage = getattr(final_message, "usage", None)
        if raw_usage is not None:
            usage = {
                "input_tokens": getattr(raw_usage, "input_tokens", 0),
                "output_tokens": getattr(raw_usage, "output_tokens", 0),
            }
        stop_reason = getattr(final_message, "stop_reason", "end_turn") or "end_turn"

        yield MessageStop(stop_reason=stop_reason, usage=usage)
        yield StreamDone(
            response=CompletionResponse(
                content=content,
                stop_reason=stop_reason,
                usage=usage,
                raw=final_message,
            )
        )

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> TokenCount:
        translated = self.translate_messages(messages)
        payload: dict[str, Any] = {"model": model, "messages": translated}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self.translate_tools(tools)
        response = await self._client.messages.count_tokens(**payload)
        return TokenCount(input_tokens=int(getattr(response, "input_tokens", 0)))

    @classmethod
    def default_retryable(cls) -> tuple[type[BaseException], ...]:
        import anthropic  # lazy — core install must not require the SDK

        return (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        )
