"""Google Gemini adapter (google-genai SDK)."""

from __future__ import annotations

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


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google's Gemini API via the `google-genai` SDK (>= 0.3)."""

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        from google import genai  # lazy

        self._client = genai.Client(api_key=api_key, **client_kwargs)

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
        from google.genai import types as genai_types

        contents = self.translate_messages(messages)
        config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = self.translate_tools(tools)
        config = genai_types.GenerateContentConfig(**config_kwargs)
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
            **kwargs,
        )
        return self.parse_response(response)

    @staticmethod
    def translate_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue  # system goes into config.system_instruction
            role = "model" if msg.role == "assistant" else "user"
            if isinstance(msg.content, str):
                out.append({"role": role, "parts": [{"text": msg.content}]})
                continue
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append({"text": block.text})
                elif isinstance(block, ToolUseBlock):
                    parts.append(
                        {
                            "function_call": {
                                "name": block.name,
                                "args": block.input,
                            }
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    parts.append(
                        {
                            "function_response": {
                                "name": block.tool_use_id,
                                "response": {"result": block.content},
                            }
                        }
                    )
            out.append({"role": role, "parts": parts})
        return out

    @staticmethod
    def translate_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                    for t in tools
                ]
            }
        ]

    @staticmethod
    def parse_response(response: Any) -> CompletionResponse:
        content: list[ContentBlock] = []
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return CompletionResponse(content=[], stop_reason="empty", raw=response)
        candidate = candidates[0]
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        for idx, part in enumerate(parts):
            text = getattr(part, "text", None)
            fn_call = getattr(part, "function_call", None)
            if text:
                content.append(TextBlock(text=text))
            elif fn_call is not None:
                name = getattr(fn_call, "name", "")
                args_raw = getattr(fn_call, "args", {}) or {}
                args = dict(args_raw) if not isinstance(args_raw, dict) else args_raw
                content.append(ToolUseBlock(id=f"{name}_{idx}", name=name, input=args))
        finish = getattr(candidate, "finish_reason", None)
        if finish is not None and hasattr(finish, "name"):
            stop_reason = finish.name.lower()
        else:
            stop_reason = str(finish or "stop")
        usage = None
        raw_usage = getattr(response, "usage_metadata", None)
        if raw_usage is not None:
            usage = {
                "input_tokens": getattr(raw_usage, "prompt_token_count", 0),
                "output_tokens": getattr(raw_usage, "candidates_token_count", 0),
            }
        return CompletionResponse(
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            raw=response,
        )
