"""Translation-layer tests for provider adapters.

These exercise the static translate_messages / translate_tools / parse_response
methods with plain dicts and mock-shaped responses — no SDK client is constructed,
so no network calls or API keys are needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.providers.gemini import GeminiAdapter
from llmstitch.providers.groq import GroqAdapter
from llmstitch.providers.openai import OpenAIAdapter
from llmstitch.types import (
    Message,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)

# ---------- Anthropic ---------- #


def test_anthropic_translate_messages_string_and_blocks() -> None:
    msgs = [
        Message(role="system", content="sys"),  # dropped — goes via top-level system=
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content=[
                TextBlock(text="thinking"),
                ToolUseBlock(id="u1", name="f", input={"x": 1}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="u1", content="42")],
        ),
    ]
    out = AnthropicAdapter.translate_messages(msgs)
    assert out[0] == {"role": "user", "content": "hi"}
    assistant_msg = out[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0] == {"type": "text", "text": "thinking"}
    assert assistant_msg["content"][1]["type"] == "tool_use"
    assert assistant_msg["content"][1]["id"] == "u1"
    tool_result_msg = out[2]
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "u1"


def test_anthropic_translate_tools() -> None:
    tools = [ToolDefinition(name="f", description="d", input_schema={"type": "object"})]
    out = AnthropicAdapter.translate_tools(tools)
    assert out == [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]


def test_anthropic_parse_response() -> None:
    raw = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="u1", name="f", input={"a": 1}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    resp = AnthropicAdapter.parse_response(raw)
    assert resp.stop_reason == "tool_use"
    assert resp.usage == {"input_tokens": 10, "output_tokens": 5}
    assert isinstance(resp.content[0], TextBlock)
    assert isinstance(resp.content[1], ToolUseBlock)
    assert resp.content[1].id == "u1"
    assert resp.content[1].input == {"a": 1}


# ---------- OpenAI ---------- #


def test_openai_translate_messages_with_system_and_tools() -> None:
    msgs = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content=[
                TextBlock(text="ok"),
                ToolUseBlock(id="call_1", name="f", input={"x": 1}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="42")],
        ),
    ]
    out = OpenAIAdapter.translate_messages(msgs, system="be helpful")
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}
    assistant_msg = out[2]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "ok"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "f"
    assert json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    tool_msg = out[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": "42"}


def test_openai_translate_tools() -> None:
    tools = [ToolDefinition(name="f", description="d", input_schema={"type": "object"})]
    out = OpenAIAdapter.translate_tools(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "f"
    assert out[0]["function"]["parameters"] == {"type": "object"}


def test_openai_parse_response_text_and_tool_calls() -> None:
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="f", arguments=json.dumps({"a": 2})),
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content="hello", tool_calls=[tool_call]),
        finish_reason="tool_calls",
    )
    raw = SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )
    resp = OpenAIAdapter.parse_response(raw)
    assert resp.stop_reason == "tool_calls"
    assert resp.usage == {"input_tokens": 3, "output_tokens": 4}
    assert isinstance(resp.content[0], TextBlock)
    assert isinstance(resp.content[1], ToolUseBlock)
    assert resp.content[1].input == {"a": 2}


def test_openai_parse_response_handles_bad_json_arguments() -> None:
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="f", arguments="not json"),
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[tool_call]),
        finish_reason="tool_calls",
    )
    raw = SimpleNamespace(choices=[choice], usage=None)
    resp = OpenAIAdapter.parse_response(raw)
    assert resp.content[0].input == {"_raw": "not json"}  # type: ignore[union-attr]


# ---------- Gemini ---------- #


def test_gemini_translate_messages_maps_roles() -> None:
    msgs = [
        Message(role="system", content="sys"),  # dropped
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content=[
                TextBlock(text="thinking"),
                ToolUseBlock(id="f_0", name="f", input={"x": 1}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="f", content="42")],
        ),
    ]
    out = GeminiAdapter.translate_messages(msgs)
    assert out[0] == {"role": "user", "parts": [{"text": "hi"}]}
    assert out[1]["role"] == "model"
    assert out[1]["parts"][0] == {"text": "thinking"}
    assert out[1]["parts"][1]["function_call"]["name"] == "f"
    assert out[2]["role"] == "user"
    assert out[2]["parts"][0]["function_response"]["name"] == "f"
    assert out[2]["parts"][0]["function_response"]["response"] == {"result": "42"}


def test_gemini_translate_tools() -> None:
    tools = [ToolDefinition(name="f", description="d", input_schema={"type": "object"})]
    out = GeminiAdapter.translate_tools(tools)
    assert len(out) == 1
    assert out[0]["function_declarations"][0]["name"] == "f"
    assert out[0]["function_declarations"][0]["parameters"] == {"type": "object"}


def test_gemini_parse_response_with_text_and_function_call() -> None:
    text_part = SimpleNamespace(text="hello", function_call=None)
    fn = SimpleNamespace(name="f", args={"a": 1})
    fn_call_part = SimpleNamespace(text=None, function_call=fn)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[text_part, fn_call_part]),
        finish_reason=SimpleNamespace(name="STOP"),
    )
    raw = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=2),
    )
    resp = GeminiAdapter.parse_response(raw)
    assert resp.stop_reason == "stop"
    assert resp.usage == {"input_tokens": 7, "output_tokens": 2}
    assert isinstance(resp.content[0], TextBlock)
    assert isinstance(resp.content[1], ToolUseBlock)
    assert resp.content[1].name == "f"
    assert resp.content[1].input == {"a": 1}
    # Synthetic ids are `{name}_{idx}`; idx for the second part is 1.
    assert resp.content[1].id == "f_1"


def test_gemini_parse_response_empty_candidates() -> None:
    raw = SimpleNamespace(candidates=[], usage_metadata=None)
    resp = GeminiAdapter.parse_response(raw)
    assert resp.content == []
    assert resp.stop_reason == "empty"


# ---------- Groq ---------- #


def test_groq_reuses_openai_translation() -> None:
    # Groq's wire format matches OpenAI's, so the adapter inherits translation
    # directly. Verify the static methods resolve to the same implementations.
    assert GroqAdapter.translate_messages is OpenAIAdapter.translate_messages
    assert GroqAdapter.translate_tools is OpenAIAdapter.translate_tools
    assert GroqAdapter.parse_response is OpenAIAdapter.parse_response


def test_groq_translate_messages_with_tools() -> None:
    msgs = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="f", input={"x": 1})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="42")],
        ),
    ]
    out = GroqAdapter.translate_messages(msgs, system="be helpful")
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}
