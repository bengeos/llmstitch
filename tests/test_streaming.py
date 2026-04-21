"""Tests for streaming: `Agent.run_stream` + per-adapter event assembly.

Adapter-level tests inject hand-rolled async iterators to mimic each vendor's
SDK stream shape — no network and no real SDK objects are required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from llmstitch import Agent, tool
from llmstitch.providers.anthropic import AnthropicAdapter
from llmstitch.providers.gemini import GeminiAdapter
from llmstitch.providers.openai import OpenAIAdapter
from llmstitch.types import (
    CompletionResponse,
    MessageStop,
    StreamDone,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
    ToolUseStart,
    ToolUseStop,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# ---------- Agent.run_stream ---------- #


async def test_run_stream_text_only_terminates(fake_adapter_factory: Any) -> None:
    adapter = fake_adapter_factory(
        [CompletionResponse(content=[TextBlock(text="Hello!")], stop_reason="end_turn")]
    )
    agent = Agent(provider=adapter, model="test-model")

    events = [e async for e in agent.run_stream("hi")]

    # FakeAdapter emits: TextDelta, MessageStop, StreamDone.
    assert [type(e).__name__ for e in events] == ["TextDelta", "MessageStop", "StreamDone"]
    assert isinstance(events[0], TextDelta) and events[0].text == "Hello!"
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.response.text() == "Hello!"


async def test_run_stream_tool_roundtrip_yields_two_turns(fake_adapter_factory: Any) -> None:
    adapter = fake_adapter_factory(
        [
            CompletionResponse(
                content=[ToolUseBlock(id="u1", name="multiply", input={"a": 3, "b": 4})],
                stop_reason="tool_use",
            ),
            CompletionResponse(
                content=[TextBlock(text="12")],
                stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(provider=adapter, model="test-model")
    agent.tools.register(multiply)

    events = [e async for e in agent.run_stream("3*4?")]

    # Two model turns → two StreamDone events.
    stream_dones = [e for e in events if isinstance(e, StreamDone)]
    assert len(stream_dones) == 2
    assert stream_dones[0].response.tool_uses()[0].input == {"a": 3, "b": 4}
    assert stream_dones[1].response.text() == "12"

    # Tool-use bracketing appears in the first turn.
    starts = [e for e in events if isinstance(e, ToolUseStart)]
    stops = [e for e in events if isinstance(e, ToolUseStop)]
    assert len(starts) == 1 and len(stops) == 1
    assert starts[0].id == stops[0].id == "u1"

    # FakeAdapter recorded two stream() calls — one per model turn.
    stream_calls = [c for c in adapter.calls if c["method"] == "stream"]
    assert len(stream_calls) == 2


async def test_run_stream_raises_when_adapter_omits_stream_done() -> None:
    class BadAdapter:
        """Adapter whose stream() forgets to emit StreamDone."""

        async def stream(self, **kwargs: Any) -> AsyncIterator[Any]:
            yield TextDelta(text="oops")
            yield MessageStop(stop_reason="stop")

        async def complete(self, **kwargs: Any) -> CompletionResponse:
            raise AssertionError("unused")

    agent = Agent(provider=BadAdapter(), model="test-model")  # type: ignore[arg-type]

    async def consume() -> None:
        async for _ in agent.run_stream("x"):
            pass

    with pytest.raises(RuntimeError, match="StreamDone"):
        await consume()


async def test_run_stream_feeds_tool_results_between_turns(fake_adapter_factory: Any) -> None:
    adapter = fake_adapter_factory(
        [
            CompletionResponse(
                content=[ToolUseBlock(id="u1", name="multiply", input={"a": 2, "b": 5})],
                stop_reason="tool_use",
            ),
            CompletionResponse(content=[TextBlock(text="10")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=adapter, model="test-model")
    agent.tools.register(multiply)

    async for _ in agent.run_stream("2*5?"):
        pass

    # Second stream() call receives the tool-result message in history.
    second_call = [c for c in adapter.calls if c["method"] == "stream"][1]
    history = second_call["messages"]
    tool_result_msg = history[2]
    assert isinstance(tool_result_msg.content, list)
    assert isinstance(tool_result_msg.content[0], ToolResultBlock)
    assert tool_result_msg.content[0].content == "10"


# ---------- OpenAI stream assembly ---------- #


def _async_iter(items: list[Any]) -> AsyncIterator[Any]:
    async def _gen() -> AsyncIterator[Any]:
        for item in items:
            yield item

    return _gen()


def _openai_chunk(
    *,
    text: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeOpenAIClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._chunks = chunks
        self.create_payload: dict[str, Any] | None = None

    async def _create(self, **payload: Any) -> AsyncIterator[Any]:
        self.create_payload = payload
        return _async_iter(self._chunks)


async def test_openai_stream_assembles_text_and_tool_call() -> None:
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    tc_delta_first = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="multiply", arguments='{"a":'),
    )
    tc_delta_more = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name=None, arguments=' 3, "b": 4}'),
    )
    chunks = [
        _openai_chunk(text="hel"),
        _openai_chunk(text="lo"),
        _openai_chunk(tool_calls=[tc_delta_first]),
        _openai_chunk(tool_calls=[tc_delta_more]),
        _openai_chunk(finish_reason="tool_calls"),
        _openai_chunk(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=11)),
    ]
    adapter._client = _FakeOpenAIClient(chunks)  # type: ignore[attr-defined]

    events = [e async for e in adapter.stream(model="gpt-test", messages=[])]

    # Expected order: TextDelta×2, ToolUseStart, ToolUseDelta×2, ToolUseStop,
    # MessageStop, StreamDone.
    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "TextDelta",
        "TextDelta",
        "ToolUseStart",
        "ToolUseDelta",
        "ToolUseDelta",
        "ToolUseStop",
        "MessageStop",
        "StreamDone",
    ]

    assembled = events[-1]
    assert isinstance(assembled, StreamDone)
    final = assembled.response
    assert final.text() == "hello"
    tu = final.tool_uses()[0]
    assert tu.id == "call_1"
    assert tu.name == "multiply"
    assert tu.input == {"a": 3, "b": 4}
    assert final.stop_reason == "tool_calls"
    assert final.usage == {"input_tokens": 7, "output_tokens": 11}


async def test_openai_stream_enables_stream_flags() -> None:
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    client = _FakeOpenAIClient([_openai_chunk(text="ok", finish_reason="stop")])
    adapter._client = client  # type: ignore[attr-defined]

    async for _ in adapter.stream(model="gpt-test", messages=[]):
        pass

    assert client.create_payload is not None
    assert client.create_payload["stream"] is True
    assert client.create_payload["stream_options"] == {"include_usage": True}


# ---------- Anthropic stream assembly ---------- #


class _FakeAnthropicStream:
    def __init__(
        self,
        events: list[Any],
        final_message: Any,
    ) -> None:
        self._events = events
        self._final_message = final_message

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        return _async_iter(list(self._events))

    async def get_final_message(self) -> Any:
        return self._final_message


class _FakeAnthropicClient:
    def __init__(self, stream: _FakeAnthropicStream) -> None:
        self._stream = stream
        self.messages = SimpleNamespace(stream=self._open)

    def _open(self, **payload: Any) -> _FakeAnthropicStream:
        return self._stream


async def test_anthropic_stream_assembles_text_and_tool_call() -> None:
    events = [
        SimpleNamespace(
            type="content_block_start", index=0, content_block=SimpleNamespace(type="text")
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="hi "),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="there"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="u1", name="multiply"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"a":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json=' 3, "b": 4}'),
        ),
        SimpleNamespace(type="content_block_stop", index=1),
    ]
    final_message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hi there"),
            SimpleNamespace(type="tool_use", id="u1", name="multiply", input={"a": 3, "b": 4}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=20, output_tokens=5),
    )
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter._client = _FakeAnthropicClient(_FakeAnthropicStream(events, final_message))  # type: ignore[attr-defined]

    collected = [e async for e in adapter.stream(model="c-test", messages=[])]

    kinds = [type(e).__name__ for e in collected]
    assert kinds == [
        "TextDelta",
        "TextDelta",
        "ToolUseStart",
        "ToolUseDelta",
        "ToolUseDelta",
        "ToolUseStop",
        "MessageStop",
        "StreamDone",
    ]
    done = collected[-1]
    assert isinstance(done, StreamDone)
    assert done.response.text() == "hi there"
    assert done.response.tool_uses()[0].input == {"a": 3, "b": 4}
    assert done.response.stop_reason == "tool_use"


# ---------- Gemini stream assembly ---------- #


class _FakeGeminiClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content_stream=self._start))
        self._chunks = chunks

    async def _start(self, **kwargs: Any) -> AsyncIterator[Any]:
        return _async_iter(self._chunks)


async def test_gemini_stream_emits_function_call_burst() -> None:
    text_part = SimpleNamespace(text="hello ", function_call=None)
    fn_call_part = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="multiply", args={"a": 3, "b": 4}),
    )
    text_part_2 = SimpleNamespace(text="world", function_call=None)
    chunk_1 = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[text_part]),
                finish_reason=None,
            )
        ],
        usage_metadata=None,
    )
    chunk_2 = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[fn_call_part, text_part_2]),
                finish_reason=SimpleNamespace(name="STOP"),
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=4, candidates_token_count=6),
    )
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter._client = _FakeGeminiClient([chunk_1, chunk_2])  # type: ignore[attr-defined]

    collected = [e async for e in adapter.stream(model="gemini-test", messages=[])]

    kinds = [type(e).__name__ for e in collected]
    # TextDelta, ToolUseStart+Delta+Stop, TextDelta, MessageStop, StreamDone
    assert kinds == [
        "TextDelta",
        "ToolUseStart",
        "ToolUseDelta",
        "ToolUseStop",
        "TextDelta",
        "MessageStop",
        "StreamDone",
    ]
    done = collected[-1]
    assert isinstance(done, StreamDone)
    assert done.response.text() == "hello world"
    tu = done.response.tool_uses()[0]
    assert tu.name == "multiply"
    assert tu.input == {"a": 3, "b": 4}
    # Synthetic id is "{name}_{idx}" — first tool call in the stream has idx 0.
    assert tu.id == "multiply_0"
    assert done.response.usage == {"input_tokens": 4, "output_tokens": 6}

    # Sanity: the ToolUseDelta partial_json should round-trip through json.loads.
    tool_delta = collected[2]
    assert isinstance(tool_delta, ToolUseDelta)
    assert json.loads(tool_delta.partial_json) == {"a": 3, "b": 4}
