"""Tests for the Agent run loop against a scripted FakeAdapter."""

from __future__ import annotations

import pytest

from llmstitch import Agent, tool
from llmstitch.agent import MaxIterationsExceeded
from llmstitch.types import (
    CompletionResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def say(msg: str) -> str:
    return msg


async def test_terminates_on_text_response(fake_adapter_factory) -> None:
    adapter = fake_adapter_factory(
        [CompletionResponse(content=[TextBlock(text="Hello!")], stop_reason="end_turn")]
    )
    agent = Agent(provider=adapter, model="test-model")
    history = await agent.run("hi")

    assert len(adapter.calls) == 1
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert isinstance(history[1].content, list)
    assert history[1].content[0] == TextBlock(text="Hello!")


async def test_tool_roundtrip(fake_adapter_factory) -> None:
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

    history = await agent.run("what is 3 * 4?")

    assert len(adapter.calls) == 2
    # user prompt, assistant tool_use, user tool_result, final assistant text
    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]
    tool_result_msg = history[2]
    assert isinstance(tool_result_msg.content, list)
    result_block = tool_result_msg.content[0]
    assert isinstance(result_block, ToolResultBlock)
    assert result_block.tool_use_id == "u1"
    assert result_block.content == "12"
    assert result_block.is_error is False


async def test_max_iterations_exceeded(fake_adapter_factory) -> None:
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="u", name="say", input={"msg": "again"})],
        stop_reason="tool_use",
    )
    adapter = fake_adapter_factory([tool_use_response] * 10)
    agent = Agent(provider=adapter, model="test-model", max_iterations=3)
    agent.tools.register(say)

    with pytest.raises(MaxIterationsExceeded):
        await agent.run("loop forever")

    assert len(adapter.calls) == 3


async def test_parallel_tool_calls(fake_adapter_factory) -> None:
    adapter = fake_adapter_factory(
        [
            CompletionResponse(
                content=[
                    ToolUseBlock(id="a", name="multiply", input={"a": 2, "b": 3}),
                    ToolUseBlock(id="b", name="multiply", input={"a": 5, "b": 7}),
                ],
                stop_reason="tool_use",
            ),
            CompletionResponse(
                content=[TextBlock(text="done")],
                stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(provider=adapter, model="test-model")
    agent.tools.register(multiply)

    history = await agent.run("compute two things")

    # history[2] is the user message carrying both tool results, in the original call order.
    tool_result_msg = history[2]
    assert isinstance(tool_result_msg.content, list)
    blocks = tool_result_msg.content
    assert len(blocks) == 2
    assert all(isinstance(b, ToolResultBlock) for b in blocks)
    assert [b.tool_use_id for b in blocks] == ["a", "b"]  # type: ignore[attr-defined]
    assert [b.content for b in blocks] == ["6", "35"]  # type: ignore[attr-defined]


async def test_system_prompt_is_forwarded(fake_adapter_factory) -> None:
    adapter = fake_adapter_factory(
        [CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")]
    )
    agent = Agent(provider=adapter, model="test-model", system="you are helpful")
    await agent.run("hi")

    assert adapter.calls[0]["system"] == "you are helpful"


async def test_list_prompt_passthrough(fake_adapter_factory) -> None:
    adapter = fake_adapter_factory(
        [CompletionResponse(content=[TextBlock(text="ack")], stop_reason="end_turn")]
    )
    agent = Agent(provider=adapter, model="test-model")
    prompt = [Message(role="user", content="first"), Message(role="assistant", content="second")]
    history = await agent.run(prompt)

    # Original two messages + one new assistant message.
    assert len(history) == 3
    assert history[0].content == "first"
    assert history[1].content == "second"
