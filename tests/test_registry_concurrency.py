"""Tests for ToolRegistry concurrent execution semantics."""

from __future__ import annotations

import asyncio
import time

from llmstitch import ToolRegistry, tool
from llmstitch.types import ToolUseBlock


@tool
async def slow(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return f"slept {seconds}"


@tool
async def boom() -> str:
    raise ValueError("kaboom")


@tool
def echo(msg: str) -> str:
    return msg


async def test_fanout_is_concurrent() -> None:
    reg = ToolRegistry()
    reg.register(slow)

    calls = [ToolUseBlock(id=f"t{i}", name="slow", input={"seconds": 0.1}) for i in range(3)]
    start = time.perf_counter()
    results = await reg.run(calls)
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    assert all(not r.is_error for r in results)
    # If executed serially this would be ~0.3s; concurrent should be well under.
    assert elapsed < 0.25, f"expected concurrent execution, took {elapsed:.3f}s"


async def test_order_preservation() -> None:
    reg = ToolRegistry()
    reg.register(slow)

    # The 0.2s call finishes last in wall-clock time, but must still be third in the result list.
    calls = [
        ToolUseBlock(id="fast1", name="slow", input={"seconds": 0.01}),
        ToolUseBlock(id="fast2", name="slow", input={"seconds": 0.01}),
        ToolUseBlock(id="slow1", name="slow", input={"seconds": 0.2}),
    ]
    results = await reg.run(calls)
    assert [r.tool_use_id for r in results] == ["fast1", "fast2", "slow1"]


async def test_timeout_marks_result_as_error() -> None:
    reg = ToolRegistry()
    reg.register(slow)

    calls = [ToolUseBlock(id="t", name="slow", input={"seconds": 1.0})]
    results = await reg.run(calls, timeout=0.05)

    assert len(results) == 1
    r = results[0]
    assert r.is_error is True
    assert "timed out" in r.content


async def test_exceptions_captured_per_call() -> None:
    reg = ToolRegistry()
    reg.register(boom)
    reg.register(echo)

    calls = [
        ToolUseBlock(id="ok", name="echo", input={"msg": "hi"}),
        ToolUseBlock(id="bad", name="boom", input={}),
    ]
    results = await reg.run(calls)
    assert results[0].is_error is False
    assert results[0].content == "hi"
    assert results[1].is_error is True
    assert "kaboom" in results[1].content


async def test_unknown_tool_is_error() -> None:
    reg = ToolRegistry()
    reg.register(echo)

    calls = [ToolUseBlock(id="x", name="does_not_exist", input={})]
    results = await reg.run(calls)
    assert results[0].is_error is True
    assert "Unknown tool" in results[0].content
