"""Tests for tool concurrency flags, sequential fallback, and read-only subset."""

from __future__ import annotations

import asyncio
import time

from llmstitch import Tool, ToolRegistry, tool
from llmstitch.types import ToolUseBlock


@tool
async def slow_safe(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return f"safe {seconds}"


@tool(is_concurrency_safe=False)
async def slow_unsafe(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return f"unsafe {seconds}"


@tool(is_read_only=True)
def read_thing() -> str:
    return "r"


@tool(is_read_only=True, is_concurrency_safe=True)
def list_things() -> str:
    return "rr"


@tool
def write_thing(msg: str) -> str:
    return f"wrote {msg}"


def test_tool_defaults_backwards_compatible() -> None:
    t = slow_safe
    assert isinstance(t, Tool)
    assert t.is_read_only is False
    assert t.is_concurrency_safe is True


def test_tool_decorator_flags_threaded_through() -> None:
    assert slow_unsafe.is_concurrency_safe is False
    assert slow_unsafe.is_read_only is False
    assert read_thing.is_read_only is True
    assert read_thing.is_concurrency_safe is True


async def test_registry_run_parallel_when_all_safe() -> None:
    reg = ToolRegistry()
    reg.register(slow_safe)

    calls = [ToolUseBlock(id=f"t{i}", name="slow_safe", input={"seconds": 0.1}) for i in range(3)]
    start = time.perf_counter()
    results = await reg.run(calls)
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    assert all(not r.is_error for r in results)
    assert elapsed < 0.25, f"expected concurrent execution, took {elapsed:.3f}s"


async def test_registry_run_sequential_when_any_unsafe() -> None:
    reg = ToolRegistry()
    reg.register(slow_safe)
    reg.register(slow_unsafe)

    calls = [
        ToolUseBlock(id="a", name="slow_safe", input={"seconds": 0.1}),
        ToolUseBlock(id="b", name="slow_unsafe", input={"seconds": 0.1}),
        ToolUseBlock(id="c", name="slow_safe", input={"seconds": 0.1}),
    ]
    start = time.perf_counter()
    results = await reg.run(calls)
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    assert [r.tool_use_id for r in results] == ["a", "b", "c"]
    # Sequential — at least 3 * 0.1s.
    assert elapsed >= 0.28, f"expected sequential execution, took {elapsed:.3f}s"


async def test_registry_run_preserves_order_in_sequential_mode() -> None:
    reg = ToolRegistry()
    reg.register(slow_unsafe)

    calls = [
        ToolUseBlock(id="first", name="slow_unsafe", input={"seconds": 0.01}),
        ToolUseBlock(id="second", name="slow_unsafe", input={"seconds": 0.01}),
    ]
    results = await reg.run(calls)
    assert [r.tool_use_id for r in results] == ["first", "second"]


async def test_unknown_tool_forces_sequential_dispatch() -> None:
    """An unknown tool name is treated as unsafe for the concurrency decision."""
    reg = ToolRegistry()
    reg.register(slow_safe)

    calls = [
        ToolUseBlock(id="a", name="slow_safe", input={"seconds": 0.1}),
        ToolUseBlock(id="b", name="nope", input={}),
    ]
    results = await reg.run(calls)
    assert results[1].is_error is True
    assert "Unknown tool" in results[1].content


def test_read_only_subset_filters_correctly() -> None:
    reg = ToolRegistry()
    reg.register(read_thing)
    reg.register(list_things)
    reg.register(write_thing)
    reg.register(slow_safe)

    subset = reg.read_only_subset()
    assert "read_thing" in subset
    assert "list_things" in subset
    assert "write_thing" not in subset
    assert "slow_safe" not in subset
    assert len(subset) == 2


def test_read_only_subset_returns_independent_registry() -> None:
    reg = ToolRegistry()
    reg.register(read_thing)
    reg.register(write_thing)

    subset = reg.read_only_subset()
    subset.register(slow_safe)  # mutate the subset

    assert "slow_safe" in subset
    assert "slow_safe" not in reg  # original unaffected


def test_read_only_subset_shares_tool_instances() -> None:
    reg = ToolRegistry()
    reg.register(read_thing)
    subset = reg.read_only_subset()
    assert subset.get("read_thing") is reg.get("read_thing")


async def test_tool_flag_explicit_name_respects_flags() -> None:
    @tool(name="x_tool", is_read_only=True, is_concurrency_safe=False)
    def _func() -> str:
        return ""

    assert _func.name == "x_tool"
    assert _func.is_read_only is True
    assert _func.is_concurrency_safe is False
