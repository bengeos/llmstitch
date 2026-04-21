"""Tool concurrency flags + `read_only_subset()`: safe parallel dispatch.

`Tool` and the `@tool` decorator accept two flags:

- `is_concurrency_safe=True` (default) — tool can run in parallel with its
  siblings in the same batch. Read-only operations, pure computations,
  idempotent RPCs fit here.
- `is_concurrency_safe=False` — the tool touches shared state (filesystem,
  DB rows, caches) in a way that races with concurrent writes. If any
  call in a batch targets an unsafe tool, `ToolRegistry.run` runs the
  entire batch sequentially, preserving input order.
- `is_read_only=True` — marks the tool as safe for a planning-phase agent
  that shouldn't be able to mutate anything. `registry.read_only_subset()`
  returns a view containing only these tools.

What this example shows:

1. **Mixed-safety batch goes sequential.** Wall-clock timing proves it.
2. **All-safe batch parallelizes.** Same tools, different batch → gather.
3. **Two-phase pipeline.** A "planner" agent sees only read-only tools and
   proposes a plan; an "executor" agent sees the full registry and carries
   it out.

Usage:
    python examples/tool_concurrency.py      # runs entirely offline — no API key
"""

from __future__ import annotations

import asyncio
import time

from llmstitch import ToolRegistry, tool
from llmstitch.types import ToolUseBlock


@tool(is_read_only=True)
async def read_file(path: str) -> str:
    """Pretend to read a file — purely illustrative."""
    await asyncio.sleep(0.1)
    return f"<contents of {path}>"


@tool(is_read_only=True)
async def list_files(dir: str) -> str:
    """Pretend to list directory contents."""
    await asyncio.sleep(0.1)
    return f"<files in {dir}>"


@tool(is_concurrency_safe=False)
async def write_file(path: str, contents: str) -> str:
    """Pretend to write a file — not safe to run concurrently with peers."""
    await asyncio.sleep(0.1)
    return f"wrote {len(contents)} bytes to {path}"


@tool
async def compute(x: int, y: int) -> int:
    """Pure math — default concurrency-safe."""
    await asyncio.sleep(0.1)
    return x + y


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(read_file)
    reg.register(list_files)
    reg.register(write_file)
    reg.register(compute)
    return reg


async def demo_mixed_batch_runs_sequentially() -> None:
    """Batch with an unsafe tool — total time ≈ sum, not max."""
    reg = build_registry()
    calls = [
        ToolUseBlock(id="a", name="read_file", input={"path": "/a.txt"}),
        ToolUseBlock(id="b", name="write_file", input={"path": "/b.txt", "contents": "hi"}),
        ToolUseBlock(id="c", name="read_file", input={"path": "/c.txt"}),
    ]
    start = time.perf_counter()
    results = await reg.run(calls)
    elapsed = time.perf_counter() - start
    print(f"[mixed]  3 calls, one unsafe: elapsed = {elapsed:.2f}s (expected ~0.3s sequential)")
    for r in results:
        print(f"   {r.tool_use_id}: {r.content}")


async def demo_all_safe_batch_parallelizes() -> None:
    """Batch of concurrency-safe tools — total time ≈ max."""
    reg = build_registry()
    calls = [
        ToolUseBlock(id="a", name="read_file", input={"path": "/a.txt"}),
        ToolUseBlock(id="b", name="list_files", input={"dir": "/etc"}),
        ToolUseBlock(id="c", name="compute", input={"x": 2, "y": 3}),
    ]
    start = time.perf_counter()
    results = await reg.run(calls)
    elapsed = time.perf_counter() - start
    print(f"[all-safe] 3 calls, all safe: elapsed = {elapsed:.2f}s (expected ~0.1s parallel)")
    for r in results:
        print(f"   {r.tool_use_id}: {r.content}")


async def demo_read_only_subset() -> None:
    """Planner uses only read-only tools; executor uses the full set."""
    full = build_registry()
    readonly = full.read_only_subset()

    print("\n[planner tools] — read-only subset:")
    for name in readonly._tools:  # internal peek for demo purposes
        print(f"   - {name}")

    print("\n[executor tools] — full set:")
    for name in full._tools:
        print(f"   - {name}")

    # A planner calling write_file gets an "unknown tool" error — by design.
    plan_calls = [
        ToolUseBlock(id="p1", name="read_file", input={"path": "/a.txt"}),
        ToolUseBlock(id="p2", name="write_file", input={"path": "/x", "contents": "oops"}),
    ]
    results = await readonly.run(plan_calls)
    print("\n[planner attempted write_file]:")
    for r in results:
        status = "ERROR" if r.is_error else "OK"
        print(f"   {r.tool_use_id} [{status}]: {r.content}")


async def main() -> None:
    await demo_mixed_batch_runs_sequentially()
    print()
    await demo_all_safe_batch_parallelizes()
    await demo_read_only_subset()


if __name__ == "__main__":
    asyncio.run(main())
