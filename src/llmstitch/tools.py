"""Tools, schema generation, registry, and skills."""

from __future__ import annotations

import asyncio
import inspect
import re
import types as _pytypes
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .types import ToolDefinition, ToolResultBlock, ToolUseBlock

ToolFunc = Callable[..., Any]

_PRIMITIVE_SCHEMA: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """If `tp` is Optional[X] / X | None, return (X, True). Otherwise (tp, False)."""
    origin = get_origin(tp)
    if origin is Union or origin is _pytypes.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return args[0], True
    return tp, False


def _type_to_schema(tp: Any) -> dict[str, Any]:
    inner, _optional = _unwrap_optional(tp)
    if inner in _PRIMITIVE_SCHEMA:
        return dict(_PRIMITIVE_SCHEMA[inner])
    origin = get_origin(inner)
    if origin is Literal:
        values = list(get_args(inner))
        # Infer JSON schema type from the literal values (string-only case is most common).
        types_seen = {type(v) for v in values}
        if types_seen == {str}:
            return {"type": "string", "enum": values}
        if types_seen == {int}:
            return {"type": "integer", "enum": values}
        return {"enum": values}
    if origin in (list, tuple, set, frozenset):
        args = get_args(inner)
        items_schema = _type_to_schema(args[0]) if args else {}
        return {"type": "array", "items": items_schema}
    if origin is dict:
        args = get_args(inner)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _type_to_schema(args[1])}
        return {"type": "object"}
    # Fallback — leave the schema open rather than guessing.
    return {}


_ARGS_HEADER_RE = re.compile(r"^\s*(Args|Arguments|Parameters|Params):\s*$", re.MULTILINE)
_ARG_LINE_RE = re.compile(r"^\s{2,}(\w+)\s*(?:\([^)]+\))?\s*:\s*(.+?)\s*$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {param_name: description}) from a docstring. Best-effort."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary_lines: list[str] = []
    param_docs: dict[str, str] = {}
    i = 0
    while i < len(lines) and not _ARGS_HEADER_RE.match(lines[i]):
        summary_lines.append(lines[i])
        i += 1
    summary = " ".join(line.strip() for line in summary_lines if line.strip())
    if i < len(lines):
        i += 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                break
            if _ARGS_HEADER_RE.match(line):
                break
            # Dedent a bit for the relaxed "    name: desc" shape common in Google-style docs.
            m = _ARG_LINE_RE.match("  " + line.lstrip())
            if m:
                param_docs[m.group(1)] = m.group(2)
            i += 1
    return summary, param_docs


def build_schema(fn: ToolFunc) -> tuple[str, dict[str, Any]]:
    """Build (description, JSON Schema) for a function."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    summary, param_docs = _parse_docstring(fn.__doc__)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(pname, param.annotation)
        if annotation is inspect.Parameter.empty:
            schema: dict[str, Any] = {}
        else:
            schema = _type_to_schema(annotation)
        _inner, optional = (
            _unwrap_optional(annotation)
            if annotation is not inspect.Parameter.empty
            else (None, False)
        )
        if pname in param_docs:
            schema = {**schema, "description": param_docs[pname]}
        properties[pname] = schema
        if param.default is inspect.Parameter.empty and not optional:
            required.append(pname)
    schema_obj: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema_obj["required"] = required
    return summary, schema_obj


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFunc
    is_async: bool

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def call(self, **kwargs: Any) -> Any:
        if self.is_async:
            return await self.fn(**kwargs)
        return await asyncio.to_thread(self.fn, **kwargs)


def tool(
    fn: ToolFunc | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that turns a function into a `Tool`.

    Can be used as `@tool` or `@tool(name="x", description="y")`.
    """

    def _wrap(f: ToolFunc) -> Tool:
        summary, schema = build_schema(f)
        return Tool(
            name=name or f.__name__,
            description=description or summary or f.__name__,
            input_schema=schema,
            fn=f,
            is_async=inspect.iscoroutinefunction(f),
        )

    if fn is None:
        return _wrap
    return _wrap(fn)


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, item: Tool | ToolFunc) -> Tool:
        t = item if isinstance(item, Tool) else tool(item)
        self._tools[t.name] = t
        return t

    def register_many(self, items: Iterable[Tool | ToolFunc]) -> None:
        for item in items:
            self.register(item)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition() for t in self._tools.values()]

    async def run(
        self,
        calls: list[ToolUseBlock],
        *,
        timeout: float | None = None,
    ) -> list[ToolResultBlock]:
        """Execute tool calls concurrently, preserving input order."""

        async def _one(use: ToolUseBlock) -> ToolResultBlock:
            tool_obj = self._tools.get(use.name)
            if tool_obj is None:
                return ToolResultBlock(
                    tool_use_id=use.id,
                    content=f"Unknown tool: {use.name}",
                    is_error=True,
                )
            try:
                coro: Awaitable[Any] = tool_obj.call(**use.input)
                if timeout is not None:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro
            except asyncio.TimeoutError:
                return ToolResultBlock(
                    tool_use_id=use.id,
                    content=f"Tool '{use.name}' timed out after {timeout}s",
                    is_error=True,
                )
            except Exception as exc:  # noqa: BLE001 — surface any tool error to the model
                return ToolResultBlock(
                    tool_use_id=use.id,
                    content=f"{type(exc).__name__}: {exc}",
                    is_error=True,
                )
            return ToolResultBlock(tool_use_id=use.id, content=_stringify(result))

        return list(await asyncio.gather(*(_one(c) for c in calls)))


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, default=str)
    except Exception:
        return str(value)


class Skill:
    """A bundle of a system prompt and a set of tools.

    Supports two construction styles:

        Skill(name="x", description="y", system_prompt="z", tools=[...])

        class MySkill(Skill):
            name = "x"
            description = "y"
            system_prompt = "z"
            tools = [...]
    """

    # Class-level defaults picked up by subclasses.
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    tools: list[Tool] = []  # noqa: RUF012 — intentional class-level default

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        cls = type(self)
        self.name = name if name is not None else cls.name
        self.description = description if description is not None else cls.description
        self.system_prompt = system_prompt if system_prompt is not None else cls.system_prompt
        self.tools = list(tools) if tools is not None else list(cls.tools)

    def extend(self, other: Skill) -> Skill:
        """Return a new Skill that concatenates prompts and merges tools by name."""
        merged_prompt = (
            f"{self.system_prompt}\n\n{other.system_prompt}".strip()
            if self.system_prompt and other.system_prompt
            else (self.system_prompt or other.system_prompt)
        )
        by_name: dict[str, Tool] = {t.name: t for t in self.tools}
        for t in other.tools:
            by_name[t.name] = t  # later wins
        combined_name = (
            f"{self.name}+{other.name}" if self.name and other.name else (self.name or other.name)
        )
        return Skill(
            name=combined_name,
            description=other.description or self.description,
            system_prompt=merged_prompt,
            tools=list(by_name.values()),
        )

    def into_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register_many(self.tools)
        return reg
