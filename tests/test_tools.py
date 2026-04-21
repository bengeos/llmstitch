"""Tests for the @tool decorator and schema generation."""

from __future__ import annotations

from typing import Literal, Optional

from llmstitch import tool
from llmstitch.tools import Tool, build_schema


def test_tool_schema_basic() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two integers."
    assert add.input_schema["type"] == "object"
    assert add.input_schema["properties"]["a"]["type"] == "integer"
    assert add.input_schema["properties"]["b"]["type"] == "integer"
    assert set(add.input_schema["required"]) == {"a", "b"}


def test_tool_schema_with_defaults() -> None:
    @tool
    def greet(name: str, greeting: str = "hello") -> str:
        return f"{greeting}, {name}"

    assert greet.input_schema["required"] == ["name"]
    assert "greeting" in greet.input_schema["properties"]


def test_tool_schema_optional() -> None:
    @tool
    def ping(host: str, port: Optional[int] = None) -> str:  # noqa: UP007, UP045 — intentional
        return f"{host}:{port}"

    # Optional with default should not be required.
    assert ping.input_schema["required"] == ["host"]
    assert ping.input_schema["properties"]["port"]["type"] == "integer"


def test_tool_schema_literal_enum() -> None:
    @tool
    def set_mode(mode: Literal["fast", "slow"]) -> str:
        return mode

    schema = set_mode.input_schema["properties"]["mode"]
    assert schema["type"] == "string"
    assert set(schema["enum"]) == {"fast", "slow"}


def test_tool_schema_list_and_dict() -> None:
    @tool
    def bulk(xs: list[int], meta: dict[str, str]) -> int:
        return len(xs) + len(meta)

    xs_schema = bulk.input_schema["properties"]["xs"]
    assert xs_schema["type"] == "array"
    assert xs_schema["items"]["type"] == "integer"
    meta_schema = bulk.input_schema["properties"]["meta"]
    assert meta_schema["type"] == "object"
    assert meta_schema["additionalProperties"]["type"] == "string"


def test_tool_schema_async() -> None:
    @tool
    async def fetch(url: str) -> str:
        return url

    assert fetch.is_async is True


def test_tool_explicit_name_and_description() -> None:
    @tool(name="custom", description="overridden")
    def raw(x: int) -> int:
        return x

    assert raw.name == "custom"
    assert raw.description == "overridden"


def test_tool_docstring_args_parsed() -> None:
    @tool
    def echo(msg: str, times: int = 1) -> str:
        """Repeat a message.

        Args:
            msg: The message to repeat.
            times: How many times.
        """
        return msg * times

    props = echo.input_schema["properties"]
    assert props["msg"].get("description") == "The message to repeat."
    assert props["times"].get("description") == "How many times."


def test_build_schema_plain_function() -> None:
    def plain(a: int) -> int:
        return a

    desc, schema = build_schema(plain)
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["required"] == ["a"]
    assert desc == ""
