"""Tests for Skill construction, .extend(), and into_registry()."""

from __future__ import annotations

from llmstitch import Skill, tool


@tool
def add(a: int, b: int) -> int:
    return a + b


@tool
def sub(a: int, b: int) -> int:
    return a - b


@tool(name="add")
def add_v2(a: int, b: int) -> int:
    """Newer add."""
    return a + b + 1


def test_skill_functional_construction() -> None:
    s = Skill(
        name="math",
        description="arithmetic",
        system_prompt="You can do math.",
        tools=[add, sub],
    )
    assert s.name == "math"
    assert s.description == "arithmetic"
    assert s.system_prompt == "You can do math."
    assert {t.name for t in s.tools} == {"add", "sub"}


def test_skill_class_construction() -> None:
    class Arithmetic(Skill):
        name = "math"
        description = "arithmetic"
        system_prompt = "You can do math."
        tools = [add]

    s = Arithmetic()
    assert s.name == "math"
    assert s.description == "arithmetic"
    assert s.system_prompt == "You can do math."
    assert [t.name for t in s.tools] == ["add"]


def test_skill_extend_merges_prompts_and_tools() -> None:
    a = Skill(name="a", system_prompt="alpha", tools=[add])
    b = Skill(name="b", system_prompt="beta", tools=[sub])
    c = a.extend(b)
    assert c.system_prompt == "alpha\n\nbeta"
    assert {t.name for t in c.tools} == {"add", "sub"}


def test_skill_extend_dedups_tools_later_wins() -> None:
    a = Skill(name="a", system_prompt="alpha", tools=[add])
    b = Skill(name="b", system_prompt="beta", tools=[add_v2])
    c = a.extend(b)
    # Both tools named "add" — later (add_v2) wins.
    add_tool = next(t for t in c.tools if t.name == "add")
    assert add_tool.description == "Newer add."


def test_skill_into_registry() -> None:
    s = Skill(name="math", system_prompt="", tools=[add, sub])
    reg = s.into_registry()
    assert "add" in reg
    assert "sub" in reg
    assert len(reg) == 2
