"""Provider-neutral domain types for llmstitch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(slots=True)
class Message:
    role: Role
    content: str | list[ContentBlock]

    def text_blocks(self) -> list[TextBlock]:
        if isinstance(self.content, str):
            return [TextBlock(text=self.content)]
        return [b for b in self.content if isinstance(b, TextBlock)]

    def tool_uses(self) -> list[ToolUseBlock]:
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class CompletionResponse:
    content: list[ContentBlock]
    stop_reason: str
    usage: dict[str, int] | None = None
    raw: Any = field(default=None, repr=False)

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))
