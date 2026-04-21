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


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Incremental text chunk emitted during streaming."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolUseStart:
    """A new tool-use block is beginning during streaming."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolUseDelta:
    """A partial fragment of a tool-use block's JSON-encoded input."""

    id: str
    partial_json: str


@dataclass(frozen=True, slots=True)
class ToolUseStop:
    """A tool-use block has finished streaming."""

    id: str


@dataclass(frozen=True, slots=True)
class MessageStop:
    """The model has finished producing this message."""

    stop_reason: str
    usage: dict[str, int] | None = None


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


@dataclass(frozen=True, slots=True)
class StreamDone:
    """Terminal streaming event carrying the fully-assembled response.

    Adapters must emit exactly one `StreamDone` as the final event so callers
    (including `Agent.run_stream`) can feed the complete message back into the
    tool-execution loop without re-accumulating from deltas.
    """

    response: CompletionResponse


StreamEvent = TextDelta | ToolUseStart | ToolUseDelta | ToolUseStop | MessageStop | StreamDone
