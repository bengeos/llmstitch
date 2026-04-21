"""llmstitch — a provider-agnostic LLM toolkit."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .agent import Agent, MaxIterationsExceeded
from .tools import Skill, Tool, ToolRegistry, tool
from .types import (
    CompletionResponse,
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)

try:
    __version__ = version("llmstitch")
except PackageNotFoundError:  # pragma: no cover — only hit in unusual dev layouts
    __version__ = "0.0.0+local"

__all__ = [
    "Agent",
    "MaxIterationsExceeded",
    "tool",
    "Tool",
    "Skill",
    "ToolRegistry",
    "Message",
    "Role",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    "ToolDefinition",
    "CompletionResponse",
    "__version__",
]
