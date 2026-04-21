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
class TokenCount:
    """Result of a pre-call token count.

    `output_tokens` is `None` for pre-call counts (only the request is known).
    `details` is a loose bag for provider-specific breakdowns (cache hits, etc.)
    so we can grow the shape without breaking the dataclass.
    """

    input_tokens: int
    output_tokens: int | None = None
    details: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class Pricing:
    """Per-model input/output token rates, expressed as USD per 1M tokens.

    Matches how Anthropic, OpenAI, and Gemini publish their rate cards, so
    users can paste numbers directly from a vendor pricing page without
    scaling. Cached-input, reasoning, and batch-discount rates are out of
    scope in v0.1.3 — if you need them, compute from raw token counts.
    """

    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True, slots=True)
class Cost:
    """Cost breakdown for a span of token usage, in USD."""

    input_cost: float
    output_cost: float

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost


@dataclass(slots=True)
class UsageTally:
    """Running totals for an `Agent` across its lifetime.

    Tracks both token consumption (`input_tokens`, `output_tokens`, `turns`)
    and call activity (`api_calls`, `retries`). Mutated in place by
    `Agent.run` / `Agent.run_stream`. Call `reset()` to zero the counters
    (e.g. between logical sessions). Not safe to mutate from multiple
    coroutines running against the same agent concurrently — neither is the
    agent's run loop itself.

    On a successful run with usage-reporting providers,
    `api_calls == turns + retries`. On a run that exhausts the retry policy
    and raises, `turns` is one lower than that identity would predict.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    api_calls: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, usage: dict[str, int] | None) -> None:
        """Fold one provider-reported `usage` dict into the tally.

        Silently ignores `None` so adapters that don't report usage don't
        force the caller to branch.
        """
        if usage is None:
            return
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.turns += 1

    def record_call(self) -> None:
        """Record one `provider.complete(...)` / `provider.stream(...)` invocation."""
        self.api_calls += 1

    def record_retry(self) -> None:
        """Record one retry attempt fired by the agent's retry policy."""
        self.retries += 1

    def cost(self, pricing: Pricing) -> Cost:
        """Compute the USD cost of the tokens tallied so far at `pricing`'s rates."""
        return Cost(
            input_cost=self.input_tokens * pricing.input_per_mtok / 1_000_000,
            output_cost=self.output_tokens * pricing.output_per_mtok / 1_000_000,
        )

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.turns = 0
        self.api_calls = 0
        self.retries = 0


@dataclass(frozen=True, slots=True)
class StreamDone:
    """Terminal streaming event carrying the fully-assembled response.

    Adapters must emit exactly one `StreamDone` as the final event so callers
    (including `Agent.run_stream`) can feed the complete message back into the
    tool-execution loop without re-accumulating from deltas.
    """

    response: CompletionResponse


StreamEvent = TextDelta | ToolUseStart | ToolUseDelta | ToolUseStop | MessageStop | StreamDone
