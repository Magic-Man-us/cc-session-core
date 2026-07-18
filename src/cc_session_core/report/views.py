"""View models and shared render/order helpers for the session layer.

These are the presentation-facing projections that :class:`cc_session_core.session.Session`
builds and the :mod:`cc_session_core.render` formatters consume: a timeline of decomposed
parts, paired tool calls, and cost rollups. Each timeline part is a variant of a
discriminated union that renders its own one-line form, so formatting is a
branch-free ``part.render(width)`` rather than a dispatch on a kind tag.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, computed_field
from pydantic_core import to_json

from .. import types as t
from ..models import AssistantMessage, SnakeModel, TextBlock, Usage, UserMessage

# --- module-local primitives (absent from types.py) ----------------------
EntryType = Annotated[
    str, Field(title="Entry type", description="A record's type, echoed on the entry.")
]

_PART_INDENT = "        "

# datetimes without a zone are read as UTC; a missing timestamp sorts first.
_EPOCH = datetime.min.replace(tzinfo=UTC)


def clip(text: str, width: int) -> str:
    """Collapse whitespace and truncate to ``width`` with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def sort_key(ts: t.Timestamp | None) -> datetime:
    """Order key: a missing timestamp sorts first; a naive datetime is read as UTC."""
    if ts is None:
        return _EPOCH
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


class _McpTextItem(BaseModel):
    """An MCP-style ``{"type": "text", "text": "..."}`` content item — the common shape
    for a tool_result's list content. Matched structurally so its own ``text`` (already
    a fully-formed string, markdown or otherwise) is used as-is instead of getting
    wrapped in another layer of JSON."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["text"]
    text: str


def message_text(message: AssistantMessage | UserMessage) -> str:
    """Flatten a user/assistant message's content to plain text: as-is if it's
    already a string, else the ``TextBlock``s joined with newlines."""
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(b.text for b in content if isinstance(b, TextBlock))


def tool_result_text(content: str | list[JsonValue]) -> str:
    """Flatten a ``tool_result`` content (str, or a list of blocks) to plain text.
    A list item shaped like an MCP text block contributes its own ``text`` field
    directly; anything else falls back to a compact JSON dump of the whole item."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        try:
            parts.append(_McpTextItem.model_validate(item).text)
        except ValidationError:
            parts.append(to_json(item).decode())
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# timeline parts (a self-rendering discriminated union)
# --------------------------------------------------------------------------- #
class TextPart(SnakeModel):
    """Plain assistant/user text."""

    type: Literal["text"] = "text"
    text: t.ContentText = ""

    def render(self, width: int) -> str:
        return f"{_PART_INDENT}text: {clip(self.text, width)}"

    def as_text(self) -> str:
        return self.text

    def as_markdown(self) -> str:
        return self.text


class ThinkingPart(SnakeModel):
    """Assistant thinking (or a redacted-thinking placeholder)."""

    type: Literal["thinking"] = "thinking"
    text: t.ContentText = ""

    def render(self, width: int) -> str:
        return f"{_PART_INDENT}thinking: {clip(self.text, width)}"

    def as_text(self) -> str:
        return f"[thinking]\n{self.text}"

    def as_markdown(self) -> str:
        return f"*(thinking)*\n\n{self.text}"


class ToolUsePart(SnakeModel):
    """A tool invocation."""

    type: Literal["tool_use"] = "tool_use"
    tool_name: t.ToolName = ""
    tool_use_id: t.ToolUseId = ""
    tool_input: JsonValue = None

    def render(self, width: int) -> str:
        arg = clip(to_json(self.tool_input).decode(), width)
        return f"{_PART_INDENT}tool_use {self.tool_name} #{self.tool_use_id}: {arg}"

    def as_text(self) -> str:
        return f"[tool_use {self.tool_name} #{self.tool_use_id}]\n{to_json(self.tool_input, indent=2).decode()}"

    def as_markdown(self) -> str:
        body = to_json(self.tool_input, indent=2).decode()
        return f"**tool_use — {self.tool_name}** `#{self.tool_use_id}`\n\n```json\n{body}\n```"


class ToolResultPart(SnakeModel):
    """A tool result paired back to its call."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: t.ToolUseId = ""
    tool_name: t.ToolName | None = None
    is_error: bool | None = None
    text: t.ContentText = ""
    result_structured: JsonValue | None = None

    def render(self, width: int) -> str:
        flag = "error" if self.is_error else "ok"
        name = f" {self.tool_name}" if self.tool_name else ""
        return (
            f"{_PART_INDENT}tool_result{name} #{self.tool_use_id} "
            f"({flag}): {clip(self.text, width)}"
        )

    def as_text(self) -> str:
        flag = "error" if self.is_error else "ok"
        return f"[tool_result {self.tool_name or ''} ({flag})]\n{self.text}"

    def as_markdown(self) -> str:
        flag = "error" if self.is_error else "ok"
        return f"**tool_result — {self.tool_name or ''}** ({flag})\n\n```\n{self.text}\n```"


class ImagePart(SnakeModel):
    """An image block (payload omitted from the text view)."""

    type: Literal["image"] = "image"

    def render(self, _width: int) -> str:
        return f"{_PART_INDENT}image: [omitted]"

    def as_text(self) -> str:
        return "[image]"

    def as_markdown(self) -> str:
        return "*(image omitted)*"


class OtherPart(SnakeModel):
    """Any block kind without a dedicated line form; carries its JSON."""

    type: Literal["other"] = "other"
    text: t.ContentText = ""

    def render(self, width: int) -> str:
        return f"{_PART_INDENT}other: {clip(self.text, width)}"

    def as_text(self) -> str:
        return self.text

    def as_markdown(self) -> str:
        return f"```\n{self.text}\n```"


TimelinePart = Annotated[
    TextPart | ThinkingPart | ToolUsePart | ToolResultPart | ImagePart | OtherPart,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# entry / call / cost views
# --------------------------------------------------------------------------- #
class TimelineEntry(SnakeModel):
    index: t.StepIndex
    step: t.StepIndex = 0
    type_step: t.StepIndex = 0
    type: EntryType
    role: t.Role | None = None
    timestamp: t.Timestamp | None = None
    uuid: t.RecordUuid | None = None
    parent_uuid: t.ParentUuid | None = None
    model: t.ModelId | None = None
    request_id: t.RequestId | None = None
    usage: Usage | None = None
    cost_usd: t.CostUsd | None = None
    cost_source: t.CostSource | None = None
    summary: t.ContentText | None = None
    parts: list[TimelinePart] = []
    raw: JsonValue | None = None


class ToolCall(SnakeModel):
    tool_use_id: t.ToolUseId
    name: t.ToolName
    input: JsonValue = None
    # Built-in tools -> their typed *Input model; MCP / unknown / scalar -> raw value.
    input_typed: Any = None
    reason: t.ContentText | None = None
    call_uuid: t.RecordUuid | None = None
    call_timestamp: t.Timestamp | None = None
    result_uuid: t.RecordUuid | None = None
    result_timestamp: t.Timestamp | None = None
    step: t.StepIndex = 0
    is_error: bool | None = None
    result_text: t.ContentText | None = None
    result_structured: JsonValue | None = None
    # Built-in tools -> their typed *Result model; MCP / unknown / scalar -> raw value.
    result_typed: Any = None
    duration_ms: t.DurationMs | None = None


def token_total(
    input_tokens: t.TokenCount,
    output_tokens: t.TokenCount,
    cache_creation_input_tokens: t.TokenCount,
    cache_read_input_tokens: t.TokenCount,
) -> t.TokenCount:
    return input_tokens + output_tokens + cache_creation_input_tokens + cache_read_input_tokens


class ModelUsage(SnakeModel):
    model: t.ModelId
    requests: t.Count = 0
    input_tokens: t.TokenCount = 0
    output_tokens: t.TokenCount = 0
    cache_creation_input_tokens: t.TokenCount = 0
    cache_read_input_tokens: t.TokenCount = 0
    cost_usd: t.CostUsd | None = None

    @computed_field
    @property
    def total_tokens(self) -> t.TokenCount:
        return token_total(
            self.input_tokens,
            self.output_tokens,
            self.cache_creation_input_tokens,
            self.cache_read_input_tokens,
        )


class CostSummary(SnakeModel):
    priced: bool
    requests: t.Count = 0
    input_tokens: t.TokenCount = 0
    output_tokens: t.TokenCount = 0
    cache_creation_input_tokens: t.TokenCount = 0
    cache_read_input_tokens: t.TokenCount = 0
    total_cost_usd: t.CostUsd | None = None
    by_model: list[ModelUsage] = []

    @computed_field
    @property
    def total_tokens(self) -> t.TokenCount:
        return token_total(
            self.input_tokens,
            self.output_tokens,
            self.cache_creation_input_tokens,
            self.cache_read_input_tokens,
        )


class SessionInfo(SnakeModel):
    """One-line summary of a session, for listing/picking investigations."""

    id: t.SessionId = ""
    source: t.FilePath | None = None
    title: t.SessionTitle = ""
    started: t.Timestamp | None = None
    ended: t.Timestamp | None = None
    records: t.Count = 0
    tool_calls: t.Count = 0
    total_cost_usd: t.CostUsd | None = None
