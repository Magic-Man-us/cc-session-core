"""Typed, lossless models for Codex rollout transcripts.

Codex persists sessions as rollout JSONL under ``$CODEX_HOME/sessions``.  Each
line has a common timestamp/type/payload envelope.  The nested ``response_item``
and ``event_msg`` payloads are independently discriminated unions.

The rollout format is intentionally treated as an evolving interface: known
records receive useful typed fields, while unknown record, event, response-item,
and content types retain their complete payload through ``extra="allow"``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, cast

from pydantic import Discriminator, Field, JsonValue, Tag, TypeAdapter

from .. import types as t
from ..models import SnakeModel


def _known_or_unknown(known: frozenset[str]) -> Callable[[object], str]:
    def tag(value: object) -> str:
        if isinstance(value, dict):
            raw: object = cast("dict[str, object]", value).get("type")
        else:
            raw = getattr(value, "type", None)
        return "known" if raw in known else "unknown"

    return tag


# ---------------------------------------------------------------------------
# Response content
# ---------------------------------------------------------------------------
class InputTextContent(SnakeModel):
    type: Literal["input_text"]
    text: t.ContentText


class OutputTextContent(SnakeModel):
    type: Literal["output_text"]
    text: t.ContentText


class InputImageContent(SnakeModel):
    type: Literal["input_image"]
    image_url: t.Url
    detail: str | None = None


class InputAudioContent(SnakeModel):
    type: Literal["input_audio"]
    audio_url: t.Url


class UnknownContent(SnakeModel):
    type: str


_CONTENT_TAGS = frozenset({"input_text", "output_text", "input_image", "input_audio"})
_KnownContent = Annotated[
    InputTextContent | OutputTextContent | InputImageContent | InputAudioContent,
    Field(discriminator="type"),
]
CodexContent = Annotated[
    Annotated[_KnownContent, Tag("known")] | Annotated[UnknownContent, Tag("unknown")],
    Discriminator(_known_or_unknown(_CONTENT_TAGS)),
]


class ReasoningText(SnakeModel):
    type: Literal["summary_text", "reasoning_text", "text"]
    text: t.ContentText


class InternalMessageMetadata(SnakeModel):
    turn_id: t.MessageId | None = None


# ---------------------------------------------------------------------------
# Responses API items persisted by Codex
# ---------------------------------------------------------------------------
class ResponseItemBase(SnakeModel):
    id: t.ApiMessageId | None = None
    internal_chat_message_metadata_passthrough: InternalMessageMetadata | None = None


class MessageResponseItem(ResponseItemBase):
    type: Literal["message"]
    role: t.Role
    content: list[CodexContent]
    phase: str | None = None


class AgentMessageContent(SnakeModel):
    type: Literal["input_text", "encrypted_content"]
    text: t.ContentText | None = None
    encrypted_content: str | None = None


class AgentMessageResponseItem(ResponseItemBase):
    type: Literal["agent_message"]
    author: str
    recipient: str
    content: list[AgentMessageContent]


class ReasoningResponseItem(ResponseItemBase):
    type: Literal["reasoning"]
    summary: list[ReasoningText] = Field(default_factory=lambda: list[ReasoningText]())
    content: list[ReasoningText] | None = None
    encrypted_content: str | None = None


class FunctionCallResponseItem(ResponseItemBase):
    type: Literal["function_call"]
    name: t.ToolName
    namespace: str | None = None
    arguments: str
    call_id: t.ToolUseId


class FunctionCallOutputResponseItem(ResponseItemBase):
    type: Literal["function_call_output"]
    call_id: t.ToolUseId
    output: JsonValue


class CustomToolCallResponseItem(ResponseItemBase):
    type: Literal["custom_tool_call"]
    status: t.StatusText | None = None
    call_id: t.ToolUseId
    name: t.ToolName
    namespace: str | None = None
    input: str


class CustomToolCallOutputResponseItem(ResponseItemBase):
    type: Literal["custom_tool_call_output"]
    call_id: t.ToolUseId
    name: t.ToolName | None = None
    output: JsonValue


class LocalShellCallResponseItem(ResponseItemBase):
    type: Literal["local_shell_call"]
    call_id: t.ToolUseId | None = None
    status: t.StatusText | None = None
    action: JsonValue


class ToolSearchCallResponseItem(ResponseItemBase):
    type: Literal["tool_search_call"]
    call_id: t.ToolUseId | None = None
    status: t.StatusText | None = None
    execution: str
    arguments: JsonValue


class ToolSearchOutputResponseItem(ResponseItemBase):
    type: Literal["tool_search_output"]
    call_id: t.ToolUseId | None = None
    status: t.StatusText
    execution: str
    tools: list[JsonValue] = Field(default_factory=lambda: list[JsonValue]())


class WebSearchCallResponseItem(ResponseItemBase):
    type: Literal["web_search_call"]
    status: t.StatusText | None = None
    action: JsonValue | None = None


class ImageGenerationCallResponseItem(ResponseItemBase):
    type: Literal["image_generation_call"]
    status: t.StatusText
    revised_prompt: t.PromptText | None = None
    result: str


class CompactionResponseItem(ResponseItemBase):
    type: Literal["compaction", "compaction_summary"]
    encrypted_content: str


class ContextCompactionResponseItem(ResponseItemBase):
    type: Literal["context_compaction"]
    encrypted_content: str | None = None


class GenericResponseItem(ResponseItemBase):
    """Known control item without a normalized timeline projection."""

    type: Literal["additional_tools", "compaction_trigger"]


class UnknownResponseItem(SnakeModel):
    type: str


_RESPONSE_ITEM_TAGS = frozenset(
    {
        "message",
        "agent_message",
        "reasoning",
        "local_shell_call",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_search_call",
        "tool_search_output",
        "web_search_call",
        "image_generation_call",
        "compaction",
        "compaction_summary",
        "context_compaction",
        "additional_tools",
        "compaction_trigger",
    }
)
_KnownResponseItem = Annotated[
    MessageResponseItem
    | AgentMessageResponseItem
    | ReasoningResponseItem
    | LocalShellCallResponseItem
    | FunctionCallResponseItem
    | FunctionCallOutputResponseItem
    | CustomToolCallResponseItem
    | CustomToolCallOutputResponseItem
    | ToolSearchCallResponseItem
    | ToolSearchOutputResponseItem
    | WebSearchCallResponseItem
    | ImageGenerationCallResponseItem
    | CompactionResponseItem
    | ContextCompactionResponseItem
    | GenericResponseItem,
    Field(discriminator="type"),
]
CodexResponseItem = Annotated[
    Annotated[_KnownResponseItem, Tag("known")] | Annotated[UnknownResponseItem, Tag("unknown")],
    Discriminator(_known_or_unknown(_RESPONSE_ITEM_TAGS)),
]


# ---------------------------------------------------------------------------
# Event messages and usage
# ---------------------------------------------------------------------------
class CodexTokenUsage(SnakeModel):
    input_tokens: t.TokenCount = 0
    cached_input_tokens: t.TokenCount = 0
    cache_write_input_tokens: t.TokenCount = 0
    output_tokens: t.TokenCount = 0
    reasoning_output_tokens: t.TokenCount = 0
    total_tokens: t.TokenCount = 0


class CodexTokenUsageInfo(SnakeModel):
    total_token_usage: CodexTokenUsage
    last_token_usage: CodexTokenUsage
    model_context_window: t.TokenCount | None = None


class UserMessageEvent(SnakeModel):
    type: Literal["user_message"]
    client_id: str | None = None
    message: t.ContentText
    images: list[t.Url] | None = None
    image_details: list[JsonValue] = Field(default_factory=lambda: list[JsonValue]())
    local_images: list[t.FilePath] = Field(default_factory=list)
    local_image_details: list[JsonValue] = Field(default_factory=lambda: list[JsonValue]())
    audio: list[t.Url] | None = None
    local_audio: list[t.FilePath] = Field(default_factory=list)
    text_elements: list[JsonValue] = Field(default_factory=lambda: list[JsonValue]())


class AgentMessageEvent(SnakeModel):
    type: Literal["agent_message"]
    message: t.ContentText
    phase: str | None = None
    memory_citation: JsonValue | None = None


class AgentReasoningEvent(SnakeModel):
    type: Literal["agent_reasoning", "agent_reasoning_raw_content"]
    text: t.ContentText


class TokenCountEvent(SnakeModel):
    type: Literal["token_count"]
    info: CodexTokenUsageInfo | None = None
    rate_limits: JsonValue | None = None


class TurnStartedEvent(SnakeModel):
    type: Literal["task_started", "turn_started"]
    turn_id: t.MessageId
    trace_id: str | None = None
    started_at: int | None = None
    model_context_window: t.TokenCount | None = None
    collaboration_mode_kind: str | None = None


class TurnCompleteEvent(SnakeModel):
    type: Literal["task_complete", "turn_complete"]
    turn_id: t.MessageId
    last_agent_message: t.ContentText | None = None
    error: JsonValue | None = None
    started_at: int | None = None
    completed_at: int | None = None
    duration_ms: t.DurationMs | None = None
    time_to_first_token_ms: t.DurationMs | None = None


class ThreadSettings(SnakeModel):
    model: t.ModelId | None = None
    model_provider_id: str | None = None
    service_tier: t.ServiceTier | None = None
    cwd: t.Cwd | None = None


class ThreadSettingsAppliedEvent(SnakeModel):
    type: Literal["thread_settings_applied"]
    thread_settings: ThreadSettings


class RawResponseCompletedEvent(SnakeModel):
    type: Literal["raw_response_completed"]
    response_id: t.RequestId
    token_usage: CodexTokenUsage | None = None


class ErrorEvent(SnakeModel):
    type: Literal["error"]
    message: t.ErrorText
    codex_error_info: JsonValue | None = None


class GenericEvent(SnakeModel):
    """Known Codex event retained losslessly when no analysis projection is needed."""

    type: Literal[
        "warning",
        "guardian_warning",
        "guardian_assessment",
        "context_compacted",
        "thread_rolled_back",
        "thread_goal_updated",
        "turn_aborted",
        "item_started",
        "item_completed",
        "hook_started",
        "hook_completed",
        "entered_review_mode",
        "exited_review_mode",
        "session_configured",
        "environment_connected",
        "environment_disconnected",
        "mcp_startup_update",
        "mcp_startup_complete",
        "mcp_tool_call_begin",
        "mcp_tool_call_end",
        "web_search_begin",
        "web_search_end",
        "image_generation_begin",
        "image_generation_end",
        "exec_command_begin",
        "exec_command_output_delta",
        "terminal_interaction",
        "exec_command_end",
        "view_image_tool_call",
        "exec_approval_request",
        "request_permissions",
        "request_user_input",
        "dynamic_tool_call_request",
        "dynamic_tool_call_response",
        "elicitation_request",
        "apply_patch_approval_request",
        "deprecation_notice",
        "stream_error",
        "patch_apply_begin",
        "patch_apply_updated",
        "patch_apply_end",
        "turn_diff",
        "plan_update",
        "shutdown_complete",
        "raw_response_item",
        "agent_message_content_delta",
        "plan_delta",
        "reasoning_content_delta",
        "reasoning_raw_content_delta",
        "collab_agent_spawn_begin",
        "collab_agent_spawn_end",
        "collab_agent_interaction_begin",
        "collab_agent_interaction_end",
        "collab_waiting_begin",
        "collab_waiting_end",
        "collab_close_begin",
        "collab_close_end",
        "collab_resume_begin",
        "collab_resume_end",
        "sub_agent_activity",
        "realtime_conversation_started",
        "realtime_conversation_realtime",
        "realtime_conversation_closed",
        "realtime_conversation_sdp",
        "realtime_conversation_list_voices_response",
        "model_reroute",
        "model_verification",
        "turn_moderation_metadata",
        "safety_buffering",
        "agent_reasoning_section_break",
    ]


class UnknownEvent(SnakeModel):
    type: str


_EVENT_TAGS = frozenset(
    {
        "user_message",
        "agent_message",
        "agent_reasoning",
        "agent_reasoning_raw_content",
        "token_count",
        "task_started",
        "turn_started",
        "task_complete",
        "turn_complete",
        "thread_settings_applied",
        "raw_response_completed",
        "error",
        "warning",
        "guardian_warning",
        "guardian_assessment",
        "context_compacted",
        "thread_rolled_back",
        "thread_goal_updated",
        "turn_aborted",
        "item_started",
        "item_completed",
        "hook_started",
        "hook_completed",
        "entered_review_mode",
        "exited_review_mode",
        "session_configured",
        "environment_connected",
        "environment_disconnected",
        "mcp_startup_update",
        "mcp_startup_complete",
        "mcp_tool_call_begin",
        "mcp_tool_call_end",
        "web_search_begin",
        "web_search_end",
        "image_generation_begin",
        "image_generation_end",
        "exec_command_begin",
        "exec_command_output_delta",
        "terminal_interaction",
        "exec_command_end",
        "view_image_tool_call",
        "exec_approval_request",
        "request_permissions",
        "request_user_input",
        "dynamic_tool_call_request",
        "dynamic_tool_call_response",
        "elicitation_request",
        "apply_patch_approval_request",
        "deprecation_notice",
        "stream_error",
        "patch_apply_begin",
        "patch_apply_updated",
        "patch_apply_end",
        "turn_diff",
        "plan_update",
        "shutdown_complete",
        "raw_response_item",
        "agent_message_content_delta",
        "plan_delta",
        "reasoning_content_delta",
        "reasoning_raw_content_delta",
        "collab_agent_spawn_begin",
        "collab_agent_spawn_end",
        "collab_agent_interaction_begin",
        "collab_agent_interaction_end",
        "collab_waiting_begin",
        "collab_waiting_end",
        "collab_close_begin",
        "collab_close_end",
        "collab_resume_begin",
        "collab_resume_end",
        "sub_agent_activity",
        "realtime_conversation_started",
        "realtime_conversation_realtime",
        "realtime_conversation_closed",
        "realtime_conversation_sdp",
        "realtime_conversation_list_voices_response",
        "model_reroute",
        "model_verification",
        "turn_moderation_metadata",
        "safety_buffering",
        "agent_reasoning_section_break",
    }
)
_KnownEvent = Annotated[
    UserMessageEvent
    | AgentMessageEvent
    | AgentReasoningEvent
    | TokenCountEvent
    | TurnStartedEvent
    | TurnCompleteEvent
    | ThreadSettingsAppliedEvent
    | RawResponseCompletedEvent
    | ErrorEvent
    | GenericEvent,
    Field(discriminator="type"),
]
CodexEvent = Annotated[
    Annotated[_KnownEvent, Tag("known")] | Annotated[UnknownEvent, Tag("unknown")],
    Discriminator(_known_or_unknown(_EVENT_TAGS)),
]


# ---------------------------------------------------------------------------
# Rollout envelopes
# ---------------------------------------------------------------------------
class GitInfo(SnakeModel):
    commit_hash: t.GitSha | None = None
    branch: t.GitBranch | None = None
    repository_url: t.Url | None = None


class SessionMetaPayload(SnakeModel):
    session_id: t.SessionId | None = None
    id: t.SessionId | None = None
    forked_from_id: t.SessionId | None = None
    parent_thread_id: t.SessionId | None = None
    timestamp: t.Timestamp | None = None
    cwd: t.Cwd | None = None
    originator: str | None = None
    cli_version: t.Version | None = None
    source: JsonValue | None = None
    thread_source: JsonValue | None = None
    agent_nickname: t.AgentName | None = None
    agent_role: str | None = None
    agent_path: str | None = None
    model_provider: str | None = None
    base_instructions: JsonValue | None = None
    dynamic_tools: list[JsonValue] | None = None
    selected_capability_roots: list[JsonValue] = Field(default_factory=lambda: list[JsonValue]())
    memory_mode: str | None = None
    history_mode: str | None = None
    history_base: JsonValue | None = None
    subagent_history_start_ordinal: t.StepIndex | None = None
    multi_agent_version: str | None = None
    context_window: JsonValue | None = None
    git: GitInfo | None = None


class TurnContextPayload(SnakeModel):
    turn_id: t.MessageId | None = None
    cwd: t.Cwd | None = None
    workspace_roots: list[t.FilePath] | None = None
    current_date: str | None = None
    timezone: str | None = None
    model: t.ModelId | None = None
    comp_hash: str | None = None
    effort: str | None = None
    summary: JsonValue | None = None
    personality: str | None = None
    approval_policy: JsonValue | None = None
    approvals_reviewer: JsonValue | None = None
    sandbox_policy: JsonValue | None = None
    permission_profile: JsonValue | None = None
    network: JsonValue | None = None
    file_system_sandbox_policy: JsonValue | None = None
    collaboration_mode: JsonValue | None = None
    multi_agent_version: str | None = None
    multi_agent_mode: JsonValue | None = None
    realtime_active: bool | None = None


class CompactedPayload(SnakeModel):
    message: t.ContentText
    replacement_history: list[CodexResponseItem] | None = None
    window_number: t.Count | None = None
    first_window_id: t.SessionId | None = None
    previous_window_id: t.SessionId | None = None
    window_id: t.SessionId | None = None


class WorldStatePayload(SnakeModel):
    full: bool
    state: JsonValue


class RolloutBase(SnakeModel):
    timestamp: t.Timestamp
    ordinal: t.StepIndex | None = None


class SessionMetaRecord(RolloutBase):
    type: Literal["session_meta"]
    payload: SessionMetaPayload


class ResponseItemRecord(RolloutBase):
    type: Literal["response_item"]
    payload: CodexResponseItem


class EventMessageRecord(RolloutBase):
    type: Literal["event_msg"]
    payload: CodexEvent


class TurnContextRecord(RolloutBase):
    type: Literal["turn_context"]
    payload: TurnContextPayload


class CompactedRecord(RolloutBase):
    type: Literal["compacted"]
    payload: CompactedPayload


class WorldStateRecord(RolloutBase):
    type: Literal["world_state"]
    payload: WorldStatePayload


class InterAgentCommunicationRecord(RolloutBase):
    type: Literal["inter_agent_communication", "inter_agent_communication_metadata"]
    payload: JsonValue


class UnknownCodexRecord(RolloutBase):
    type: str
    payload: JsonValue | None = None


_ROLLOUT_TAGS = frozenset(
    {
        "session_meta",
        "response_item",
        "event_msg",
        "turn_context",
        "compacted",
        "world_state",
        "inter_agent_communication",
        "inter_agent_communication_metadata",
    }
)
CODEX_RECORD_TYPES = _ROLLOUT_TAGS
_KnownCodexRecord = Annotated[
    SessionMetaRecord
    | ResponseItemRecord
    | EventMessageRecord
    | TurnContextRecord
    | CompactedRecord
    | WorldStateRecord
    | InterAgentCommunicationRecord,
    Field(discriminator="type"),
]
CodexRecord = Annotated[
    Annotated[_KnownCodexRecord, Tag("known")] | Annotated[UnknownCodexRecord, Tag("unknown")],
    Discriminator(_known_or_unknown(_ROLLOUT_TAGS)),
]

CODEX_RECORD_ADAPTER: TypeAdapter[CodexRecord] = TypeAdapter(CodexRecord)
