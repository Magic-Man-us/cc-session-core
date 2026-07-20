"""Typed models for every Claude Code transcript record kind.

Layering, each layer a discriminated union on its own ``type`` field:

* content blocks   — text | thinking | tool_use | tool_result | image
* attachments      — 34 UI / hook / IDE payload kinds
* top-level record — 16 kinds (assistant, user, attachment, system, …)

``RECORD_ADAPTER`` is the boundary ``TypeAdapter`` — feed it one transcript line
and it returns the right model. Tool/hook-owned bodies (``tool_use.input``,
``toolUseResult``, attachment delta payloads) are typed ``JsonValue``: their
schema is owned by individual tools, not the transcript format, so a forced
discriminated union there would be fiction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, cast

from pydantic import BaseModel, Discriminator, Field, JsonValue, Tag, TypeAdapter

from . import types as t


class SnakeModel(BaseModel):
    model_config = t.SNAKE_CONFIG


class CamelModel(BaseModel):
    model_config = t.CAMEL_CONFIG


# ======================================================================
# Content blocks (Anthropic API passthrough — snake_case)
# ======================================================================
class TextBlock(SnakeModel):
    type: Literal["text"]
    text: t.ContentText


class ThinkingBlock(SnakeModel):
    type: Literal["thinking"]
    thinking: t.ContentText
    signature: t.Signature | None = None


class RedactedThinkingBlock(SnakeModel):
    type: Literal["redacted_thinking"]
    data: t.ContentText | None = None


class Caller(SnakeModel):
    """Who invoked a tool — e.g. a programmatic (code-execution) caller."""

    type: t.CallerType | None = None


class ToolUseBlock(SnakeModel):
    type: Literal["tool_use"]
    id: t.ToolUseId
    name: t.ToolName
    input: JsonValue
    caller: Caller | None = None


class ToolResultBlock(SnakeModel):
    type: Literal["tool_result"]
    tool_use_id: t.ToolUseId
    content: str | list[JsonValue]
    is_error: bool | None = None


class ImageBlock(SnakeModel):
    type: Literal["image"]
    source: JsonValue


class FallbackBlock(SnakeModel):
    """Model-fallback marker written when a request is retried on another model."""

    type: Literal["fallback"]
    from_: JsonValue = Field(alias="from")
    to: JsonValue


class UnknownBlock(SnakeModel):
    """Any content-block ``type`` not modeled above; full payload kept via extra."""

    type: str


def _known_or_unknown(known: frozenset[str]) -> Callable[[object], str]:
    """Build a callable discriminator that routes a payload to ``"known"`` when its
    ``type`` is one we model and ``"unknown"`` otherwise, so a new kind lands in an
    ``Unknown*`` carrier instead of raising. It inspects raw input at the validation
    boundary — a dict on first parse, a model on revalidation — the one place
    reflective access over an open ``type`` field is warranted."""

    def tag(value: object) -> str:
        if isinstance(value, dict):
            raw: object = cast("dict[str, object]", value).get("type")
        else:
            raw = getattr(value, "type", None)
        return "known" if raw in known else "unknown"

    return tag


_BLOCK_TAGS = frozenset(
    {"text", "thinking", "redacted_thinking", "tool_use", "tool_result", "image", "fallback"}
)

_KnownBlock = Annotated[
    TextBlock
    | ThinkingBlock
    | RedactedThinkingBlock
    | ToolUseBlock
    | ToolResultBlock
    | ImageBlock
    | FallbackBlock,
    Field(discriminator="type"),
]

ContentBlock = Annotated[
    Annotated[_KnownBlock, Tag("known")] | Annotated[UnknownBlock, Tag("unknown")],
    Discriminator(_known_or_unknown(_BLOCK_TAGS)),
]


# ======================================================================
# Usage (snake_case)
# ======================================================================
class CacheCreation(SnakeModel):
    # Defaulted, not required: a pre-cache-era assistant line's usage carries only
    # input_tokens/output_tokens, with no cache_creation object at all — Usage below
    # falls back to a zeroed CacheCreation for it rather than rejecting the whole record.
    ephemeral_1h_input_tokens: t.TokenCount = 0
    ephemeral_5m_input_tokens: t.TokenCount = 0


class ServerToolUse(SnakeModel):
    web_fetch_requests: t.Count
    web_search_requests: t.Count


class UsageIteration(SnakeModel):
    type: t.IterationType
    input_tokens: t.TokenCount
    output_tokens: t.TokenCount
    cache_read_input_tokens: t.TokenCount
    cache_creation_input_tokens: t.TokenCount
    cache_creation: CacheCreation


class Usage(SnakeModel):
    input_tokens: t.TokenCount
    output_tokens: t.TokenCount
    # Cache accounting postdates input_tokens/output_tokens; a pre-cache-era usage block
    # carries only those two, so every cache field here defaults to its zero value rather
    # than requiring one — the lossless contract (nothing raises, nothing is dropped) would
    # otherwise reject the whole record over fields that simply didn't exist yet.
    cache_read_input_tokens: t.TokenCount = 0
    cache_creation_input_tokens: t.TokenCount = 0
    cache_creation: CacheCreation = Field(default_factory=CacheCreation)
    server_tool_use: ServerToolUse | None = None
    iterations: list[UsageIteration] | None = None
    service_tier: t.ServiceTier | None = None
    speed: t.Speed | None = None
    inference_geo: t.InferenceGeo | None = None


# ======================================================================
# Message payloads (snake_case)
# ======================================================================
class AssistantMessage(SnakeModel):
    type: Literal["message"]
    role: t.Role
    id: t.ApiMessageId
    model: t.ModelId
    content: list[ContentBlock]
    usage: Usage
    stop_reason: t.StopReason | None = None
    stop_sequence: t.StopSequence | None = None
    stop_details: JsonValue | None = None
    container: JsonValue | None = None
    context_management: JsonValue | None = None
    diagnostics: JsonValue | None = None


class UserMessage(SnakeModel):
    role: t.Role
    content: str | list[ContentBlock]


# ======================================================================
# Attachments (Claude Code — camelCase, discriminated on `type`)
# ======================================================================
class HookSuccess(CamelModel):
    type: Literal["hook_success"]
    command: t.CommandLine
    content: t.ContentText
    duration_ms: t.DurationMs
    exit_code: t.ExitCode
    hook_event: t.HookEvent
    hook_name: t.HookName
    stderr: t.ShellOutput
    stdout: t.ShellOutput
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class HookNonBlockingError(CamelModel):
    type: Literal["hook_non_blocking_error"]
    command: t.CommandLine
    duration_ms: t.DurationMs
    exit_code: t.ExitCode
    hook_event: t.HookEvent
    hook_name: t.HookName
    stderr: t.ShellOutput
    stdout: t.ShellOutput
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class HookAdditionalContext(CamelModel):
    type: Literal["hook_additional_context"]
    content: list[JsonValue]
    hook_event: t.HookEvent
    hook_name: t.HookName
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class HookDeferredTool(CamelModel):
    type: Literal["hook_deferred_tool"]
    hook_event: t.HookEvent
    hook_name: t.HookName
    permission_mode: t.PermissionMode
    tool_input: JsonValue
    tool_name: t.ToolName
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class TaskReminder(CamelModel):
    type: Literal["task_reminder"]
    content: list[JsonValue]
    item_count: t.Count


class DiagnosticsAttachment(CamelModel):
    type: Literal["diagnostics"]
    files: list[JsonValue]
    is_new: bool


class EditedTextFile(CamelModel):
    type: Literal["edited_text_file"]
    filename: t.FilePath
    snippet: t.ContentText
    display_path: t.DisplayPath | None = None


class QueuedCommand(CamelModel):
    type: Literal["queued_command"]
    command_mode: t.CommandMode
    prompt: t.PromptText | list[JsonValue]
    image_paste_ids: list[t.ImagePasteId] | None = None
    is_meta: bool | None = None
    origin: JsonValue | None = None
    timestamp: t.Timestamp | None = None


class SkillListing(CamelModel):
    type: Literal["skill_listing"]
    content: t.ContentText
    is_initial: bool
    names: list[str]
    skill_count: t.Count


class DeferredToolsDelta(CamelModel):
    type: Literal["deferred_tools_delta"]
    added_lines: list[JsonValue]
    added_names: list[str]
    readded_names: list[str]
    removed_names: list[str]
    pending_mcp_servers: list[JsonValue] | None = None
    needs_auth_mcp_servers: list[JsonValue] | None = None


class OpenedFileInIde(CamelModel):
    type: Literal["opened_file_in_ide"]
    filename: t.FilePath
    display_path: t.DisplayPath | None = None


class FileAttachment(CamelModel):
    type: Literal["file"]
    content: JsonValue
    display_path: t.DisplayPath
    filename: t.FilePath


class McpInstructionsDelta(CamelModel):
    type: Literal["mcp_instructions_delta"]
    added_blocks: list[JsonValue]
    added_names: list[str]
    removed_names: list[str]


class DateChange(CamelModel):
    type: Literal["date_change"]
    new_date: t.DateString


class SelectedLinesInIde(CamelModel):
    type: Literal["selected_lines_in_ide"]
    content: t.ContentText
    display_path: t.DisplayPath
    filename: t.FilePath
    ide_name: t.IdeName
    line_end: t.LineNumber
    line_start: t.LineNumber


class AgentListingDelta(CamelModel):
    type: Literal["agent_listing_delta"]
    added_lines: list[JsonValue]
    added_types: list[str]
    is_initial: bool
    removed_types: list[str]
    show_concurrency_note: bool


class CompactFileReference(CamelModel):
    type: Literal["compact_file_reference"]
    display_path: t.DisplayPath
    filename: t.FilePath


class CommandPermissions(CamelModel):
    type: Literal["command_permissions"]
    allowed_tools: list[str]


class InvokedSkills(CamelModel):
    type: Literal["invoked_skills"]
    skills: list[JsonValue]


class NestedMemory(CamelModel):
    type: Literal["nested_memory"]
    content: JsonValue
    display_path: t.DisplayPath
    path: t.FilePath


class PlanModeExit(CamelModel):
    type: Literal["plan_mode_exit"]
    plan_exists: bool
    plan_file_path: t.FilePath


class PlanMode(CamelModel):
    type: Literal["plan_mode"]
    is_sub_agent: bool
    plan_exists: bool
    plan_file_path: t.FilePath
    reminder_type: t.ReminderType


class StructuredOutput(CamelModel):
    type: Literal["structured_output"]
    data: JsonValue
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class PlanFileReference(CamelModel):
    type: Literal["plan_file_reference"]
    plan_content: t.ContentText
    plan_file_path: t.FilePath


class MaxTurnsReached(CamelModel):
    type: Literal["max_turns_reached"]
    max_turns: t.Count
    turn_count: t.Count


class HookBlockingError(CamelModel):
    type: Literal["hook_blocking_error"]
    blocking_error: JsonValue
    hook_event: t.HookEvent
    hook_name: t.HookName
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class HookSystemMessage(CamelModel):
    type: Literal["hook_system_message"]
    content: t.ContentText
    hook_event: t.HookEvent
    hook_name: t.HookName
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class DynamicSkill(CamelModel):
    type: Literal["dynamic_skill"]
    display_path: t.DisplayPath
    skill_dir: t.SkillDir
    skill_names: list[str]


class PlanModeReentry(CamelModel):
    type: Literal["plan_mode_reentry"]
    plan_file_path: t.FilePath


class AgentMention(CamelModel):
    type: Literal["agent_mention"]
    agent_type: t.AgentType


class TaskStatus(CamelModel):
    type: Literal["task_status"]
    task_id: t.TaskId
    task_type: t.TaskTypeName
    status: t.StatusText
    description: t.DescriptionText
    output_file_path: t.OutputFilePath
    delta_summary: JsonValue | None = None


class WorkflowKeywordRequest(CamelModel):
    type: Literal["workflow_keyword_request"]


class TotalTokensReminder(CamelModel):
    type: Literal["total_tokens_reminder"]
    text: t.ContentText


class AutoMode(CamelModel):
    type: Literal["auto_mode"]


class ReadTruncationNotice(CamelModel):
    type: Literal["read_truncation_notice"]
    banner: t.TruncationBanner
    tool_use_id: t.ToolUseId = Field(alias="toolUseID")


class UnknownAttachment(CamelModel):
    """Any attachment ``type`` not modeled above; full payload kept via extra."""

    type: str


_KnownAttachment = Annotated[
    HookSuccess
    | HookNonBlockingError
    | HookAdditionalContext
    | HookDeferredTool
    | HookBlockingError
    | HookSystemMessage
    | TaskReminder
    | DiagnosticsAttachment
    | EditedTextFile
    | QueuedCommand
    | SkillListing
    | DeferredToolsDelta
    | DynamicSkill
    | OpenedFileInIde
    | FileAttachment
    | McpInstructionsDelta
    | DateChange
    | SelectedLinesInIde
    | AgentListingDelta
    | AgentMention
    | CompactFileReference
    | CommandPermissions
    | InvokedSkills
    | NestedMemory
    | PlanModeExit
    | PlanMode
    | PlanModeReentry
    | StructuredOutput
    | PlanFileReference
    | MaxTurnsReached
    | TaskStatus
    | WorkflowKeywordRequest
    | TotalTokensReminder
    | AutoMode
    | ReadTruncationNotice,
    Field(discriminator="type"),
]

_ATTACHMENT_TAGS = frozenset(
    {
        "hook_success",
        "hook_non_blocking_error",
        "hook_additional_context",
        "hook_deferred_tool",
        "hook_blocking_error",
        "hook_system_message",
        "task_reminder",
        "diagnostics",
        "edited_text_file",
        "queued_command",
        "skill_listing",
        "deferred_tools_delta",
        "dynamic_skill",
        "opened_file_in_ide",
        "file",
        "mcp_instructions_delta",
        "date_change",
        "selected_lines_in_ide",
        "agent_listing_delta",
        "agent_mention",
        "compact_file_reference",
        "command_permissions",
        "invoked_skills",
        "nested_memory",
        "plan_mode_exit",
        "plan_mode",
        "plan_mode_reentry",
        "structured_output",
        "plan_file_reference",
        "max_turns_reached",
        "task_status",
        "workflow_keyword_request",
        "total_tokens_reminder",
        "auto_mode",
        "read_truncation_notice",
    }
)

Attachment = Annotated[
    Annotated[_KnownAttachment, Tag("known")] | Annotated[UnknownAttachment, Tag("unknown")],
    Discriminator(_known_or_unknown(_ATTACHMENT_TAGS)),
]


# ======================================================================
# Top-level records (Claude Code — camelCase, discriminated on `type`)
# ======================================================================
class ConvBase(CamelModel):
    """Shared envelope for full conversation records."""

    session_id: t.SessionId | None = None
    uuid: t.RecordUuid | None = None
    parent_uuid: t.ParentUuid | None = None
    timestamp: t.Timestamp | None = None
    is_sidechain: bool | None = None
    user_type: t.UserType | None = None
    entrypoint: t.EntryPoint | None = None
    cwd: t.Cwd | None = None
    git_branch: t.GitBranch | None = None
    version: t.Version | None = None
    slug: t.Slug | None = None
    session_kind: t.SessionKind | None = None
    agent_id: t.AgentId | None = None
    # A snake ``session_id`` distinct from the camel ``sessionId`` envelope: it
    # references another session a record was carried in from (differs from the
    # current session in ~2.8k of ~8.7k occurrences). Kept as its own field.
    source_session_id: t.SessionId | None = Field(default=None, alias="session_id")


class AssistantRecord(ConvBase):
    type: Literal["assistant"]
    message: AssistantMessage
    effort: t.Effort | None = None
    request_id: t.RequestId | None = None
    attribution_agent: t.AttributionRef | None = None
    attribution_mcp_server: t.AttributionRef | None = None
    attribution_mcp_tool: t.AttributionRef | None = None
    attribution_plugin: t.AttributionRef | None = None
    attribution_skill: t.AttributionRef | None = None
    api_error_status: t.HttpStatus | None = None
    error: t.ErrorText | None = None
    is_api_error_message: bool | None = None


class UserRecord(ConvBase):
    type: Literal["user"]
    message: UserMessage
    classifier_meta_lines: t.ClassifierMetaLines | None = None
    tool_use_result: JsonValue | None = None
    image_paste_ids: list[t.ImagePasteId] | None = None
    interrupted_message_id: t.MessageId | None = None
    is_compact_summary: bool | None = None
    is_meta: bool | None = None
    is_visible_in_transcript_only: bool | None = None
    mcp_meta: JsonValue | None = None
    origin: JsonValue | None = None
    permission_mode: t.PermissionMode | None = None
    prompt_id: t.PromptId | None = None
    prompt_source: t.PromptSource | None = None
    queue_priority: t.QueuePriority | None = None
    source_tool_assistant_uuid: t.RecordUuid | None = Field(
        default=None, alias="sourceToolAssistantUUID"
    )
    source_tool_use_id: t.ToolUseId | None = Field(default=None, alias="sourceToolUseID")
    tool_ends_turn: bool | None = None
    tool_denial_kind: t.ToolDenialKind | None = None


class AttachmentRecord(ConvBase):
    type: Literal["attachment"]
    attachment: Attachment


class SystemErrorConnection(CamelModel):
    code: t.ErrorCode | None = None
    message: t.ErrorText | None = None
    is_ssl_error: bool | None = None


class SystemError(CamelModel):
    message: t.ErrorText | None = None
    formatted: t.FormattedError | None = None
    status: t.HttpStatus | None = None
    is_network_down: bool | None = None
    connection: SystemErrorConnection | None = None
    rate_limits: JsonValue | None = None


class CompactPreservedSegment(CamelModel):
    anchor_uuid: t.RecordUuid | None = None
    head_uuid: t.RecordUuid | None = None
    tail_uuid: t.RecordUuid | None = None


class CompactPreservedMessages(CamelModel):
    anchor_uuid: t.RecordUuid | None = None
    all_uuids: list[t.RecordUuid] | None = None
    uuids: list[t.RecordUuid] | None = None


class CompactMetadata(CamelModel):
    trigger: t.Trigger | None = None
    pre_tokens: t.TokenCount | None = None
    post_tokens: t.TokenCount | None = None
    duration_ms: t.DurationMs | None = None
    messages_summarized: t.Count | None = None
    cumulative_dropped_tokens: t.TokenCount | None = None
    pre_compact_discovered_tools: list[JsonValue] | None = None
    preserved_segment: CompactPreservedSegment | None = None
    preserved_messages: CompactPreservedMessages | None = None


class SystemRecord(ConvBase):
    type: Literal["system"]
    subtype: t.Subtype
    content: t.ContentText | None = None
    duration_ms: t.DurationMs | None = None
    message_count: t.Count | None = None
    is_meta: bool | None = None
    level: t.LogLevel | None = None
    compact_metadata: CompactMetadata | None = None
    error: SystemError | None = None
    has_output: bool | None = None
    hook_additional_context: list[JsonValue] | None = None
    hook_count: t.Count | None = None
    hook_errors: list[JsonValue] | None = None
    hook_infos: list[JsonValue] | None = None
    logical_parent_uuid: t.RecordUuid | None = None
    max_retries: t.Count | None = None
    pending_background_agent_count: t.Count | None = None
    pending_workflow_count: t.Count | None = None
    prevented_continuation: bool | None = None
    retry_attempt: t.Count | None = None
    retry_in_ms: t.RetryMs | None = None
    stop_reason: t.StopReason | None = None
    tool_use_id: t.ToolUseId | None = Field(default=None, alias="toolUseID")
    trigger: t.Trigger | None = None
    direction: t.Direction | None = None
    original_model: t.ModelId | None = None
    fallback_model: t.ModelId | None = None
    request_id: t.RequestId | None = None
    api_refusal_category: t.RefusalCategory | None = None
    api_refusal_explanation: t.RefusalExplanation | None = None
    retracted_message_uuids: list[t.RecordUuid] | None = None
    refused_user_message_uuid: t.RecordUuid | None = None


# --- lightweight pointer / state records (no full envelope) -------------
class LastPromptRecord(CamelModel):
    type: Literal["last-prompt"]
    leaf_uuid: t.LeafUuid
    session_id: t.SessionId
    last_prompt: t.PromptText | None = None


class ModeRecord(CamelModel):
    type: Literal["mode"]
    mode: t.Mode
    session_id: t.SessionId


class AiTitleRecord(CamelModel):
    type: Literal["ai-title"]
    ai_title: t.AiTitle
    session_id: t.SessionId


class PermissionModeRecord(CamelModel):
    type: Literal["permission-mode"]
    permission_mode: t.PermissionMode
    session_id: t.SessionId


class FileHistorySnapshotRecord(CamelModel):
    type: Literal["file-history-snapshot"]
    message_id: t.MessageId
    is_snapshot_update: bool
    snapshot: JsonValue


class FileHistoryDeltaRecord(CamelModel):
    type: Literal["file-history-delta"]
    message_id: t.MessageId
    snapshot_message_id: t.MessageId
    tracking_path: t.FilePath
    backup: JsonValue
    timestamp: t.Timestamp


class QueueOperationRecord(CamelModel):
    type: Literal["queue-operation"]
    operation: t.Operation
    session_id: t.SessionId
    timestamp: t.Timestamp
    content: t.ContentText | None = None


class AgentNameRecord(CamelModel):
    type: Literal["agent-name"]
    agent_name: t.AgentName
    session_id: t.SessionId


class PrLinkRecord(CamelModel):
    type: Literal["pr-link"]
    pr_number: t.PrNumber
    pr_repository: t.PrRepository
    pr_url: t.PrUrl
    session_id: t.SessionId
    timestamp: t.Timestamp


class StartedRecord(CamelModel):
    type: Literal["started"]
    agent_id: t.AgentId
    key: t.WorkKey


class ResultRecord(CamelModel):
    type: Literal["result"]
    agent_id: t.AgentId
    key: t.WorkKey
    result: JsonValue


class WorktreeStateRecord(CamelModel):
    type: Literal["worktree-state"]
    session_id: t.SessionId
    worktree_session: JsonValue | None = None


class AgentSettingRecord(CamelModel):
    type: Literal["agent-setting"]
    agent_setting: t.AgentSetting
    session_id: t.SessionId


class UnknownRecord(CamelModel):
    """Any top-level record ``type`` not modeled above; full payload kept via extra."""

    type: str


_KnownRecord = Annotated[
    AssistantRecord
    | UserRecord
    | AttachmentRecord
    | SystemRecord
    | LastPromptRecord
    | ModeRecord
    | AiTitleRecord
    | PermissionModeRecord
    | FileHistorySnapshotRecord
    | FileHistoryDeltaRecord
    | QueueOperationRecord
    | AgentNameRecord
    | PrLinkRecord
    | StartedRecord
    | ResultRecord
    | WorktreeStateRecord
    | AgentSettingRecord,
    Field(discriminator="type"),
]

_RECORD_TAGS = frozenset(
    {
        "assistant",
        "user",
        "attachment",
        "system",
        "last-prompt",
        "mode",
        "ai-title",
        "permission-mode",
        "file-history-snapshot",
        "file-history-delta",
        "queue-operation",
        "agent-name",
        "pr-link",
        "started",
        "result",
        "worktree-state",
        "agent-setting",
    }
)

Record = Annotated[
    Annotated[_KnownRecord, Tag("known")] | Annotated[UnknownRecord, Tag("unknown")],
    Discriminator(_known_or_unknown(_RECORD_TAGS)),
]

RECORD_ADAPTER: TypeAdapter[Record] = TypeAdapter(Record)
