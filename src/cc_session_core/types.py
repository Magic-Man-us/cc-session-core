"""Domain primitives and shared model configs for Claude Code transcript records.

Every domain concept is a named ``Annotated`` alias carrying its constraint and
schema metadata, so no model below declares a bare ``str``/``int``/``float``.
Booleans are left as plain ``bool`` — a flag is fully described by its field name
and carries no constraint or format an alias would add.

Two config presets are centralized here:

* ``SNAKE_CONFIG`` — for models whose JSON keys are already snake_case
  (the Anthropic API passthrough: ``message``, ``usage``, content blocks).
* ``CAMEL_CONFIG`` — for Claude Code's own envelope/attachment records, whose
  keys are camelCase; a ``to_camel`` alias generator maps them, with explicit
  per-field overrides for the all-caps ``...ID``/``...UUID`` keys it can't derive.

Both keep unknown fields (``extra="allow"``): parsing is lossless, so a record
kind or field the models miss lands in ``model_extra`` rather than being dropped
or raising. The coverage gate moves to the schema audit, which reports anything
that fell into ``extra`` or an ``Unknown*`` fallback so gaps stay visible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

SNAKE_CONFIG = ConfigDict(extra="allow", validate_by_name=True)
CAMEL_CONFIG = ConfigDict(
    extra="allow", alias_generator=to_camel, validate_by_name=True, validate_by_alias=True
)

# --- identifiers ---------------------------------------------------------
SessionId = Annotated[str, Field(title="Session ID", description="UUID of the session.")]
RecordUuid = Annotated[str, Field(title="Record UUID", description="UUID of this entry.")]
ParentUuid = Annotated[str, Field(title="Parent UUID", description="UUID of the parent entry.")]
LeafUuid = Annotated[str, Field(title="Leaf UUID", description="UUID a pointer record refers to.")]
MessageId = Annotated[str, Field(title="Message ID", description="ID tying a record to a turn.")]
ApiMessageId = Annotated[str, Field(title="API message ID", description="Anthropic msg_… id.")]
RequestId = Annotated[str, Field(title="Request ID", description="Anthropic req_… id.")]
PromptId = Annotated[str, Field(title="Prompt ID", description="Originating user prompt.")]
ToolUseId = Annotated[str, Field(title="Tool-use ID", description="Links a call to its result.")]
AgentId = Annotated[str, Field(title="Agent ID", description="(Sub)agent that emitted this.")]
WorkKey = Annotated[str, Field(title="Work key", description="Key of a background work item.")]

# --- roles / attribution -------------------------------------------------
Role = Annotated[str, Field(title="Message role", description="user | assistant.")]
MessageType = Annotated[str, Field(title="Message type", description="Always 'message'.")]
IterationType = Annotated[str, Field(title="Usage iteration type")]
AttributionRef = Annotated[str, Field(title="Attribution")]
PromptSource = Annotated[str, Field(title="Prompt source")]
QueuePriority = Annotated[str, Field(title="Queue priority")]

# --- model / service -----------------------------------------------------
ModelId = Annotated[str, Field(title="Model ID", description="Claude model id.")]
ServiceTier = Annotated[str, Field(title="Service tier", description="Inference service tier.")]
Speed = Annotated[str, Field(title="Speed", description="standard | fast.")]
InferenceGeo = Annotated[str, Field(title="Inference geo", description="Region inference ran in.")]
StopReason = Annotated[str, Field(title="Stop reason", description="Why generation stopped.")]
StopSequence = Annotated[str, Field(title="Stop sequence", description="Matched stop sequence.")]
Signature = Annotated[str, Field(title="Thinking signature")]

# --- counts / measures ---------------------------------------------------
TokenCount = Annotated[int, Field(ge=0, title="Token count")]
Count = Annotated[int, Field(ge=0, title="Count")]
DurationMs = Annotated[int, Field(ge=0, title="Duration (ms)")]
DurationSeconds = Annotated[float, Field(ge=0, title="Duration (s)")]
ByteSize = Annotated[int, Field(ge=0, title="Size (bytes)")]
ByteOffset = Annotated[int, Field(ge=0, title="Byte offset")]
ExitCode = Annotated[int, Field(title="Process exit code")]
LineNumber = Annotated[int, Field(ge=0, title="Line number")]
RetryMs = Annotated[float, Field(ge=0, title="Retry delay (ms)")]
HttpStatus = Annotated[int, Field(title="HTTP status code")]

# --- environment / paths -------------------------------------------------
Cwd = Annotated[str, Field(title="Working directory")]
GitBranch = Annotated[str, Field(title="Git branch")]
Version = Annotated[str, Field(title="Claude Code version")]
EntryPoint = Annotated[str, Field(title="Entry point", description="How the session was started.")]
UserType = Annotated[str, Field(title="User type", description="external | …")]
Slug = Annotated[str, Field(title="Session slug")]
SessionKind = Annotated[str, Field(title="Session kind")]
FilePath = Annotated[str, Field(title="File path")]
DisplayPath = Annotated[str, Field(title="Display path")]

# --- free text / labels --------------------------------------------------
Mode = Annotated[str, Field(title="Mode")]
PermissionMode = Annotated[str, Field(title="Permission mode")]
AiTitle = Annotated[str, Field(title="AI-generated title")]
SessionTitle = Annotated[str, Field(title="Session title")]
AgentName = Annotated[str, Field(title="Agent name")]
AgentSetting = Annotated[str, Field(title="Agent setting")]
Subtype = Annotated[str, Field(title="System record subtype")]
Operation = Annotated[str, Field(title="Queue operation")]
LogLevel = Annotated[str, Field(title="Log level")]
Trigger = Annotated[str, Field(title="Trigger", description="What triggered a system event.")]
Direction = Annotated[str, Field(title="Direction", description="Summarization direction.")]
RefusalCategory = Annotated[str, Field(title="API refusal category")]
RefusalExplanation = Annotated[str, Field(title="API refusal explanation")]
ToolDenialKind = Annotated[str, Field(title="Tool denial kind")]
CallerType = Annotated[str, Field(title="Caller type", description="Who invoked a tool.")]
ErrorCode = Annotated[str, Field(title="Error code")]
FormattedError = Annotated[str, Field(title="Formatted error text")]
PromptText = Annotated[str, Field(title="Prompt text")]
ContentText = Annotated[str, Field(title="Content text")]
ShellOutput = Annotated[str, Field(title="Shell output")]
CommandLine = Annotated[str, Field(title="Command line")]
CommandName = Annotated[str, Field(title="Command name")]
CommandMode = Annotated[str, Field(title="Command mode")]
ToolName = Annotated[str, Field(title="Tool name")]
HookName = Annotated[str, Field(title="Hook name")]
HookEvent = Annotated[str, Field(title="Hook event")]
SkillName = Annotated[str, Field(title="Skill name")]
ErrorText = Annotated[str, Field(title="Error text")]
IdeName = Annotated[str, Field(title="IDE name")]
ReminderType = Annotated[str, Field(title="Reminder type")]
ImagePasteId = Annotated[int, Field(title="Image paste ID", description="Pasted-image handle.")]
SkillDir = Annotated[str, Field(title="Skill directory")]
OutputFilePath = Annotated[str, Field(title="Output file path")]
TaskId = Annotated[str, Field(title="Task ID")]
TaskTypeName = Annotated[str, Field(title="Task type")]
StatusText = Annotated[str, Field(title="Status")]
AgentType = Annotated[str, Field(title="Agent type")]
DescriptionText = Annotated[str, Field(title="Description")]

# --- pr-link -------------------------------------------------------------
PrNumber = Annotated[int, Field(gt=0, title="PR number")]
PrUrl = Annotated[str, Field(title="PR URL")]
PrRepository = Annotated[str, Field(title="PR repository")]

# --- time ----------------------------------------------------------------
Timestamp = Annotated[datetime, Field(title="Timestamp", description="ISO-8601 event time.")]
ScheduledForEpoch = Annotated[int, Field(title="Scheduled-for epoch")]
DateString = Annotated[str, Field(title="Calendar date")]

# --- tool-call payloads --------------------------------------------------
Url = Annotated[str, Field(title="URL")]
SearchQuery = Annotated[str, Field(title="Search query")]
Pattern = Annotated[str, Field(title="Search pattern")]
ResultText = Annotated[str, Field(title="Result text")]
MaxResults = Annotated[int, Field(ge=0, title="Max results")]
BackgroundTaskId = Annotated[str, Field(title="Background task ID")]
Base64Data = Annotated[str, Field(title="Base64 data")]
MimeType = Annotated[str, Field(title="MIME type")]
PatchLine = Annotated[str, Field(title="Unified-diff line")]
OutputMode = Annotated[str, Field(title="Output mode")]
NoticeText = Annotated[str, Field(title="Notice text")]
ActionName = Annotated[str, Field(title="Action")]
SyncMethod = Annotated[str, Field(title="Sync method")]
ProjectId = Annotated[str, Field(title="Project ID")]
PlanId = Annotated[str, Field(title="Plan ID")]
DisplayName = Annotated[str, Field(title="Display name")]
ContentType = Annotated[str, Field(title="Content type")]
DelaySeconds = Annotated[int, Field(title="Delay (seconds)")]
ReasonText = Annotated[str, Field(title="Reason")]
WorkflowName = Annotated[str, Field(title="Workflow name")]
RunId = Annotated[str, Field(title="Run ID")]
TranscriptDir = Annotated[str, Field(title="Transcript directory")]
TeammateId = Annotated[str, Field(title="Teammate ID")]
TeamName = Annotated[str, Field(title="Team name")]
ColorName = Annotated[str, Field(title="Color")]
TmuxRef = Annotated[str, Field(title="tmux ref")]
RecipientRef = Annotated[str, Field(title="Message recipient")]
GitAction = Annotated[str, Field(title="Git action")]
GitRef = Annotated[str, Field(title="Git ref")]
GitSha = Annotated[str, Field(title="Git SHA")]
GitCommitKind = Annotated[str, Field(title="Git commit kind")]
SkillArgs = Annotated[str, Field(title="Skill args")]
UsdPerMillion = Annotated[float, Field(ge=0, title="USD per 1M tokens")]
CostUsd = Annotated[float, Field(title="Cost (USD)")]
CostSource = Annotated[str, Field(title="Cost source", description="logged | computed.")]
StepIndex = Annotated[int, Field(ge=0, title="Step index")]
