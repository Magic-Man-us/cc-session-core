"""Typed models for built-in tool inputs and results, dispatched by tool name.

``tool_use.input`` and ``toolUseResult`` have no in-payload discriminator — the
tool *name* is the discriminator (on the sibling ``tool_use`` block; for results,
reached via ``tool_use_id``). So instead of one ``Field(discriminator=...)`` union,
this is a **name-keyed registry** of ``TypeAdapter``s plus two resolvers:

* ``parse_tool_input(name, value)``  → typed input model, or the raw value
* ``parse_tool_result(name, value)`` → typed result model, or the raw value

Coverage is the built-in file/shell/web/task tools. Unknown tools (MCP, custom,
``StructuredOutput`` whose schema is per-call) and non-dict payloads fall through
to ``JsonValue`` — their schema is owned by the tool, not by us, so typing them
would be a guess. Models use ``extra="allow"`` (payloads are model-generated and
occasionally carry extra/partial keys); a payload that can't validate falls back
to the raw value rather than raising.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from .. import types as t
from ..models import (
    AssistantRecord,
    CamelModel,
    Record,
    SnakeModel,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserRecord,
)


class ToolInput(SnakeModel):
    pass


class ToolResult(CamelModel):
    pass


class ToolResultSnake(SnakeModel):
    pass


# ======================================================================
# Inputs (snake_case)
# ======================================================================
class BashInput(ToolInput):
    command: t.CommandLine
    description: t.DescriptionText | None = None
    timeout: t.DurationMs | None = None
    run_in_background: bool | None = None
    dangerously_disable_sandbox: bool | None = None


class ReadInput(ToolInput):
    file_path: t.FilePath
    limit: t.Count | None = None
    offset: t.Count | None = None


class EditInput(ToolInput):
    file_path: t.FilePath
    old_string: t.ContentText
    new_string: t.ContentText
    replace_all: bool | None = None


class WriteInput(ToolInput):
    file_path: t.FilePath
    content: t.ContentText


class GlobInput(ToolInput):
    pattern: t.Pattern
    path: t.FilePath | None = None


class WebFetchInput(ToolInput):
    url: t.Url
    prompt: t.PromptText


class WebSearchInput(ToolInput):
    query: t.SearchQuery


class ToolSearchInput(ToolInput):
    query: t.SearchQuery
    max_results: t.MaxResults | None = None


class AgentInput(ToolInput):
    description: t.DescriptionText
    prompt: t.PromptText
    subagent_type: t.AgentType
    name: t.AgentName | None = None
    model: t.ModelId | None = None
    isolation: t.ResultText | None = None
    run_in_background: bool | None = None


class AskUserQuestionInput(ToolInput):
    questions: list[JsonValue]


class DesignSyncInput(ToolInput):
    method: t.SyncMethod | None = None
    name: t.DisplayName | None = None
    project_id: t.ProjectId | None = None
    plan_id: t.PlanId | None = None
    local_dir: t.FilePath | None = None
    path: t.FilePath | None = None
    paths: list[t.FilePath] | None = None
    files: list[JsonValue] | None = None
    writes: list[JsonValue] | None = None
    deletes: list[JsonValue] | None = None
    counts: JsonValue | None = None


class EnterWorktreeInput(ToolInput):
    path: t.FilePath


class ExitPlanModeInput(ToolInput):
    plan: t.ContentText | None = None
    plan_file_path: t.FilePath | None = None
    allowed_prompts: list[JsonValue] | None = None


class ExitWorktreeInput(ToolInput):
    action: t.GitAction | None = None


class GrepInput(ToolInput):
    """Loose/mixed shape — ripgrep flag keys (``-A``/``-B``/``-C``/``-i``/``-n``)
    need explicit aliases; extras (e.g. ``command``, ``query``) flow to model_extra."""

    pattern: t.Pattern | None = None
    path: t.FilePath | None = None
    output_mode: t.OutputMode | None = None
    glob: t.Pattern | None = None
    type: t.MimeType | None = None
    head_limit: t.Count | None = None
    multiline: bool | None = None
    context_after: t.Count | None = Field(default=None, alias="-A")
    context_before: t.Count | None = Field(default=None, alias="-B")
    context: t.Count | None = Field(default=None, alias="-C")
    case_insensitive: bool | None = Field(default=None, alias="-i")
    line_numbers: bool | None = Field(default=None, alias="-n")


class MonitorInput(ToolInput):
    """Loose shape spanning background-task, search, and message-wait uses."""

    command: t.CommandLine | None = None
    description: t.DescriptionText | None = None
    query: t.SearchQuery | None = None
    target: t.RecipientRef | None = None
    until: t.ResultText | None = None
    max_results: t.MaxResults | None = None
    timeout: t.DurationMs | None = None
    timeout_ms: t.DurationMs | None = None
    persistent: bool | None = None


class ScheduleWakeupInput(ToolInput):
    delay_seconds: t.DelaySeconds
    prompt: t.PromptText | None = None
    reason: t.ReasonText | None = None


class SendMessageInput(ToolInput):
    """Loose shape — ``message`` is str in some calls, a dict in others."""

    to: t.RecipientRef | None = None
    recipient: t.RecipientRef | None = None
    type: t.ContentType | None = None
    content: t.ContentText | None = None
    message: JsonValue | None = None
    summary: t.ResultText | None = None
    request_id: t.RequestId | None = None
    approve: bool | None = None


class SkillInput(ToolInput):
    skill: t.SkillName | None = None
    args: t.SkillArgs | None = None


class TaskCreateInput(ToolInput):
    subject: t.DescriptionText | None = None
    description: t.DescriptionText | None = None
    active_form: t.DescriptionText | None = None
    prompt: t.PromptText | None = None


class TaskGetInput(ToolInput):
    task_id: t.TaskId | None = Field(default=None, alias="taskId")


class TaskOutputInput(ToolInput):
    task_id: t.TaskId | None = None
    timeout: t.DurationMs | None = None
    block: bool | None = None


class TaskStopInput(ToolInput):
    task_id: t.TaskId | None = None


class TaskUpdateInput(ToolInput):
    """Corpus has both ``taskId`` and ``task_id`` keys across records."""

    task_id: t.TaskId | None = None
    task_id_camel: t.TaskId | None = Field(default=None, alias="taskId")
    subject: t.DescriptionText | None = None
    description: t.DescriptionText | None = None
    status: t.StatusText | None = None
    owner: t.AgentId | None = None
    add_blocked_by: list[JsonValue] | None = Field(default=None, alias="addBlockedBy")


class WorkflowInput(ToolInput):
    script: t.ContentText | None = None
    script_path: t.FilePath | None = None


# ======================================================================
# Nested result payloads
# ======================================================================
class FileReadPayload(ToolResult):
    file_path: t.FilePath | None = None  # absent for image reads (base64 payload)
    content: t.ContentText | None = None
    num_lines: t.Count | None = None
    start_line: t.Count | None = None
    total_lines: t.Count | None = None
    type: t.MimeType | None = None
    base64: t.Base64Data | None = None
    dimensions: JsonValue | None = None
    original_size: t.ByteSize | None = None
    truncated_by_token_cap: bool | None = None


class PatchHunk(ToolResult):
    old_start: t.Count
    old_lines: t.Count
    new_start: t.Count
    new_lines: t.Count
    lines: list[t.PatchLine]


class GitBranchOp(ToolResult):
    action: t.GitAction
    ref: t.GitRef


class GitCommitOp(ToolResult):
    kind: t.GitCommitKind
    sha: t.GitSha


class GitPrOp(ToolResult):
    action: t.GitAction
    number: t.PrNumber
    url: t.PrUrl | None = None  # absent while a PR op is still resolving (18 in corpus)


class GitPushOp(ToolResult):
    branch: t.GitBranch


class GitOperation(ToolResult):
    branch: GitBranchOp | None = None
    commit: GitCommitOp | None = None
    pr: GitPrOp | None = None
    push: GitPushOp | None = None


class StatusChange(ToolResult):
    from_: t.StatusText = Field(alias="from")
    to: t.StatusText


class TaskRef(ToolResult):
    id: t.TaskId | None = None
    subject: t.DescriptionText | None = None


class AgentToolStats(ToolResult):
    read_count: t.Count
    search_count: t.Count
    bash_count: t.Count
    edit_file_count: t.Count
    lines_added: t.Count
    lines_removed: t.Count
    other_tool_count: t.Count


# ======================================================================
# Results (camelCase unless noted)
# ======================================================================
class BashResult(ToolResult):
    stdout: t.ShellOutput
    stderr: t.ShellOutput
    interrupted: bool
    is_image: bool | None = None  # absent in an older result shape (~3 of 22.8k)
    no_output_expected: bool | None = None
    return_code_interpretation: t.ResultText | None = None
    background_task_id: t.BackgroundTaskId | None = None
    assistant_auto_backgrounded: bool | None = None
    backgrounded_by_user: bool | None = None
    dangerously_disable_sandbox: bool | None = None
    git_operation: GitOperation | None = None
    persisted_output_path: t.FilePath | None = None
    persisted_output_size: t.ByteSize | None = None
    stale_read_file_state_hint: t.ResultText | None = None


class ReadResult(ToolResult):
    type: t.MimeType
    file: FileReadPayload


class EditResult(ToolResult):
    file_path: t.FilePath
    old_string: t.ContentText
    new_string: t.ContentText
    replace_all: bool
    original_file: t.ContentText | None = None
    structured_patch: list[PatchHunk]
    user_modified: bool


class WriteResult(ToolResult):
    file_path: t.FilePath
    content: t.ContentText
    type: t.MimeType
    original_file: t.ContentText | None = None
    structured_patch: list[PatchHunk]
    user_modified: bool


class WebFetchResult(ToolResult):
    url: t.Url
    result: t.ResultText
    code: t.HttpStatus
    code_text: t.ResultText
    bytes: t.ByteSize
    duration_ms: t.DurationMs


class WebSearchResult(ToolResult):
    query: t.SearchQuery
    results: list[JsonValue]
    search_count: t.Count
    duration_seconds: t.DurationSeconds


class ToolSearchResult(ToolResultSnake):
    query: t.SearchQuery
    matches: list[JsonValue]
    total_deferred_tools: t.Count


class AgentResult(ToolResult):
    """Covers both shapes the Agent tool returns: a subagent run (status, prompt,
    usage/toolStats, …) and a teammate/fleet spawn (teammate_id, tmux refs, …)."""

    status: t.StatusText
    prompt: t.PromptText
    agent_id: t.AgentId | None = None
    agent_type: t.AgentType | None = None
    resolved_model: t.ModelId | None = None
    description: t.DescriptionText | None = None
    content: list[JsonValue] | None = None
    usage: Usage | None = None
    tool_stats: AgentToolStats | None = None
    total_tokens: t.TokenCount | None = None
    total_duration_ms: t.DurationMs | None = None
    total_tool_use_count: t.Count | None = None
    is_async: bool | None = None
    output_file: t.OutputFilePath | None = None
    can_read_output_file: bool | None = None
    # SHAPE 2 — teammate/fleet spawn
    teammate_id: t.TeammateId | None = None
    model: t.ModelId | None = None
    name: t.AgentName | None = None
    color: t.ColorName | None = None
    tmux_session_name: t.TmuxRef | None = None
    tmux_window_name: t.TmuxRef | None = None
    tmux_pane_id: t.TmuxRef | None = None
    team_name: t.TeamName | None = None
    is_splitpane: bool | None = None
    plan_mode_required: bool | None = None


class TaskCreateResult(ToolResult):
    task: TaskRef


class TaskUpdateResult(ToolResult):
    success: bool
    task_id: t.TaskId
    updated_fields: list[str]
    status_change: StatusChange | None = None
    error: t.ErrorText | None = None


class TaskListResult(ToolResult):
    tasks: list[JsonValue]


class AskUserQuestionResult(ToolResult):
    questions: list[JsonValue]
    answers: JsonValue
    annotations: JsonValue | None = None
    afk_timeout_ms: t.DurationMs | None = None


class CronListResult(ToolResult):
    jobs: list[JsonValue]


class DesignSyncResult(ToolResult):
    method: t.SyncMethod | None = None
    name: t.DisplayName | None = None
    notice: t.NoticeText | None = None
    project_id: t.ProjectId | None = None
    plan_id: t.PlanId | None = None
    projects: list[JsonValue] | None = None
    owner_display_name: t.DisplayName | None = None
    path: t.FilePath | None = None
    paths: list[t.FilePath] | None = None
    content: t.ContentText | None = None
    content_type: t.ContentType | None = None
    type: t.ContentType | None = None
    is_base64: bool | None = None
    truncated: bool | None = None
    writes: list[JsonValue] | None = None
    deletes: list[JsonValue] | None = None
    written: t.Count | None = None
    deleted: t.Count | None = None


class EnterPlanModeResult(ToolResult):
    message: t.ResultText | None = None


class EnterWorktreeResult(ToolResult):
    worktree_path: t.FilePath
    worktree_branch: t.GitBranch
    message: t.ResultText | None = None


class ExitPlanModeResult(ToolResult):
    plan: t.ContentText
    is_agent: bool | None = None
    file_path: t.FilePath | None = None
    has_task_tool: bool | None = None
    plan_was_edited: bool | None = None


class ExitWorktreeResult(ToolResult):
    action: t.GitAction | None = None
    original_cwd: t.Cwd | None = None
    worktree_path: t.FilePath | None = None
    worktree_branch: t.GitBranch | None = None
    message: t.ResultText | None = None


class GlobResult(ToolResult):
    filenames: list[t.FilePath]
    duration_ms: t.DurationMs
    num_files: t.Count
    truncated: bool | None = None
    total_matches: t.Count | None = None
    count_is_complete: bool | None = None


class GrepResult(ToolResult):
    mode: t.OutputMode | None = None
    num_files: t.Count | None = None
    filenames: list[t.FilePath] | None = None
    content: t.ContentText | None = None
    num_lines: t.Count | None = None


class MonitorResult(ToolResult):
    task_id: t.TaskId | None = None
    timeout_ms: t.DurationMs | None = None
    persistent: bool | None = None


class ScheduleWakeupResult(ToolResult):
    scheduled_for: t.ScheduledForEpoch
    clamped_delay_seconds: t.DelaySeconds | None = None
    was_clamped: bool | None = None


class SendMessageResult(ToolResultSnake):
    success: bool
    message: t.ResultText | None = None
    routing: JsonValue | None = None
    request_id: t.RequestId | None = None
    target: t.RecipientRef | None = None
    msg_id: t.MessageId | None = None


class SkillResult(ToolResult):
    success: bool | None = None
    command_name: t.CommandName | None = None
    allowed_tools: list[str] | None = None
    status: t.StatusText | None = None
    agent_id: t.AgentId | None = None
    result: JsonValue | None = None


class TaskGetResult(ToolResult):
    task: TaskRef | None = None


class TaskOutputResult(ToolResultSnake):
    retrieval_status: t.StatusText | None = None
    task: TaskRef | None = None


class TaskStopResult(ToolResultSnake):
    message: t.ResultText | None = None
    task_id: t.TaskId | None = None
    task_type: t.TaskTypeName | None = None
    command: t.CommandLine | None = None


class WorkflowResult(ToolResult):
    status: t.StatusText | None = None
    task_id: t.TaskId | None = None
    task_type: t.TaskTypeName | None = None
    workflow_name: t.WorkflowName | None = None
    run_id: t.RunId | None = None
    summary: t.ResultText | None = None
    transcript_dir: t.TranscriptDir | None = None
    script_path: t.FilePath | None = None


# ======================================================================
# Registries + resolvers
# ======================================================================
_INPUTS: dict[str, type[BaseModel]] = {
    "Bash": BashInput,
    "Read": ReadInput,
    "Edit": EditInput,
    "Write": WriteInput,
    "Glob": GlobInput,
    "WebFetch": WebFetchInput,
    "WebSearch": WebSearchInput,
    "ToolSearch": ToolSearchInput,
    "Agent": AgentInput,
    "AskUserQuestion": AskUserQuestionInput,
    "DesignSync": DesignSyncInput,
    "EnterWorktree": EnterWorktreeInput,
    "ExitPlanMode": ExitPlanModeInput,
    "ExitWorktree": ExitWorktreeInput,
    "Grep": GrepInput,
    "Monitor": MonitorInput,
    "ScheduleWakeup": ScheduleWakeupInput,
    "SendMessage": SendMessageInput,
    "Skill": SkillInput,
    "TaskCreate": TaskCreateInput,
    "TaskGet": TaskGetInput,
    "TaskOutput": TaskOutputInput,
    "TaskStop": TaskStopInput,
    "TaskUpdate": TaskUpdateInput,
    "Workflow": WorkflowInput,
}
_RESULTS: dict[str, type[BaseModel]] = {
    "Bash": BashResult,
    "Read": ReadResult,
    "Edit": EditResult,
    "Write": WriteResult,
    "WebFetch": WebFetchResult,
    "WebSearch": WebSearchResult,
    "ToolSearch": ToolSearchResult,
    "Agent": AgentResult,
    "TaskCreate": TaskCreateResult,
    "TaskUpdate": TaskUpdateResult,
    "TaskList": TaskListResult,
    "AskUserQuestion": AskUserQuestionResult,
    "CronList": CronListResult,
    "DesignSync": DesignSyncResult,
    "EnterPlanMode": EnterPlanModeResult,
    "EnterWorktree": EnterWorktreeResult,
    "ExitPlanMode": ExitPlanModeResult,
    "ExitWorktree": ExitWorktreeResult,
    "Glob": GlobResult,
    "Grep": GrepResult,
    "Monitor": MonitorResult,
    "ScheduleWakeup": ScheduleWakeupResult,
    "SendMessage": SendMessageResult,
    "Skill": SkillResult,
    "TaskGet": TaskGetResult,
    "TaskOutput": TaskOutputResult,
    "TaskStop": TaskStopResult,
    "Workflow": WorkflowResult,
}

TOOL_INPUT_ADAPTERS: dict[str, TypeAdapter[BaseModel]] = {
    n: TypeAdapter(m) for n, m in _INPUTS.items()
}
TOOL_RESULT_ADAPTERS: dict[str, TypeAdapter[BaseModel]] = {
    n: TypeAdapter(m) for n, m in _RESULTS.items()
}

MODELED_INPUT_TOOLS = frozenset(_INPUTS)
MODELED_RESULT_TOOLS = frozenset(_RESULTS)


def parse_tool_input(name: str, value: JsonValue) -> BaseModel | JsonValue:
    """Typed input model for a known tool, else the raw value."""
    adapter = TOOL_INPUT_ADAPTERS.get(name)
    if adapter is None or not isinstance(value, dict):
        return value
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return value


def parse_tool_result(name: str | None, value: JsonValue) -> BaseModel | JsonValue:
    """Typed result model for a known tool, else the raw value (str results pass through)."""
    if name is None or not isinstance(value, dict):
        return value
    adapter = TOOL_RESULT_ADAPTERS.get(name)
    if adapter is None:
        return value
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return value


def tool_name_index(records: Iterable[Record]) -> dict[str, str]:
    """Map each ``tool_use`` id to the tool name that produced it."""
    idx: dict[str, str] = {}
    for rec in records:
        if isinstance(rec, AssistantRecord):
            for block in rec.message.content:
                if isinstance(block, ToolUseBlock):
                    idx[block.id] = block.name
    return idx


def result_tool_name(record: UserRecord, index: dict[str, str]) -> str | None:
    """Resolve which tool a user record's ``toolUseResult`` came from, via ``tool_use_id``."""
    content = record.message.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, ToolResultBlock):
                return index.get(block.tool_use_id)
    return None
