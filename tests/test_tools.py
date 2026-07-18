"""Tool inputs/results resolve to typed models, dispatched by tool name."""

from __future__ import annotations

import pytest
from conftest import tool_inputs, tool_results
from pydantic import BaseModel, JsonValue

from cc_session_core import parse_tool_input, parse_tool_result
from cc_session_core.parsing.tools import (
    AgentResult,
    AgentToolStats,
    CronListResult,
    DesignSyncResult,
    EnterPlanModeResult,
    EnterWorktreeResult,
    ExitPlanModeResult,
    ExitWorktreeResult,
    GitOperation,
    GlobResult,
    GrepInput,
    GrepResult,
    MonitorResult,
    ScheduleWakeupResult,
    SendMessageResult,
    SkillResult,
    StatusChange,
    TaskGetResult,
    TaskOutputResult,
    TaskRef,
    TaskStopResult,
    TaskUpdateInput,
    WorkflowResult,
)

INPUTS = tool_inputs()
RESULTS = tool_results()


@pytest.mark.parametrize("name", sorted(INPUTS), ids=lambda n: n)
def test_input_resolves_to_model(name: str) -> None:
    parsed = parse_tool_input(name, INPUTS[name])
    assert isinstance(parsed, BaseModel)


@pytest.mark.parametrize("name", sorted(RESULTS), ids=lambda n: n)
def test_result_resolves_to_model(name: str) -> None:
    parsed = parse_tool_result(name, RESULTS[name])
    assert isinstance(parsed, BaseModel)


def test_known_input_fields_typed() -> None:
    parsed = parse_tool_input("Bash", INPUTS["Bash"])
    assert isinstance(parsed.command, str)  # type: ignore[union-attr]


def test_edit_result_exposes_structured_patch() -> None:
    parsed = parse_tool_result("Edit", RESULTS["Edit"])
    from cc_session_core.parsing.tools import EditResult

    assert isinstance(parsed, EditResult)
    for hunk in parsed.structured_patch:
        assert isinstance(hunk.new_lines, int)
        assert isinstance(hunk.lines, list)


def test_unknown_tool_falls_back_to_raw() -> None:
    raw: JsonValue = {"anything": [1, 2, 3]}
    assert parse_tool_input("mcp__some__tool", raw) == raw
    assert parse_tool_result("mcp__some__tool", raw) == raw


def test_malformed_input_falls_back_not_raises() -> None:
    # Bash with no command (interrupted/partial call) must not raise
    result = parse_tool_input("Bash", {})
    assert result == {}  # raw dict back, not a model, no exception


def test_string_result_passes_through() -> None:
    assert parse_tool_result("Bash", "plain error text") == "plain error text"


# ======================================================================
# Result models added for the extended tool set — each keyed by its
# camelCase/snake_case wire shape (ToolResult vs ToolResultSnake).
# ======================================================================
EXTRA_RESULTS: dict[str, tuple[dict[str, JsonValue], type[BaseModel]]] = {
    "Glob": (
        {"filenames": ["/a.py", "/b.py"], "durationMs": 12, "numFiles": 2, "truncated": False},
        GlobResult,
    ),
    "Grep": (
        {
            "mode": "content",
            "numFiles": 1,
            "filenames": ["/a.py"],
            "content": "match",
            "numLines": 1,
        },
        GrepResult,
    ),
    "Skill": (
        {"success": True, "commandName": "think", "allowedTools": ["Read"], "status": "completed"},
        SkillResult,
    ),
    "Workflow": (
        {"status": "completed", "taskId": "t1", "workflowName": "release", "runId": "r1"},
        WorkflowResult,
    ),
    "TaskStop": (
        {"message": "stopped", "task_id": "t1", "task_type": "agent", "command": "kill"},
        TaskStopResult,
    ),
    "SendMessage": (
        {"success": True, "message": "sent", "target": "team-lead", "msg_id": "m1"},
        SendMessageResult,
    ),
    "ScheduleWakeup": (
        {"scheduledFor": 1750000000, "clampedDelaySeconds": 60, "wasClamped": False},
        ScheduleWakeupResult,
    ),
    "Monitor": ({"taskId": "t1", "timeoutMs": 5000, "persistent": True}, MonitorResult),
    "DesignSync": ({"method": "pull", "name": "sync1", "projectId": "p1"}, DesignSyncResult),
    "CronList": ({"jobs": [{"id": "cron1"}]}, CronListResult),
    "EnterPlanMode": ({"message": "entered plan mode"}, EnterPlanModeResult),
    "ExitPlanMode": ({"plan": "do the thing", "isAgent": False}, ExitPlanModeResult),
    "EnterWorktree": (
        {"worktreePath": "/tmp/wt", "worktreeBranch": "feat/x"},
        EnterWorktreeResult,
    ),
    "ExitWorktree": ({"action": "merge", "originalCwd": "/home/x"}, ExitWorktreeResult),
    "TaskGet": ({"task": {"id": "t1", "subject": "fix bug"}}, TaskGetResult),
    "TaskOutput": ({"retrieval_status": "ready", "task": {"id": "t1"}}, TaskOutputResult),
}


@pytest.mark.parametrize("name", sorted(EXTRA_RESULTS), ids=lambda n: n)
def test_extra_result_types_to_its_model(name: str) -> None:
    payload, expected = EXTRA_RESULTS[name]
    parsed = parse_tool_result(name, payload)
    assert isinstance(parsed, expected)


def test_agent_result_teammate_fleet_shape_parses() -> None:
    """The Agent tool also returns a teammate/fleet-spawn shape, distinct from a
    subagent-run shape — both must type into the same AgentResult."""
    payload: dict[str, JsonValue] = {
        "status": "spawned",
        "prompt": "investigate the bug",
        "teammateId": "tm-1",
        "model": "claude-opus-4-8",
        "name": "worker",
        "color": "blue",
        "tmuxSessionName": "sess",
        "tmuxWindowName": "win",
        "tmuxPaneId": "pane1",
        "teamName": "team-x",
        "isSplitpane": False,
        "planModeRequired": False,
    }
    parsed = parse_tool_result("Agent", payload)
    assert isinstance(parsed, AgentResult)
    assert parsed.teammate_id == "tm-1"
    assert parsed.team_name == "team-x"
    assert parsed.is_splitpane is False


def test_agent_result_tool_stats_nested_type() -> None:
    payload: dict[str, JsonValue] = {
        "status": "completed",
        "prompt": "investigate the bug",
        "toolStats": {
            "readCount": 3,
            "searchCount": 1,
            "bashCount": 2,
            "editFileCount": 1,
            "linesAdded": 10,
            "linesRemoved": 2,
            "otherToolCount": 0,
        },
    }
    parsed = parse_tool_result("Agent", payload)
    assert isinstance(parsed, AgentResult)
    assert isinstance(parsed.tool_stats, AgentToolStats)
    assert parsed.tool_stats.read_count == 3


def test_task_update_result_status_change_nested_type() -> None:
    payload: dict[str, JsonValue] = {
        "success": True,
        "taskId": "t1",
        "updatedFields": ["status"],
        "statusChange": {"from": "pending", "to": "in_progress"},
    }
    from cc_session_core.parsing.tools import TaskUpdateResult

    parsed = parse_tool_result("TaskUpdate", payload)
    assert isinstance(parsed, TaskUpdateResult)
    assert isinstance(parsed.status_change, StatusChange)
    assert parsed.status_change.from_ == "pending"
    assert parsed.status_change.to == "in_progress"


def test_git_operation_pr_without_url_still_types() -> None:
    """A PR op still resolving (no url yet) must not be over-strict."""
    op = GitOperation.model_validate({"pr": {"action": "open", "number": 42}})
    assert op.pr is not None
    assert op.pr.url is None


def test_task_ref_missing_id_and_subject_still_types() -> None:
    ref = TaskRef.model_validate({})
    assert ref.id is None
    assert ref.subject is None


def test_task_update_input_resolves_task_id_from_camel_alias() -> None:
    parsed = TaskUpdateInput.model_validate({"taskId": "t9", "status": "done"})
    assert parsed.task_id_camel == "t9"
    assert parsed.status == "done"


def test_grep_input_resolves_context_after_flag() -> None:
    parsed = GrepInput.model_validate({"pattern": "foo", "-A": 3})
    assert parsed.context_after == 3
