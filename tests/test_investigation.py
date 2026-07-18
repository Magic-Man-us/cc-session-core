"""build_investigation / render_investigation_markdown compose the existing typed
views (tool calls, cost summary, timeline) into one investigation record."""

from __future__ import annotations

import json

from cc_session_core import Session, build_investigation, parse_line, render_investigation_markdown

_ASSISTANT_OK = json.dumps(
    {
        "type": "assistant",
        "uuid": "a1",
        "parentUuid": None,
        "sessionId": "s",
        "timestamp": "2026-01-01T00:00:00Z",
        "isSidechain": False,
        "userType": "external",
        "entrypoint": "cli",
        "cwd": "/w",
        "gitBranch": "main",
        "version": "1",
        "requestId": "req1",
        "message": {
            "type": "message",
            "role": "assistant",
            "id": "m1",
            "model": "claude-opus-4-8",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "content": [
                {"type": "text", "text": "checking the file listing"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                    "caller": None,
                },
            ],
        },
    }
)
_USER_OK = json.dumps(
    {
        "type": "user",
        "uuid": "u1",
        "parentUuid": "a1",
        "sessionId": "s",
        "timestamp": "2026-01-01T00:00:01Z",
        "isSidechain": False,
        "userType": "external",
        "entrypoint": "cli",
        "cwd": "/w",
        "gitBranch": "main",
        "version": "1",
        "toolUseResult": {"stdout": "file.txt", "stderr": "", "interrupted": False},
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "file.txt",
                    "is_error": False,
                },
            ],
        },
    }
)
_ASSISTANT_ERR = json.dumps(
    {
        "type": "assistant",
        "uuid": "a2",
        "parentUuid": "u1",
        "sessionId": "s",
        "timestamp": "2026-01-01T00:00:02Z",
        "isSidechain": False,
        "userType": "external",
        "entrypoint": "cli",
        "cwd": "/w",
        "gitBranch": "main",
        "version": "1",
        "requestId": "req2",
        "message": {
            "type": "message",
            "role": "assistant",
            "id": "m2",
            "model": "claude-opus-4-8",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "content": [
                {"type": "text", "text": "now remove it"},
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Bash",
                    "input": {"command": "rm file.txt"},
                    "caller": None,
                },
            ],
        },
    }
)
_USER_ERR = json.dumps(
    {
        "type": "user",
        "uuid": "u2",
        "parentUuid": "a2",
        "sessionId": "s",
        "timestamp": "2026-01-01T00:00:03Z",
        "isSidechain": False,
        "userType": "external",
        "entrypoint": "cli",
        "cwd": "/w",
        "gitBranch": "main",
        "version": "1",
        "toolUseResult": {"stdout": "", "stderr": "permission denied", "interrupted": False},
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "permission denied",
                    "is_error": True,
                },
            ],
        },
    }
)


def _session() -> Session:
    lines = (_ASSISTANT_OK, _USER_OK, _ASSISTANT_ERR, _USER_ERR)
    return Session([parse_line(line) for line in lines])


def test_report_carries_every_tool_call_with_its_reason() -> None:
    report = build_investigation(_session())
    assert [c.name for c in report.tool_calls] == ["Bash", "Bash"]
    assert report.tool_calls[0].reason == "checking the file listing"
    assert report.tool_calls[0].is_error is False
    assert report.tool_calls[1].reason == "now remove it"
    assert report.tool_calls[1].is_error is True


def test_errors_only_lists_the_failed_call() -> None:
    report = build_investigation(_session())
    assert len(report.errors) == 1
    assert report.errors[0].tool_use_id == "t2"


def test_tool_stats_rolls_up_calls_and_errors_per_tool() -> None:
    report = build_investigation(_session())
    assert len(report.tool_stats) == 1
    assert report.tool_stats[0].name == "Bash"
    assert report.tool_stats[0].calls == 2
    assert report.tool_stats[0].errors == 1


def test_cost_summary_and_info_are_populated() -> None:
    report = build_investigation(_session())
    assert report.cost_summary.requests == 2
    assert report.info.tool_calls == 2


def test_markdown_render_includes_reason_args_and_result() -> None:
    text = render_investigation_markdown(build_investigation(_session()))
    assert "checking the file listing" in text
    assert '"command": "ls"' in text
    assert "permission denied" in text
    assert "## Errors" in text
    assert "mixed" in text  # Bash: 1 of 2 calls errored
