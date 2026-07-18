"""ExportSpec selects records/parts and renders to text/markdown/json/jsonl."""

from __future__ import annotations

import json

from cc_session_core import ExportSpec, Session, parse_line
from cc_session_core.report.export import render

_ASSISTANT = json.dumps(
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
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "content": [
                {"type": "text", "text": "let me look"},
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
_USER = json.dumps(
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


def _session() -> Session:
    return Session([parse_line(_ASSISTANT), parse_line(_USER)])


def test_no_filter_includes_everything() -> None:
    text = render(_session(), ExportSpec(), "text")
    assert "let me look" in text and "tool_use Bash" in text and "tool_result" in text


def test_part_filter_keeps_only_selected_kinds() -> None:
    text = render(_session(), ExportSpec(parts={"tool_use", "tool_result"}), "text")
    assert "let me look" not in text  # text part dropped
    assert "tool_use Bash" in text and "file.txt" in text
    # a no-part entry (none here) or a non-matching entry never appears
    assert "[text]" not in text


def test_tool_filter_restricts_to_named_tools() -> None:
    assert "Bash" in render(_session(), ExportSpec(tools={"Bash"}), "text")
    assert render(_session(), ExportSpec(tools={"Edit"}), "text").strip() == ""


def test_type_and_uuid_filters() -> None:
    only_user = render(_session(), ExportSpec(types={"user"}), "text")
    assert "[user]" in only_user and "[assistant]" not in only_user
    picked = render(_session(), ExportSpec(uuids={"a1"}), "text")
    assert "#a1" in picked and "#u1" not in picked


def test_json_and_jsonl_roundtrip() -> None:
    payload = json.loads(render(_session(), ExportSpec(), "json"))
    assert isinstance(payload, list) and payload[0]["type"] == "assistant"
    lines = render(_session(), ExportSpec(), "jsonl").splitlines()
    assert len(lines) == 2
    assert parse_line(lines[0]).type == "assistant"  # raw records re-parse
