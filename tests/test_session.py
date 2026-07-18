"""Session assembles records into tool_calls/timeline/cost_summary views."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from cc_session_core.parsing.tools import BashResult
from cc_session_core.session import Session, iter_raw

ENVELOPE = {
    "sessionId": "s1",
    "isSidechain": False,
    "userType": "external",
    "entrypoint": "cli",
    "cwd": "/repo",
    "gitBranch": "main",
    "version": "1.0.0",
}


def _line(**overrides: JsonValue) -> str:
    return json.dumps({**ENVELOPE, **overrides})


USER_PROMPT = _line(
    type="user",
    uuid="u0",
    parentUuid=None,
    timestamp="2026-01-01T00:00:00Z",
    message={"role": "user", "content": "Please investigate the repo"},
)

ASSISTANT_TOOL_USE = _line(
    type="assistant",
    uuid="u1",
    parentUuid="u0",
    timestamp="2026-01-01T00:00:01Z",
    requestId="req1",
    message={
        "type": "message",
        "role": "assistant",
        "id": "msg1",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "text", "text": "Let me run a command."},
            {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}},
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        },
    },
)

# Same request id, a different uuid — a later streamed record of the same request.
# Real transcripts grow output_tokens across these records; the request must count
# once, at the last record's (complete) usage.
ASSISTANT_RETRY = _line(
    type="assistant",
    uuid="u1b",
    parentUuid="u1",
    timestamp="2026-01-01T00:00:01.500Z",
    requestId="req1",
    message={
        "type": "message",
        "role": "assistant",
        "id": "msg1",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}},
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 12,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        },
    },
)

USER_TOOL_RESULT = _line(
    type="user",
    uuid="u2",
    parentUuid="u1",
    timestamp="2026-01-01T00:00:02Z",
    message={
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu1",
                "content": "file1\nfile2",
                "is_error": False,
            },
        ],
    },
    toolUseResult={"stdout": "file1\nfile2", "stderr": "", "interrupted": False},
)


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    path = tmp_path / "session.jsonl"
    lines = [USER_PROMPT, ASSISTANT_TOOL_USE, ASSISTANT_RETRY, USER_TOOL_RESULT, USER_TOOL_RESULT]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_dedupe_drops_repeated_uuid(transcript: Path) -> None:
    session = Session.load(transcript)
    uuids = [getattr(r, "uuid", None) for r in session.records]
    assert uuids.count("u2") == 1


def test_tool_calls_pairs_use_with_result_and_narration(transcript: Path) -> None:
    session = Session.load(transcript)
    calls = session.tool_calls()
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "Bash"
    assert call.reason == "Let me run a command."
    assert call.is_error is False
    assert call.result_text == "file1\nfile2"
    assert isinstance(call.result_typed, BashResult)
    assert call.result_typed.stdout == "file1\nfile2"


def test_timeline_decomposes_parts_by_kind(transcript: Path) -> None:
    session = Session.load(transcript)
    entries = {e.uuid: e for e in session.timeline()}
    assistant_kinds = [p.type for p in entries["u1"].parts]
    assert assistant_kinds == ["text", "tool_use"]
    user_result_kinds = [p.type for p in entries["u2"].parts]
    assert user_result_kinds == ["tool_result"]


def test_cost_summary_dedups_by_request_id(transcript: Path) -> None:
    session = Session.load(transcript)
    summary = session.cost_summary()
    assert summary.requests == 1
    assert summary.input_tokens == 10
    assert summary.priced is True
    assert summary.total_cost_usd is not None


def test_dedup_keeps_the_last_streamed_record_per_request(transcript: Path) -> None:
    session = Session.load(transcript)
    requests = session.assistant_requests()
    assert [r.uuid for r in requests] == ["u1b"]
    assert requests[0].message.usage is not None
    assert requests[0].message.usage.output_tokens == 12
    assert session.cost_summary().output_tokens == 12


def test_label_prefers_first_real_user_prompt(transcript: Path) -> None:
    session = Session.load(transcript)
    assert session.label() == "Please investigate the repo"


def test_info_reports_session_summary(transcript: Path) -> None:
    session = Session.load(transcript)
    info = session.info()
    assert info.title == "Please investigate the repo"
    assert info.tool_calls == 1
    assert info.records == 4


def test_timeline_populates_raw_for_a_non_conversation_record(tmp_path: Path) -> None:
    attachment_line = _line(
        type="attachment",
        uuid="u3",
        parentUuid="u2",
        timestamp="2026-01-01T00:00:03Z",
        attachment={"type": "auto_mode"},
    )
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join([USER_PROMPT, attachment_line]) + "\n", encoding="utf-8")

    session = Session.load(path)
    entries = {e.uuid: e for e in session.timeline()}
    entry = entries["u3"]
    assert entry.type == "attachment"
    assert entry.timestamp is not None
    assert isinstance(entry.raw, dict)
    assert entry.raw["attachment"] == {"type": "auto_mode"}


def test_iter_raw_never_raises_on_a_torn_encoding_or_a_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    good = json.dumps({"type": "user", "uuid": "u1"}).encode()
    path.write_bytes(
        good
        + b"\n"
        + b'{"type": "not json"'
        + b"\n"  # malformed JSON, no fallback container here
        + "café".encode("latin-1")
        + b"\n"  # not valid UTF-8
        + good
        + b"\n"
    )

    rows = list(iter_raw(path))

    assert rows == [{"type": "user", "uuid": "u1"}, {"type": "user", "uuid": "u1"}]
