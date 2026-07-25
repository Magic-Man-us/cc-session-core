"""Codex rollout parsing, shared discrimination, and canonical normalization."""

from __future__ import annotations

import json
from pathlib import Path

from cc_session_core import (
    CodexSession,
    Session,
    UnknownTranscriptRecord,
    iter_transcript_records,
    parse_transcript_line,
)
from cc_session_core.codex.models import (
    EventMessageRecord,
    FunctionCallResponseItem,
    ResponseItemRecord,
    SessionMetaRecord,
    UnknownCodexRecord,
)
from cc_session_core.models import AssistantRecord
from cc_session_core.parsing.parse import ParseFailure

FIXTURES = Path(__file__).parent / "fixtures"
ROLLOUT = FIXTURES / "codex_rollout.jsonl"


def test_shared_union_routes_claude_and_codex_without_a_format_flag() -> None:
    claude_line = next(
        line
        for line in (FIXTURES / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "assistant"
    )
    codex_line = ROLLOUT.read_text(encoding="utf-8").splitlines()[0]

    assert isinstance(parse_transcript_line(claude_line), AssistantRecord)
    assert isinstance(parse_transcript_line(codex_line), SessionMetaRecord)


def test_shared_union_uses_neutral_lossless_fallback_for_new_types() -> None:
    line = json.dumps(
        {
            "timestamp": "2026-07-25T14:00:00Z",
            "type": "future_provider_record",
            "payload": {"kept": True},
        }
    )

    record = parse_transcript_line(line)

    assert isinstance(record, UnknownTranscriptRecord)
    assert record.model_extra == {
        "timestamp": "2026-07-25T14:00:00Z",
        "payload": {"kept": True},
    }


def test_codex_rollout_nested_unions_discriminate() -> None:
    records = [
        record
        for record in iter_transcript_records(ROLLOUT)
        if not isinstance(record, ParseFailure)
    ]

    assert len(records) == 11
    call_record = records[6]
    usage_record = records[8]
    assert isinstance(call_record, ResponseItemRecord)
    assert isinstance(call_record.payload, FunctionCallResponseItem)
    assert call_record.payload.call_id == "call-1"
    assert isinstance(usage_record, EventMessageRecord)
    assert usage_record.payload.type == "token_count"


def test_codex_specific_union_retains_an_unknown_record() -> None:
    record = UnknownCodexRecord.model_validate(
        {
            "timestamp": "2026-07-25T14:00:00Z",
            "type": "future_codex_record",
            "payload": {"kept": True},
        }
    )

    assert record.payload == {"kept": True}


def test_session_load_auto_detects_and_normalizes_codex() -> None:
    session = Session.load(ROLLOUT)

    assert isinstance(session, CodexSession)
    assert session.session_id == "019c0000-0000-7000-8000-000000000001"
    assert session.label() == "Implement the Codex parser."
    assert session.errors == []

    calls = session.tool_calls()
    assert len(calls) == 1
    assert calls[0].tool_use_id == "call-1"
    assert calls[0].name == "exec_command"
    assert calls[0].input == {"cmd": "rg --files"}
    assert calls[0].result_text == "README.md\npyproject.toml"
    assert calls[0].reason == "I will inspect the repository."
    assert calls[0].duration_ms == 1000


def test_codex_usage_maps_cached_input_without_double_counting() -> None:
    session = Session.load(ROLLOUT)
    summary = session.cost_summary()

    assert summary.requests == 1
    assert summary.input_tokens == 40
    assert summary.cache_read_input_tokens == 60
    assert summary.output_tokens == 20
    assert summary.total_tokens == 120
    assert summary.by_model[0].model == "gpt-5.6-sol"
    assert summary.priced is False


def test_codex_info_uses_thread_id_and_raw_rollout_count() -> None:
    session = Session.load(ROLLOUT)
    info = session.info()

    assert info.id == "019c0000-0000-7000-8000-000000000001"
    assert info.records == 11
    assert info.tool_calls == 1


def test_legacy_event_only_codex_session_still_builds_a_timeline(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    lines = [
        {
            "timestamp": "2026-07-25T14:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": "legacy-session",
                "timestamp": "2026-07-25T14:00:00Z",
                "cwd": "/repo",
                "originator": "codex-cli",
                "cli_version": "0.1.0",
                "source": "cli",
            },
        },
        {
            "timestamp": "2026-07-25T14:00:01Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "legacy prompt"},
        },
        {
            "timestamp": "2026-07-25T14:00:02Z",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "legacy answer"},
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    session = Session.load(path)

    assert isinstance(session, CodexSession)
    assert session.label() == "legacy prompt"
    assert [entry.role for entry in session.timeline()] == ["user", "assistant"]


def test_detection_skips_an_unknown_leading_record(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    unknown = {
        "timestamp": "2026-07-25T13:59:59Z",
        "type": "future_rollout_header",
        "payload": {"kept": True},
    }
    known = json.loads(ROLLOUT.read_text(encoding="utf-8").splitlines()[0])
    path.write_text(
        f"{json.dumps(unknown)}\n{json.dumps(known)}\n",
        encoding="utf-8",
    )

    session = Session.load(path)

    assert isinstance(session, CodexSession)
    assert isinstance(session.codex_records[0], UnknownTranscriptRecord)
