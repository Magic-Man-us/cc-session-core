"""Every record kind discriminates to the right model and round-trips."""

from __future__ import annotations

import json

import pytest
from conftest import record_lines, record_type
from pydantic import BaseModel

from cc_session_core import RECORD_ADAPTER, parse_line
from cc_session_core.models import (
    AssistantRecord,
    SystemRecord,
    UnknownBlock,
    UnknownRecord,
    UserRecord,
)

LINES = record_lines()


@pytest.mark.parametrize("line", LINES, ids=record_type)
def test_record_discriminates_to_its_type(line: str) -> None:
    rec = parse_line(line)
    assert isinstance(rec, BaseModel)
    assert rec.type == record_type(line)


@pytest.mark.parametrize("line", LINES, ids=record_type)
def test_record_round_trips(line: str) -> None:
    rec = parse_line(line)
    again = RECORD_ADAPTER.validate_json(rec.model_dump_json(by_alias=True))
    assert type(again) is type(rec)
    assert again.type == rec.type


def test_all_seventeen_kinds_present() -> None:
    kinds = {record_type(line) for line in LINES}
    assert kinds == {
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


def test_new_envelope_fields_are_typed() -> None:
    """The 2026-07-era envelope fields parse into typed attributes, not model_extra."""
    obj = json.loads(next(line for line in LINES if record_type(line) == "assistant"))
    obj["effort"] = "xhigh"
    a = parse_line(json.dumps(obj))
    assert isinstance(a, AssistantRecord)
    assert a.effort == "xhigh"
    assert not a.model_extra
    obj = json.loads(next(line for line in LINES if record_type(line) == "user"))
    obj["classifierMetaLines"] = "meta"
    u = parse_line(json.dumps(obj))
    assert isinstance(u, UserRecord)
    assert u.classifier_meta_lines == "meta"
    assert not u.model_extra
    obj = json.loads(next(line for line in LINES if record_type(line) == "system"))
    obj["hookAdditionalContext"] = []
    s = parse_line(json.dumps(obj))
    assert isinstance(s, SystemRecord)
    assert s.hook_additional_context == []
    assert not s.model_extra


def test_unknown_record_type_falls_back_losslessly() -> None:
    """A new top-level kind lands in UnknownRecord with its payload kept, not raised."""
    rec = parse_line(json.dumps({"type": "totally-new-kind", "sessionId": "s", "wild": 1}))
    assert isinstance(rec, UnknownRecord)
    assert rec.type == "totally-new-kind"
    assert rec.model_extra == {"sessionId": "s", "wild": 1}


def test_extra_field_is_kept() -> None:
    """An unmodeled field on a known kind is preserved in model_extra, not dropped."""
    obj = json.loads(LINES[[record_type(line) for line in LINES].index("mode")])
    obj["bogusFieldNeverSeen"] = 1
    rec = parse_line(json.dumps(obj))
    assert rec.type == "mode"
    assert rec.model_extra is not None and rec.model_extra["bogusFieldNeverSeen"] == 1


def test_unknown_block_type_falls_back_losslessly() -> None:
    """A new content-block kind lands in UnknownBlock with its payload kept, not raised."""
    obj = json.loads(next(line for line in LINES if record_type(line) == "user"))
    obj["message"]["content"] = [{"type": "brand_new_block_kind", "payload": [1, 2]}]
    rec = parse_line(json.dumps(obj))
    assert isinstance(rec, UserRecord)
    block = rec.message.content[0]
    assert isinstance(block, UnknownBlock)
    assert block.type == "brand_new_block_kind"
    assert block.model_extra == {"payload": [1, 2]}


def test_pre_cache_era_usage_parses_losslessly() -> None:
    """An assistant usage block with only input/output tokens (no cache_creation object
    at all) predates cache accounting; it must parse, not fall into ParseFailure."""
    obj = json.loads(next(line for line in LINES if record_type(line) == "assistant"))
    obj["message"]["usage"] = {"input_tokens": 42, "output_tokens": 7}
    rec = parse_line(json.dumps(obj))
    assert isinstance(rec, AssistantRecord)
    usage = rec.message.usage
    assert usage.input_tokens == 42
    assert usage.output_tokens == 7
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_creation.ephemeral_5m_input_tokens == 0
    assert usage.cache_creation.ephemeral_1h_input_tokens == 0
