"""Shared fixtures: load the frozen sample lines/payloads for parametrized tests."""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue, TypeAdapter

FIXTURES = Path(__file__).parent / "fixtures"

_TOOL_MAP: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_RECORD: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def record_lines() -> list[str]:
    return (FIXTURES / "records.jsonl").read_text(encoding="utf-8").splitlines()


def attachment_lines() -> list[str]:
    return (FIXTURES / "attachments.jsonl").read_text(encoding="utf-8").splitlines()


def tool_inputs() -> dict[str, JsonValue]:
    return _TOOL_MAP.validate_json((FIXTURES / "tool_inputs.json").read_bytes())


def tool_results() -> dict[str, JsonValue]:
    return _TOOL_MAP.validate_json((FIXTURES / "tool_results.json").read_bytes())


def record_type(line: str) -> str:
    value = _RECORD.validate_json(line)["type"]
    assert isinstance(value, str)
    return value


def attachment_type(line: str) -> str:
    attachment = _RECORD.validate_json(line)["attachment"]
    assert isinstance(attachment, dict)
    value = attachment["type"]
    assert isinstance(value, str)
    return value
