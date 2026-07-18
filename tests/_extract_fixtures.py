#!/usr/bin/env python3
"""Freeze representative, scrubbed fixtures from a real transcript corpus.

One example per record kind, per attachment kind, and per modeled tool
input/result. Free-text, paths, and base64 are scrubbed so fixtures are safe to
commit — tests assert structure/types/discrimination, never content.

Run once to (re)generate tests/fixtures/*:
    CC_SESSION_CORPUS=~/.claude/projects python tests/_extract_fixtures.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError  # noqa: E402
from pydantic_core import to_json  # noqa: E402

from cc_session_core import (  # noqa: E402
    ParseFailure,
    iter_records,
    parse_tool_input,
    parse_tool_result,
)
from cc_session_core.models import AssistantRecord, ToolUseBlock  # noqa: E402

FIX = Path(__file__).parent / "fixtures"

_LINE: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])

# Keys whose value is a dict with arbitrary, content-bearing keys of its own (e.g. an
# AskUserQuestion answers map keyed by the real question text) -- scrub() only ever
# redacts dict *values*, so a dict shaped this way needs collapsing outright rather
# than recursing key-by-key, or its keys leak untouched.
OPAQUE_MAPS = {"answers"}
# Keys whose *value* is always safe to keep verbatim: fixed-vocabulary discriminators
# and enums (Claude Code's own record/block/attachment types, built-in tool names,
# statuses), opaque identifiers (uuid/id-shaped -- pair calls to results, reveal no
# content), and timestamps/versions. A blocklist of "sensitive" field names reliably
# misses fields (unlisted keys, list items) that turn out to carry real corpus content
# -- an allowlist can't silently miss a new one, it only ever under-includes.
ALLOWED_KEYS = {
    "type",
    "role",
    "mode",
    "permissionMode",
    "userType",
    "entrypoint",
    "operation",
    "subtype",
    "promptSource",
    "commandMode",
    "reminderType",
    "hookEvent",
    "hookName",
    "retrieval_status",
    "task_type",
    "taskType",
    "status",
    "issue_class",
    "confidence",
    "severity",
    "stop_reason",
    "service_tier",
    "speed",
    "inference_geo",
    "model",
    "resolvedModel",
    "action",
    "output_mode",
    "tool",
    # opaque identifiers
    "uuid",
    "parentUuid",
    "sessionId",
    "session_id",
    "messageId",
    "snapshotMessageId",
    "leafUuid",
    "requestId",
    "promptId",
    "toolUseID",
    "tool_use_id",
    "id",
    "taskId",
    "task_id",
    "projectId",
    "runId",
    # timestamps / versions
    "timestamp",
    "newDate",
    "backupTime",
    "updatedAt",
    "version",
}
PLACEHOLDER = "redacted"


def scrub(value: JsonValue, key: str | None = None) -> JsonValue:
    """Replace every string not on the explicit safe-key allowlist with a placeholder,
    so fixtures carry no content -- redact-by-default, including strings nested in lists
    (the parent key is threaded through) and dicts keyed by free text of their own."""
    if isinstance(value, dict):
        if key in OPAQUE_MAPS:
            return {PLACEHOLDER: PLACEHOLDER} if value else value
        return {k: scrub(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, key) for v in value]
    if isinstance(value, str) and key not in ALLOWED_KEYS:
        return PLACEHOLDER
    return value


def _write_jsonl(path: Path, values: list[JsonValue]) -> None:
    path.write_bytes(b"\n".join(to_json(v) for v in values) + b"\n")


def _write_json(path: Path, value: JsonValue) -> None:
    path.write_bytes(to_json(value, indent=2))


def main() -> None:
    corpus = os.environ.get("CC_SESSION_CORPUS")
    if not corpus:
        print("set CC_SESSION_CORPUS to the projects dir", file=sys.stderr)
        raise SystemExit(2)
    files = sorted(glob.glob(f"{corpus}/**/*.jsonl", recursive=True))

    records: dict[str, JsonValue] = {}
    attachments: dict[str, JsonValue] = {}
    tool_inputs: dict[str, JsonValue] = {}
    tool_results: dict[str, JsonValue] = {}

    for f in files:
        recs = [r for r in iter_records(Path(f)) if not isinstance(r, ParseFailure)]
        index: dict[str, str] = {}
        for r in recs:
            if isinstance(r, AssistantRecord):
                for block in r.message.content:
                    if isinstance(block, ToolUseBlock):
                        index[block.id] = block.name

        for raw_line in Path(f).read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = _LINE.validate_json(line)
            except ValidationError:
                continue
            record_type = record.get("type")
            if not isinstance(record_type, str):
                continue
            if record_type not in records:
                records[record_type] = scrub(record)

            if record_type == "attachment":
                attachment = record.get("attachment")
                att_type = attachment.get("type") if isinstance(attachment, dict) else None
                if isinstance(att_type, str) and att_type not in attachments:
                    attachments[att_type] = scrub(record)

            elif record_type == "assistant":
                message = record.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_use":
                            continue
                        name, tool_input = block.get("name"), block.get("input")
                        if (
                            isinstance(name, str)
                            and name not in tool_inputs
                            and isinstance(tool_input, dict)
                            and isinstance(parse_tool_input(name, tool_input), BaseModel)
                        ):
                            tool_inputs[name] = scrub(tool_input)

            elif record_type == "user":
                tool_use_result = record.get("toolUseResult")
                if not isinstance(tool_use_result, dict):
                    continue
                message = record.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                result_tool: str | None = None
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_use_id = block.get("tool_use_id")
                            if isinstance(tool_use_id, str):
                                result_tool = index.get(tool_use_id)
                if (
                    isinstance(result_tool, str)
                    and result_tool not in tool_results
                    and isinstance(parse_tool_result(result_tool, tool_use_result), BaseModel)
                ):
                    tool_results[result_tool] = scrub(tool_use_result)

    FIX.mkdir(exist_ok=True)
    _write_jsonl(FIX / "records.jsonl", list(records.values()))
    _write_jsonl(FIX / "attachments.jsonl", list(attachments.values()))
    _write_json(FIX / "tool_inputs.json", tool_inputs)
    _write_json(FIX / "tool_results.json", tool_results)
    print(
        f"records={len(records)} attachments={len(attachments)} "
        f"tool_inputs={len(tool_inputs)} tool_results={len(tool_results)}"
    )


if __name__ == "__main__":
    main()
