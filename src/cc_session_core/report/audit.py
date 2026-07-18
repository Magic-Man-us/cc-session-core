"""Coverage gate: where the typed models still fall short of real transcript data.

The models are lossless (``extra="allow"``; unmodeled record/block/attachment
kinds fall back to ``Unknown*``), so nothing raises on new data. This module is
what used to be ``extra="forbid"``: it reports, across the files you point it
at, every record/block/attachment kind that fell back to an ``Unknown*``
carrier, every field that landed in ``model_extra`` instead of a named field,
and the observed shape of each tool's ``toolUseResult``.

It reports field NAMES and value-TYPES only — never field values — so the
report is safe to share without leaking transcript content.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, JsonValue

from ..models import (
    AssistantMessage,
    AssistantRecord,
    Attachment,
    AttachmentRecord,
    ContentBlock,
    Record,
    UnknownAttachment,
    UnknownBlock,
    UnknownRecord,
    UserMessage,
    UserRecord,
)
from ..parsing.parse import ParseFailure, iter_records
from ..parsing.tools import result_tool_name, tool_name_index
from ..types import SNAKE_CONFIG


def _value_type(v: Any) -> str:
    """Reduce a raw value to a type label — never the value itself."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _merge_extra(acc: dict[str, set[str]], obj: BaseModel) -> None:
    """Fold a model's ``model_extra`` (fields it didn't recognize) into {field: {types}}."""
    for name, val in (obj.model_extra or {}).items():
        acc.setdefault(name, set()).add(_value_type(val))


def _merge_extra_grouped(acc: dict[str, dict[str, set[str]]], group: str, obj: BaseModel) -> None:
    """Like ``_merge_extra``, but only creates the group's entry when there's an extra
    field to report — an empty ``model_extra`` must not leave a stray empty group."""
    if obj.model_extra:
        _merge_extra(acc.setdefault(group, {}), obj)


def _merge_shape(acc: dict[str, set[str]], observed: JsonValue) -> None:
    """Fold one observed toolUseResult payload into a merged {field: {types}} shape."""
    if isinstance(observed, dict):
        for key, val in observed.items():
            acc.setdefault(key, set()).add(_value_type(val))
    else:
        acc.setdefault("<value>", set()).add(_value_type(observed))


def _freeze_nested(d: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    return {k: {f: sorted(types) for f, types in sub.items()} for k, sub in sorted(d.items())}


class SchemaAudit(BaseModel):
    model_config = SNAKE_CONFIG

    files: int = 0
    lines: int = 0
    record_types: dict[str, int] = Field(default_factory=dict)
    unmodeled_record_types: list[str] = Field(default_factory=list)
    record_extra_fields: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    message_extra_fields: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    usage_extra_fields: dict[str, list[str]] = Field(default_factory=dict)
    block_types: dict[str, int] = Field(default_factory=dict)
    unmodeled_block_types: list[str] = Field(default_factory=list)
    block_extra_fields: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    attachment_types: dict[str, int] = Field(default_factory=dict)
    unmodeled_attachment_types: list[str] = Field(default_factory=list)
    attachment_extra_fields: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    tool_result_shapes: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    models_seen: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class _Accumulator(BaseModel):
    """Mutable working state for one audit run; ``freeze()`` yields the report."""

    model_config = SNAKE_CONFIG

    files: int = 0
    lines: int = 0
    errors: list[str] = Field(default_factory=list)
    record_types: dict[str, int] = Field(default_factory=dict)
    unmodeled_records: set[str] = Field(default_factory=set)
    rec_extra: dict[str, dict[str, set[str]]] = Field(default_factory=dict)
    msg_extra: dict[str, dict[str, set[str]]] = Field(default_factory=dict)
    usage_extra: dict[str, set[str]] = Field(default_factory=dict)
    block_types: dict[str, int] = Field(default_factory=dict)
    unmodeled_blocks: set[str] = Field(default_factory=set)
    blk_extra: dict[str, dict[str, set[str]]] = Field(default_factory=dict)
    attachment_types: dict[str, int] = Field(default_factory=dict)
    unmodeled_attachments: set[str] = Field(default_factory=set)
    att_extra: dict[str, dict[str, set[str]]] = Field(default_factory=dict)
    tool_shapes: dict[str, dict[str, set[str]]] = Field(default_factory=dict)
    models_seen: set[str] = Field(default_factory=set)

    def note_record(self, rec: Record) -> None:
        rt = rec.type
        self.record_types[rt] = self.record_types.get(rt, 0) + 1
        if isinstance(rec, UnknownRecord):
            self.unmodeled_records.add(rt)
        _merge_extra_grouped(self.rec_extra, rt, rec)

        if isinstance(rec, (AssistantRecord, UserRecord)):
            self._note_message(rec.message)
        elif isinstance(rec, AttachmentRecord):
            self.note_attachment(rec.attachment)

    def _note_message(self, msg: AssistantMessage | UserMessage) -> None:
        _merge_extra_grouped(self.msg_extra, msg.role, msg)
        if isinstance(msg.content, list):
            for blk in msg.content:
                self.note_block(blk)
        if isinstance(msg, AssistantMessage):
            _merge_extra(self.usage_extra, msg.usage)
            self.models_seen.add(msg.model)

    def note_block(self, blk: ContentBlock) -> None:
        bt = blk.type
        self.block_types[bt] = self.block_types.get(bt, 0) + 1
        if isinstance(blk, UnknownBlock):
            self.unmodeled_blocks.add(bt)
        _merge_extra_grouped(self.blk_extra, bt, blk)

    def note_attachment(self, att: Attachment) -> None:
        at = att.type
        self.attachment_types[at] = self.attachment_types.get(at, 0) + 1
        if isinstance(att, UnknownAttachment):
            self.unmodeled_attachments.add(at)
        _merge_extra_grouped(self.att_extra, at, att)

    def note_tool_result(self, name: str, value: JsonValue) -> None:
        _merge_shape(self.tool_shapes.setdefault(name, {}), value)

    def freeze(self) -> SchemaAudit:
        return SchemaAudit(
            files=self.files,
            lines=self.lines,
            record_types=dict(sorted(self.record_types.items())),
            unmodeled_record_types=sorted(self.unmodeled_records),
            record_extra_fields=_freeze_nested(self.rec_extra),
            message_extra_fields=_freeze_nested(self.msg_extra),
            usage_extra_fields={f: sorted(types) for f, types in sorted(self.usage_extra.items())},
            block_types=dict(sorted(self.block_types.items())),
            unmodeled_block_types=sorted(self.unmodeled_blocks),
            block_extra_fields=_freeze_nested(self.blk_extra),
            attachment_types=dict(sorted(self.attachment_types.items())),
            unmodeled_attachment_types=sorted(self.unmodeled_attachments),
            attachment_extra_fields=_freeze_nested(self.att_extra),
            tool_result_shapes=_freeze_nested(self.tool_shapes),
            models_seen=sorted(self.models_seen),
            errors=self.errors,
        )


def _ingest(acc: _Accumulator, items: Iterable[Record | ParseFailure]) -> None:
    valid: list[Record] = []
    for item in items:
        acc.lines += 1
        if isinstance(item, ParseFailure):
            acc.errors.append(f"{item.file}:{item.line_number}: {item.error}")
            continue
        acc.note_record(item)
        valid.append(item)

    index = tool_name_index(valid)
    for rec in valid:
        if isinstance(rec, UserRecord) and rec.tool_use_result is not None:
            name = result_tool_name(rec, index)
            if name is not None:
                acc.note_tool_result(name, rec.tool_use_result)


def audit_files(paths: Iterable[Path]) -> SchemaAudit:
    """Audit every transcript file, tool-use ids resolved within each file."""
    acc = _Accumulator()
    for path in paths:
        acc.files += 1
        _ingest(acc, iter_records(path))
    return acc.freeze()


def _dump_extras(out: list[str], title: str, groups: dict[str, dict[str, list[str]]]) -> None:
    if not groups:
        return
    out.append(title + "   (* = present in data, not typed yet)")
    for group, fields in groups.items():
        out.append(f"  {group}:")
        for name, types in fields.items():
            out.append(f"      * {name}: {'|'.join(types)}")
    out.append("")


def format_audit(a: SchemaAudit) -> str:
    """Readable text report. Field names and value-types only; no values emitted."""
    out: list[str] = []
    out.append("SCHEMA AUDIT  (field names and value-types only; no values emitted)")
    out.append(f"files: {a.files}   records+errors: {a.lines}   parse failures: {len(a.errors)}")
    out.append("")

    out.append("record types:")
    for rt, n in a.record_types.items():
        tag = "  <- UNMODELED, add a record model" if rt in a.unmodeled_record_types else ""
        out.append(f"  {rt:<28} {n:>6}{tag}")
    out.append("")

    _dump_extras(out, "untyped fields by record type", a.record_extra_fields)
    _dump_extras(out, "untyped fields by message role", a.message_extra_fields)
    if a.usage_extra_fields:
        out.append("untyped usage fields:   (* = present in data, not typed yet)")
        for name, types in a.usage_extra_fields.items():
            out.append(f"      * {name}: {'|'.join(types)}")
        out.append("")

    out.append("content block types:")
    for bt, n in a.block_types.items():
        tag = "  <- UNMODELED, add a block model" if bt in a.unmodeled_block_types else ""
        out.append(f"  {bt:<28} {n:>6}{tag}")
    out.append("")
    _dump_extras(out, "untyped fields by block type", a.block_extra_fields)

    out.append("attachment types:")
    for at, n in a.attachment_types.items():
        tag = (
            "  <- UNMODELED, add an attachment model" if at in a.unmodeled_attachment_types else ""
        )
        out.append(f"  {at:<28} {n:>6}{tag}")
    out.append("")
    _dump_extras(out, "untyped fields by attachment type", a.attachment_extra_fields)

    out.append("toolUseResult shape by tool:")
    if not a.tool_result_shapes:
        out.append("  (none observed)")
    for tool, shape in a.tool_result_shapes.items():
        fields = ", ".join(f"{f}: {'|'.join(types)}" for f, types in shape.items())
        out.append(f"  {tool}: {{{fields}}}")
    out.append("")

    out.append("model strings seen:")
    for m in a.models_seen:
        out.append(f"  {m}")

    if a.errors:
        out.append("")
        out.append("parse failures:")
        for err in a.errors:
            out.append(f"  {err}")
    return "\n".join(out)
