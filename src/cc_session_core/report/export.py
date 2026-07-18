"""Select and export parts of a parsed session to a file.

An :class:`ExportSpec` says *what* to include — record types, content kinds, tool
names, specific message uuids, main-vs-sidechain — and :func:`render` / :func:`export`
turn the selection into ``text``, ``markdown``, structured ``json``, or raw
``jsonl``. Everything is derived from the typed :meth:`Session.timeline`, so a
filter is a predicate over typed entries and their discriminated-union parts,
never a dict poke.

Filtering is AND across the set fields (omit a field to include everything for
it). ``parts``/``tools`` filter *within* an entry; ``types``/``uuids``/``main_only``
filter *whole* entries. For ``jsonl`` the raw matching record is emitted whole —
tool blocks live inside a message, so they can't be split out as raw lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json

from .. import types as t
from ..models import AssistantRecord, AttachmentRecord, Record, SystemRecord, UserRecord
from ..session import Session
from .views import (
    TimelineEntry,
    TimelinePart,
    ToolResultPart,
    ToolUsePart,
)

PartKind = Literal["text", "thinking", "tool_use", "tool_result", "image", "other"]
ExportFormat = Literal["text", "markdown", "json", "jsonl"]

_CONV_RECORDS = (AssistantRecord, UserRecord, AttachmentRecord, SystemRecord)


class ExportSpec(BaseModel):
    """What to include when exporting a session. Omit a field to keep everything."""

    model_config = ConfigDict(extra="forbid")

    types: set[str] | None = None
    parts: set[PartKind] | None = None
    tools: set[t.ToolName] | None = None
    uuids: set[t.RecordUuid] | None = None
    main_only: bool = False


def _is_sidechain(rec: Record) -> bool:
    return bool(rec.is_sidechain) if isinstance(rec, _CONV_RECORDS) else False


def _keep_part(part: TimelinePart, spec: ExportSpec) -> bool:
    if spec.parts is not None and part.type not in spec.parts:
        return False
    if spec.tools is not None:
        if isinstance(part, ToolUsePart | ToolResultPart):
            return part.tool_name in spec.tools
        return False
    return True


def select(session: Session, spec: ExportSpec) -> list[TimelineEntry]:
    """The session's timeline, filtered to the selection: whole-entry filters
    (types/uuids/main_only) drop entries; part/tool filters trim each entry's
    parts and drop an entry left with none."""
    records = session.ordered()
    part_filter = spec.parts is not None or spec.tools is not None
    out: list[TimelineEntry] = []
    for entry in session.timeline():
        if spec.types is not None and entry.type not in spec.types:
            continue
        if spec.uuids is not None and entry.uuid not in spec.uuids:
            continue
        if spec.main_only and _is_sidechain(records[entry.index][1]):
            continue
        kept = [p for p in entry.parts if _keep_part(p, spec)]
        if part_filter and not kept:
            continue
        out.append(entry.model_copy(update={"parts": kept}))
    return out


# --------------------------------------------------------------------------- #
# renderers (full fidelity — no clipping)
# --------------------------------------------------------------------------- #
def _entry_head(entry: TimelineEntry) -> str:
    ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S") if entry.timestamp else "--"
    model = f"  {entry.model}" if entry.model else ""
    return f"===== [{entry.type}] {ts}{model}  #{entry.uuid or ''} ====="


def _render_text(entries: list[TimelineEntry]) -> str:
    blocks: list[str] = []
    for entry in entries:
        chunk = [_entry_head(entry)]
        if entry.summary is not None:
            chunk.append(f"summary: {entry.summary}")
        chunk.extend(p.as_text() for p in entry.parts)
        blocks.append("\n".join(chunk))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _render_markdown(entries: list[TimelineEntry]) -> str:
    blocks: list[str] = []
    for entry in entries:
        head = f"### {entry.type}" + (f" · {entry.model}" if entry.model else "")
        chunk = [head]
        if entry.summary is not None:
            chunk.append(f"> {entry.summary}")
        chunk.extend(p.as_markdown() for p in entry.parts)
        blocks.append("\n\n".join(chunk))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _render_json(entries: list[TimelineEntry]) -> str:
    return to_json([e.model_dump(mode="json") for e in entries], indent=2).decode()


def _render_jsonl(session: Session, entries: list[TimelineEntry]) -> str:
    records = session.ordered()
    lines = [records[e.index][1].model_dump_json(by_alias=True) for e in entries]
    return "\n".join(lines) + ("\n" if lines else "")


def render(session: Session, spec: ExportSpec, fmt: ExportFormat) -> str:
    """Render the selection to a string in the given format."""
    entries = select(session, spec)
    if fmt == "text":
        return _render_text(entries)
    if fmt == "markdown":
        return _render_markdown(entries)
    if fmt == "json":
        return _render_json(entries)
    return _render_jsonl(session, entries)


def export(session: Session, spec: ExportSpec, fmt: ExportFormat, path: str | Path) -> Path:
    """Render the selection and write it to ``path``; returns the path written."""
    out = Path(path)
    out.write_text(render(session, spec, fmt), encoding="utf-8")
    return out
