"""Assemble typed transcript records into a :class:`Session` and its views.

The models in :mod:`cc_session_core.models` type one line each; this module assembles a
whole file into a :class:`Session` and derives the views defined in
:mod:`cc_session_core.views`:

* :meth:`Session.timeline`    — ordered, decomposed, human-readable events
* :meth:`Session.tool_calls`  — every ``tool_use`` paired with its ``tool_result``
* :meth:`Session.cost_summary` — usage + cost rollup, per model and total

De-duplication mirrors the log: records are keyed by ``uuid`` and cost is keyed by
request id, so a repeated line and a re-counted API request are each counted once.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_json

from . import types as t
from .cost.pricing import EXAMPLE_PRICING, PriceTable, request_cost
from .models import (
    AiTitleRecord,
    AssistantRecord,
    AttachmentRecord,
    ImageBlock,
    LastPromptRecord,
    PrLinkRecord,
    QueueOperationRecord,
    Record,
    RedactedThinkingBlock,
    SystemRecord,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserRecord,
)
from .parsing.parse import ParseFailure, iter_records
from .parsing.tools import parse_tool_input, parse_tool_result
from .report.views import (
    CostSummary,
    ImagePart,
    ModelUsage,
    OtherPart,
    SessionInfo,
    TextPart,
    ThinkingPart,
    TimelineEntry,
    TimelinePart,
    ToolCall,
    ToolResultPart,
    ToolUsePart,
    message_text,
    sort_key,
    tool_result_text,
)

_RAW_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def _record_timestamp(rec: Record) -> t.Timestamp | None:
    if isinstance(
        rec,
        (
            AssistantRecord,
            UserRecord,
            AttachmentRecord,
            SystemRecord,
            QueueOperationRecord,
            PrLinkRecord,
        ),
    ):
        return rec.timestamp
    return None


def request_key(rec: AssistantRecord) -> str | None:
    """The dedup key for one API request: request id, then message id, then record uuid.
    Streaming writes several records per request that re-echo the same usage; every
    accounting layer collapses to one row per key."""
    return rec.request_id or rec.message.id or rec.uuid


def _record_uuid(rec: Record) -> t.RecordUuid | None:
    if isinstance(rec, (AssistantRecord, UserRecord, AttachmentRecord, SystemRecord)):
        return rec.uuid
    return None


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #
class Session:
    """A parsed session: ordered, de-duplicated records plus derived views."""

    def __init__(
        self,
        records: list[Record],
        source: Path | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.source = source
        self.errors: list[str] = errors or []
        self.records: list[Record] = self._dedupe(records)

    # ---- loading ---------------------------------------------------------- #

    @classmethod
    def load(cls, path: str | Path, strict: bool = False) -> Session:
        """Parse one .jsonl file. strict=True raises on a bad line; else it is collected."""
        path = Path(path)
        records: list[Record] = []
        errors: list[str] = []
        for item in iter_records(path):
            if isinstance(item, ParseFailure):
                msg = f"{item.file}:{item.line_number}: {item.error}"
                if strict:
                    raise ValueError(msg)
                errors.append(msg)
            else:
                records.append(item)
        return cls(records, source=path, errors=errors)

    @staticmethod
    def _dedupe(records: list[Record]) -> list[Record]:
        """Drop repeated lines by uuid, preserving first-seen order. Records without
        a uuid (the lightweight pointer/state kinds) are always kept."""
        seen: set[str] = set()
        out: list[Record] = []
        for rec in records:
            uid = _record_uuid(rec)
            if uid:
                if uid in seen:
                    continue
                seen.add(uid)
            out.append(rec)
        return out

    # ---- ordering --------------------------------------------------------- #

    @cached_property
    def _ordered(self) -> list[tuple[int, Record]]:
        indexed = list(enumerate(self.records))
        indexed.sort(key=lambda p: (sort_key(_record_timestamp(p[1])), p[0]))
        return indexed

    def ordered(self) -> list[tuple[int, Record]]:
        """Records in timeline order (timestamp, then original position)."""
        return self._ordered

    # ---- tool call index -------------------------------------------------- #

    @cached_property
    def _tool_uses(self) -> dict[str, tuple[ToolUseBlock, AssistantRecord]]:
        index: dict[str, tuple[ToolUseBlock, AssistantRecord]] = {}
        for rec in self.records:
            if isinstance(rec, AssistantRecord):
                for blk in rec.message.content:
                    if isinstance(blk, ToolUseBlock) and blk.id:
                        index[blk.id] = (blk, rec)
        return index

    @cached_property
    def _tool_reasons(self) -> dict[str, str]:
        """Map each tool_use_id to the assistant's narration that fired it — the
        "why". One assistant turn is split across records sharing a message id, so
        the narrating text and the tool_use it explains land in different records;
        group by that id and attribute the turn's text to every tool_use in it."""
        text_by_turn: dict[str, list[str]] = {}
        ids_by_turn: dict[str, list[str]] = {}
        for rec in self.records:
            if not isinstance(rec, AssistantRecord):
                continue
            key = rec.message.id or rec.request_id
            if not key:
                continue
            for blk in rec.message.content:
                if isinstance(blk, TextBlock) and blk.text:
                    text_by_turn.setdefault(key, []).append(blk.text)
                elif isinstance(blk, ToolUseBlock) and blk.id:
                    ids_by_turn.setdefault(key, []).append(blk.id)

        reasons: dict[str, str] = {}
        for key, ids in ids_by_turn.items():
            why = "\n".join(text_by_turn.get(key, [])).strip()
            for tuid in ids:
                reasons[tuid] = why
        return reasons

    @cached_property
    def _tool_calls(self) -> list[ToolCall]:
        uses = self._tool_uses
        reasons = self._tool_reasons
        results: dict[str, tuple[ToolResultBlock, UserRecord]] = {}
        for rec in self.records:
            if isinstance(rec, UserRecord) and isinstance(rec.message.content, list):
                for blk in rec.message.content:
                    if isinstance(blk, ToolResultBlock) and blk.tool_use_id:
                        results[blk.tool_use_id] = (blk, rec)

        calls: list[ToolCall] = []
        for tuid, (use, call_rec) in uses.items():
            call = ToolCall(
                tool_use_id=tuid,
                name=use.name,
                input=use.input,
                input_typed=parse_tool_input(use.name, use.input),
                reason=reasons.get(tuid) or None,
                call_uuid=call_rec.uuid,
                call_timestamp=call_rec.timestamp,
            )
            pair = results.get(tuid)
            if pair is not None:
                res_blk, res_rec = pair
                call.result_uuid = res_rec.uuid
                call.result_timestamp = res_rec.timestamp
                call.is_error = res_blk.is_error
                call.result_text = tool_result_text(res_blk.content)
                call.result_structured = res_rec.tool_use_result
                call.result_typed = parse_tool_result(use.name, res_rec.tool_use_result)
                if call.call_timestamp is not None and call.result_timestamp is not None:
                    delta = call.result_timestamp - call.call_timestamp
                    call.duration_ms = max(int(delta.total_seconds() * 1000), 0)
            calls.append(call)

        calls.sort(key=lambda c: sort_key(c.call_timestamp))
        for step, call in enumerate(calls, start=1):
            call.step = step
        return calls

    def tool_calls(self) -> list[ToolCall]:
        """Every tool_use paired with its tool_result, in call order."""
        return self._tool_calls

    # ---- timeline --------------------------------------------------------- #

    @cached_property
    def _timeline(self) -> list[TimelineEntry]:
        return self._build_timeline(EXAMPLE_PRICING)

    def timeline(self, price_table: PriceTable | None = None) -> list[TimelineEntry]:
        """Ordered, decomposed events. The default price table's result is cached."""
        if price_table is None:
            return self._timeline
        return self._build_timeline(price_table)

    def _build_timeline(self, table: PriceTable) -> list[TimelineEntry]:
        uses = self._tool_uses  # tool_use_id -> (block, record), for naming results
        entries: list[TimelineEntry] = []

        for position, (_, rec) in enumerate(self.ordered()):
            if isinstance(rec, AssistantRecord):
                entries.append(self._assistant_entry(position, rec, table))
            elif isinstance(rec, UserRecord):
                entries.append(self._user_entry(position, rec, uses))
            elif isinstance(rec, SystemRecord):
                entries.append(self._system_entry(position, rec))
            else:
                # Every other record kind (AttachmentRecord, AiTitleRecord, UnknownRecord, ...)
                # has no dedicated TimelineEntry parts, but its full typed payload — e.g. an
                # attachment's hook output or edited-file snippet — stays reachable via `raw`
                # rather than silently disappearing from the timeline.
                entries.append(
                    TimelineEntry(
                        index=position,
                        type=rec.type,
                        timestamp=_record_timestamp(rec),
                        uuid=_record_uuid(rec),
                        raw=rec.model_dump(mode="json"),
                    )
                )

        per_type: dict[str, int] = {}
        for step, entry in enumerate(entries, start=1):
            entry.step = step
            per_type[entry.type] = per_type.get(entry.type, 0) + 1
            entry.type_step = per_type[entry.type]
        return entries

    @staticmethod
    def _assistant_entry(position: int, rec: AssistantRecord, table: PriceTable) -> TimelineEntry:
        parts: list[TimelinePart] = []
        for blk in rec.message.content:
            if isinstance(blk, TextBlock):
                parts.append(TextPart(text=blk.text))
            elif isinstance(blk, ThinkingBlock):
                parts.append(ThinkingPart(text=blk.thinking))
            elif isinstance(blk, RedactedThinkingBlock):
                parts.append(ThinkingPart(text="[redacted]"))
            elif isinstance(blk, ToolUseBlock):
                parts.append(
                    ToolUsePart(tool_name=blk.name, tool_use_id=blk.id, tool_input=blk.input)
                )
            elif isinstance(blk, ImageBlock):
                parts.append(ImagePart())
            else:
                parts.append(OtherPart(text=to_json(blk.model_dump(mode="json")).decode()))

        cost = request_cost(rec.message.model, rec.message.usage, table)
        source: t.CostSource | None = "computed" if cost is not None else None
        return TimelineEntry(
            index=position,
            type="assistant",
            role="assistant",
            timestamp=rec.timestamp,
            uuid=rec.uuid,
            parent_uuid=rec.parent_uuid,
            model=rec.message.model,
            request_id=rec.request_id,
            usage=rec.message.usage,
            cost_usd=cost,
            cost_source=source,
            parts=parts,
        )

    @staticmethod
    def _user_entry(
        position: int,
        rec: UserRecord,
        uses: dict[str, tuple[ToolUseBlock, AssistantRecord]],
    ) -> TimelineEntry:
        parts: list[TimelinePart] = []
        content = rec.message.content
        if isinstance(content, str):
            parts.append(TextPart(text=content))
        else:
            for blk in content:
                if isinstance(blk, TextBlock):
                    parts.append(TextPart(text=blk.text))
                elif isinstance(blk, ToolResultBlock):
                    use = uses.get(blk.tool_use_id)
                    parts.append(
                        ToolResultPart(
                            tool_use_id=blk.tool_use_id,
                            tool_name=use[0].name if use else None,
                            is_error=blk.is_error,
                            text=tool_result_text(blk.content),
                            result_structured=rec.tool_use_result,
                        )
                    )
                elif isinstance(blk, ImageBlock):
                    parts.append(ImagePart())
                else:
                    parts.append(OtherPart(text=to_json(blk.model_dump(mode="json")).decode()))

        return TimelineEntry(
            index=position,
            type="user",
            role="user",
            timestamp=rec.timestamp,
            uuid=rec.uuid,
            parent_uuid=rec.parent_uuid,
            parts=parts,
        )

    @staticmethod
    def _system_entry(position: int, rec: SystemRecord) -> TimelineEntry:
        return TimelineEntry(
            index=position,
            type="system",
            timestamp=rec.timestamp,
            uuid=rec.uuid,
            parent_uuid=rec.parent_uuid,
            parts=[OtherPart(text=rec.content or "")],
        )

    # ---- cost ------------------------------------------------------------- #

    @cached_property
    def _assistant_requests(self) -> list[AssistantRecord]:
        slot: dict[str, int] = {}
        out: list[AssistantRecord] = []
        for rec in self.records:
            if not isinstance(rec, AssistantRecord):
                continue
            key = request_key(rec)
            if key is None:
                out.append(rec)
            elif key in slot:
                out[slot[key]] = rec
            else:
                slot[key] = len(out)
                out.append(rec)
        return out

    def assistant_requests(self) -> list[AssistantRecord]:
        """One assistant record per API request (last-seen), keyed by request id.
        Streaming writes several records per request whose usage grows as output
        streams; the last record carries the full counts, so keeping it counts
        tokens and cost once and completely."""
        return self._assistant_requests

    @cached_property
    def _cost_summary(self) -> CostSummary:
        return self._build_cost_summary(EXAMPLE_PRICING)

    def cost_summary(self, price_table: PriceTable | None = None) -> CostSummary:
        """Roll up usage and cost. Each API request (by request id) is counted once.
        The default price table's result is cached; a custom table computes fresh."""
        if price_table is None:
            return self._cost_summary
        return self._build_cost_summary(price_table)

    def _build_cost_summary(self, table: PriceTable) -> CostSummary:
        by_model: dict[str, ModelUsage] = {}
        any_priced = False

        for rec in self.assistant_requests():
            usage = rec.message.usage
            model = rec.message.model or "unknown"
            mu = by_model.setdefault(model, ModelUsage(model=model))
            mu.requests += 1
            mu.input_tokens += usage.input_tokens
            mu.output_tokens += usage.output_tokens
            mu.cache_creation_input_tokens += usage.cache_creation_input_tokens
            mu.cache_read_input_tokens += usage.cache_read_input_tokens

            cost = request_cost(model, usage, table)
            if cost is not None:
                any_priced = True
                mu.cost_usd = (mu.cost_usd or 0.0) + cost

        summary = CostSummary(
            priced=any_priced,
            by_model=sorted(by_model.values(), key=lambda m: m.model),
        )
        for mu in summary.by_model:
            summary.requests += mu.requests
            summary.input_tokens += mu.input_tokens
            summary.output_tokens += mu.output_tokens
            summary.cache_creation_input_tokens += mu.cache_creation_input_tokens
            summary.cache_read_input_tokens += mu.cache_read_input_tokens
            if mu.cost_usd is not None:
                summary.total_cost_usd = (summary.total_cost_usd or 0.0) + mu.cost_usd
        return summary

    # ---- identity --------------------------------------------------------- #

    def label(self) -> str:
        """Best human-readable name: the AI-generated title, else the first real
        user prompt, else the last recorded prompt, else untitled."""
        ai = last = None
        for rec in self.records:
            if isinstance(rec, AiTitleRecord) and rec.ai_title:
                ai = rec.ai_title
            elif isinstance(rec, LastPromptRecord) and rec.last_prompt:
                last = rec.last_prompt
        if ai:
            return ai
        for _, rec in self.ordered():
            if isinstance(rec, UserRecord):
                text = message_text(rec.message).strip()
                if text and not text.startswith("<"):
                    return text
        return last or "(untitled)"

    def info(self) -> SessionInfo:
        """One-line summary for listing/picking this investigation."""
        stamps = [ts for ts in (_record_timestamp(r) for r in self.records) if ts is not None]
        return SessionInfo(
            id=self.source.stem if self.source else "",
            source=str(self.source) if self.source else None,
            title=self.label(),
            started=min(stamps, key=sort_key) if stamps else None,
            ended=max(stamps, key=sort_key) if stamps else None,
            records=len(self.records),
            tool_calls=len(self.tool_calls()),
            total_cost_usd=self.cost_summary().total_cost_usd,
        )


# --------------------------------------------------------------------------- #
# convenience loaders
# --------------------------------------------------------------------------- #
DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"
"""Where Claude Code writes session transcripts, one directory per project."""


def session_files(target: str | Path) -> list[Path]:
    """Resolve a target to session files: a directory yields its ``.jsonl`` files
    (recursively, path-sorted); a single path yields just itself."""
    path = Path(target)
    return sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]


def resolve_session_file(session: str, projects_root: Path = DEFAULT_PROJECTS_ROOT) -> Path | None:
    """A ``.jsonl`` path as-is, or the first session file whose name matches an
    id/prefix under ``projects_root``; ``None`` when nothing matches."""
    path = Path(session).expanduser()
    if path.is_file():
        return path
    matches = sorted(projects_root.rglob(f"{session}*.jsonl"))
    return matches[0] if matches else None


def load_project(directory: str | Path, strict: bool = False) -> list[Session]:
    """Parse every .jsonl session file under a project directory (recursively)."""
    return [Session.load(p, strict=strict) for p in session_files(directory)]


def iter_raw(path: str | Path) -> Iterator[dict[str, JsonValue]]:
    """Yield raw line dicts without building models. Skips blank lines and, like
    :func:`~cc_session_core.parsing.parse.iter_records`, never raises: a byte sequence
    the file's declared encoding can't decode is replaced rather than aborting the read,
    and a line that isn't a JSON object is skipped rather than propagating a
    ``ValidationError`` (there is no typed fallback container here, unlike
    ``iter_records``'s ``ParseFailure``/``Unknown*`` variants — a malformed raw line is
    simply dropped from this stream)."""
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield _RAW_ADAPTER.validate_json(stripped)
            except ValidationError:
                continue
