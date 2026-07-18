"""The full investigation record for one session: every tool call with its
stated reason, arguments, result, error, and timing, alongside the narrative
timeline and cost rollup — everything :class:`~cc_session_core.session.Session`
already derives, composed into one document instead of read piecemeal.

A tool call's "why" is mechanical, not narrated: the assistant's own text/thinking
from the turn that fired it (:attr:`ToolCall.reason`, built by
:class:`Session._tool_reasons`) — never an inferred or LLM-summarized explanation.
"""

from __future__ import annotations

from pydantic_core import to_json

from .. import types as t
from ..cost.pricing import PriceTable
from ..models import SnakeModel
from ..session import Session
from .views import CostSummary, SessionInfo, TimelineEntry, ToolCall, clip


class ToolCallStat(SnakeModel):
    """Per-tool-name rollup: how often it was called and how often it errored."""

    name: t.ToolName
    calls: t.Count = 0
    errors: t.Count = 0


class InvestigationReport(SnakeModel):
    """Every tool called, why, with what arguments, what came back, and every
    parse/tool error — plus the full narrative timeline and cost rollup, for
    reconstructing exactly what an agent did in one session and why."""

    info: SessionInfo
    cost_summary: CostSummary
    tool_stats: list[ToolCallStat] = []
    tool_calls: list[ToolCall] = []
    errors: list[ToolCall] = []
    parse_errors: list[str] = []
    timeline: list[TimelineEntry] = []


def build_investigation(
    session: Session, price_table: PriceTable | None = None
) -> InvestigationReport:
    """Assemble the investigation record from a loaded session."""
    calls = session.tool_calls()
    cost_summary = session.cost_summary(price_table)
    info = session.info().model_copy(update={"total_cost_usd": cost_summary.total_cost_usd})

    stats: dict[str, ToolCallStat] = {}
    for call in calls:
        stat = stats.setdefault(call.name, ToolCallStat(name=call.name))
        stat.calls += 1
        if call.is_error:
            stat.errors += 1

    return InvestigationReport(
        info=info,
        cost_summary=cost_summary,
        tool_stats=sorted(stats.values(), key=lambda s: s.name),
        tool_calls=calls,
        errors=[c for c in calls if c.is_error],
        parse_errors=session.errors,
        timeline=session.timeline(price_table),
    )


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def _fmt_duration(started: t.Timestamp | None, ended: t.Timestamp | None) -> str:
    if started is None or ended is None:
        return "unknown"
    seconds = (ended - started).total_seconds()
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


def _render_header(report: InvestigationReport) -> str:
    info = report.info
    lines = [
        f"# Investigation record — {info.title}",
        "",
        f"- **session id:** `{info.id}`",
        f"- **source:** `{info.source or 'unknown'}`",
        f"- **started:** {info.started.isoformat() if info.started else 'unknown'}",
        f"- **ended:** {info.ended.isoformat() if info.ended else 'unknown'}",
        f"- **duration:** {_fmt_duration(info.started, info.ended)}",
        f"- **records:** {info.records}",
        f"- **tool calls:** {info.tool_calls}",
    ]
    if info.total_cost_usd is not None:
        lines.append(f"- **cost:** ${info.total_cost_usd:.4f}")
    return "\n".join(lines)


def _render_cost_summary(summary: CostSummary) -> str:
    lines = ["## Cost summary", ""]
    if not summary.by_model:
        return "\n".join([*lines, "_no assistant requests in this session_"])
    lines.append("| model | requests | input | output | cache write | cache read | cost |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mu in summary.by_model:
        cost = f"${mu.cost_usd:.4f}" if mu.cost_usd is not None else "n/a"
        lines.append(
            f"| {mu.model} | {mu.requests} | {mu.input_tokens} | {mu.output_tokens} | "
            f"{mu.cache_creation_input_tokens} | {mu.cache_read_input_tokens} | {cost} |"
        )
    total = f"${summary.total_cost_usd:.4f}" if summary.total_cost_usd is not None else "n/a"
    lines.append(
        f"| **total** | {summary.requests} | {summary.input_tokens} | {summary.output_tokens} | "
        f"{summary.cache_creation_input_tokens} | {summary.cache_read_input_tokens} | {total} |"
    )
    if not summary.priced:
        lines.append("")
        lines.append("_no model in the price table was matched; cost is unavailable._")
    return "\n".join(lines)


def _render_tool_stats(stats: list[ToolCallStat]) -> str:
    lines = ["## Tools called", ""]
    if not stats:
        return "\n".join([*lines, "_no tools were called_"])
    lines.append("| tool | calls | errors | pathway |")
    lines.append("|---|---:|---:|---|")
    for s in stats:
        pathway = (
            "worked" if s.errors == 0 else ("failed every time" if s.errors == s.calls else "mixed")
        )
        lines.append(f"| {s.name} | {s.calls} | {s.errors} | {pathway} |")
    return "\n".join(lines)


def _render_errors(errors: list[ToolCall]) -> str:
    lines = ["## Errors", ""]
    if not errors:
        return "\n".join([*lines, "_no tool errors_"])
    for c in errors:
        lines.append(f"### [step {c.step}] {c.name} `#{c.tool_use_id}`")
        if c.reason:
            lines.append(f"**why:** {clip(c.reason, 400)}")
        lines.append(f"**result:** {clip(c.result_text or '(no result text)', 1000)}")
        lines.append("")
    return "\n".join(lines)


def _render_investigation_log(calls: list[ToolCall]) -> str:
    """Every tool call, why it was made, what it was called with, and what came
    back — full fidelity, unclipped, in call order."""
    lines = ["## Investigation log", ""]
    if not calls:
        return "\n".join([*lines, "_no tool calls_"])
    for c in calls:
        flag = "?" if c.is_error is None else ("error" if c.is_error else "ok")
        dur = f" · {c.duration_ms}ms" if c.duration_ms is not None else ""
        lines.append(f"### [step {c.step}] {c.name} — {flag}{dur} `#{c.tool_use_id}`")
        if c.reason:
            lines.append("**why:**")
            lines.append("")
            lines.append(c.reason)
            lines.append("")
        lines.append("**called with:**")
        lines.append("")
        lines.append("```json")
        lines.append(to_json(c.input, indent=2).decode())
        lines.append("```")
        if c.result_text is not None:
            lines.append("")
            lines.append("**returned:**")
            lines.append("")
            lines.append("```")
            lines.append(c.result_text)
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_narrative(entries: list[TimelineEntry]) -> str:
    lines = ["## Full narrative timeline", ""]
    for entry in entries:
        head = f"### [{entry.step}] {entry.type}" + (f" · {entry.model}" if entry.model else "")
        lines.append(head)
        if entry.summary is not None:
            lines.append(f"> {entry.summary}")
        for part in entry.parts:
            lines.append(part.as_markdown())
        if not entry.parts and entry.raw is not None:
            lines.append(f"```json\n{to_json(entry.raw, indent=2).decode()}\n```")
        lines.append("")
    return "\n".join(lines)


def render_investigation_markdown(report: InvestigationReport) -> str:
    """The full investigation record as one Markdown document: session
    metadata, cost, per-tool pathway summary, every error, the complete
    why/queried/returned log, then the raw narrative timeline."""
    sections = [
        _render_header(report),
        _render_cost_summary(report.cost_summary),
        _render_tool_stats(report.tool_stats),
        _render_errors(report.errors),
        _render_investigation_log(report.tool_calls),
        _render_narrative(report.timeline),
    ]
    if report.parse_errors:
        sections.append("## Unparsed lines\n\n" + "\n".join(f"- {e}" for e in report.parse_errors))
    return "\n\n".join(sections) + "\n"
