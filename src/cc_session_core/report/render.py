"""Text renderers over the cc_session_core view models.

Pure formatters (model in, str out) for the timeline, cost summary, and tool
call/query views built by :mod:`cc_session_core.session`. Kept separate from the
library so the data layer stays presentation-free; the CLI is the only caller.
"""

from __future__ import annotations

from pydantic_core import to_json

from .views import CostSummary, SessionInfo, TimelineEntry, ToolCall, clip, sort_key


def format_timeline(entries: list[TimelineEntry], width: int = 100) -> str:
    lines: list[str] = []
    for e in entries:
        ts = e.timestamp.strftime("%H:%M:%S") if e.timestamp else "--:--:--"
        head = f"[{e.step:>4}] {ts}  {e.type}#{e.type_step}"
        if e.model:
            head += f"  {e.model}"
        if e.cost_usd is not None:
            head += f"  ${e.cost_usd:.4f}"
            if e.cost_source:
                head += f" ({e.cost_source})"
        if e.usage:
            head += (
                f"  [in {e.usage.input_tokens} out {e.usage.output_tokens}"
                f" cw {e.usage.cache_creation_input_tokens}"
                f" cr {e.usage.cache_read_input_tokens}]"
            )
        lines.append(head)

        if e.summary is not None:
            lines.append(f"        summary: {clip(e.summary, width)}")
        lines.extend(part.render(width) for part in e.parts)
    return "\n".join(lines)


def format_cost_summary(summary: CostSummary) -> str:
    lines = ["cost summary"]
    for mu in summary.by_model:
        cost = f"${mu.cost_usd:.4f}" if mu.cost_usd is not None else "n/a"
        lines.append(
            f"  {mu.model:<32} requests {mu.requests:>4}  "
            f"in {mu.input_tokens:>8}  out {mu.output_tokens:>8}  "
            f"cw {mu.cache_creation_input_tokens:>8}  cr {mu.cache_read_input_tokens:>9}  "
            f"cost {cost}"
        )
    total = f"${summary.total_cost_usd:.4f}" if summary.total_cost_usd is not None else "n/a"
    lines.append(
        f"  {'TOTAL':<32} requests {summary.requests:>4}  "
        f"in {summary.input_tokens:>8}  out {summary.output_tokens:>8}  "
        f"cw {summary.cache_creation_input_tokens:>8}  cr {summary.cache_read_input_tokens:>9}  "
        f"cost {total}"
    )
    if not summary.priced:
        lines.append("  (no model in the price table was matched; populate one for cost)")
    return "\n".join(lines)


def format_queries(calls: list[ToolCall]) -> str:
    """Full, unclipped query timeline: for each call, the why (assistant's
    narration), the query, and what it returned. Built for reading an
    investigation end to end."""

    def block(text: str, pad: str = "    ") -> list[str]:
        return [pad + ln for ln in text.splitlines()] or [pad]

    lines: list[str] = []
    for c in calls:
        flag = "?" if c.is_error is None else ("error" if c.is_error else "ok")
        dur = f"  {c.duration_ms}ms" if c.duration_ms is not None else ""
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"[step {c.step}]  {c.name}  ({flag}){dur}  #{c.tool_use_id}")
        if c.reason:
            lines.append("  why:")
            lines.extend(block(c.reason))
        lines.append("  queried:")
        lines.extend(block(to_json(c.input, indent=2).decode()))
        if c.result_text is not None:
            lines.append("  returned:")
            lines.extend(block(c.result_text))
    return "\n".join(lines)


def format_session_list(infos: list[SessionInfo]) -> str:
    """A pickable index of investigations, most recent first."""
    rows = sorted(infos, key=lambda i: sort_key(i.started), reverse=True)
    lines = [f"{len(rows)} investigation(s):", ""]
    for info in rows:
        when = info.started.strftime("%Y-%m-%d %H:%M") if info.started else "------ --:--"
        cost = f"${info.total_cost_usd:.2f}" if info.total_cost_usd is not None else "  n/a"
        lines.append(
            f"  {info.id[:12]:<12}  {when}  {info.tool_calls:>4} calls  {cost:>8}  "
            f"{clip(info.title, 70)}"
        )
    return "\n".join(lines)
