"""Local MCP server exposing cc_session_core over the Model Context Protocol.

A thin, read-only adapter: each tool wraps an existing cc_session_core function so an MCP
client (Claude Code, etc.) can list, summarize, and *selectively export* Claude
Code sessions without shelling out. Outputs are bounded — ``export_session``
truncates and leans on the ``ExportSpec`` filters — so a tool call can't blow the
caller's context with a whole transcript.

Optional dependency: ``pip install cc-session-core[mcp]``.
Run: ``cc-session-mcp`` (stdio), or ``python -m cc_session_core.interface.mcp``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic_core import to_json

from ..models import SnakeModel
from ..report.audit import SchemaAudit, audit_files
from ..report.export import ExportFormat, ExportSpec, render
from ..report.investigation import build_investigation, render_investigation_markdown
from ..report.views import CostSummary, SessionInfo, ToolCall
from ..session import DEFAULT_PROJECTS_ROOT, Session, resolve_session_file, session_files

server = FastMCP("cc-session")
_MAX_CHARS = 20_000


class SessionSummary(SnakeModel):
    """Token + cost rollup paired with a session's one-line info."""

    info: SessionInfo
    cost: CostSummary


def _resolve(session: str) -> Path:
    """A .jsonl path as-is, or a session id/prefix under ~/.claude/projects."""
    path = resolve_session_file(session)
    if path is None:
        raise ValueError(f"no session file found for {session!r}")
    return path


@server.tool()
def list_sessions(limit: int = 20) -> list[SessionInfo]:
    """List the most-recently-modified Claude Code sessions (id, title, cost, tool calls)."""
    files = sorted(
        session_files(DEFAULT_PROJECTS_ROOT), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return [Session.load(f).info() for f in files[:limit]]


@server.tool()
def session_summary(session: str) -> SessionSummary:
    """Token + cost rollup and one-line info for a session (by id/prefix or path)."""
    parsed = Session.load(_resolve(session))
    return SessionSummary(info=parsed.info(), cost=parsed.cost_summary())


@server.tool()
def investigate_session(session: str, fmt: Literal["markdown", "json"] = "markdown") -> str:
    """The full investigation record for a session: every tool call with its stated
    reason (the assistant's own preceding text/thinking — never inferred), arguments,
    result, error, and timing, a per-tool pathway summary (worked / mixed / failed
    every time), and the complete narrative timeline. Unbounded — a large session
    can produce a very long document; narrow with ``export_session`` instead if you
    only need a slice."""
    report = build_investigation(Session.load(_resolve(session)))
    if fmt == "json":
        return to_json(report.model_dump(mode="json"), indent=2).decode()
    return render_investigation_markdown(report)


@server.tool()
def tool_calls(session: str, tool: str | None = None, limit: int = 50) -> list[ToolCall]:
    """Paired tool_use/tool_result calls for a session, optionally filtered to one tool name."""
    calls = Session.load(_resolve(session)).tool_calls()
    if tool is not None:
        calls = [c for c in calls if c.name == tool]
    return calls[:limit]


@server.tool()
def export_session(
    session: str,
    spec: ExportSpec | None = None,
    fmt: ExportFormat = "text",
    max_chars: int = _MAX_CHARS,
) -> str:
    """Export a filtered slice of a session as text/markdown/json/jsonl.

    ``spec`` selects what to include (omit any field to keep everything): ``parts``
    (text/thinking/tool_use/tool_result/image/other), ``tools`` (restrict tool parts
    to these names), ``types`` (record kinds), ``uuids``, ``main_only`` (drop
    sidechain turns). Output is truncated to ``max_chars`` — narrow the filters for
    a complete, smaller slice.
    """
    out = render(Session.load(_resolve(session)), spec or ExportSpec(), fmt)
    if len(out) > max_chars:
        return out[:max_chars] + f"\n\n[truncated {len(out) - max_chars} chars — narrow the filter]"
    return out


@server.tool()
def audit(directory: str | None = None) -> SchemaAudit:
    """Schema-coverage audit over a session file or a project dir (field names + value
    types only, never values). Reports any unmodeled kinds or untyped fields."""
    target = Path(directory).expanduser() if directory else DEFAULT_PROJECTS_ROOT
    files = session_files(target)
    return audit_files(files)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
