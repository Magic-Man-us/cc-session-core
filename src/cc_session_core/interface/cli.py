"""Command-line entry point: render a transcript file or project directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError
from pydantic_core import to_json

from .. import types as t
from ..models import SnakeModel
from ..report.audit import audit_files, format_audit
from ..report.export import ExportSpec, export, render
from ..report.investigation import build_investigation, render_investigation_markdown
from ..report.render import (
    clip,
    format_cost_summary,
    format_queries,
    format_session_list,
    format_timeline,
)
from ..report.views import CostSummary, TimelineEntry, ToolCall
from ..session import Session, load_project, session_files

_SELECT_SET_KEYS = frozenset({"types", "parts", "tools", "uuids"})


class SessionExport(SnakeModel):
    """Full session dump for ``--json``: source, parse errors, timeline, tool calls, cost."""

    source: t.FilePath | None
    errors: list[str]
    timeline: list[TimelineEntry]
    tool_calls: list[ToolCall]
    cost_summary: CostSummary


def _parse_select(tokens: list[str]) -> ExportSpec:
    """Boundary converter: ``key=v1,v2`` CLI tokens into a validated ExportSpec."""
    data: dict[str, object] = {}
    for tok in tokens:
        key, sep, val = tok.partition("=")
        if not sep:
            raise SystemExit(f"--select expects key=value, got {tok!r}")
        if key == "main_only":
            data[key] = val.strip().lower() in ("1", "true", "yes", "y")
        elif key in _SELECT_SET_KEYS:
            data[key] = {v for v in val.split(",") if v}
        else:
            raise SystemExit(f"unknown --select key {key!r}")
    try:
        return ExportSpec.model_validate(data)
    except ValidationError as exc:
        raise SystemExit(f"invalid --select: {exc}") from exc


def _print_session(session: Session, args: argparse.Namespace) -> None:
    if args.json:
        export = SessionExport(
            source=str(session.source) if session.source else None,
            errors=session.errors,
            timeline=session.timeline(),
            tool_calls=session.tool_calls(),
            cost_summary=session.cost_summary(),
        )
        print(export.model_dump_json(indent=2))
        return

    if session.source:
        print(f"# {session.source}  ({len(session.records)} records)")
    if session.errors:
        print(f"# {len(session.errors)} unparsed line(s):")
        for err in session.errors:
            print(f"#   {err}")

    if args.queries:
        print(format_queries(session.tool_calls()))
        return

    if args.tools:
        for c in session.tool_calls():
            flag = "?" if c.is_error is None else ("error" if c.is_error else "ok")
            dur = f" {c.duration_ms}ms" if c.duration_ms is not None else ""
            print(f"[{c.step:>4}] {c.name} #{c.tool_use_id} ({flag}){dur}")
            if c.reason:
                print(f"    why: {clip(c.reason, 96)}")
            print(f"    in:  {clip(to_json(c.input).decode(), 96)}")
            if c.result_text is not None:
                print(f"    out: {clip(c.result_text, 96)}")
        print()

    print(format_timeline(session.timeline()))
    print()
    print(format_cost_summary(session.cost_summary()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse Claude Code session .jsonl logs into a clean timeline."
    )
    parser.add_argument("path", help="a .jsonl session file, or a project directory")
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    parser.add_argument("--tools", action="store_true", help="list paired tool calls")
    parser.add_argument(
        "--queries",
        action="store_true",
        help="full query timeline: why / queried / returned, per step, unclipped",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="report where the models fall short of your real data (no values emitted)",
    )
    parser.add_argument(
        "--investigate",
        action="store_true",
        help="full investigation record for one session: every tool call, why, "
        "args, result, errors, and the narrative timeline (markdown, or --json)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sessions",
        help="list the investigations in a directory (id, title, calls, cost)",
    )
    parser.add_argument("--strict", action="store_true", help="fail on a malformed line")
    parser.add_argument(
        "--export",
        choices=["text", "markdown", "json", "jsonl"],
        help="export a filtered selection of one session (see --select)",
    )
    parser.add_argument(
        "--select",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="filters: types=, parts=, tools=, uuids= (comma-separated), main_only=true",
    )
    parser.add_argument("-o", "--out", help="write --export output here (default: stdout)")
    args = parser.parse_args(argv)

    path = Path(args.path)

    if args.export:
        if path.is_dir():
            raise SystemExit("--export takes a single .jsonl session file, not a directory")
        session = Session.load(path, strict=args.strict)
        spec = _parse_select(args.select)
        if args.out:
            export(session, spec, args.export, args.out)
            print(f"wrote {args.out}")
        else:
            print(render(session, spec, args.export))
        return 0

    if args.investigate:
        if path.is_dir():
            raise SystemExit("--investigate takes a single .jsonl session file, not a directory")
        report = build_investigation(Session.load(path, strict=args.strict))
        text = (
            to_json(report.model_dump(mode="json"), indent=2).decode()
            if args.json
            else render_investigation_markdown(report)
        )
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0

    sessions = (
        load_project(path, strict=args.strict)
        if path.is_dir()
        else [Session.load(path, strict=args.strict)]
    )

    if args.list_sessions:
        infos = [s.info() for s in sessions]
        if args.json:
            print(to_json([i.model_dump(mode="json") for i in infos], indent=2).decode())
        else:
            print(format_session_list(infos))
        return 0

    if args.audit:
        files = session_files(path)
        report = audit_files(files)
        if args.json:
            print(to_json(report.model_dump(mode="json"), indent=2).decode())
        else:
            print(format_audit(report))
        return 0

    for i, session in enumerate(sessions):
        if i:
            print("\n" + "=" * 100 + "\n")
        _print_session(session, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
