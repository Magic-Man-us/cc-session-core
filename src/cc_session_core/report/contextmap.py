#!/usr/bin/env python3
"""Context map over Claude Code transcripts, built on the typed cc_session_core layer.

Every figure here comes from a validated field on a typed record — no dict
poking. Aggregates per transcript and overall: turns (main vs sidechain), tool
usage, token usage split by kind, server web tools, and computed cost from
per-model published rates. Emits a text report plus map.json and map.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from .. import types as t
from ..models import UserRecord
from ..parsing.tools import BashResult, EditResult, PatchHunk, WebFetchResult, WriteResult
from ..session import DEFAULT_PROJECTS_ROOT, Session
from ..types import SNAKE_CONFIG
from .views import ToolCall, token_total

TOP_TRANSCRIPTS = 25
TOP_TOOLS = 30


class SessionMap(BaseModel):
    model_config = SNAKE_CONFIG

    file: t.FilePath
    assistant_turns: t.Count = 0
    user_turns: t.Count = 0
    sidechain_turns: t.Count = 0
    parse_failures: t.Count = 0
    input_tokens: t.TokenCount = 0
    output_tokens: t.TokenCount = 0
    cache_write_tokens: t.TokenCount = 0
    cache_read_tokens: t.TokenCount = 0
    web_search_requests: t.Count = 0
    web_fetch_requests: t.Count = 0
    bash_interrupted: t.Count = 0
    diff_lines_added: t.Count = 0
    diff_lines_removed: t.Count = 0
    web_fetch_bytes: t.ByteSize = 0
    cost_usd: t.CostUsd = 0.0
    models: dict[t.ModelId, t.Count] = Field(default_factory=dict)
    tools: dict[t.ToolName, t.Count] = Field(default_factory=dict)


class Totals(BaseModel):
    """Field-wise rollup of every :class:`SessionMap` for the overall report."""

    model_config = SNAKE_CONFIG

    input_tokens: t.TokenCount = 0
    output_tokens: t.TokenCount = 0
    cache_write_tokens: t.TokenCount = 0
    cache_read_tokens: t.TokenCount = 0
    cost_usd: t.CostUsd = 0.0
    web_search_requests: t.Count = 0
    web_fetch_requests: t.Count = 0
    bash_interrupted: t.Count = 0
    diff_lines_added: t.Count = 0
    diff_lines_removed: t.Count = 0
    web_fetch_bytes: t.ByteSize = 0
    models: dict[t.ModelId, t.Count] = Field(default_factory=dict)
    tools: dict[t.ToolName, t.Count] = Field(default_factory=dict)

    @classmethod
    def from_maps(cls, maps: list[SessionMap]) -> Totals:
        models: Counter[str] = Counter()
        tools: Counter[str] = Counter()
        tot = cls()
        for sm in maps:
            models.update(sm.models)
            tools.update(sm.tools)
            tot.input_tokens += sm.input_tokens
            tot.output_tokens += sm.output_tokens
            tot.cache_write_tokens += sm.cache_write_tokens
            tot.cache_read_tokens += sm.cache_read_tokens
            tot.cost_usd += sm.cost_usd
            tot.web_search_requests += sm.web_search_requests
            tot.web_fetch_requests += sm.web_fetch_requests
            tot.bash_interrupted += sm.bash_interrupted
            tot.diff_lines_added += sm.diff_lines_added
            tot.diff_lines_removed += sm.diff_lines_removed
            tot.web_fetch_bytes += sm.web_fetch_bytes
        tot.models = dict(models)
        tot.tools = dict(tools)
        return tot

    @property
    def token_total(self) -> t.TokenCount:
        # cache_write_tokens is cost.cache_creation_input_tokens under a shorter name.
        return token_total(
            self.input_tokens,
            self.output_tokens,
            self.cache_write_tokens,
            self.cache_read_tokens,
        )


_MAPS_ADAPTER: TypeAdapter[list[SessionMap]] = TypeAdapter(list[SessionMap])


class _ResultCounts(BaseModel):
    """Metrics derived from tool results — the four tool-owned payloads we read."""

    model_config = SNAKE_CONFIG

    bash_interrupted: t.Count = 0
    diff_lines_added: t.Count = 0
    diff_lines_removed: t.Count = 0
    web_fetch_bytes: t.ByteSize = 0


def _patch_lines(patch: list[PatchHunk]) -> tuple[int, int]:
    added = removed = 0
    for hunk in patch:
        for line in hunk.lines:
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
    return added, removed


def _result_counts(calls: list[ToolCall]) -> _ResultCounts:
    """Fold the already-parsed, de-duplicated tool results into the four metrics."""
    bash_interrupted = diff_added = diff_removed = web_bytes = 0
    for call in calls:
        result = call.result_typed
        if isinstance(result, BashResult):
            bash_interrupted += int(result.interrupted)
        elif isinstance(result, (EditResult, WriteResult)):
            added, removed = _patch_lines(result.structured_patch)
            diff_added += added
            diff_removed += removed
        elif isinstance(result, WebFetchResult):
            web_bytes += result.bytes
    return _ResultCounts(
        bash_interrupted=bash_interrupted,
        diff_lines_added=diff_added,
        diff_lines_removed=diff_removed,
        web_fetch_bytes=web_bytes,
    )


def map_file(path: Path) -> SessionMap:
    session = Session.load(path)
    cost = session.cost_summary()
    calls = session.tool_calls()
    requests = session.assistant_requests()
    server = [
        r.message.usage.server_tool_use
        for r in requests
        if r.message.usage.server_tool_use is not None
    ]
    counts = _result_counts(calls)

    return SessionMap(
        file=path.name,
        assistant_turns=cost.requests,
        user_turns=sum(isinstance(r, UserRecord) for r in session.records),
        sidechain_turns=sum(1 for r in requests if r.is_sidechain),
        parse_failures=len(session.errors),
        input_tokens=cost.input_tokens,
        output_tokens=cost.output_tokens,
        cache_write_tokens=cost.cache_creation_input_tokens,
        cache_read_tokens=cost.cache_read_input_tokens,
        web_search_requests=sum(s.web_search_requests for s in server),
        web_fetch_requests=sum(s.web_fetch_requests for s in server),
        bash_interrupted=counts.bash_interrupted,
        diff_lines_added=counts.diff_lines_added,
        diff_lines_removed=counts.diff_lines_removed,
        web_fetch_bytes=counts.web_fetch_bytes,
        cost_usd=cost.total_cost_usd or 0.0,
        models={mu.model: mu.requests for mu in cost.by_model},
        tools=dict(Counter(c.name for c in calls)),
    )


def _report(target: str, maps: list[SessionMap], totals: Totals) -> None:
    print(f"=== CONTEXT MAP: {len(maps)} transcripts under {target} ===\n")
    print(f"{'transcript':<40} {'a-turns':>7} {'sub':>5} {'tools':>6} {'cost($)':>10}")
    print("-" * 74)
    for sm in sorted(maps, key=lambda s: -s.cost_usd)[:TOP_TRANSCRIPTS]:
        print(
            f"{sm.file[:38]:<40} {sm.assistant_turns:>7} {sm.sidechain_turns:>5} "
            f"{sum(sm.tools.values()):>6} {sm.cost_usd:>10.2f}"
        )
    if len(maps) > TOP_TRANSCRIPTS:
        print(f"... and {len(maps) - TOP_TRANSCRIPTS} more (full detail in map.json)")

    print("\n=== TOKENS (all transcripts) ===")
    total = totals.token_total
    for label, value in (
        ("input", totals.input_tokens),
        ("output", totals.output_tokens),
        ("cache_write", totals.cache_write_tokens),
        ("cache_read", totals.cache_read_tokens),
    ):
        print(f"  {label:<12} {value:>16,}")
    print(f"  {'TOTAL':<12} {total:>16,}")
    if total:
        print(f"  cache-read share: {totals.cache_read_tokens / total:.1%}")

    print("\n=== MODELS ===")
    for model, count in Counter(totals.models).most_common():
        print(f"  {count:>7,}  {model}")

    print(
        f"\n=== SERVER WEB TOOLS ===  "
        f"web_search={totals.web_search_requests:,}  web_fetch={totals.web_fetch_requests:,}"
    )

    print("\n=== DERIVED FROM TYPED TOOL RESULTS ===")
    print(f"  bash calls interrupted:  {totals.bash_interrupted:,}")
    print(f"  diff lines added:        {totals.diff_lines_added:,}")
    print(f"  diff lines removed:      {totals.diff_lines_removed:,}")
    print(f"  web_fetch bytes fetched: {totals.web_fetch_bytes:,}")

    print(f"\n=== TOOL USAGE (top {TOP_TOOLS} of {len(totals.tools)}) ===")
    for name, count in Counter(totals.tools).most_common(TOP_TOOLS):
        print(f"  {count:>7,}  {name}")

    print(f"\n=== ESTIMATED TOTAL COST: ${totals.cost_usd:,.2f} ===")


def _write_artifacts(out_dir: Path, maps: list[SessionMap]) -> None:
    (out_dir / "map.json").write_bytes(_MAPS_ADAPTER.dump_json(maps, indent=2))
    cols = [name for name in SessionMap.model_fields if name not in ("models", "tools")]
    with (out_dir / "map.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for sm in maps:
            row = sm.model_dump(exclude={"models", "tools"})
            writer.writerow([row[c] for c in cols])
    print(f"\nwrote {out_dir / 'map.json'} and {out_dir / 'map.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context map over Claude Code transcripts: a text report plus map.json/map.csv."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_PROJECTS_ROOT),
        help="a projects directory to map (default: ~/.claude/projects)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path(),
        help="directory to write map.json and map.csv into (default: current directory)",
    )
    args = parser.parse_args()
    files = sorted(glob.glob(f"{args.target}/**/*.jsonl", recursive=True))
    maps = [map_file(Path(f)) for f in files]
    totals = Totals.from_maps(maps)

    _report(args.target, maps, totals)
    _write_artifacts(args.out_dir, maps)


if __name__ == "__main__":
    main()
