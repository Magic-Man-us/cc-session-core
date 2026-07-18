# cc-session-core

[![PyPI](https://img.shields.io/pypi/v/cc-session-core.svg)](https://pypi.org/project/cc-session-core/)
[![Python versions](https://img.shields.io/pypi/pyversions/cc-session-core.svg)](https://pypi.org/project/cc-session-core/)
[![CI](https://github.com/Magic-Man-us/cc-session-core/actions/workflows/publish.yml/badge.svg)](https://github.com/Magic-Man-us/cc-session-core/actions/workflows/publish.yml)
[![codecov](https://codecov.io/gh/Magic-Man-us/cc-session-core/branch/main/graph/badge.svg)](https://codecov.io/gh/Magic-Man-us/cc-session-core)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/Magic-Man-us/cc-session-core/blob/main/.github/dependabot.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Magic-Man-us/cc-session-core/blob/main/LICENSE)

Typed, lossless parser for Claude Code session transcripts (`~/.claude/projects/**/*.jsonl`), plus a session-analysis layer (timeline, tool-call pairing, cost) and a context-map tool built on it.

Each transcript line is validated by a Pydantic `TypeAdapter` over a discriminated union keyed on the record `type`. Three layers, each a discriminated union: top-level records, `message.content` blocks, and attachments. `usage`, `message`, and the built-in tools' inputs/results are fully typed; `cc_session_core.parsing.tools` dispatches per-tool models by tool name, with a raw fallback for MCP/unknown tools.

Parsing is **lossless**: models keep unknown fields (`extra="allow"`), and an unmodeled record/block/attachment `type` lands in an `Unknown*` carrier that still holds its payload — nothing is dropped or raised. The coverage gate is the **schema audit** (`cc_session_core.report.audit`), which reports anything that fell into `extra` or an `Unknown*` fallback (field names and value-types only, never values).

## Install

```bash
pip install -e .          # or: uv pip install -e .
```

## Library

Parse lines into typed records:

```python
from pathlib import Path
from cc_session_core import iter_records, parse_line, ParseFailure

rec = parse_line(line)                       # one JSON line -> typed record

for rec in iter_records(Path("session.jsonl")):
    if isinstance(rec, ParseFailure):
        ...                                  # file, line_number, error, raw
    elif rec.type == "assistant":
        rec.message.usage.input_tokens
```

Per-tool input/result resolution:

```python
from cc_session_core import parse_tool_input, parse_tool_result, tool_name_index, result_tool_name

typed_input = parse_tool_input(block.name, block.input)         # model, or raw value
index = tool_name_index(records)                                # tool_use_id -> tool name
typed_result = parse_tool_result(result_tool_name(rec, index), rec.tool_use_result)
```

Analyze a whole session:

```python
from cc_session_core import Session

s = Session.load("session.jsonl")
s.timeline()          # ordered, decomposed events (text / thinking / tool_use / tool_result / ...)
s.tool_calls()        # every tool_use paired with its tool_result, plus the assistant's "why"
s.cost_summary()      # token + cost rollup per model (one API request counted once)
s.label(); s.info()   # human title + one-line summary
```

Cost uses `cc_session_core.cost.pricing` (published list rates in `EXAMPLE_PRICING`); pass your own `PriceTable` for a different valuation. Rates are a usage valuation, not a billed amount.

## CLI

```bash
cc-session PATH [--tools] [--queries] [--audit] [--list] [--json] [--strict]
cc-session PATH --export <text|markdown|json|jsonl> [--select k=v ...] [-o OUT]
```

`PATH` is a `.jsonl` file or a project directory (directories load recursively). `--tools` lists paired tool calls; `--queries` prints the full why/queried/returned timeline; `--list` indexes the sessions in a directory; `--audit` reports schema coverage over the target (field names + value-types only, safe to share).

`--export` writes a filtered slice of the session; `--select` narrows it (space-separated `key=comma,values`): `parts=` (`text,thinking,tool_use,tool_result,image,other`), `tools=`, `types=`, `uuids=`, `main_only=true`. `-o` writes to a file instead of stdout.

```bash
cc-session-map [TRANSCRIPTS_DIR] [-o OUT_DIR]   # default: ~/.claude/projects, .
```

Aggregates per transcript and overall: turns (main vs sidechain), tool usage, token usage by kind, server web tools, and cost; writes `map.json` and `map.csv` into `OUT_DIR`.

## MCP server

`cc-session-mcp` (stdio) exposes `list_sessions`, `session_summary`, `tool_calls`, `export_session`, and `audit`. It needs the `mcp` extra (`pip install "cc-session-core[mcp]"`). Register in a Claude Code `.mcp.json`:

```json
{
  "mcpServers": {
    "cc-session": { "command": "cc-session-mcp" }
  }
}
```

## Tests

```bash
uv sync --all-groups                                    # pytest, ruff, pyright, mcp
uv run pytest                                           # fast unit tests on frozen, scrubbed fixtures
CC_SESSION_CORPUS=~/.claude/projects uv run pytest -m corpus   # opt-in: asserts zero Unknown*/extra/parse-failures on real data
```

The corpus test is the lossless coverage gate: it fails if any real line lands in an `Unknown*` fallback, leaves a field in `model_extra`, or a modeled built-in tool result falls back to raw. Fixtures are regenerated with `python tests/_extract_fixtures.py` (`CC_SESSION_CORPUS` set); free-text, paths, and base64 are scrubbed.
