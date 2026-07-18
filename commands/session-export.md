---
description: Export a filtered slice of a Claude Code session (tool calls, messages, etc.) to a file
argument-hint: "<session> --export <text|markdown|json|jsonl> [--select <k=v…>] [-o <out>]"
allowed-tools: Bash(cc-session:*)
---
Export result:

!`cc-session $ARGUMENTS`

Report the path it wrote above (or relay any error). Arguments pass straight to the `cc-session` CLI:

- `<session>`: a `.jsonl` path or a session id/prefix
- `--export`: `text` | `markdown` | `json` | `jsonl`
- `--select` (space-separated `key=comma,values`, omit to include all):
  `parts=` (`text,thinking,tool_use,tool_result,image,other`), `tools=`, `types=`, `uuids=`, `main_only=true`
- `-o`: output file (omit for stdout)

If `$ARGUMENTS` is empty or plainly not CLI flags, ask for the session and what to include instead of rerunning.
