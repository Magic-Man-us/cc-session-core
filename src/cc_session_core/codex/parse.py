"""Validation boundary for Codex rollout JSONL."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from ..parsing.parse import ParseFailure, failure
from .models import CODEX_RECORD_ADAPTER, CodexRecord


def parse_codex_line(line: str) -> CodexRecord:
    """Validate one Codex rollout line."""
    return CODEX_RECORD_ADAPTER.validate_json(line)


def iter_codex_records(path: Path) -> Iterator[CodexRecord | ParseFailure]:
    """Yield typed Codex records, retaining malformed lines as ``ParseFailure``."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield CODEX_RECORD_ADAPTER.validate_json(line)
            except ValidationError as exc:
                yield failure(path.name, number, exc, line)
