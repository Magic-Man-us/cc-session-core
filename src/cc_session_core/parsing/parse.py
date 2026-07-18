"""Boundary: validate transcript lines into typed records.

``iter_records`` is the line-by-line ``TypeAdapter`` boundary — it converts each
JSON line to the right ``Record`` model and translates any ``ValidationError``
into a typed ``ParseFailure`` carrying the file, line number, and raw line, so a
caller can decide whether to stop or collect coverage gaps.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..models import RECORD_ADAPTER, Record
from ..types import SNAKE_CONFIG, ErrorText, FilePath, LineNumber


class ParseFailure(BaseModel):
    model_config = SNAKE_CONFIG

    file: FilePath
    line_number: LineNumber
    error: ErrorText
    raw: str


def parse_line(line: str) -> Record:
    """Validate one transcript line into its record model (raises on mismatch)."""
    return RECORD_ADAPTER.validate_json(line)


def failure(file: str, line_number: int, exc: ValidationError, line: str) -> ParseFailure:
    """Fold a ``ValidationError`` into the typed carrier every line reader yields."""
    return ParseFailure(
        file=file,
        line_number=line_number,
        error="; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()),
        raw=line[:600],
    )


def iter_records(path: Path) -> Iterator[Record | ParseFailure]:
    """Yield a typed record per line; a ``ParseFailure`` where validation fails."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for n, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield RECORD_ADAPTER.validate_json(line)
            except ValidationError as exc:
                yield failure(path.name, n, exc, line)
