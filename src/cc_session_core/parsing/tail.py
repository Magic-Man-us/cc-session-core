"""Incremental reads over a transcript file that is still being written.

``tail_records`` is the follow-mode counterpart to ``iter_records``: instead of consuming
the whole file, it parses only the complete lines appended past a byte-offset cursor
and hands back the cursor for the next call. A line caught mid-write (no trailing
newline yet) is left unconsumed so it arrives whole on a later read, and a file that
shrank underneath the cursor (rewrite/rotation) restarts from the top rather than
yielding garbage from the middle of a line.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..models import RECORD_ADAPTER, Record
from ..types import SNAKE_CONFIG, ByteOffset, LineNumber
from .parse import ParseFailure, failure


class TailRecord(BaseModel):
    """One parsed record paired with its 1-based line number in the file."""

    model_config = SNAKE_CONFIG

    line: LineNumber
    record: Record | ParseFailure


class TailBatch(BaseModel):
    """Records appended past a cursor, plus the cursor to poll from next.

    ``offset`` is the byte position after the last complete line consumed and
    ``line`` its 1-based line number — feed both back into the next call.
    ``restarted`` marks a read where the file was smaller than the cursor, so the
    batch starts over from the beginning of the file.
    """

    model_config = SNAKE_CONFIG

    records: list[TailRecord]
    offset: ByteOffset
    line: LineNumber
    restarted: bool = False


def tail_records(path: Path, offset: int = 0, line: int = 0) -> TailBatch:
    """Parse the complete lines appended past ``offset``; cursors default to the top.

    Raises ``OSError`` (e.g. ``FileNotFoundError``) when the file cannot be read —
    a tailed transcript disappearing is the caller's policy decision, not ours.
    """
    restarted = False
    with path.open("rb") as fh:
        # Size the handle we read from, not the path — a rotation between a stat()
        # and the open would otherwise let the cursor point past EOF.
        if os.fstat(fh.fileno()).st_size < offset:
            offset = 0
            line = 0
            restarted = True
        fh.seek(offset)
        data = fh.read()
    end = data.rfind(b"\n")
    if end < 0:
        return TailBatch(records=[], offset=offset, line=line, restarted=restarted)
    complete = data[: end + 1]
    records: list[TailRecord] = []
    for raw in complete.split(b"\n")[:-1]:
        line += 1
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            records.append(TailRecord(line=line, record=RECORD_ADAPTER.validate_json(text)))
        except ValidationError as exc:
            records.append(TailRecord(line=line, record=failure(path.name, line, exc, text)))
    return TailBatch(records=records, offset=offset + len(complete), line=line, restarted=restarted)
