"""``tail_records`` reads only completed appends and survives rewrites."""

from __future__ import annotations

from pathlib import Path

from conftest import record_lines
from pydantic import BaseModel

from cc_session_core import ParseFailure, tail_records

LINES = record_lines()


def write(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def test_reads_whole_file_from_default_cursor(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write(path, "".join(line + "\n" for line in LINES))
    batch = tail_records(path)
    assert len(batch.records) == len(LINES)
    assert all(isinstance(r.record, BaseModel) for r in batch.records)
    assert [r.line for r in batch.records] == list(range(1, len(LINES) + 1))
    assert batch.offset == path.stat().st_size
    assert batch.line == len(LINES)
    assert not batch.restarted


def test_appended_lines_arrive_on_the_next_call(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write(path, LINES[0] + "\n")
    first = tail_records(path)
    assert len(first.records) == 1

    write(path, LINES[1] + "\n" + LINES[2] + "\n")
    second = tail_records(path, first.offset, first.line)
    assert len(second.records) == 2
    assert [r.line for r in second.records] == [2, 3]
    assert second.line == 3
    assert second.offset == path.stat().st_size

    third = tail_records(path, second.offset, second.line)
    assert third.records == []
    assert third.offset == second.offset


def test_partial_line_is_left_for_the_next_read(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write(path, LINES[0] + "\n" + LINES[1][:20])
    first = tail_records(path)
    assert len(first.records) == 1
    assert first.offset == len((LINES[0] + "\n").encode())

    write(path, LINES[1][20:] + "\n")
    second = tail_records(path, first.offset, first.line)
    assert len(second.records) == 1
    assert second.line == 2


def test_shrunken_file_restarts_from_the_top(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write(path, "".join(line + "\n" for line in LINES))
    first = tail_records(path)

    path.write_text(LINES[0] + "\n", encoding="utf-8")
    second = tail_records(path, first.offset, first.line)
    assert second.restarted
    assert len(second.records) == 1
    assert second.line == 1


def test_bad_line_yields_parse_failure_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write(path, LINES[0] + "\n\n{not json}\n")
    batch = tail_records(path)
    assert len(batch.records) == 2
    assert batch.records[1].line == 3
    bad = batch.records[1].record
    assert isinstance(bad, ParseFailure)
    assert bad.line_number == 3
    assert bad.file == "s.jsonl"
