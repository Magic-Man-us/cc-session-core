"""Opt-in invariants over the real transcript corpus.

Skipped unless CC_SESSION_CORPUS points at a transcripts dir (e.g. ~/.claude/projects).
The parser is lossless by design (extra="allow"; unmodeled kinds fall back to
Unknown*), so nothing here can raise. These are the "fully typed, not just
non-raising" guarantees: the schema audit finds zero gaps, and every
built-in tool's dict result actually types instead of silently falling back.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from cc_session_core import (
    MODELED_RESULT_TOOLS,
    ParseFailure,
    iter_records,
    parse_tool_result,
    result_tool_name,
    tool_name_index,
)
from cc_session_core.models import UserRecord
from cc_session_core.report.audit import audit_files, format_audit

CORPUS = os.environ.get("CC_SESSION_CORPUS")
pytestmark = pytest.mark.corpus

requires_corpus = pytest.mark.skipif(not CORPUS, reason="set CC_SESSION_CORPUS to run")


def _files() -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(f"{CORPUS}/**/*.jsonl", recursive=True))]


@requires_corpus
def test_corpus_is_fully_typed() -> None:
    """The schema audit must find zero gaps: no unmodeled kinds, no untyped fields,
    no parse failures. A failure here is a real coverage gap, not a flaky test —
    it means the models have fallen behind the corpus and need extending."""
    audit = audit_files(_files())
    assert audit.errors == [], format_audit(audit)
    assert audit.unmodeled_record_types == [], format_audit(audit)
    assert audit.unmodeled_block_types == [], format_audit(audit)
    assert audit.unmodeled_attachment_types == [], format_audit(audit)
    assert audit.record_extra_fields == {}, format_audit(audit)
    assert audit.message_extra_fields == {}, format_audit(audit)
    assert audit.block_extra_fields == {}, format_audit(audit)
    assert audit.attachment_extra_fields == {}, format_audit(audit)
    assert audit.usage_extra_fields == {}, format_audit(audit)


@requires_corpus
def test_no_modeled_tool_result_gaps() -> None:
    """Every dict toolUseResult whose tool is modeled must type cleanly — a modeled
    tool silently falling back to a raw dict is a coverage gap in tools.py."""
    gaps: list[str] = []
    for f in _files():
        records = [r for r in iter_records(f) if not isinstance(r, ParseFailure)]
        index = tool_name_index(records)
        for rec in records:
            if isinstance(rec, UserRecord) and isinstance(rec.tool_use_result, dict):
                name = result_tool_name(rec, index)
                if name in MODELED_RESULT_TOOLS:
                    parsed = parse_tool_result(name, rec.tool_use_result)
                    if not isinstance(parsed, BaseModel):
                        gaps.append(f"{name} in {f.name}")
    assert not gaps, f"{len(gaps)} modeled-tool dict results failed to type; e.g. {gaps[:3]}"
