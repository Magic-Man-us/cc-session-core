"""Schema audit: finds gaps in already-modeled fixtures, reports safely."""

from __future__ import annotations

from conftest import FIXTURES

from cc_session_core.report.audit import SchemaAudit, audit_files, format_audit


def test_audit_of_fully_modeled_fixture_finds_no_gaps() -> None:
    audit = audit_files([FIXTURES / "records.jsonl"])
    assert isinstance(audit, SchemaAudit)
    assert audit.files == 1
    assert audit.lines > 0
    assert audit.errors == []
    assert audit.unmodeled_record_types == []
    assert audit.unmodeled_block_types == []
    assert audit.unmodeled_attachment_types == []
    assert audit.record_extra_fields == {}
    assert audit.message_extra_fields == {}


def test_format_audit_documents_its_own_privacy_guarantee() -> None:
    audit = audit_files([FIXTURES / "records.jsonl"])
    report = format_audit(audit)
    assert "field names and value-types only" in report
