"""Every attachment kind discriminates within an attachment record."""

from __future__ import annotations

import pytest
from conftest import attachment_lines, attachment_type

from cc_session_core import parse_line
from cc_session_core.models import AttachmentRecord

LINES = attachment_lines()


@pytest.mark.parametrize("line", LINES, ids=attachment_type)
def test_attachment_discriminates(line: str) -> None:
    rec = parse_line(line)
    assert isinstance(rec, AttachmentRecord)
    assert rec.attachment.type == attachment_type(line)


def test_all_attachment_kinds_present() -> None:
    kinds = {attachment_type(line) for line in LINES}
    assert len(kinds) == 34
    for required in (
        "hook_success",
        "task_reminder",
        "dynamic_skill",
        "hook_system_message",
        "total_tokens_reminder",
        "auto_mode",
    ):
        assert required in kinds
