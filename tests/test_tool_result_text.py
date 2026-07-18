"""tool_result_text flattens a tool_result's content to plain text, extracting an
MCP text item's own ``text`` instead of re-wrapping it in another layer of JSON."""

from __future__ import annotations

from pydantic import JsonValue

from cc_session_core.report.views import tool_result_text


def test_string_content_passes_through() -> None:
    assert tool_result_text("file.txt") == "file.txt"


def test_mcp_text_item_contributes_its_own_text() -> None:
    content: list[JsonValue] = [{"type": "text", "text": '### Result\n"{\\n  \\"ok\\": true\\n}"'}]
    assert tool_result_text(content) == '### Result\n"{\\n  \\"ok\\": true\\n}"'


def test_multiple_mcp_text_items_join_with_newlines() -> None:
    content: list[JsonValue] = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert tool_result_text(content) == "first\nsecond"


def test_non_text_item_falls_back_to_a_compact_json_dump() -> None:
    content: list[JsonValue] = [{"type": "image", "source": "base64..."}]
    assert tool_result_text(content) == '{"type":"image","source":"base64..."}'


def test_plain_string_items_pass_through() -> None:
    assert tool_result_text(["a", "b"]) == "a\nb"
