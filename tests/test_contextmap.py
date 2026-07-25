"""Context-map CLI artifact writing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cc_session_core.report.contextmap import main


def test_main_creates_missing_out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """-o pointing at a directory that does not exist yet must not crash;
    the text report has already streamed by the time artifacts are written,
    so a FileNotFoundError here loses the json/csv half of the run."""
    target = tmp_path / "projects"
    target.mkdir()
    (target / "s.jsonl").write_text(
        '{"type": "mode", "mode": "normal", "sessionId": "s"}\n', encoding="utf-8"
    )
    out = tmp_path / "not" / "yet" / "created"
    monkeypatch.setattr(sys, "argv", ["cc-session-map", "-o", str(out), str(target)])
    main()
    assert (out / "map.json").is_file()
    assert (out / "map.csv").is_file()
