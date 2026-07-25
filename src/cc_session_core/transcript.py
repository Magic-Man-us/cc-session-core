"""Shared transcript union spanning Claude Code and Codex rollout records."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, cast

from pydantic import Discriminator, Tag, TypeAdapter

from .codex.models import CODEX_RECORD_TYPES, CodexRecord
from .models import CLAUDE_RECORD_TYPES, Record, UnknownRecord


class UnknownTranscriptRecord(UnknownRecord):
    """A lossless line whose new ``type`` cannot yet be attributed to a provider."""


def _provider_discriminator() -> Callable[[object], str]:
    def tag(value: object) -> str:
        if isinstance(value, dict):
            record_type: object = cast("dict[str, object]", value).get("type")
        else:
            record_type = getattr(value, "type", None)
        if record_type in CLAUDE_RECORD_TYPES:
            return "claude"
        if record_type in CODEX_RECORD_TYPES:
            return "codex"
        return "unknown"

    return tag


TranscriptRecord = Annotated[
    Annotated[Record, Tag("claude")]
    | Annotated[CodexRecord, Tag("codex")]
    | Annotated[UnknownTranscriptRecord, Tag("unknown")],
    Discriminator(_provider_discriminator()),
]

TRANSCRIPT_RECORD_ADAPTER: TypeAdapter[TranscriptRecord] = TypeAdapter(TranscriptRecord)
