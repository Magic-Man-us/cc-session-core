"""Normalize typed Codex rollout records into the package's shared session views."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import cast

from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_json

from .. import types as t
from ..models import (
    AssistantMessage,
    AssistantRecord,
    CacheCreation,
    ContentBlock,
    ImageBlock,
    Record,
    SystemRecord,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UnknownBlock,
    Usage,
    UserMessage,
    UserRecord,
)
from ..parsing.parse import ParseFailure, iter_transcript_records
from ..report.views import SessionInfo
from ..session import Session
from ..transcript import TranscriptRecord
from .models import (
    AgentMessageEvent,
    AgentMessageResponseItem,
    AgentReasoningEvent,
    CodexResponseItem,
    CodexTokenUsage,
    CompactedRecord,
    CompactionResponseItem,
    ContextCompactionResponseItem,
    CustomToolCallOutputResponseItem,
    CustomToolCallResponseItem,
    ErrorEvent,
    EventMessageRecord,
    FunctionCallOutputResponseItem,
    FunctionCallResponseItem,
    ImageGenerationCallResponseItem,
    InputAudioContent,
    InputImageContent,
    InputTextContent,
    LocalShellCallResponseItem,
    MessageResponseItem,
    OutputTextContent,
    RawResponseCompletedEvent,
    ReasoningResponseItem,
    ResponseItemRecord,
    RolloutBase,
    SessionMetaRecord,
    ThreadSettingsAppliedEvent,
    TokenCountEvent,
    ToolSearchCallResponseItem,
    ToolSearchOutputResponseItem,
    TurnContextRecord,
    TurnStartedEvent,
    UserMessageEvent,
    WebSearchCallResponseItem,
)

_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def _json_or_text(value: str) -> JsonValue:
    try:
        return _JSON_VALUE_ADAPTER.validate_json(value)
    except ValidationError:
        return value


def _result_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return to_json(value).decode()


def _message_turn_id(item: CodexResponseItem, current: str | None, position: int) -> str:
    metadata = getattr(item, "internal_chat_message_metadata_passthrough", None)
    if metadata is not None and metadata.turn_id:
        return metadata.turn_id
    return current or f"codex-turn-{position}"


def _content_blocks(item: MessageResponseItem) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for content in item.content:
        if isinstance(content, (InputTextContent, OutputTextContent)):
            blocks.append(TextBlock(type="text", text=content.text))
        elif isinstance(content, InputImageContent):
            blocks.append(
                ImageBlock(
                    type="image",
                    source={"type": "url", "url": content.image_url, "detail": content.detail},
                )
            )
        elif isinstance(content, InputAudioContent):
            blocks.append(
                UnknownBlock.model_validate({"type": "input_audio", "audio_url": content.audio_url})
            )
        else:
            blocks.append(UnknownBlock.model_validate(content.model_dump(mode="json")))
    return blocks


def _usage_from_codex(value: CodexTokenUsage) -> Usage:
    # Responses API input_tokens includes cached_input_tokens.  The canonical
    # Claude-shaped view stores uncached and cached tokens separately.
    uncached = max(value.input_tokens - value.cached_input_tokens, 0)
    return Usage(
        input_tokens=uncached,
        output_tokens=value.output_tokens,
        cache_read_input_tokens=value.cached_input_tokens,
        cache_creation_input_tokens=value.cache_write_input_tokens,
        cache_creation=CacheCreation(),
        reasoning_output_tokens=value.reasoning_output_tokens,
        total_tokens=value.total_tokens,
    )


def _usage_delta(current: CodexTokenUsage, previous: CodexTokenUsage) -> CodexTokenUsage:
    return CodexTokenUsage(
        input_tokens=max(current.input_tokens - previous.input_tokens, 0),
        cached_input_tokens=max(current.cached_input_tokens - previous.cached_input_tokens, 0),
        cache_write_input_tokens=max(
            current.cache_write_input_tokens - previous.cache_write_input_tokens, 0
        ),
        output_tokens=max(current.output_tokens - previous.output_tokens, 0),
        reasoning_output_tokens=max(
            current.reasoning_output_tokens - previous.reasoning_output_tokens, 0
        ),
        total_tokens=max(current.total_tokens - previous.total_tokens, 0),
    )


def _has_usage(value: CodexTokenUsage) -> bool:
    return any(
        (
            value.input_tokens,
            value.cached_input_tokens,
            value.cache_write_input_tokens,
            value.output_tokens,
            value.reasoning_output_tokens,
            value.total_tokens,
        )
    )


class _Normalizer:
    def __init__(self, records: list[TranscriptRecord]) -> None:
        self.source_records = records
        self.records: list[AssistantRecord | UserRecord | SystemRecord] = []
        self.session_id: str | None = None
        self.current_turn_id: str | None = None
        self.current_model = "unknown"
        self.previous_usage = CodexTokenUsage()
        self.last_uuid: str | None = None
        self.has_raw_usage = any(
            isinstance(record, EventMessageRecord)
            and isinstance(record.payload, RawResponseCompletedEvent)
            and record.payload.token_usage is not None
            for record in records
        )
        self.response_text = {
            (record.payload.role, self._message_text(record.payload))
            for record in records
            if isinstance(record, ResponseItemRecord)
            and isinstance(record.payload, MessageResponseItem)
        }

    @staticmethod
    def _message_text(item: MessageResponseItem) -> str:
        return "\n".join(
            content.text
            for content in item.content
            if isinstance(content, (InputTextContent, OutputTextContent))
        )

    def _uuid(self, position: int, kind: str, item_id: str | None = None) -> str:
        return item_id or f"codex-{position}-{kind}"

    def _append(self, record: AssistantRecord | UserRecord | SystemRecord) -> None:
        if record.parent_uuid is None:
            record.parent_uuid = self.last_uuid
        self.records.append(record)
        self.last_uuid = record.uuid

    def _assistant(
        self,
        *,
        position: int,
        timestamp: t.Timestamp,
        turn_id: str,
        blocks: list[ContentBlock],
        item_id: str | None = None,
        usage: Usage | None = None,
        request_id: str | None = None,
    ) -> None:
        self._append(
            AssistantRecord(
                type="assistant",
                session_id=self.session_id,
                uuid=self._uuid(position, "assistant", item_id),
                timestamp=timestamp,
                message=AssistantMessage(
                    type="message",
                    role="assistant",
                    id=turn_id,
                    model=self.current_model,
                    content=blocks,
                    usage=usage or Usage(input_tokens=0, output_tokens=0),
                ),
                request_id=request_id,
            )
        )

    def _user(
        self,
        *,
        position: int,
        timestamp: t.Timestamp,
        content: str | list[ContentBlock],
        item_id: str | None = None,
        tool_result: JsonValue | None = None,
    ) -> None:
        self._append(
            UserRecord(
                type="user",
                session_id=self.session_id,
                uuid=self._uuid(position, "user", item_id),
                timestamp=timestamp,
                message=UserMessage(role="user", content=content),
                tool_use_result=tool_result,
            )
        )

    def _system(
        self,
        *,
        position: int,
        timestamp: t.Timestamp,
        subtype: str,
        content: str,
    ) -> None:
        self._append(
            SystemRecord(
                type="system",
                session_id=self.session_id,
                uuid=self._uuid(position, subtype),
                timestamp=timestamp,
                subtype=subtype,
                content=content,
            )
        )

    def normalize(self) -> list[AssistantRecord | UserRecord | SystemRecord]:
        for position, record in enumerate(self.source_records):
            if not isinstance(record, RolloutBase):
                continue
            if isinstance(record, SessionMetaRecord):
                self.session_id = record.payload.session_id or record.payload.id
                continue
            if isinstance(record, TurnContextRecord):
                self.current_turn_id = record.payload.turn_id or self.current_turn_id
                self.current_model = record.payload.model or self.current_model
                continue
            if isinstance(record, CompactedRecord):
                self._system(
                    position=position,
                    timestamp=record.timestamp,
                    subtype="codex_compacted",
                    content=record.payload.message,
                )
                continue
            if isinstance(record, ResponseItemRecord):
                self._response(position, record)
                continue
            if isinstance(record, EventMessageRecord):
                self._event(position, record)
        return self.records

    def _response(self, position: int, record: ResponseItemRecord) -> None:
        item = record.payload
        turn_id = _message_turn_id(item, self.current_turn_id, position)
        if isinstance(item, MessageResponseItem):
            blocks = _content_blocks(item)
            if item.role == "assistant":
                self._assistant(
                    position=position,
                    timestamp=record.timestamp,
                    turn_id=turn_id,
                    blocks=blocks,
                    item_id=item.id,
                )
            elif item.role == "user":
                self._user(
                    position=position,
                    timestamp=record.timestamp,
                    content=blocks,
                    item_id=item.id,
                )
            else:
                self._system(
                    position=position,
                    timestamp=record.timestamp,
                    subtype=f"codex_{item.role}",
                    content=self._message_text(item),
                )
        elif isinstance(item, AgentMessageResponseItem):
            text = "\n".join(part.text or "" for part in item.content if part.text)
            self._assistant(
                position=position,
                timestamp=record.timestamp,
                turn_id=turn_id,
                blocks=[TextBlock(type="text", text=text)] if text else [],
                item_id=item.id,
            )
        elif isinstance(item, ReasoningResponseItem):
            detail = item.summary or item.content or []
            text = "\n".join(part.text for part in detail)
            self._assistant(
                position=position,
                timestamp=record.timestamp,
                turn_id=turn_id,
                blocks=[ThinkingBlock(type="thinking", thinking=text)] if text else [],
                item_id=item.id,
            )
        elif isinstance(item, FunctionCallResponseItem):
            self._tool_use(
                position,
                record.timestamp,
                turn_id,
                item.call_id,
                item.name,
                item.arguments,
                item.id,
            )
        elif isinstance(item, CustomToolCallResponseItem):
            self._tool_use(
                position, record.timestamp, turn_id, item.call_id, item.name, item.input, item.id
            )
        elif isinstance(item, LocalShellCallResponseItem):
            call_id = item.call_id or item.id or self._uuid(position, "local-shell")
            self._tool_use(
                position,
                record.timestamp,
                turn_id,
                call_id,
                "local_shell",
                item.action,
                item.id,
            )
        elif isinstance(item, ToolSearchCallResponseItem):
            call_id = item.call_id or item.id or self._uuid(position, "tool-search")
            tool_input: JsonValue = {
                "execution": item.execution,
                "arguments": item.arguments,
            }
            self._tool_use(
                position, record.timestamp, turn_id, call_id, "tool_search", tool_input, item.id
            )
        elif isinstance(item, WebSearchCallResponseItem):
            call_id = item.id or self._uuid(position, "web-search")
            self._tool_use(
                position,
                record.timestamp,
                turn_id,
                call_id,
                "web_search",
                item.action,
                item.id,
            )
        elif isinstance(item, ImageGenerationCallResponseItem):
            call_id = item.id or self._uuid(position, "image-generation")
            self._tool_use(
                position,
                record.timestamp,
                turn_id,
                call_id,
                "image_generation",
                {"revised_prompt": item.revised_prompt},
                item.id,
            )
        elif isinstance(item, (FunctionCallOutputResponseItem, CustomToolCallOutputResponseItem)):
            self._tool_result(position, record.timestamp, item.call_id, item.output, item.id)
        elif isinstance(item, ToolSearchOutputResponseItem):
            call_id = item.call_id or item.id or self._uuid(position, "tool-search")
            output: JsonValue = {
                "status": item.status,
                "execution": item.execution,
                "tools": item.tools,
            }
            self._tool_result(position, record.timestamp, call_id, output, item.id)
        elif isinstance(item, (CompactionResponseItem, ContextCompactionResponseItem)):
            self._system(
                position=position,
                timestamp=record.timestamp,
                subtype="codex_compaction",
                content="[encrypted compaction state]",
            )

    def _tool_use(
        self,
        position: int,
        timestamp: t.Timestamp,
        turn_id: str,
        call_id: str,
        name: str,
        raw_input: str | JsonValue,
        item_id: str | None,
    ) -> None:
        tool_input = _json_or_text(raw_input) if isinstance(raw_input, str) else raw_input
        self._assistant(
            position=position,
            timestamp=timestamp,
            turn_id=turn_id,
            blocks=[ToolUseBlock(type="tool_use", id=call_id, name=name, input=tool_input)],
            item_id=item_id,
        )

    def _tool_result(
        self,
        position: int,
        timestamp: t.Timestamp,
        call_id: str,
        output: JsonValue,
        item_id: str | None,
    ) -> None:
        self._user(
            position=position,
            timestamp=timestamp,
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id=call_id,
                    content=_result_text(output),
                )
            ],
            item_id=item_id,
            tool_result=output,
        )

    def _event(self, position: int, record: EventMessageRecord) -> None:
        event = record.payload
        if isinstance(event, TurnStartedEvent):
            self.current_turn_id = event.turn_id
        elif isinstance(event, ThreadSettingsAppliedEvent):
            self.current_model = event.thread_settings.model or self.current_model
        elif isinstance(event, UserMessageEvent):
            if ("user", event.message) not in self.response_text:
                self._user(
                    position=position,
                    timestamp=record.timestamp,
                    content=event.message,
                )
        elif isinstance(event, AgentMessageEvent):
            if ("assistant", event.message) not in self.response_text:
                self._assistant(
                    position=position,
                    timestamp=record.timestamp,
                    turn_id=self.current_turn_id or f"codex-turn-{position}",
                    blocks=[TextBlock(type="text", text=event.message)],
                )
        elif isinstance(event, AgentReasoningEvent):
            self._assistant(
                position=position,
                timestamp=record.timestamp,
                turn_id=self.current_turn_id or f"codex-turn-{position}",
                blocks=[ThinkingBlock(type="thinking", thinking=event.text)],
            )
        elif isinstance(event, RawResponseCompletedEvent) and event.token_usage is not None:
            self._usage_record(
                position,
                record.timestamp,
                event.token_usage,
                request_id=event.response_id,
            )
        elif (
            isinstance(event, TokenCountEvent) and event.info is not None and not self.has_raw_usage
        ):
            delta = _usage_delta(event.info.total_token_usage, self.previous_usage)
            self.previous_usage = event.info.total_token_usage
            if _has_usage(delta):
                self._usage_record(position, record.timestamp, delta)
        elif isinstance(event, ErrorEvent):
            self._system(
                position=position,
                timestamp=record.timestamp,
                subtype="codex_error",
                content=event.message,
            )

    def _usage_record(
        self,
        position: int,
        timestamp: t.Timestamp,
        usage: CodexTokenUsage,
        request_id: str | None = None,
    ) -> None:
        turn_id = self.current_turn_id or f"codex-turn-{position}"
        canonical_usage = _usage_from_codex(usage)
        canonical_request_id = request_id or f"codex-usage-{position}"
        for record in reversed(self.records):
            if (
                isinstance(record, AssistantRecord)
                and record.message.id == turn_id
                and record.request_id is None
            ):
                record.message.usage = canonical_usage
                record.request_id = canonical_request_id
                return
        self._assistant(
            position=position,
            timestamp=timestamp,
            turn_id=turn_id,
            blocks=[],
            usage=canonical_usage,
            request_id=canonical_request_id,
        )


class CodexSession(Session):
    """A Codex rollout presented through the same timeline/tool/cost API as Claude."""

    provider = "codex"

    def __init__(
        self,
        codex_records: list[TranscriptRecord],
        source: Path | None = None,
        errors: list[str] | None = None,
    ) -> None:
        normalizer = _Normalizer(codex_records)
        normalized = normalizer.normalize()
        super().__init__(cast("list[Record]", normalized), source=source, errors=errors)
        self.codex_records = codex_records
        self.session_id = normalizer.session_id

    @classmethod
    def load(cls, path: str | Path, strict: bool = False) -> CodexSession:
        source = Path(path)
        return cls.from_parsed(source, list(iter_transcript_records(source)), strict=strict)

    @classmethod
    def from_parsed(
        cls,
        path: Path,
        items: list[TranscriptRecord | ParseFailure],
        strict: bool = False,
    ) -> CodexSession:
        records: list[TranscriptRecord] = []
        errors: list[str] = []
        for item in items:
            if isinstance(item, ParseFailure):
                message = f"{item.file}:{item.line_number}: {item.error}"
                if strict:
                    raise ValueError(message)
                errors.append(message)
            else:
                records.append(item)
        return cls(records, source=path, errors=errors)

    def info(self) -> SessionInfo:
        info = super().info()
        if self.session_id:
            info.id = self.session_id
        return info

    def record_count(self) -> int:
        return len(self.codex_records)

    def assistant_requests(self) -> list[AssistantRecord]:
        """Return only synthetic usage rows, not zero-usage content projections."""
        return [
            record
            for record in self.records
            if isinstance(record, AssistantRecord) and record.request_id is not None
        ]

    @cached_property
    def _tool_reasons(self) -> dict[str, str]:
        """Bind each call to narration observed before it in the same Codex turn."""
        narration: dict[str, list[str]] = {}
        reasons: dict[str, str] = {}
        for record in self.records:
            if not isinstance(record, AssistantRecord):
                continue
            turn_id = record.message.id
            for block in record.message.content:
                if isinstance(block, TextBlock) and block.text:
                    narration.setdefault(turn_id, []).append(block.text)
                elif isinstance(block, ToolUseBlock):
                    reason = "\n".join(narration.get(turn_id, [])).strip()
                    if reason:
                        reasons[block.id] = reason
        return reasons
