from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


MemoryType = Literal["group_fact", "self_experience", "conversation"]
CanonicalMemoryType = Literal["fact", "episode", "preference", "constraint", "goal"]
MemoryStatus = Literal["active", "candidate", "archived", "legacy"]
MemorySourceKind = Literal["message", "summary", "topic", "manual", "migration"]

MEMORY_STATUS_ACTIVE: MemoryStatus = "active"
MEMORY_STATUS_CANDIDATE: MemoryStatus = "candidate"
MEMORY_STATUS_ARCHIVED: MemoryStatus = "archived"
MEMORY_STATUS_LEGACY: MemoryStatus = "legacy"

KEYED_CANONICAL_TYPES: tuple[CanonicalMemoryType, ...] = (
    "fact",
    "preference",
    "constraint",
    "goal",
)


def is_recall_eligible_status(status: str) -> bool:
    clean = str(status or "").strip().lower()
    return clean in {MEMORY_STATUS_ACTIVE, MEMORY_STATUS_LEGACY}


@dataclass
class MessageLog:
    message_id: int
    group_id: int
    user_id: int
    nickname: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MemoryItem:
    id: int
    group_id: int
    mem_type: MemoryType
    content: str
    canonical_type: CanonicalMemoryType = "fact"
    status: MemoryStatus = MEMORY_STATUS_CANDIDATE
    evidence_count: int = 1
    source_kind: MemorySourceKind = "message"
    source_ref: str = ""
    fact_key: str = ""
    user_id: int = 0
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def effective_status(self) -> MemoryStatus:
        clean = str(self.status or "").strip().lower()
        if clean in {
            MEMORY_STATUS_ACTIVE,
            MEMORY_STATUS_CANDIDATE,
            MEMORY_STATUS_ARCHIVED,
            MEMORY_STATUS_LEGACY,
        }:
            return clean  # type: ignore[return-value]
        return MEMORY_STATUS_LEGACY

    def recall_eligible(self) -> bool:
        return is_recall_eligible_status(self.effective_status())

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "group_id": int(self.group_id),
            "mem_type": str(self.mem_type),
            "content": str(self.content),
            "canonical_type": str(self.canonical_type),
            "status": str(self.status),
            "evidence_count": int(self.evidence_count),
            "source_kind": str(self.source_kind),
            "source_ref": str(self.source_ref),
            "fact_key": str(self.fact_key),
            "user_id": int(self.user_id or 0),
            "importance": float(self.importance),
            "access_count": int(self.access_count),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, row: dict) -> "MemoryItem | None":
        if not isinstance(row, dict):
            return None
        try:
            memory_id = int(row.get("id", 0))
            group_id = int(row.get("group_id", 0))
        except (TypeError, ValueError):
            return None
        if memory_id <= 0 or group_id < 0:
            return None

        created_at = _parse_time(row.get("created_at"))
        updated_at = _parse_time(row.get("updated_at")) or created_at
        if created_at is None:
            created_at = datetime.utcnow()
        if updated_at is None:
            updated_at = created_at

        canonical_type = str(row.get("canonical_type", "fact") or "fact").strip().lower()
        if canonical_type not in {"fact", "episode", "preference", "constraint", "goal"}:
            canonical_type = "fact"
        status = str(row.get("status", MEMORY_STATUS_CANDIDATE) or MEMORY_STATUS_CANDIDATE).strip().lower()
        if status not in {
            MEMORY_STATUS_ACTIVE,
            MEMORY_STATUS_CANDIDATE,
            MEMORY_STATUS_ARCHIVED,
            MEMORY_STATUS_LEGACY,
        }:
            status = MEMORY_STATUS_CANDIDATE
        mem_type = str(row.get("mem_type", "conversation") or "conversation").strip().lower()
        if mem_type not in {"group_fact", "self_experience", "conversation"}:
            mem_type = "conversation"
        source_kind = str(row.get("source_kind", "message") or "message").strip().lower()
        if source_kind not in {"message", "summary", "topic", "manual", "migration"}:
            source_kind = "message"

        try:
            evidence_count = max(1, int(row.get("evidence_count", 1)))
        except (TypeError, ValueError):
            evidence_count = 1
        try:
            importance = float(row.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        try:
            access_count = max(0, int(row.get("access_count", 0)))
        except (TypeError, ValueError):
            access_count = 0
        try:
            user_id = int(row.get("user_id", 0) or 0)
        except (TypeError, ValueError):
            user_id = 0

        return cls(
            id=memory_id,
            group_id=group_id,
            mem_type=mem_type,  # type: ignore[arg-type]
            content=str(row.get("content", "") or "").strip(),
            canonical_type=canonical_type,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            evidence_count=evidence_count,
            source_kind=source_kind,  # type: ignore[arg-type]
            source_ref=str(row.get("source_ref", "") or "").strip(),
            fact_key=str(row.get("fact_key", "") or "").strip(),
            user_id=user_id,
            importance=importance,
            access_count=access_count,
            created_at=created_at,
            updated_at=updated_at,
        )


def _parse_time(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _coerce_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 6:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = _coerce_json_value(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_coerce_json_value(item, depth=depth + 1) for item in list(value)]
    return str(value)


@dataclass
class ToolCallLog:
    id: int
    session_id: str
    message_type: str
    group_id: int
    user_id: int
    speaker: str
    step: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    result: Any = None
    error: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "session_id": str(self.session_id or "").strip(),
            "message_type": str(self.message_type or "").strip(),
            "group_id": int(self.group_id or 0),
            "user_id": int(self.user_id or 0),
            "speaker": str(self.speaker or "").strip(),
            "step": int(self.step or 0),
            "tool_call_id": str(self.tool_call_id or "").strip(),
            "tool_name": str(self.tool_name or "").strip(),
            "arguments": _coerce_json_value(self.arguments if isinstance(self.arguments, dict) else {}),
            "success": bool(self.success),
            "result": _coerce_json_value(self.result),
            "error": str(self.error or ""),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ToolCallLog | None:
        if not isinstance(row, dict):
            return None
        try:
            row_id = int(row.get("id", 0) or 0)
        except (TypeError, ValueError):
            return None
        if row_id <= 0:
            return None
        try:
            group_id = int(row.get("group_id", 0) or 0)
        except (TypeError, ValueError):
            group_id = 0
        try:
            user_id = int(row.get("user_id", 0) or 0)
        except (TypeError, ValueError):
            user_id = 0
        try:
            step = int(row.get("step", 0) or 0)
        except (TypeError, ValueError):
            step = 0
        arguments = row.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        created_at = _parse_time(row.get("created_at")) or datetime.utcnow()
        return cls(
            id=row_id,
            session_id=str(row.get("session_id", "") or "").strip(),
            message_type=str(row.get("message_type", "") or "").strip(),
            group_id=group_id,
            user_id=user_id,
            speaker=str(row.get("speaker", "") or "").strip(),
            step=step,
            tool_call_id=str(row.get("tool_call_id", "") or "").strip(),
            tool_name=str(row.get("tool_name", "") or "").strip(),
            arguments=_coerce_json_value(arguments) if isinstance(arguments, dict) else {},
            success=bool(row.get("success", False)),
            result=_coerce_json_value(row.get("result")),
            error=str(row.get("error", "") or ""),
            created_at=created_at,
        )
