from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str
    message_type: str
    speaker: str
    user_id: int | None
    is_master: bool
    created_at: datetime


@dataclass(slots=True)
class SessionSummary:
    text: str
    updated_at: datetime


@dataclass(slots=True)
class _SessionBucket:
    messages: Deque[ConversationMessage]
    dropped_count: int = 0
    total_messages: int = 0
    summary: SessionSummary | None = None


class MessageHistory:
    def __init__(self, context_window_size: int, summary_trigger_dropped: int = 6) -> None:
        self.context_window_size = max(1, context_window_size)
        # one round usually has user + assistant messages
        self._capacity = self.context_window_size * 2
        self._summary_trigger_dropped = max(1, summary_trigger_dropped)
        self._sessions: dict[str, _SessionBucket] = {}

    def append(self, session_id: str, message: ConversationMessage) -> None:
        bucket = self._bucket(session_id)
        if len(bucket.messages) >= self._capacity:
            bucket.dropped_count += 1
        bucket.messages.append(message)
        bucket.total_messages += 1

    def get_recent(self, session_id: str, limit: int | None = None) -> list[ConversationMessage]:
        bucket = self._sessions.get(session_id)
        if bucket is None:
            return []
        rows = list(bucket.messages)
        if limit is None or limit <= 0:
            return rows
        return rows[-limit:]

    def get_structured_messages(self, session_id: str) -> list[dict[str, str]]:
        rows = self.get_recent(session_id)
        if not rows:
            return []

        messages: list[dict[str, str]] = []
        for item in rows:
            tags: list[str] = []
            tags.append("GROUP" if item.message_type == "group" else "PRIVATE")
            if item.user_id is not None:
                tags.append(f"uid={item.user_id}")
            if item.is_master:
                tags.append("MASTER")
            tag_text = "][".join(tags)
            timestamp = item.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            content = f"[{tag_text}] ({timestamp}) {item.speaker}: {item.content}"
            messages.append({"role": item.role, "content": content})
        return messages

    def get_summary(self, session_id: str) -> SessionSummary | None:
        bucket = self._sessions.get(session_id)
        if bucket is None:
            return None
        return bucket.summary

    def set_summary(self, session_id: str, text: str, *, updated_at: datetime) -> None:
        bucket = self._bucket(session_id)
        bucket.summary = SessionSummary(text=text.strip(), updated_at=updated_at)
        bucket.dropped_count = 0

    def should_refresh_summary(self, session_id: str) -> bool:
        bucket = self._sessions.get(session_id)
        if bucket is None:
            return False
        if bucket.summary is None:
            return bucket.dropped_count >= self._summary_trigger_dropped
        return bucket.dropped_count >= self._summary_trigger_dropped

    def summary_prompt(self, session_id: str) -> str:
        summary = self.get_summary(session_id)
        if summary is None or not summary.text:
            return ""
        return f"历史总结（{summary.updated_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}）：{summary.text}"

    def export_session(self, session_id: str) -> dict:
        bucket = self._sessions.get(session_id)
        if bucket is None:
            return {"session_id": session_id, "messages": [], "summary": None, "dropped_count": 0}

        payload_messages = [
            {
                "role": item.role,
                "content": item.content,
                "message_type": item.message_type,
                "speaker": item.speaker,
                "user_id": item.user_id,
                "is_master": item.is_master,
                "created_at": item.created_at.isoformat(),
            }
            for item in bucket.messages
        ]
        payload_summary = None
        if bucket.summary:
            payload_summary = {
                "text": bucket.summary.text,
                "updated_at": bucket.summary.updated_at.isoformat(),
            }

        return {
            "session_id": session_id,
            "messages": payload_messages,
            "summary": payload_summary,
            "dropped_count": bucket.dropped_count,
            "total_messages": bucket.total_messages,
        }

    def _bucket(self, session_id: str) -> _SessionBucket:
        bucket = self._sessions.get(session_id)
        if bucket is not None:
            return bucket
        new_bucket = _SessionBucket(messages=deque(maxlen=self._capacity))
        self._sessions[session_id] = new_bucket
        return new_bucket
