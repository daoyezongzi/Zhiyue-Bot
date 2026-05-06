from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True, frozen=True)
class LogEvent:
    source: str
    channel: str
    message: str
    timestamp: str

    def as_dict(self) -> dict[str, str]:
        event_type = "action" if self.channel == "action" else "system"
        return {
            "source": self.source,
            "channel": self.channel,
            "type": event_type,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class LogStreamHub:
    def __init__(self, backlog_size: int = 200) -> None:
        self._backlog: deque[LogEvent] = deque(maxlen=max(1, int(backlog_size)))
        self._subscribers: set[asyncio.Queue[LogEvent]] = set()
        self._lock = asyncio.Lock()

    async def publish(
        self,
        source: str,
        message: str,
        *,
        channel: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        clean_message = str(message).rstrip()
        if not clean_message:
            return
        clean_source = str(source or "system")
        clean_channel = self._normalize_channel(channel) or self._infer_channel(clean_source, clean_message)

        event = LogEvent(
            source=clean_source,
            channel=clean_channel,
            message=clean_message,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )

        async with self._lock:
            self._backlog.append(event)
            subscribers = list(self._subscribers)

        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    async def subscribe(
        self,
        *,
        with_backlog: bool = True,
        queue_size: int = 500,
    ) -> asyncio.Queue[LogEvent]:
        queue: asyncio.Queue[LogEvent] = asyncio.Queue(maxsize=max(1, int(queue_size)))
        async with self._lock:
            self._subscribers.add(queue)
            backlog = list(self._backlog) if with_backlog else []
        for event in backlog:
            if queue.full():
                break
            queue.put_nowait(event)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[LogEvent]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    def has_subscribers(self) -> bool:
        return bool(self._subscribers)

    @staticmethod
    def normalize_event(payload: Any) -> dict[str, str]:
        if isinstance(payload, LogEvent):
            return payload.as_dict()
        if isinstance(payload, dict):
            source = str(payload.get("source", "system"))
            message = str(payload.get("message", ""))
            channel = LogStreamHub._normalize_channel(payload.get("channel"))
            if not channel:
                channel = LogStreamHub._infer_channel(source, message)
            event_type = str(payload.get("type", "")).strip().lower()
            if event_type not in {"system", "action"}:
                event_type = "action" if channel == "action" else "system"
            return {
                "source": source,
                "channel": channel,
                "type": event_type,
                "message": message,
                "timestamp": str(payload.get("timestamp", datetime.now(timezone.utc).isoformat())),
            }
        return {
            "source": "system",
            "channel": "system",
            "type": "system",
            "message": str(payload),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _normalize_channel(raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value in {"system", "action", "napcat"}:
            return value
        return ""

    @staticmethod
    def _infer_channel(source: str, message: str) -> str:
        source_l = str(source or "").strip().lower()
        message_l = str(message or "").strip().lower()
        if source_l == "napcat":
            return "napcat"
        if source_l in {"action", "plugin", "agent"}:
            return "action"
        action_tokens = (
            "rx post_type",
            "tx action",
            "message_type=",
            "post_type=",
            "queue.dispatch",
            "queue.reply",
            "queue.skipreply",
            "queue.lowenergyreply",
            "queue.silence",
            "queue.forcereply",
            "sendchain.",
            "replystatus",
            "llm.think",
            "llm.reply",
            "sticker.reply",
            "[plugins]",
            "[groups]",
            "[reset]",
            "plugin",
        )
        if any(token in message_l for token in action_tokens):
            return "action"
        return "system"
