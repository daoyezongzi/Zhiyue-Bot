from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True, frozen=True)
class LogEvent:
    source: str
    message: str
    timestamp: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
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
        timestamp: str | None = None,
    ) -> None:
        clean_message = str(message).rstrip()
        if not clean_message:
            return

        event = LogEvent(
            source=str(source or "bot"),
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
            return {
                "source": str(payload.get("source", "bot")),
                "message": str(payload.get("message", "")),
                "timestamp": str(payload.get("timestamp", datetime.now(timezone.utc).isoformat())),
            }
        return {
            "source": "bot",
            "message": str(payload),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
