from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List

from internal.memory.models import MemoryItem, MemoryType, MessageLog


class MemoryManager:
    def __init__(self) -> None:
        self._messages: Dict[int, List[MessageLog]] = defaultdict(list)
        self._memories: Dict[int, List[MemoryItem]] = defaultdict(list)
        self._id = 0
        self._lock = asyncio.Lock()

    async def add_message(self, msg: MessageLog) -> None:
        async with self._lock:
            self._messages[msg.group_id].append(msg)

    async def get_recent_messages(self, group_id: int, limit: int = 15) -> list[MessageLog]:
        async with self._lock:
            return list(self._messages[group_id][-limit:])

    async def save_memory(
        self,
        group_id: int,
        content: str,
        mem_type: MemoryType = "conversation",
        importance: float = 0.5,
    ) -> MemoryItem:
        async with self._lock:
            self._id += 1
            item = MemoryItem(
                id=self._id,
                group_id=group_id,
                mem_type=mem_type,
                content=content,
                importance=importance,
            )
            self._memories[group_id].append(item)
            return item

    async def query_memory(self, group_id: int, query: str, limit: int = 5) -> list[MemoryItem]:
        async with self._lock:
            query_lower = query.lower().strip()
            items = [m for m in self._memories[group_id] if query_lower in m.content.lower()]
            items = items[-limit:]
            for item in items:
                item.access_count += 1
            return items

    async def close(self) -> None:
        return None
