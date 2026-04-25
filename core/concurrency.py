from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class ThinkTask:
    group_id: int
    is_mention: bool


class ConcurrencyManager:
    def __init__(self, max_concurrency: int, handler: Callable[[int, bool], Awaitable[None]]) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.handler = handler
        self.queue: asyncio.Queue[ThinkTask] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._in_queue: set[int] = set()
        self._closed = False

    async def start(self) -> None:
        if self._workers:
            return
        for idx in range(self.max_concurrency):
            task = asyncio.create_task(self._worker(), name=f"think-worker-{idx}")
            self._workers.append(task)

    async def submit(self, group_id: int, is_mention: bool) -> None:
        if self._closed:
            return
        if group_id in self._in_queue:
            return
        self._in_queue.add(group_id)
        await self.queue.put(ThinkTask(group_id=group_id, is_mention=is_mention))

    async def _worker(self) -> None:
        while not self._closed:
            try:
                task = await self.queue.get()
            except asyncio.CancelledError:
                return
            self._in_queue.discard(task.group_id)
            try:
                await self.handler(task.group_id, task.is_mention)
            finally:
                self.queue.task_done()

    async def close(self) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._workers.clear()
