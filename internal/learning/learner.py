from __future__ import annotations

import asyncio
import logging

from internal.config.schema import LearningConfig


class Learner:
    def __init__(self, cfg: LearningConfig) -> None:
        self._cfg = cfg
        self._task: asyncio.Task[None] | None = None
        self._running = asyncio.Event()
        self._logger = logging.getLogger("Learner")

    async def start(self) -> None:
        if self._task or not self._cfg.enabled:
            return
        self._running.set()
        self._task = asyncio.create_task(self._run_loop(), name="learner-loop")

    async def stop(self) -> None:
        self._running.clear()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        interval = max(self._cfg.interval_minutes, 1) * 60
        while self._running.is_set():
            self._logger.debug("learning tick")
            await asyncio.sleep(interval)
