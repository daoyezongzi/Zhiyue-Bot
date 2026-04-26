from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True, frozen=True)
class StatusSnapshot:
    energy: float
    mood: float
    fatigue_mode: bool
    forced_rest: bool
    last_active_at: datetime
    updated_at: datetime


class StatusEngine:
    def __init__(
        self,
        *,
        initial_energy: float = 65.0,
        initial_mood: float = 50.0,
        heartbeat_interval_sec: int = 600,
        idle_threshold_sec: int = 180,
        recovery_step: float = 8.0,
        mood_recovery_step: float = 2.0,
    ) -> None:
        now = datetime.now(timezone.utc)
        self._energy = _clamp(float(initial_energy), 0.0, 100.0)
        self._mood = _clamp(float(initial_mood), 0.0, 100.0)
        self._last_active_at = now
        self._updated_at = now

        self._heartbeat_interval_sec = max(60, int(heartbeat_interval_sec))
        self._idle_threshold_sec = max(30, int(idle_threshold_sec))
        self._recovery_step = max(1.0, float(recovery_step))
        self._mood_recovery_step = max(0.5, float(mood_recovery_step))

        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._heartbeat_task is not None:
            return
        self._stop_event.clear()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="status-heartbeat")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def touch(self) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._last_active_at = now
            self._updated_at = now

    async def apply_user_message(self, text: str) -> StatusSnapshot:
        delta = self._estimate_sentiment_delta(text)
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._mood = _clamp(self._mood + delta, 0.0, 100.0)
            self._last_active_at = now
            self._updated_at = now
            return self._snapshot_locked()

    async def consume_reply(self, reply: str) -> StatusSnapshot:
        clean = reply.strip()
        length = len(clean)
        base_cost = 2.0 + float(length // 45)
        if length >= 220:
            base_cost += 2.0

        async with self._lock:
            now = datetime.now(timezone.utc)
            self._energy = _clamp(self._energy - base_cost, 0.0, 100.0)
            self._mood = _clamp(self._mood - min(3.0, base_cost * 0.25), 0.0, 100.0)
            self._last_active_at = now
            self._updated_at = now
            return self._snapshot_locked()

    async def apply_reply_policy(self, reply: str) -> tuple[str, bool]:
        clean = reply.strip()
        async with self._lock:
            energy = self._energy

        if energy >= 10.0:
            return clean, False

        if energy < 4.0:
            return "先不聊了，我要休息一会。", True

        clipped = self._clip_reply(clean, limit=30)
        if not clipped:
            return "有点累，先这样。", False
        if not clipped.endswith(("。", "！", "？", "!", "?")):
            clipped += "。"
        return f"{clipped}先这样。", False

    async def reset(self, *, fill_energy: bool = False, reset_mood: bool = False) -> StatusSnapshot:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if fill_energy:
                self._energy = 100.0
            if reset_mood:
                self._mood = 50.0
            self._updated_at = now
            return self._snapshot_locked()

    async def get_snapshot(self) -> StatusSnapshot:
        async with self._lock:
            return self._snapshot_locked()

    async def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._heartbeat_interval_sec)
                return
            except asyncio.TimeoutError:
                pass

            await self._recover_if_idle()

    async def _recover_if_idle(self) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            idle_seconds = (now - self._last_active_at).total_seconds()
            if idle_seconds < float(self._idle_threshold_sec):
                return

            self._energy = _clamp(self._energy + self._recovery_step, 0.0, 100.0)
            if self._mood < 50.0:
                self._mood = _clamp(self._mood + self._mood_recovery_step, 0.0, 100.0)
            elif self._mood > 50.0:
                self._mood = _clamp(self._mood - self._mood_recovery_step, 0.0, 100.0)
            self._updated_at = now

    def _snapshot_locked(self) -> StatusSnapshot:
        energy = _clamp(self._energy, 0.0, 100.0)
        mood = _clamp(self._mood, 0.0, 100.0)
        return StatusSnapshot(
            energy=energy,
            mood=mood,
            fatigue_mode=energy < 10.0,
            forced_rest=energy < 4.0,
            last_active_at=self._last_active_at,
            updated_at=self._updated_at,
        )

    @staticmethod
    def _estimate_sentiment_delta(text: str) -> float:
        clean = text.strip().lower()
        if not clean:
            return 0.0

        positive_tokens = (
            "开心",
            "喜欢",
            "谢谢",
            "棒",
            "赞",
            "牛",
            "太好了",
            "爱你",
            "舒服",
            "哈哈",
        )
        negative_tokens = (
            "烦",
            "讨厌",
            "生气",
            "难过",
            "崩溃",
            "垃圾",
            "无语",
            "累",
            "困",
            "糟糕",
            "痛苦",
            "滚",
        )

        positive_hits = sum(1 for token in positive_tokens if token in clean)
        negative_hits = sum(1 for token in negative_tokens if token in clean)
        raw = float(positive_hits - negative_hits)
        if raw == 0.0:
            return 0.0
        return _clamp(raw * 4.5, -14.0, 14.0)

    @staticmethod
    def _clip_reply(text: str, *, limit: int) -> str:
        if not text:
            return ""
        pieces = [part.strip() for part in re.split(r"[。！？!?；;\n]", text) if part.strip()]
        if not pieces:
            return text[:limit].strip()
        first = pieces[0]
        if len(first) <= limit:
            return first
        return first[:limit].strip()
