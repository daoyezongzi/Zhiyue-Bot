from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from internal.logger import get_logger


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _energy_tier(energy: float) -> str:
    if energy >= 70.0:
        return "充沛"
    if energy >= 30.0:
        return "一般"
    return "疲惫"


@dataclass(slots=True, frozen=True)
class StatusSnapshot:
    energy: float
    energy_tier: str
    fatigue_mode: bool
    forced_rest: bool
    rest_locked: bool
    last_active_at: datetime
    updated_at: datetime


class StatusEngine:
    def __init__(
        self,
        *,
        initial_energy: float = 65.0,
        heartbeat_interval_sec: int = 600,
        idle_threshold_sec: int = 180,
        recovery_step: float = 8.0,
        reply_cost_per_turn: float = 1.2,
        fatigue_silence_threshold: float = 30.0,
        rest_lock_threshold: float = 10.0,
        rest_unlock_threshold: float = 45.0,
        timezone_offset_hours: int = 8,
        active_start_hour: int = 8,
        active_end_hour: int = 21,
        active_recovery_multiplier: float = 0.9,
        active_reply_cost_multiplier: float = 0.9,
        rest_recovery_multiplier: float = 1.12,
        rest_reply_cost_multiplier: float = 1.12,
        state_file: str = "data/runtime/status_engine.json",
    ) -> None:
        now = datetime.now(timezone.utc)
        self._energy = _clamp(float(initial_energy), 0.0, 100.0)
        self._last_active_at = now
        self._updated_at = now
        self._logger = get_logger("StatusEngine")
        self._state_path = self._resolve_state_path(state_file)

        self._heartbeat_interval_sec = max(60, int(heartbeat_interval_sec))
        self._idle_threshold_sec = max(30, int(idle_threshold_sec))
        self._recovery_step = max(0.5, float(recovery_step))
        self._reply_cost_per_turn = max(0.1, float(reply_cost_per_turn))
        self._fatigue_silence_threshold = self._normalize_threshold(
            fatigue_silence_threshold,
            minimum=1.0,
            maximum=99.0,
            fallback=30.0,
        )
        self._rest_lock_threshold = self._normalize_threshold(
            rest_lock_threshold,
            minimum=0.0,
            maximum=95.0,
            fallback=10.0,
        )
        self._rest_unlock_threshold = self._normalize_threshold(
            rest_unlock_threshold,
            minimum=self._rest_lock_threshold + 1.0,
            maximum=100.0,
            fallback=max(45.0, self._rest_lock_threshold + 1.0),
        )
        self._timezone_offset_hours = self._normalize_timezone_offset_hours(timezone_offset_hours)
        self._active_start_hour, self._active_end_hour = self._normalize_active_window(
            active_start_hour,
            active_end_hour,
        )
        self._active_recovery_multiplier = self._normalize_multiplier(active_recovery_multiplier)
        self._active_reply_cost_multiplier = self._normalize_multiplier(active_reply_cost_multiplier)
        self._rest_recovery_multiplier = self._normalize_multiplier(rest_recovery_multiplier)
        self._rest_reply_cost_multiplier = self._normalize_multiplier(rest_reply_cost_multiplier)
        self._rest_locked = self._energy <= self._rest_lock_threshold

        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._load_state_if_exists()

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
        async with self._lock:
            self._persist_state_locked()

    async def touch(self) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._last_active_at = now
            self._updated_at = now
            self._persist_state_locked()

    async def apply_user_message(self, text: str) -> StatusSnapshot:
        del text
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._updated_at = now
            self._persist_state_locked()
            return self._snapshot_locked()

    async def consume_reply(self, reply: str) -> StatusSnapshot:
        del reply
        async with self._lock:
            now = datetime.now(timezone.utc)
            reply_cost = self._reply_cost_for(now)
            self._energy = _clamp(self._energy - reply_cost, 0.0, 100.0)
            self._last_active_at = now
            self._updated_at = now
            self._refresh_rest_lock_locked()
            self._persist_state_locked()
            return self._snapshot_locked()

    async def apply_reply_policy(self, reply: str) -> tuple[str, bool]:
        clean = reply.strip()
        async with self._lock:
            energy = self._energy
            rest_locked = self._rest_locked

        if rest_locked:
            return "", True
        if energy < self._fatigue_silence_threshold:
            return "", False
        return clean, False

    async def reset(self, *, fill_energy: bool = False) -> StatusSnapshot:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if fill_energy:
                self._energy = 100.0
            self._updated_at = now
            self._refresh_rest_lock_locked()
            self._persist_state_locked()
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
            recovery_step = self._recovery_step_for(now)
            self._energy = _clamp(self._energy + recovery_step, 0.0, 100.0)
            self._refresh_rest_lock_locked()
            self._updated_at = now
            self._persist_state_locked()

    def _recovery_step_for(self, now: datetime) -> float:
        multiplier = (
            self._active_recovery_multiplier
            if self._is_active_period(now)
            else self._rest_recovery_multiplier
        )
        return self._recovery_step * multiplier

    def _reply_cost_for(self, now: datetime) -> float:
        multiplier = (
            self._active_reply_cost_multiplier
            if self._is_active_period(now)
            else self._rest_reply_cost_multiplier
        )
        return self._reply_cost_per_turn * multiplier

    def _is_active_period(self, now: datetime) -> bool:
        local_hour = self._local_hour(now)
        start = self._active_start_hour
        end = self._active_end_hour
        if start == end:
            return True
        if start < end:
            return start <= local_hour < end
        return local_hour >= start or local_hour < end

    def _local_hour(self, now: datetime) -> int:
        utc_now = now.astimezone(timezone.utc)
        local_now = utc_now + timedelta(hours=self._timezone_offset_hours)
        return int(local_now.hour)

    def _snapshot_locked(self) -> StatusSnapshot:
        energy = _clamp(self._energy, 0.0, 100.0)
        return StatusSnapshot(
            energy=energy,
            energy_tier=_energy_tier(energy),
            fatigue_mode=energy < self._fatigue_silence_threshold,
            forced_rest=self._rest_locked,
            rest_locked=self._rest_locked,
            last_active_at=self._last_active_at,
            updated_at=self._updated_at,
        )

    def _load_state_if_exists(self) -> None:
        path = self._state_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._logger.warning("Status restore skipped: failed to parse state file %s", path)
            return
        if not isinstance(payload, dict):
            self._logger.warning("Status restore skipped: state payload is not object: %s", path)
            return

        restored_energy = _clamp(self._to_float(payload.get("energy"), self._energy), 0.0, 100.0)
        restored_last_active = self._parse_datetime(payload.get("last_active_at"), self._last_active_at)
        restored_updated = self._parse_datetime(payload.get("updated_at"), self._updated_at)
        restored_locked_raw = payload.get("rest_locked")
        if isinstance(restored_locked_raw, bool):
            restored_locked = restored_locked_raw
        else:
            restored_locked = restored_energy <= self._rest_lock_threshold

        self._energy = restored_energy
        self._rest_locked = restored_locked
        self._refresh_rest_lock_locked()
        self._last_active_at = restored_last_active
        self._updated_at = restored_updated
        self._logger.info("Status restored: energy=%.1f file=%s", self._energy, path)

    def _persist_state_locked(self) -> None:
        path = self._state_path
        if path is None:
            return
        payload = {
            "energy": round(_clamp(self._energy, 0.0, 100.0), 3),
            "energy_tier": _energy_tier(self._energy),
            "rest_locked": bool(self._rest_locked),
            "fatigue_silence_threshold": round(self._fatigue_silence_threshold, 3),
            "rest_lock_threshold": round(self._rest_lock_threshold, 3),
            "rest_unlock_threshold": round(self._rest_unlock_threshold, 3),
            "last_active_at": self._last_active_at.astimezone(timezone.utc).isoformat(),
            "updated_at": self._updated_at.astimezone(timezone.utc).isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=2),
                encoding="utf-8",
            )
            temp_path.replace(path)
        except Exception:
            self._logger.warning("Status persist failed: file=%s", path, exc_info=True)

    @staticmethod
    def _resolve_state_path(state_file: str) -> Path | None:
        clean = str(state_file or "").strip()
        if not clean:
            return None
        return Path(clean)

    @staticmethod
    def _normalize_timezone_offset_hours(value: Any) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return 8
        return max(-12, min(14, numeric))

    @staticmethod
    def _normalize_hour(value: Any) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, min(23, numeric))

    @classmethod
    def _normalize_active_window(cls, start_hour: Any, end_hour: Any) -> tuple[int, int]:
        return cls._normalize_hour(start_hour), cls._normalize_hour(end_hour)

    @staticmethod
    def _normalize_multiplier(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 1.0
        return _clamp(numeric, 0.1, 3.0)

    @staticmethod
    def _normalize_threshold(value: Any, *, minimum: float, maximum: float, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(fallback)
        return _clamp(numeric, minimum, maximum)

    @staticmethod
    def _to_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    @staticmethod
    def _parse_datetime(value: Any, fallback: datetime) -> datetime:
        raw = str(value or "").strip()
        if not raw:
            return fallback
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _refresh_rest_lock_locked(self) -> None:
        energy = _clamp(self._energy, 0.0, 100.0)
        if self._rest_locked:
            if energy >= self._rest_unlock_threshold:
                self._rest_locked = False
            return
        if energy <= self._rest_lock_threshold:
            self._rest_locked = True
