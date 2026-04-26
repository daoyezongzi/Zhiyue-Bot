from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

try:
    import ahocorasick
except ImportError:  # pragma: no cover - runtime fallback
    ahocorasick = None

from adapters.llm.chat import ChatLLMAdapter
from internal.logger import get_logger

JargonScope = Literal["user", "group", "public"]
LearnedCallback = Callable[[str, str], Awaitable[None]]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None

    def _decode(candidate: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
        return None

    direct = _decode(text)
    if direct is not None:
        return direct

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", text, re.IGNORECASE)
    if fenced:
        parsed = _decode(fenced.group(1))
        if parsed is not None:
            return parsed

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        parsed = _decode(text[first_obj : last_obj + 1])
        if parsed is not None:
            return parsed
    return None


@dataclass(slots=True)
class LearnedJargon:
    jargon: str
    standard: str
    meaning: str = ""
    confidence: float = 0.5
    weight: float = 1.0
    source_users: list[int] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)

    def key(self) -> str:
        return f"{self.standard.lower()}\t{self.jargon.lower()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "jargon": self.jargon,
            "standard": self.standard,
            "meaning": self.meaning,
            "confidence": self.confidence,
            "weight": self.weight,
            "source_users": self.source_users,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LearnedJargon | None:
        jargon = str(raw.get("jargon", "") or "").strip()
        standard = str(raw.get("standard", "") or "").strip()
        if not jargon or not standard:
            return None

        meaning = str(raw.get("meaning", "") or "").strip()
        confidence = 0.5
        try:
            confidence = float(raw.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = _clamp(confidence, 0.0, 1.0)

        weight = 1.0
        try:
            weight = float(raw.get("weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        weight = _clamp(weight, 0.05, 100.0)

        source_users: list[int] = []
        raw_users = raw.get("source_users", [])
        if isinstance(raw_users, list):
            for item in raw_users:
                try:
                    uid = int(item)
                except (TypeError, ValueError):
                    continue
                if uid > 0 and uid not in source_users:
                    source_users.append(uid)

        updated_at = str(raw.get("updated_at", "") or "").strip() or _now_iso()
        return cls(
            jargon=jargon,
            standard=standard,
            meaning=meaning,
            confidence=confidence,
            weight=weight,
            source_users=source_users,
            updated_at=updated_at,
        )


class JargonLexiconStore:
    def __init__(self, path: str | Path) -> None:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        self._path = file_path
        self._loaded = False
        self._lock = asyncio.Lock()
        self._spaces: dict[JargonScope, dict[str, LearnedJargon]] = {
            "user": {},
            "group": {},
            "public": {},
        }
        self._logger = get_logger("JargonLexiconStore")

    async def load(self) -> None:
        async with self._lock:
            if self._loaded:
                return

            self._spaces = {
                "user": {},
                "group": {},
                "public": {},
            }
            if not self._path.exists():
                self._loaded = True
                return

            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._logger.exception("Failed to load jargon lexicon: %s", self._path)
                self._loaded = True
                return

            raw_spaces = payload.get("spaces", {})
            if isinstance(raw_spaces, dict):
                for scope in ("user", "group", "public"):
                    raw_rows = raw_spaces.get(scope, {})
                    if not isinstance(raw_rows, dict):
                        continue
                    parsed: dict[str, LearnedJargon] = {}
                    for key, value in raw_rows.items():
                        if not isinstance(value, dict):
                            continue
                        row = LearnedJargon.from_dict(value)
                        if row is not None:
                            parsed[str(key)] = row
                    self._spaces[scope] = parsed
            self._loaded = True

    async def get_entries(self, scopes: tuple[JargonScope, ...] = ("user", "group", "public")) -> list[LearnedJargon]:
        await self.load()
        async with self._lock:
            rows: list[LearnedJargon] = []
            for scope in scopes:
                rows.extend(self._spaces.get(scope, {}).values())
            return list(rows)

    async def merge_entries(
        self,
        *,
        scope: JargonScope,
        entries: list[LearnedJargon],
        max_entries: int = 1000,
    ) -> None:
        await self.load()
        async with self._lock:
            bucket = self._spaces.setdefault(scope, {})
            for row in entries:
                key = row.key()
                existing = bucket.get(key)
                if existing is None:
                    row.updated_at = _now_iso()
                    bucket[key] = row
                    continue

                if row.meaning and (not existing.meaning or len(row.meaning) >= len(existing.meaning)):
                    existing.meaning = row.meaning
                existing.confidence = _clamp((existing.confidence * 0.7) + (row.confidence * 0.3), 0.0, 1.0)
                existing.weight = _clamp((existing.weight * 0.85) + (row.weight * 0.35), 0.05, 100.0)
                existing.updated_at = _now_iso()
                for uid in row.source_users:
                    if uid > 0 and uid not in existing.source_users:
                        existing.source_users.append(uid)

            if len(bucket) > max_entries:
                ranked = sorted(bucket.items(), key=lambda item: item[1].weight, reverse=True)
                bucket.clear()
                for key, value in ranked[:max_entries]:
                    bucket[key] = value

            await self._persist_locked()

    async def _persist_locked(self) -> None:
        payload = {
            "version": 1,
            "spaces": {
                scope: {key: row.to_dict() for key, row in bucket.items()} for scope, bucket in self._spaces.items()
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_suffix(self._path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self._path)


class JargonEvolutionEngine:
    def __init__(
        self,
        *,
        llm: ChatLLMAdapter,
        store: JargonLexiconStore,
        conversion_rate: float = 0.35,
        trigger_message_count: int = 20,
        context_limit: int = 40,
        enabled: bool = True,
        on_learned: LearnedCallback | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._enabled = enabled
        self._conversion_rate = _clamp(conversion_rate, 0.0, 1.0)
        self._trigger = max(1, trigger_message_count)
        self._context_limit = max(self._trigger, context_limit)
        self._on_learned = on_learned

        self._lock = asyncio.Lock()
        self._buffers: dict[int, list[dict[str, Any]]] = {}
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._needs_rerun: set[int] = set()

        self._automaton: Any | None = None
        self._pattern_lowers: list[str] = []
        self._pattern_display: list[str] = []
        self._entries_by_standard: dict[str, list[LearnedJargon]] = {}

        self._rng = random.Random()
        self._logger = get_logger("JargonEvolutionEngine")

    async def start(self) -> None:
        await self._store.load()
        await self.reload_automaton()

    async def stop(self) -> None:
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def observe_user_message(
        self,
        *,
        user_id: int | None,
        speaker: str,
        text: str,
        session_id: str,
        group_id: int | None,
    ) -> None:
        if not self._enabled:
            return

        uid = int(user_id or 0)
        content = str(text).strip()
        if uid <= 0 or not content:
            return

        event = {
            "speaker": speaker,
            "text": content,
            "session_id": session_id,
            "group_id": group_id,
            "at": _now_iso(),
        }

        async with self._lock:
            buffer = self._buffers.setdefault(uid, [])
            buffer.append(event)
            if len(buffer) > (self._context_limit * 4):
                del buffer[: len(buffer) - (self._context_limit * 4)]

            if len(buffer) < self._trigger:
                return

            if uid in self._tasks:
                self._needs_rerun.add(uid)
                return

            self._start_task_locked(uid)

    async def reload_automaton(self) -> None:
        rows = await self._store.get_entries()
        entries_by_standard: dict[str, list[LearnedJargon]] = {}
        for row in rows:
            standard = row.standard.strip()
            jargon = row.jargon.strip()
            if not standard or not jargon:
                continue
            entries_by_standard.setdefault(standard.lower(), []).append(row)

        automaton: Any | None
        pattern_lowers: list[str] = []
        pattern_display: list[str] = []
        if ahocorasick is not None:
            automaton = ahocorasick.Automaton()
            for lowered, bucket in entries_by_standard.items():
                idx = len(pattern_lowers)
                automaton.add_word(lowered, idx)
                pattern_lowers.append(lowered)
                pattern_display.append(bucket[0].standard)
            if pattern_lowers:
                automaton.make_automaton()
            else:
                automaton = None
        else:
            automaton = None
            for lowered, bucket in entries_by_standard.items():
                pattern_lowers.append(lowered)
                pattern_display.append(bucket[0].standard)

        async with self._lock:
            self._automaton = automaton
            self._pattern_lowers = pattern_lowers
            self._pattern_display = pattern_display
            self._entries_by_standard = entries_by_standard

    async def apply_to_reply(self, text: str) -> str:
        content = str(text).strip()
        if not content:
            return ""

        async with self._lock:
            automaton = self._automaton
            pattern_lowers = list(self._pattern_lowers)
            pattern_display = list(self._pattern_display)
            entries_by_standard = {key: list(value) for key, value in self._entries_by_standard.items()}
            conversion_rate = self._conversion_rate

        if not pattern_lowers:
            return content

        lowered = content.lower()
        matches: list[tuple[int, int, int]] = []
        if automaton is not None:
            for end_idx, pattern_idx in automaton.iter(lowered):
                if pattern_idx < 0 or pattern_idx >= len(pattern_lowers):
                    continue
                token = pattern_lowers[pattern_idx]
                start_idx = end_idx - len(token) + 1
                if start_idx >= 0:
                    matches.append((start_idx, end_idx, pattern_idx))
        else:
            for idx, token in enumerate(pattern_lowers):
                start = 0
                while True:
                    found = lowered.find(token, start)
                    if found < 0:
                        break
                    matches.append((found, found + len(token) - 1, idx))
                    start = found + 1

        if not matches:
            return content

        picks = self._leftmost_longest(matches)
        if not picks:
            return content

        chunks: list[str] = []
        cursor = 0
        for start_idx, end_idx, pattern_idx in picks:
            if start_idx < cursor:
                continue
            chunks.append(content[cursor:start_idx])

            original = content[start_idx : end_idx + 1]
            standard_key = pattern_lowers[pattern_idx]
            fallback_value = pattern_display[pattern_idx]
            candidates = entries_by_standard.get(standard_key, [])

            replacement = original
            if candidates and conversion_rate > 0 and self._rng.random() <= conversion_rate:
                chosen = self._pick_weighted(candidates)
                if chosen is not None and chosen.jargon.strip():
                    replacement = chosen.jargon.strip()
            if not replacement:
                replacement = fallback_value or original

            chunks.append(replacement)
            cursor = end_idx + 1

        chunks.append(content[cursor:])
        return "".join(chunks).strip()

    def _start_task_locked(self, user_id: int) -> None:
        task = asyncio.create_task(self._run_learn(user_id), name=f"jargon-learn-{user_id}")
        self._tasks[user_id] = task
        task.add_done_callback(lambda done: asyncio.create_task(self._on_task_done(user_id, done)))

    async def _on_task_done(self, user_id: int, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            self._logger.exception("Jargon learning task failed: user=%s", user_id)

        async with self._lock:
            self._tasks.pop(user_id, None)
            should_rerun = user_id in self._needs_rerun
            if should_rerun:
                self._needs_rerun.discard(user_id)
            buffer_len = len(self._buffers.get(user_id, []))
            if should_rerun and buffer_len >= self._trigger:
                self._start_task_locked(user_id)

    async def _run_learn(self, user_id: int) -> None:
        async with self._lock:
            rows = list(self._buffers.get(user_id, []))
        if len(rows) < self._trigger:
            return

        clip = rows[-self._context_limit :]
        transcript = "\n".join(
            f"{row['speaker']}: {row['text']}" for row in clip if str(row.get("text", "")).strip()
        )
        if not transcript:
            return

        prompt = (
            "You are a community slang extractor. Identify non-standard expressions used by users "
            "(slang, memes, catchphrases). Output JSON only.\n\n"
            "Output format:\n"
            "{\n"
            '  "items": [\n'
            '    {"jargon":"slang phrase", "standard":"standard phrase", "meaning":"short meaning", "confidence":0.0}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1. Only output terms that are clearly supported by context.\n"
            "2. standard must be a phrase that can be directly used in normal sentences.\n"
            "3. Return at most 8 items.\n"
            "4. If evidence is weak, return items=[].\n\n"
            f"Transcript:\n{transcript}"
        )
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        raw = await self._llm.generate_from_messages(messages, extra_fields={"temperature": 0.2})
        payload = _extract_json_object(raw)
        if payload is None:
            self._logger.warning("Jargon LLM result is not JSON: user=%s", user_id)
            return

        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []

        learned: list[LearnedJargon] = []
        for item in raw_items[:12]:
            if not isinstance(item, dict):
                continue
            jargon = str(item.get("jargon", "") or "").strip()
            standard = str(item.get("standard", "") or "").strip()
            meaning = str(item.get("meaning", "") or "").strip()
            if not jargon or not standard:
                continue
            if jargon.lower() == standard.lower():
                continue

            confidence = 0.5
            try:
                confidence = float(item.get("confidence", 0.5) or 0.5)
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = _clamp(confidence, 0.0, 1.0)

            learned.append(
                LearnedJargon(
                    jargon=jargon[:40],
                    standard=standard[:40],
                    meaning=meaning[:120],
                    confidence=confidence,
                    weight=1.0 + confidence,
                    source_users=[user_id],
                    updated_at=_now_iso(),
                )
            )

        if not learned:
            await self._trim_buffer(user_id)
            return

        await self._store.merge_entries(scope="user", entries=learned)
        await self.reload_automaton()
        await self._trim_buffer(user_id)

        if self._on_learned is not None:
            for row in learned:
                try:
                    await self._on_learned(row.jargon, row.meaning or row.standard)
                except Exception:
                    self._logger.exception("Failed to sync learned jargon: %s", row.jargon)

    async def _trim_buffer(self, user_id: int) -> None:
        keep = max(8, self._trigger // 2)
        async with self._lock:
            rows = self._buffers.get(user_id, [])
            if len(rows) > keep:
                self._buffers[user_id] = rows[-keep:]

    @staticmethod
    def _leftmost_longest(matches: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if not matches:
            return []

        ordered = sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        picked: list[tuple[int, int, int]] = []
        cursor = 0
        idx = 0
        while idx < len(ordered):
            while idx < len(ordered) and ordered[idx][1] < cursor:
                idx += 1
            if idx >= len(ordered):
                break

            probe = idx
            while probe < len(ordered) and ordered[probe][0] < cursor:
                probe += 1
            if probe >= len(ordered):
                break

            start = ordered[probe][0]
            best = ordered[probe]
            probe += 1
            while probe < len(ordered) and ordered[probe][0] == start:
                current = ordered[probe]
                if (current[1] - current[0]) > (best[1] - best[0]):
                    best = current
                probe += 1

            picked.append(best)
            cursor = best[1] + 1
            idx = probe

        return picked

    def _pick_weighted(self, rows: list[LearnedJargon]) -> LearnedJargon | None:
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]

        total = 0.0
        for row in rows:
            total += max(0.05, row.weight)
        if total <= 0:
            return rows[0]

        point = self._rng.random() * total
        cursor = 0.0
        for row in rows:
            cursor += max(0.05, row.weight)
            if point <= cursor:
                return row
        return rows[-1]
