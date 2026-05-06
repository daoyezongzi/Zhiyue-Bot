from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from adapters.llm.chat import ChatLLMAdapter
from adapters.llm.embedding import EmbeddingAdapter
from internal.config.schema import Config
from internal.logger import get_logger

TOPIC_ACTIVE = "active"
TOPIC_ARCHIVED = "archived"


class TopicManager:
    def __init__(self, *, cfg: Config, llm: ChatLLMAdapter) -> None:
        self.cfg = cfg
        self.llm = llm
        self._logger = get_logger("TopicManager")

        self._enabled = bool(getattr(cfg.memory, "topic_enabled", True))
        project_root = Path(__file__).resolve().parents[2]
        store_path = str(getattr(cfg.memory, "topic_store_path", "data/topics/topic_threads.json") or "")
        self._store_path = Path(store_path)
        if not self._store_path.is_absolute():
            self._store_path = project_root / self._store_path

        self._max_active = max(1, int(getattr(cfg.memory, "topic_max_active_per_group", 5)))
        self._summary_trigger = max(1, int(getattr(cfg.memory, "topic_summary_trigger_messages", 10)))
        self._archive_inactive_minutes = max(1, int(getattr(cfg.memory, "topic_archive_inactive_minutes", 180)))
        self._reuse_threshold = float(getattr(cfg.memory, "topic_reuse_threshold", 0.42))
        self._recall_top_k = max(1, int(getattr(cfg.memory, "topic_recall_top_k", 3)))
        self._tail_size = max(20, int(getattr(cfg.memory, "topic_message_tail_size", 80)))

        dim = min(max(int(getattr(cfg.memory, "vector_dim", 256) or 256), 128), 2048)
        self._embedding = EmbeddingAdapter(cfg.embedding, target_dim=dim)
        self._summary_vec_cache: dict[int, list[float]] = {}

        self._lock = asyncio.Lock()
        self._started = False
        self._next_topic_id = 1
        self._topics: dict[int, dict[str, Any]] = {}
        self._group_topics: dict[int, list[int]] = defaultdict(list)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self._enabled:
            return
        await self._load()
        self._logger.info(
            "Topic manager started: enabled=%s store=%s active_limit=%s summary_trigger=%s",
            self._enabled,
            self._store_path,
            self._max_active,
            self._summary_trigger,
        )

    async def close(self) -> None:
        if not self._enabled:
            return
        await self._save()

    async def ingest_user_message(
        self,
        *,
        group_id: int,
        message_id: int,
        user_id: int,
        speaker: str,
        content: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        if not self._enabled or group_id <= 0:
            return {"action": "disabled", "topic_id": None, "score": 0.0, "reason": "topic_disabled"}

        text = self._normalize_text(content)
        if not text:
            return {"action": "ignored", "topic_id": None, "score": 0.0, "reason": "empty_text"}

        now = self._to_utc(created_at)
        topic_id: int | None = None
        best_score = 0.0
        action = "no_topic"
        reason = "low_signal"
        trigger_summary = False

        async with self._lock:
            self._archive_inactive_locked(group_id=group_id, now=now)

            if not self._is_low_signal(text):
                topic_id, best_score = self._pick_best_active_topic_locked(group_id=group_id, user_id=user_id, text=text, now=now)
                if topic_id is not None and best_score >= self._reuse_threshold:
                    action = "reuse"
                    reason = "score_reuse"
                    topic = self._topics[topic_id]
                else:
                    topic = self._new_topic_locked(group_id=group_id, message_id=message_id, user_id=user_id, speaker=speaker, text=text, now=now)
                    topic_id = int(topic["topic_id"])
                    action = "new"
                    reason = "new_topic"

                self._append_message_locked(
                    topic=topic,
                    message_id=message_id,
                    role="user",
                    user_id=user_id,
                    speaker=speaker,
                    content=text,
                    now=now,
                )
                trigger_summary = self._unsummarized_count(topic) >= self._summary_trigger

            self._archive_overflow_locked(group_id=group_id, now=now)

        if trigger_summary and topic_id is not None:
            await self.refresh_topic_summary(topic_id=topic_id, reason="threshold")

        await self._save()
        return {"action": action, "topic_id": topic_id, "score": round(float(best_score), 4), "reason": reason}

    async def ingest_assistant_reply(
        self,
        *,
        group_id: int,
        topic_id: int | None,
        speaker: str,
        content: str,
        created_at: datetime,
    ) -> None:
        if not self._enabled or group_id <= 0 or topic_id is None:
            return

        text = self._normalize_text(content)
        if not text:
            return

        now = self._to_utc(created_at)
        trigger_summary = False
        async with self._lock:
            topic = self._topics.get(int(topic_id))
            if topic is None or int(topic.get("group_id", 0)) != group_id or str(topic.get("status", "")) != TOPIC_ACTIVE:
                return
            self._append_message_locked(
                topic=topic,
                message_id=0,
                role="assistant",
                user_id=int(self.cfg.persona.qq or 0),
                speaker=speaker,
                content=text,
                now=now,
            )
            trigger_summary = self._unsummarized_count(topic) >= self._summary_trigger

        if trigger_summary:
            await self.refresh_topic_summary(topic_id=int(topic_id), reason="assistant_progress")

        await self._save()

    async def refresh_topic_summary(self, *, topic_id: int, reason: str = "manual") -> bool:
        if not self._enabled or topic_id <= 0:
            return False

        async with self._lock:
            topic = self._topics.get(int(topic_id))
            if topic is None:
                return False
            if reason != "manual" and self._unsummarized_count(topic) <= 0:
                return False
            snapshot = self._clone(topic)
            snapshot_count = len(snapshot.get("messages", []))

        summary = await self._build_summary(snapshot=snapshot, reason=reason)
        summary_vec = await self._embedding.embed(self._summary_text(summary))

        async with self._lock:
            topic = self._topics.get(int(topic_id))
            if topic is None:
                return False
            old_summary = dict(topic.get("summary", {}))
            if old_summary.get("gist"):
                history = topic.setdefault("summary_history", [])
                history.append(old_summary)
                if len(history) > 8:
                    del history[:-8]
            topic["summary"] = summary
            topic["summary_until"] = max(int(topic.get("summary_until", 0)), snapshot_count)
            topic["updated_at"] = self._now_iso()
            self._summary_vec_cache[int(topic_id)] = summary_vec

        await self._save()
        self._logger.info("Topic summary updated: topic=%s reason=%s", topic_id, reason)
        return True

    async def build_prompt_context(
        self,
        *,
        group_id: int,
        session_id: str,
        query_text: str,
        current_topic_id: int | None,
    ) -> dict[str, Any]:
        del session_id
        if not self._enabled or group_id <= 0:
            return {"current_topic": "", "archived_topics": []}

        async with self._lock:
            active = [self._clone(self._topics[item]) for item in self._group_topics.get(group_id, []) if item in self._topics and self._topics[item].get("status") == TOPIC_ACTIVE]
            archived = [self._clone(self._topics[item]) for item in self._group_topics.get(group_id, []) if item in self._topics and self._topics[item].get("status") == TOPIC_ARCHIVED]

        active.sort(key=lambda row: str(row.get("last_message_at", "")), reverse=True)
        current = None
        if current_topic_id is not None:
            current = next((item for item in active if int(item.get("topic_id", 0)) == int(current_topic_id)), None)
        if current is None and active:
            current = active[0]

        # The latest user turn is already present in chat context.
        # Avoid duplicating that line in topic memory injection, which can cause parroting.
        current_block = self._render_topic_block(current, include_recent=False) if current else ""
        archived_blocks = await self._recall_archived(query_text=query_text, topics=archived)
        return {"current_topic": current_block, "archived_topics": archived_blocks}

    async def get_runtime_snapshot(self) -> dict[str, Any]:
        if not self._enabled:
            return {"enabled": False, "total_topics": 0, "active_topics": 0, "archived_topics": 0, "groups": {}}

        async with self._lock:
            total = len(self._topics)
            active_count = sum(1 for item in self._topics.values() if item.get("status") == TOPIC_ACTIVE)
            groups: dict[str, dict[str, int]] = {}
            for group_id, topic_ids in self._group_topics.items():
                active = 0
                archived = 0
                for topic_id in topic_ids:
                    topic = self._topics.get(topic_id)
                    if topic is None:
                        continue
                    if topic.get("status") == TOPIC_ACTIVE:
                        active += 1
                    else:
                        archived += 1
                groups[str(group_id)] = {"active": active, "archived": archived, "total": active + archived}

        return {"enabled": True, "total_topics": total, "active_topics": active_count, "archived_topics": total - active_count, "groups": groups}

    async def list_topics(
        self,
        *,
        group_id: int = 0,
        status: str = "",
        keyword: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        if not self._enabled:
            return {"items": [], "total": 0, "page": max(1, page), "page_size": max(1, page_size)}

        clean_status = str(status or "").strip().lower()
        clean_keyword = self._normalize_text(keyword).lower()
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))

        async with self._lock:
            rows = [self._clone(item) for item in self._topics.values()]

        if group_id > 0:
            rows = [item for item in rows if int(item.get("group_id", 0)) == group_id]
        if clean_status in {TOPIC_ACTIVE, TOPIC_ARCHIVED}:
            rows = [item for item in rows if str(item.get("status", "")) == clean_status]
        if clean_keyword:
            rows = [item for item in rows if clean_keyword in self._topic_search_text(item)]

        rows.sort(key=lambda row: str(row.get("last_message_at", "")), reverse=True)
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size
        return {"items": [self._topic_row(item) for item in rows[start:end]], "total": len(rows), "page": safe_page, "page_size": safe_page_size}

    async def get_topic_detail(self, *, topic_id: int, message_limit: int = 80) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        async with self._lock:
            row = self._topics.get(int(topic_id))
            if row is None:
                return None
            cloned = self._clone(row)

        data = self._topic_row(cloned)
        limit = max(1, min(int(message_limit), 500))
        data["messages"] = list(cloned.get("messages", []))[-limit:]
        data["summary_history"] = list(cloned.get("summary_history", []))
        return data

    async def set_topic_status(self, *, topic_id: int, status: str) -> bool:
        clean_status = str(status or "").strip().lower()
        if clean_status not in {TOPIC_ACTIVE, TOPIC_ARCHIVED}:
            return False
        async with self._lock:
            row = self._topics.get(int(topic_id))
            if row is None:
                return False
            row["status"] = clean_status
            row["updated_at"] = self._now_iso()
        await self._save()
        return True

    async def _recall_archived(self, *, query_text: str, topics: list[dict[str, Any]]) -> list[str]:
        if not topics:
            return []
        clean_query = self._normalize_text(query_text)
        scored: list[tuple[dict[str, Any], float]] = []
        for item in topics:
            score = self._recall_score(clean_query, item)
            if score > 0:
                scored.append((item, score))
        if not scored:
            return []

        scored.sort(key=lambda row: row[1], reverse=True)
        shortlist = scored[: max(self._recall_top_k * 3, self._recall_top_k)]
        qvec = await self._embedding.embed(clean_query) if clean_query else []

        rerank: list[tuple[dict[str, Any], float]] = []
        for topic, lex_score in shortlist:
            emb_score = 0.0
            if qvec:
                tvec = await self._summary_vector(topic)
                emb_score = self._cosine(qvec, tvec)
            rerank.append((topic, lex_score * 0.65 + emb_score * 0.35))

        rerank.sort(key=lambda row: row[1], reverse=True)
        return [self._render_topic_block(item, include_recent=False) for item, _ in rerank[: self._recall_top_k]]

    async def _summary_vector(self, topic: dict[str, Any]) -> list[float]:
        topic_id = int(topic.get("topic_id", 0))
        cached = self._summary_vec_cache.get(topic_id)
        if cached:
            return cached
        vec = await self._embedding.embed(self._summary_text(topic.get("summary", {})))
        self._summary_vec_cache[topic_id] = vec
        return vec

    async def _build_summary(self, *, snapshot: dict[str, Any], reason: str) -> dict[str, Any]:
        messages = snapshot.get("messages", [])
        summary_until = int(snapshot.get("summary_until", 0))
        new_messages = messages[max(0, summary_until) :]
        if not new_messages:
            new_messages = messages[-8:]

        old_summary = snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
        msg_lines = [self._render_turn(item.get("speaker", ""), item.get("content", "")) for item in new_messages]
        prompt_messages = [
            {
                "role": "system",
                "content": "你是群聊话题整理器。输出纯 JSON，字段必须有 title,gist,facts,participants,open_loops,recent_turns,keywords。",
            },
            {
                "role": "user",
                "content": (
                    f"群号: {snapshot.get('group_id')}\\n"
                    f"话题ID: {snapshot.get('topic_id')}\\n"
                    f"触发原因: {reason}\\n"
                    f"旧摘要: {json.dumps(old_summary, ensure_ascii=False)}\\n"
                    "新增消息:\\n"
                    + "\\n".join(msg_lines)
                ),
            },
        ]

        llm_text = ""
        try:
            llm_text = await self.llm.generate_from_messages(prompt_messages, {"temperature": 0.2})
        except Exception as exc:
            self._logger.warning("Topic summary llm failed: topic=%s err=%s", snapshot.get("topic_id"), exc)

        parsed = self._parse_summary_json(llm_text)
        if parsed is not None:
            parsed["updated_at"] = self._now_iso()
            return parsed
        return self._fallback_summary(snapshot=snapshot, old_summary=old_summary)

    def _fallback_summary(self, *, snapshot: dict[str, Any], old_summary: dict[str, Any]) -> dict[str, Any]:
        messages = list(snapshot.get("messages", []))
        text_list = [str(item.get("content", "") or "") for item in messages[-8:]]
        title = str(old_summary.get("title", "") or "").strip() or self._truncate_title(text_list[0] if text_list else "新话题")
        gist = self._truncate(str(text_list[-1] if text_list else old_summary.get("gist", "")), 140)
        participants = list(dict.fromkeys(str(item.get("speaker", "") or "").strip() for item in messages if str(item.get("speaker", "")).strip()))
        recent_turns = [self._render_turn(item.get("speaker", ""), item.get("content", "")) for item in messages[-4:]]
        keywords = self._keywords(text_list + list(old_summary.get("keywords", []) if isinstance(old_summary.get("keywords"), list) else []))
        return {
            "title": title,
            "gist": gist,
            "facts": self._normalize_list(old_summary.get("facts", []), limit=8, max_len=90),
            "participants": participants[:8],
            "open_loops": self._open_loops(text_list),
            "recent_turns": recent_turns[:8],
            "keywords": keywords,
            "updated_at": self._now_iso(),
        }

    def _parse_summary_json(self, text: str) -> dict[str, Any] | None:
        clean = str(text or "").strip()
        if not clean:
            return None
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
        try:
            row = json.loads(clean)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            return None
        title = self._truncate(str(row.get("title", "") or "").strip(), 50)
        gist = self._truncate(str(row.get("gist", "") or "").strip(), 160)
        if not title and gist:
            title = self._truncate_title(gist)
        if not gist and title:
            gist = title
        return {
            "title": title,
            "gist": gist,
            "facts": self._normalize_list(row.get("facts", []), limit=8, max_len=90),
            "participants": self._normalize_list(row.get("participants", []), limit=8, max_len=32),
            "open_loops": self._normalize_list(row.get("open_loops", []), limit=8, max_len=90),
            "recent_turns": self._normalize_list(row.get("recent_turns", []), limit=8, max_len=120),
            "keywords": self._normalize_list(row.get("keywords", []), limit=12, max_len=20),
            "updated_at": self._now_iso(),
        }

    def _pick_best_active_topic_locked(self, *, group_id: int, user_id: int, text: str, now: datetime) -> tuple[int | None, float]:
        best_id: int | None = None
        best_score = 0.0
        for topic_id in self._group_topics.get(group_id, []):
            topic = self._topics.get(topic_id)
            if topic is None or topic.get("status") != TOPIC_ACTIVE:
                continue
            score = self._reuse_score(text=text, user_id=user_id, topic=topic, now=now)
            if score > best_score:
                best_score = score
                best_id = int(topic_id)
        return best_id, best_score

    def _reuse_score(self, *, text: str, user_id: int, topic: dict[str, Any], now: datetime) -> float:
        lex = self._recall_score(text, topic)
        participant = 0.08 if user_id > 0 and str(user_id) in topic.get("participants", {}) else 0.0
        delta = max((now - self._parse_time(str(topic.get("last_message_at", "")))).total_seconds(), 0.0)
        if delta <= 120:
            recency = 0.12
        elif delta <= 600:
            recency = 0.07
        elif delta <= 1800:
            recency = 0.03
        else:
            recency = 0.0
        return lex + participant + recency

    def _recall_score(self, query_text: str, topic: dict[str, Any]) -> float:
        query = self._normalize_text(query_text).lower()
        if not query:
            return 0.0
        qt = self._tokens(query)
        if not qt:
            return 0.0
        st = self._tokens(self._summary_text(topic.get("summary", {})).lower())
        if not st:
            return 0.0
        overlap = len(qt.intersection(st))
        if overlap <= 0:
            return 0.0
        score = overlap / max(len(qt), 1)
        delta_hours = max((self._to_utc(datetime.now(timezone.utc)) - self._parse_time(str(topic.get("last_message_at", "")))).total_seconds() / 3600.0, 0.0)
        if delta_hours <= 6:
            score += 0.06
        elif delta_hours <= 24:
            score += 0.03
        return score

    def _new_topic_locked(self, *, group_id: int, message_id: int, user_id: int, speaker: str, text: str, now: datetime) -> dict[str, Any]:
        topic_id = self._next_topic_id
        self._next_topic_id += 1
        now_iso = self._format_time(now)
        topic = {
            "topic_id": topic_id,
            "group_id": group_id,
            "status": TOPIC_ACTIVE,
            "created_at": now_iso,
            "updated_at": now_iso,
            "last_message_at": now_iso,
            "last_message_id": max(0, int(message_id or 0)),
            "summary_until": 0,
            "participants": ({str(user_id): str(speaker).strip()} if user_id > 0 and str(speaker).strip() else {}),
            "summary": {
                "title": self._truncate_title(text),
                "gist": self._truncate(text, 120),
                "facts": [],
                "participants": ([str(speaker).strip()] if str(speaker).strip() else []),
                "open_loops": self._open_loops([text]),
                "recent_turns": [self._render_turn(speaker, text)],
                "keywords": self._keywords([text]),
                "updated_at": now_iso,
            },
            "summary_history": [],
            "messages": [],
        }
        self._topics[topic_id] = topic
        self._group_topics[group_id].append(topic_id)
        return topic

    def _append_message_locked(
        self,
        *,
        topic: dict[str, Any],
        message_id: int,
        role: str,
        user_id: int,
        speaker: str,
        content: str,
        now: datetime,
    ) -> None:
        messages = topic.setdefault("messages", [])
        message = {
            "message_id": max(0, int(message_id or 0)),
            "role": str(role or "user"),
            "user_id": max(0, int(user_id or 0)),
            "speaker": str(speaker or "").strip(),
            "content": self._truncate(str(content or ""), 800),
            "created_at": self._format_time(now),
        }
        messages.append(message)
        if len(messages) > self._tail_size:
            trimmed = len(messages) - self._tail_size
            del messages[:trimmed]
            topic["summary_until"] = max(0, int(topic.get("summary_until", 0)) - trimmed)

        if message["user_id"] > 0 and message["speaker"]:
            participants = topic.setdefault("participants", {})
            participants[str(message["user_id"])] = message["speaker"]

        now_iso = self._format_time(now)
        topic["updated_at"] = now_iso
        topic["last_message_at"] = now_iso
        topic["last_message_id"] = max(int(topic.get("last_message_id", 0)), int(message["message_id"]))

    def _archive_inactive_locked(self, *, group_id: int, now: datetime) -> None:
        deadline = now - timedelta(minutes=self._archive_inactive_minutes)
        for topic_id in self._group_topics.get(group_id, []):
            topic = self._topics.get(topic_id)
            if topic is None or topic.get("status") != TOPIC_ACTIVE:
                continue
            if self._parse_time(str(topic.get("last_message_at", ""))) <= deadline:
                topic["status"] = TOPIC_ARCHIVED
                topic["updated_at"] = self._format_time(now)

    def _archive_overflow_locked(self, *, group_id: int, now: datetime) -> None:
        active = [self._topics[item] for item in self._group_topics.get(group_id, []) if item in self._topics and self._topics[item].get("status") == TOPIC_ACTIVE]
        if len(active) <= self._max_active:
            return
        active.sort(key=lambda row: str(row.get("last_message_at", "")), reverse=True)
        for item in active[self._max_active :]:
            item["status"] = TOPIC_ARCHIVED
            item["updated_at"] = self._format_time(now)

    def _unsummarized_count(self, topic: dict[str, Any]) -> int:
        return max(0, len(topic.get("messages", [])) - max(0, int(topic.get("summary_until", 0))))

    def _summary_text(self, summary: Any) -> str:
        if not isinstance(summary, dict):
            return ""
        fields: list[str] = []
        for key in ("title", "gist"):
            text = str(summary.get(key, "") or "").strip()
            if text:
                fields.append(text)
        for key in ("facts", "open_loops", "keywords"):
            values = summary.get(key, [])
            if isinstance(values, list):
                for item in values:
                    text = str(item or "").strip()
                    if text:
                        fields.append(text)
        return "\n".join(fields).strip()

    def _topic_search_text(self, topic: dict[str, Any]) -> str:
        summary = topic.get("summary", {})
        pieces = [self._summary_text(summary)]
        pieces.extend(str(item.get("content", "") or "") for item in list(topic.get("messages", []))[-20:])
        return "\n".join(pieces).lower()

    def _topic_row(self, topic: dict[str, Any]) -> dict[str, Any]:
        return {
            "topic_id": int(topic.get("topic_id", 0)),
            "group_id": int(topic.get("group_id", 0)),
            "status": str(topic.get("status", "")),
            "created_at": str(topic.get("created_at", "")),
            "updated_at": str(topic.get("updated_at", "")),
            "last_message_at": str(topic.get("last_message_at", "")),
            "last_message_id": int(topic.get("last_message_id", 0)),
            "summary_until": int(topic.get("summary_until", 0)),
            "message_count": len(topic.get("messages", [])),
            "participants": list(dict(topic.get("participants", {})).values())[:12],
            "summary": self._clone(topic.get("summary", {})),
        }

    def _render_topic_block(self, topic: dict[str, Any] | None, *, include_recent: bool) -> str:
        if not topic:
            return ""
        summary = topic.get("summary", {}) if isinstance(topic.get("summary"), dict) else {}
        title = str(summary.get("title", "") or "").strip() or f"话题 {topic.get('topic_id', '')}"
        lines = [f"[话题#{topic.get('topic_id')}][{topic.get('status')}] {title}"]
        gist = str(summary.get("gist", "") or "").strip()
        if gist:
            lines.append(f"摘要: {gist}")
        facts = summary.get("facts", [])
        if isinstance(facts, list) and facts:
            lines.append("已确认: " + "；".join(str(item) for item in facts[:4]))
        loops = summary.get("open_loops", [])
        if isinstance(loops, list) and loops:
            lines.append("未完事项: " + "；".join(str(item) for item in loops[:4]))
        participants = summary.get("participants", [])
        if isinstance(participants, list) and participants:
            lines.append("参与者: " + "、".join(str(item) for item in participants[:6]))
        if include_recent:
            recent = [self._render_turn(item.get("speaker", ""), item.get("content", "")) for item in list(topic.get("messages", []))[-3:]]
            if recent:
                lines.append("最近推进: " + " | ".join(recent))
        return "\n".join(lines)

    def _keywords(self, texts: list[str], *, top_k: int = 10) -> list[str]:
        tokens: list[str] = []
        for item in texts:
            tokens.extend(self._tokens(str(item or "")))
        if not tokens:
            return []
        counter = Counter(token for token in tokens if len(token) >= 2)
        return [token for token, _ in counter.most_common(top_k)]

    def _open_loops(self, texts: list[str]) -> list[str]:
        loops: list[str] = []
        for raw in texts[-6:]:
            text = str(raw or "").strip()
            if not text:
                continue
            if "?" in text or "？" in text or any(marker in text for marker in ("待", "后续", "下一步", "回头", "还要")):
                loops.append(self._truncate(text, 80))
        return self._dedup(loops, limit=6)

    @staticmethod
    def _normalize_list(raw: Any, *, limit: int, max_len: int) -> list[str]:
        if not isinstance(raw, list):
            return []
        values = [TopicManager._truncate(str(item or "").strip(), max_len) for item in raw]
        values = [item for item in values if item]
        return TopicManager._dedup(values, limit=limit)

    @staticmethod
    def _dedup(values: list[str], *, limit: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _render_turn(speaker: Any, content: Any) -> str:
        name = str(speaker or "").strip() or "某人"
        return f"{name}: {TopicManager._truncate(str(content or ''), 90)}"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        clean = str(text or "").lower().strip()
        if not clean:
            return set()
        return {item for item in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{1,6}", clean) if item}

    @staticmethod
    def _normalize_text(text: str) -> str:
        clean = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        clean = str(text or "").strip()
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 1].rstrip() + "…"

    @staticmethod
    def _truncate_title(text: str) -> str:
        clean = re.sub(r"\s+", " ", str(text or "").strip())
        if not clean:
            return "新话题"
        sentence = re.split(r"[。！？!?；;\n]", clean, maxsplit=1)[0].strip() or clean
        return TopicManager._truncate(sentence, 28)

    @staticmethod
    def _is_low_signal(text: str) -> bool:
        clean = str(text or "").strip()
        if not clean:
            return True
        compact = re.sub(r"[\s，,。.!！?？；;:：~～…·`'\"“”‘’\-_/\(\)（）\[\]{}<>]+", "", clean)
        if len(compact) <= 1:
            return True
        if len(compact) <= 3 and compact.isdigit():
            return True
        return False

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for idx in range(size):
            lv = float(left[idx])
            rv = float(right[idx])
            dot += lv * rv
            left_norm += lv * lv
            right_norm += rv * rv
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return dot / math.sqrt(left_norm * right_norm)

    @staticmethod
    def _to_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _format_time(value: datetime) -> str:
        return TopicManager._to_utc(value).isoformat()

    @staticmethod
    def _parse_time(raw: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(raw or ""))
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _clone(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False))

    async def _load(self) -> None:
        async with self._lock:
            if not self._store_path.exists():
                self._store_path.parent.mkdir(parents=True, exist_ok=True)
                return

            raw = self._store_path.read_text(encoding="utf-8")
            if not raw.strip():
                return

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._logger.warning("Topic store decode failed: path=%s err=%s", self._store_path, exc)
                return

            rows = payload.get("topics", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                return

            topics: dict[int, dict[str, Any]] = {}
            groups: dict[int, list[int]] = defaultdict(list)
            next_id = 1
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    topic_id = int(row.get("topic_id", 0))
                    group_id = int(row.get("group_id", 0))
                except (TypeError, ValueError):
                    continue
                if topic_id <= 0 or group_id <= 0:
                    continue
                topics[topic_id] = row
                groups[group_id].append(topic_id)
                next_id = max(next_id, topic_id + 1)

            self._topics = topics
            self._group_topics = groups
            self._next_topic_id = next_id

    async def _save(self) -> None:
        if not self._enabled:
            return
        async with self._lock:
            payload = {"version": 1, "next_topic_id": self._next_topic_id, "topics": list(self._topics.values())}
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self._store_path.with_suffix(self._store_path.suffix + ".tmp")
            temp_file.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            temp_file.replace(self._store_path)
