from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List

from adapters.llm.chat import ChatLLMAdapter
from adapters.llm.embedding import EmbeddingAdapter
from internal.config.schema import Config
from internal.logger import get_logger
from internal.memory.models import MemoryItem, MemoryType, MessageLog
from internal.memory.vector_storage import ChromaVectorStorage, CollectionType, VectorRecord


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str
    speaker: str
    user_id: int | None
    created_at: datetime


@dataclass(slots=True)
class SummaryTask:
    session_id: str
    group_id: int
    reason: str
    turns: list[ConversationTurn]


@dataclass(slots=True)
class RetrievalBundle:
    history_background: list[str] = field(default_factory=list)
    related_knowledge: list[str] = field(default_factory=list)


SummaryHook = Callable[[str, str, datetime], Awaitable[None]]


class MemoryManager:
    def __init__(
        self,
        *,
        cfg: Config,
        llm: ChatLLMAdapter,
        on_summary: SummaryHook | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.on_summary = on_summary
        self._logger = get_logger("MemoryManager")

        self._messages: Dict[int, List[MessageLog]] = defaultdict(list)
        self._memories: Dict[int, List[MemoryItem]] = defaultdict(list)
        self._short_term_turns: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._session_groups: dict[str, int] = {}
        self._memory_by_id: dict[int, MemoryItem] = {}

        self._id = 0
        self._lock = asyncio.Lock()

        self._short_term_threshold = max(1, int(getattr(cfg.memory, "short_term_threshold", 20)))
        self._short_term_keep_last = max(1, int(getattr(cfg.memory, "short_term_keep_last", 3)))
        self._topic_shift_threshold = float(getattr(cfg.memory, "topic_shift_similarity_threshold", 0.35))
        self._topic_shift_min_messages = max(3, int(getattr(cfg.memory, "topic_shift_min_messages", 8)))
        self._rag_top_k = max(1, int(getattr(cfg.memory, "rag_top_k", 5)))

        embedding = EmbeddingAdapter(cfg.embedding, target_dim=cfg.memory.vector_dim)
        self._vector_storage = ChromaVectorStorage(
            persist_path=getattr(cfg.memory, "chroma_path", "./data/chroma"),
            embedding_adapter=embedding,
        )

        self._summary_queue: asyncio.Queue[SummaryTask] = asyncio.Queue()
        self._summary_worker: asyncio.Task[None] | None = None
        self._pending_sessions: set[str] = set()
        self._cleared_sessions: set[str] = set()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self._vector_storage.start()
        self._summary_worker = asyncio.create_task(self._run_summary_worker(), name="memory-summary-worker")
        self._started = True
        self._logger.info(
            "Memory manager started: short_term_threshold=%s keep_last=%s rag_top_k=%s",
            self._short_term_threshold,
            self._short_term_keep_last,
            self._rag_top_k,
        )

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
        text = content.strip()
        async with self._lock:
            self._id += 1
            item = MemoryItem(
                id=self._id,
                group_id=group_id,
                mem_type=mem_type,
                content=text,
                importance=importance,
            )
            self._memories[group_id].append(item)
            self._memory_by_id[item.id] = item

        await self._vector_storage.store(
            text,
            "user_memories",
            metadata={
                "memory_id": item.id,
                "group_id": group_id,
                "mem_type": mem_type,
                "importance": importance,
                "source": "manual_save",
                "created_at": item.created_at.isoformat(),
            },
        )
        return item

    async def query_memory(self, group_id: int, query: str, limit: int = 5) -> list[MemoryItem]:
        vector_rows = await self._vector_storage.query(
            query,
            "user_memories",
            top_k=max(limit * 2, limit),
        )
        filtered = self._filter_rows_for_group(vector_rows, group_id)
        if filtered:
            self._logger.info("向量库命中: collection=user_memories group_id=%s count=%s", group_id, len(filtered))
        out: list[MemoryItem] = []
        seen_ids: set[int] = set()
        async with self._lock:
            for row in filtered:
                memory_id = self._read_memory_id(row)
                if memory_id is not None and memory_id in self._memory_by_id:
                    item = self._memory_by_id[memory_id]
                    if item.id in seen_ids:
                        continue
                    item.access_count += 1
                    out.append(item)
                    seen_ids.add(item.id)
                    if len(out) >= limit:
                        return out

            # fallback to in-memory keyword match (keeps old behavior if vector miss)
            query_lower = query.lower().strip()
            legacy = [m for m in self._memories[group_id] if query_lower in m.content.lower()]
            legacy = legacy[-limit:]
            for item in legacy:
                if item.id in seen_ids:
                    continue
                item.access_count += 1
                out.append(item)
                if len(out) >= limit:
                    break
        return out

    async def store_external_knowledge(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        row_id = await self._vector_storage.store(text, "external_knowledge", metadata or {})
        if row_id:
            self._logger.info("External knowledge stored: id=%s", row_id)
        return row_id

    async def query_external_knowledge(self, query: str, top_k: int = 5) -> list[VectorRecord]:
        rows = await self._vector_storage.query(query, "external_knowledge", top_k=top_k)
        if rows:
            self._logger.info("向量库命中: collection=external_knowledge count=%s", len(rows))
        return rows

    async def retrieve_for_prompt(
        self,
        *,
        text: str,
        session_id: str,
        group_id: int | None,
        top_k: int | None = None,
    ) -> RetrievalBundle:
        if not text.strip():
            return RetrievalBundle()

        limit = top_k if top_k is not None else self._rag_top_k
        user_rows_task = self._vector_storage.query(text, "user_memories", top_k=limit)
        ext_rows_task = self._vector_storage.query(text, "external_knowledge", top_k=limit)
        user_rows, ext_rows = await asyncio.gather(user_rows_task, ext_rows_task)

        history_rows = self._filter_rows_for_session(user_rows, session_id, group_id)
        history_lines = [self._format_row_text(row) for row in history_rows if self._format_row_text(row)]
        knowledge_lines = [self._format_row_text(row) for row in ext_rows if self._format_row_text(row)]

        self._logger.info(
            "向量库命中: session=%s history=%s knowledge=%s",
            session_id,
            len(history_lines),
            len(knowledge_lines),
        )
        return RetrievalBundle(
            history_background=history_lines,
            related_knowledge=knowledge_lines,
        )

    async def record_conversation_turn(
        self,
        *,
        session_id: str,
        group_id: int,
        role: str,
        content: str,
        speaker: str,
        user_id: int | None,
        created_at: datetime,
    ) -> None:
        text = content.strip()
        if not text:
            return

        reason: str | None = None
        async with self._lock:
            turns = self._short_term_turns[session_id]
            self._cleared_sessions.discard(session_id)
            turns.append(
                ConversationTurn(
                    role=role,
                    content=text,
                    speaker=speaker,
                    user_id=user_id,
                    created_at=created_at,
                ),
            )
            self._session_groups[session_id] = group_id

            if len(turns) >= self._short_term_threshold:
                reason = "threshold"

        if reason is None and role == "user":
            reason = await self._detect_topic_shift(session_id)

        if reason is None:
            return

        await self._enqueue_summary_task(session_id, group_id, reason)

    async def close(self) -> None:
        self._started = False
        if self._summary_worker is not None:
            self._summary_worker.cancel()
            try:
                await self._summary_worker
            except asyncio.CancelledError:
                pass
            self._summary_worker = None
        await self._vector_storage.close()

    async def clear_session_memory(self, session_id: str) -> bool:
        clean = str(session_id).strip()
        if not clean:
            return False
        async with self._lock:
            existed = clean in self._short_term_turns or clean in self._session_groups
            self._short_term_turns.pop(clean, None)
            self._session_groups.pop(clean, None)
            self._pending_sessions.discard(clean)
            self._cleared_sessions.add(clean)
            return existed

    async def get_short_term_snapshot(self, max_sessions: int = 8, turn_limit: int = 4) -> list[dict]:
        session_limit = max(1, int(max_sessions))
        keep_turns = max(1, int(turn_limit))
        async with self._lock:
            rows = list(self._short_term_turns.items())
            session_groups = dict(self._session_groups)

        rows.sort(
            key=lambda item: item[1][-1].created_at if item[1] else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        out: list[dict] = []
        for session_id, turns in rows[:session_limit]:
            trimmed = turns[-keep_turns:]
            out.append(
                {
                    "session_id": session_id,
                    "group_id": session_groups.get(session_id, 0),
                    "turn_count": len(turns),
                    "latest_at": trimmed[-1].created_at.isoformat() if trimmed else "",
                    "recent_turns": [
                        {
                            "role": item.role,
                            "speaker": item.speaker,
                            "content": item.content,
                            "created_at": item.created_at.isoformat(),
                        }
                        for item in trimmed
                    ],
                },
            )
        return out

    async def _enqueue_summary_task(self, session_id: str, group_id: int, reason: str) -> None:
        async with self._lock:
            if session_id in self._cleared_sessions:
                return
            if session_id in self._pending_sessions:
                return
            turns = list(self._short_term_turns.get(session_id, []))
            if len(turns) < self._short_term_keep_last + 1:
                return

            if len(turns) > self._short_term_threshold:
                turns = turns[-self._short_term_threshold :]

            self._pending_sessions.add(session_id)

        await self._summary_queue.put(
            SummaryTask(
                session_id=session_id,
                group_id=group_id,
                reason=reason,
                turns=turns,
            ),
        )
        self._logger.info(
            "Memory sediment queued: session=%s reason=%s turns=%s",
            session_id,
            reason,
            len(turns),
        )

    async def _run_summary_worker(self) -> None:
        while True:
            try:
                task = await self._summary_queue.get()
            except asyncio.CancelledError:
                return

            success = False
            try:
                async with self._lock:
                    if task.session_id in self._cleared_sessions:
                        continue
                summary = await self._summarize_turns(task)
                if summary:
                    memory_item = await self._store_summary_memory(task, summary)
                    await self._trim_short_term(task.session_id)
                    success = True
                    self._logger.info(
                        "记忆沉淀成功: session=%s memory_id=%s reason=%s",
                        task.session_id,
                        memory_item.id,
                        task.reason,
                    )

                    if self.on_summary is not None:
                        await self.on_summary(task.session_id, summary, datetime.now(timezone.utc))
            except Exception:
                self._logger.exception("Memory sediment task failed: session=%s", task.session_id)
            finally:
                async with self._lock:
                    self._pending_sessions.discard(task.session_id)
                self._summary_queue.task_done()
                if not success:
                    self._logger.warning("Memory sediment skipped/failed: session=%s", task.session_id)

    async def _summarize_turns(self, task: SummaryTask) -> str:
        dialog_lines = []
        for idx, turn in enumerate(task.turns, start=1):
            timestamp = turn.created_at.astimezone().strftime("%m-%d %H:%M:%S")
            dialog_lines.append(f"{idx}. [{timestamp}] {turn.speaker}({turn.role}): {turn.content}")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是记忆代谢器。请把对话压缩为长期记忆，必须输出纯文本，包含三段："
                    "核心事件、用户偏好、重要约定。只保留高价值事实，去掉寒暄和重复。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"会话ID: {task.session_id}\n"
                    f"触发原因: {task.reason}\n"
                    "请总结以下对话：\n"
                    + "\n".join(dialog_lines)
                ),
            },
        ]
        summary = await self.llm.generate_from_messages(messages, {"temperature": 0.2})
        summary = summary.strip()
        if summary:
            return summary
        return self._fallback_summary(task.turns)

    async def _store_summary_memory(self, task: SummaryTask, summary: str) -> MemoryItem:
        async with self._lock:
            self._id += 1
            item = MemoryItem(
                id=self._id,
                group_id=task.group_id,
                mem_type="conversation",
                content=summary,
                importance=0.8,
                created_at=datetime.now(timezone.utc),
            )
            self._memories[task.group_id].append(item)
            self._memory_by_id[item.id] = item

        await self._vector_storage.store(
            summary,
            "user_memories",
            metadata={
                "memory_id": item.id,
                "group_id": task.group_id,
                "session_id": task.session_id,
                "reason": task.reason,
                "source": "metabolized_summary",
                "created_at": item.created_at.isoformat(),
            },
        )
        return item

    async def _trim_short_term(self, session_id: str) -> None:
        async with self._lock:
            turns = self._short_term_turns.get(session_id, [])
            if not turns:
                return
            if len(turns) <= self._short_term_keep_last:
                return
            self._short_term_turns[session_id] = turns[-self._short_term_keep_last :]

    async def _detect_topic_shift(self, session_id: str) -> str | None:
        async with self._lock:
            turns = list(self._short_term_turns.get(session_id, []))

        if len(turns) < self._topic_shift_min_messages:
            return None

        current = turns[-1].content
        previous = ""
        for item in reversed(turns[:-1]):
            if item.role == "user":
                previous = item.content
                break
        if not previous:
            return None

        lexical_score = self._lexical_similarity(previous, current)
        embedding_score = await self._embedding_similarity(previous, current)
        combined = lexical_score
        if embedding_score >= 0:
            combined = (lexical_score + embedding_score) / 2.0

        if combined < self._topic_shift_threshold:
            self._logger.info(
                "Topic shift detected: session=%s score=%.3f threshold=%.3f",
                session_id,
                combined,
                self._topic_shift_threshold,
            )
            return "topic_shift"
        return None

    async def _embedding_similarity(self, left: str, right: str) -> float:
        left_vec, right_vec = await asyncio.gather(
            self._vector_storage.embedding_adapter.embed(left),
            self._vector_storage.embedding_adapter.embed(right),
        )
        if not left_vec or not right_vec:
            return -1.0
        size = min(len(left_vec), len(right_vec))
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for idx in range(size):
            lv = float(left_vec[idx])
            rv = float(right_vec[idx])
            dot += lv * rv
            left_norm += lv * lv
            right_norm += rv * rv
        if left_norm <= 0 or right_norm <= 0:
            return -1.0
        return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))

    @staticmethod
    def _lexical_similarity(left: str, right: str) -> float:
        left_tokens = set(left.lower().split())
        right_tokens = set(right.lower().split())
        if not left_tokens or not right_tokens:
            return 0.0
        inter = left_tokens.intersection(right_tokens)
        union = left_tokens.union(right_tokens)
        return float(len(inter)) / float(len(union))

    def _filter_rows_for_group(self, rows: list[VectorRecord], group_id: int) -> list[VectorRecord]:
        out: list[VectorRecord] = []
        for row in rows:
            meta_group = row.metadata.get("group_id")
            if meta_group is None:
                out.append(row)
                continue
            try:
                if int(meta_group) == int(group_id):
                    out.append(row)
            except (TypeError, ValueError):
                continue
        return out

    def _filter_rows_for_session(
        self,
        rows: list[VectorRecord],
        session_id: str,
        group_id: int | None,
    ) -> list[VectorRecord]:
        out: list[VectorRecord] = []
        for row in rows:
            row_session = str(row.metadata.get("session_id", "")).strip()
            row_group_raw = row.metadata.get("group_id")
            if row_session and row_session == session_id:
                out.append(row)
                continue
            if row_group_raw is not None and group_id is not None:
                try:
                    if int(row_group_raw) == int(group_id):
                        out.append(row)
                        continue
                except (TypeError, ValueError):
                    pass
            if not row_session and row_group_raw is None:
                out.append(row)
        return out

    @staticmethod
    def _format_row_text(row: VectorRecord) -> str:
        text = row.text.strip()
        if not text:
            return ""
        distance = row.distance
        if distance is None:
            return text
        return f"{text} (distance={distance:.3f})"

    @staticmethod
    def _read_memory_id(row: VectorRecord) -> int | None:
        raw = row.metadata.get("memory_id")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fallback_summary(turns: list[ConversationTurn]) -> str:
        previews = [f"{turn.speaker}: {turn.content}" for turn in turns[-6:]]
        return (
            "核心事件：最近讨论集中在同一话题上。\n"
            "用户偏好：根据近期对话继续保持当前风格回应。\n"
            "重要约定："
            + " | ".join(previews)
        )
