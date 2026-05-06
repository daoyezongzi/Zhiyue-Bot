from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from adapters.llm.chat import ChatLLMAdapter
from adapters.llm.embedding import EmbeddingAdapter
from internal.config.schema import Config
from internal.logger import get_logger
from internal.memory.models import (
    KEYED_CANONICAL_TYPES,
    MEMORY_STATUS_ACTIVE,
    MEMORY_STATUS_ARCHIVED,
    MEMORY_STATUS_CANDIDATE,
    MEMORY_STATUS_LEGACY,
    CanonicalMemoryType,
    MemoryItem,
    MemorySourceKind,
    MemoryStatus,
    MemoryType,
    MessageLog,
    ToolCallLog,
)
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
    _SOURCE_PROMOTE_SET = {"summary", "topic"}

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
        self._indexed_memory_ids: set[int] = set()

        self._id = 0
        self._lock = asyncio.Lock()

        self._short_term_threshold = max(1, int(getattr(cfg.memory, "short_term_threshold", 20)))
        self._short_term_keep_last = max(1, int(getattr(cfg.memory, "short_term_keep_last", 3)))
        self._topic_shift_threshold = float(getattr(cfg.memory, "topic_shift_similarity_threshold", 0.35))
        self._topic_shift_min_messages = max(3, int(getattr(cfg.memory, "topic_shift_min_messages", 8)))
        self._rag_top_k = max(1, int(getattr(cfg.memory, "rag_top_k", 5)))
        self._knowledge_exclude_dirs = self._normalize_exclude_dirs(getattr(cfg.paths, "knowledge_exclude_dirs", []))

        project_root = Path(__file__).resolve().parents[2]
        store_path = str(getattr(cfg.memory, "memory_store_path", "data/memory/memory_items.json") or "")
        self._store_path = Path(store_path)
        if not self._store_path.is_absolute():
            self._store_path = project_root / self._store_path
        tool_call_store_path = str(getattr(cfg.memory, "tool_call_store_path", "data/memory/tool_calls.json") or "")
        self._tool_call_store_path = Path(tool_call_store_path)
        if not self._tool_call_store_path.is_absolute():
            self._tool_call_store_path = project_root / self._tool_call_store_path
        self._tool_call_max_entries = max(100, int(getattr(cfg.memory, "tool_call_max_entries", 5000)))

        self._auto_ingest_enabled = bool(getattr(cfg.memory, "memory_auto_ingest_enabled", True))
        self._convergence_interval_minutes = max(1, int(getattr(cfg.memory, "memory_convergence_interval_minutes", 15)))
        self._candidate_grace_hours = max(1, int(getattr(cfg.memory, "memory_candidate_grace_hours", 72)))
        self._candidate_promote_evidence = max(1, int(getattr(cfg.memory, "memory_candidate_promote_evidence", 2)))

        embedding = EmbeddingAdapter(cfg.embedding, target_dim=cfg.memory.vector_dim)
        self._vector_storage = ChromaVectorStorage(
            persist_path=getattr(cfg.memory, "chroma_path", "./data/chroma"),
            embedding_adapter=embedding,
        )

        self._summary_queue: asyncio.Queue[SummaryTask] = asyncio.Queue()
        self._summary_worker: asyncio.Task[None] | None = None
        self._convergence_worker: asyncio.Task[None] | None = None
        self._pending_sessions: set[str] = set()
        self._cleared_sessions: set[str] = set()
        self._tool_calls: list[ToolCallLog] = []
        self._tool_call_by_id: dict[int, ToolCallLog] = {}
        self._tool_call_id = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self._vector_storage.start()
        await self._load_memory_store()
        await self._load_tool_call_store()
        self._summary_worker = asyncio.create_task(self._run_summary_worker(), name="memory-summary-worker")
        self._convergence_worker = asyncio.create_task(
            self._run_convergence_worker(),
            name="memory-convergence-worker",
        )
        self._started = True
        self._logger.info(
            (
                "Memory manager started: short_term_threshold=%s keep_last=%s rag_top_k=%s "
                "auto_ingest=%s convergence_interval=%smin"
            ),
            self._short_term_threshold,
            self._short_term_keep_last,
            self._rag_top_k,
            self._auto_ingest_enabled,
            self._convergence_interval_minutes,
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
        item, _ = await self._upsert_memory_candidate(
            group_id=group_id,
            user_id=0,
            content=content,
            source_kind="manual",
            source_ref=f"manual:{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            mem_type_hint=mem_type,
            canonical_type_hint=self._infer_canonical_type(content),
            force_status=MEMORY_STATUS_ACTIVE,
            importance_override=importance,
        )
        return item

    async def save_governed_memory(
        self,
        *,
        group_id: int,
        user_id: int = 0,
        content: str,
        mem_type: MemoryType = "conversation",
        canonical_type: CanonicalMemoryType = "fact",
        status: MemoryStatus = MEMORY_STATUS_CANDIDATE,
        source_kind: MemorySourceKind = "manual",
        source_ref: str = "",
        importance: float = 0.0,
    ) -> MemoryItem:
        clean_content = self._normalize_text(content)
        if not clean_content:
            raise ValueError("memory content is empty")
        item, _ = await self._upsert_memory_candidate(
            group_id=group_id,
            user_id=user_id,
            content=clean_content,
            source_kind=source_kind,
            source_ref=source_ref or f"manual:{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            mem_type_hint=mem_type,
            canonical_type_hint=canonical_type,
            force_status=status,
            importance_override=importance if importance > 0 else None,
        )
        return item

    async def ingest_message_memory(
        self,
        *,
        group_id: int,
        user_id: int | None,
        content: str,
        source_ref: str,
        source_kind: MemorySourceKind = "message",
    ) -> MemoryItem | None:
        if not self._auto_ingest_enabled:
            return None
        clean_text = self._normalize_text(content)
        if self._is_low_signal_text(clean_text):
            return None

        try:
            item, action = await self._upsert_memory_candidate(
                group_id=group_id,
                user_id=int(user_id or 0),
                content=clean_text,
                source_kind=source_kind,
                source_ref=source_ref,
            )
            self._logger.info(
                "Memory ingest: group=%s user=%s action=%s memory_id=%s status=%s canonical=%s",
                group_id,
                int(user_id or 0),
                action,
                item.id,
                item.status,
                item.canonical_type,
            )
            return item
        except Exception:
            self._logger.exception("Memory ingest failed: group=%s source_ref=%s", group_id, source_ref)
            return None

    async def query_memory(self, group_id: int, query: str, limit: int = 5) -> list[MemoryItem]:
        vector_rows = await self._vector_storage.query(
            query,
            "user_memories",
            top_k=max(limit * 3, limit),
        )
        filtered = self._filter_rows_for_group(vector_rows, group_id)
        if filtered:
            self._logger.info("Vector hit: collection=user_memories group_id=%s count=%s", group_id, len(filtered))

        out: list[MemoryItem] = []
        seen_ids: set[int] = set()
        async with self._lock:
            for row in filtered:
                memory_id = self._read_memory_id(row)
                if memory_id is None:
                    continue
                item = self._memory_by_id.get(memory_id)
                if item is None:
                    continue
                if item.id in seen_ids:
                    continue
                if not item.recall_eligible():
                    continue
                if int(item.group_id) != int(group_id):
                    continue
                item.access_count += 1
                out.append(item)
                seen_ids.add(item.id)
                if len(out) >= limit:
                    return out

            query_lower = query.lower().strip()
            legacy = [
                m
                for m in self._memories.get(group_id, [])
                if m.recall_eligible() and query_lower in m.content.lower()
            ]
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
            self._logger.info("Vector hit: collection=external_knowledge count=%s", len(rows))
        return rows

    async def reindex_external_knowledge(self, knowledge_dir: str | Path) -> dict[str, Any]:
        root = Path(knowledge_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        await self._vector_storage.clear_collection("external_knowledge")

        files_seen = 0
        files_indexed = 0
        chunks_indexed = 0
        skipped_files: list[str] = []
        allowed_extensions = {
            ".txt",
            ".md",
            ".markdown",
            ".rst",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".log",
        }

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            files_seen += 1
            relative_path = str(file_path.relative_to(root))
            if self._should_skip_knowledge_file(root, file_path):
                skipped_files.append(relative_path)
                continue
            if file_path.suffix.lower() not in allowed_extensions:
                skipped_files.append(relative_path)
                continue

            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                skipped_files.append(relative_path)
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                skipped_files.append(relative_path)
                continue

            for idx, chunk in enumerate(chunks, start=1):
                await self.store_external_knowledge(
                    chunk,
                    metadata={
                        "source": "knowledge_file",
                        "file_path": relative_path,
                        "chunk_index": idx,
                        "chunk_total": len(chunks),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                chunks_indexed += 1

            files_indexed += 1

        self._logger.info(
            "Knowledge reindex finished: dir=%s seen=%s indexed=%s chunks=%s skipped=%s",
            root,
            files_seen,
            files_indexed,
            chunks_indexed,
            len(skipped_files),
        )
        return {
            "knowledge_dir": str(root),
            "files_seen": files_seen,
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "skipped_files": skipped_files,
        }

    @staticmethod
    def _normalize_exclude_dirs(values: Any) -> set[str]:
        if not isinstance(values, list):
            return set()
        out: set[str] = set()
        for item in values:
            text = str(item).strip().replace("\\", "/").strip("/")
            if not text:
                continue
            out.add(text.casefold())
        return out

    def _should_skip_knowledge_file(self, root: Path, file_path: Path) -> bool:
        if not self._knowledge_exclude_dirs:
            return False
        relative = file_path.relative_to(root)
        for part in relative.parts[:-1]:
            if part.casefold() in self._knowledge_exclude_dirs:
                return True
        return False

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
        user_rows_task = self._vector_storage.query(text, "user_memories", top_k=max(limit * 3, limit))
        ext_rows_task = self._vector_storage.query(text, "external_knowledge", top_k=limit)
        user_rows, ext_rows = await asyncio.gather(user_rows_task, ext_rows_task)

        history_rows = self._filter_rows_for_session(user_rows, session_id, group_id)
        history_lines = await self._rows_to_memory_lines(history_rows, session_id=session_id, group_id=group_id, limit=limit)
        knowledge_lines = [self._format_row_text(row) for row in ext_rows if self._format_row_text(row)]

        self._logger.info(
            "Vector hit: session=%s history=%s knowledge=%s",
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
                    created_at=self._ensure_utc(created_at),
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
        if self._convergence_worker is not None:
            self._convergence_worker.cancel()
            try:
                await self._convergence_worker
            except asyncio.CancelledError:
                pass
            self._convergence_worker = None
        await self._save_memory_store()
        await self._save_tool_call_store()
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

    async def get_runtime_snapshot(self) -> dict[str, Any]:
        async with self._lock:
            rows = list(self._memory_by_id.values())
        total = len(rows)
        status_counts = {
            MEMORY_STATUS_ACTIVE: 0,
            MEMORY_STATUS_CANDIDATE: 0,
            MEMORY_STATUS_ARCHIVED: 0,
            MEMORY_STATUS_LEGACY: 0,
        }
        canonical_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        for item in rows:
            status_key = item.effective_status()
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            canonical_key = str(item.canonical_type)
            canonical_counts[canonical_key] = canonical_counts.get(canonical_key, 0) + 1
            group_key = str(int(item.group_id))
            group_counts[group_key] = group_counts.get(group_key, 0) + 1
        return {
            "total": total,
            "status": status_counts,
            "canonical": canonical_counts,
            "groups": group_counts,
            "candidate_grace_hours": self._candidate_grace_hours,
            "candidate_promote_evidence": self._candidate_promote_evidence,
            "auto_ingest_enabled": self._auto_ingest_enabled,
        }

    async def record_tool_call(
        self,
        *,
        session_id: str,
        message_type: str,
        group_id: int,
        user_id: int,
        speaker: str,
        step: int,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        success: bool = False,
        result: Any = None,
        error: str = "",
    ) -> ToolCallLog:
        clean_session_id = str(session_id or "").strip()
        clean_message_type = str(message_type or "").strip() or "private"
        clean_speaker = str(speaker or "").strip()
        clean_tool_name = str(tool_name or "").strip()
        clean_tool_call_id = str(tool_call_id or "").strip()
        safe_arguments = self._sanitize_tool_payload(arguments if isinstance(arguments, dict) else {})
        safe_result = self._sanitize_tool_payload(result)
        clean_error = str(error or "").strip()

        async with self._lock:
            self._tool_call_id += 1
            item = ToolCallLog(
                id=self._tool_call_id,
                session_id=clean_session_id,
                message_type=clean_message_type,
                group_id=int(group_id or 0),
                user_id=int(user_id or 0),
                speaker=clean_speaker,
                step=max(0, int(step or 0)),
                tool_call_id=clean_tool_call_id,
                tool_name=clean_tool_name,
                arguments=safe_arguments if isinstance(safe_arguments, dict) else {},
                success=bool(success),
                result=safe_result,
                error=clean_error,
                created_at=datetime.now(timezone.utc),
            )
            self._tool_calls.append(item)
            self._tool_call_by_id[item.id] = item
            self._trim_tool_calls_locked()

        await self._save_tool_call_store()
        return item

    async def list_tool_calls(
        self,
        *,
        keyword: str = "",
        tool_name: str = "",
        session_id: str = "",
        group_id: int = 0,
        success: bool | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        clean_keyword = str(keyword or "").strip().lower()
        clean_tool_name = str(tool_name or "").strip().lower()
        clean_session_id = str(session_id or "").strip().lower()
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))

        async with self._lock:
            rows = list(self._tool_calls)

        filtered: list[ToolCallLog] = []
        for item in rows:
            if clean_tool_name and clean_tool_name not in str(item.tool_name or "").lower():
                continue
            if clean_session_id and clean_session_id not in str(item.session_id or "").lower():
                continue
            if int(group_id or 0) > 0 and int(item.group_id or 0) != int(group_id):
                continue
            if success is not None and bool(item.success) != bool(success):
                continue
            if clean_keyword:
                search_blocks = [
                    str(item.tool_name or ""),
                    str(item.session_id or ""),
                    str(item.speaker or ""),
                    str(item.error or ""),
                    str(item.tool_call_id or ""),
                    json.dumps(item.arguments, ensure_ascii=False, default=str),
                    json.dumps(item.result, ensure_ascii=False, default=str),
                ]
                haystack = "\n".join(search_blocks).lower()
                if clean_keyword not in haystack:
                    continue
            filtered.append(item)

        filtered.sort(key=lambda row: int(row.id), reverse=True)
        total = len(filtered)
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size
        page_rows = filtered[start:end]
        return {
            "items": [self._tool_call_row(item) for item in page_rows],
            "total": total,
            "page": safe_page,
            "page_size": safe_page_size,
        }

    async def get_tool_call_detail(self, tool_call_log_id: int) -> dict[str, Any] | None:
        clean_id = int(tool_call_log_id or 0)
        if clean_id <= 0:
            return None
        async with self._lock:
            item = self._tool_call_by_id.get(clean_id)
            if item is None:
                return None
            return item.to_dict()

    async def delete_tool_call(self, tool_call_log_id: int) -> bool:
        clean_id = int(tool_call_log_id or 0)
        if clean_id <= 0:
            return False
        removed = False
        async with self._lock:
            item = self._tool_call_by_id.pop(clean_id, None)
            if item is not None:
                self._tool_calls = [row for row in self._tool_calls if int(row.id) != clean_id]
                removed = True
        if removed:
            await self._save_tool_call_store()
        return removed

    async def clear_tool_calls(
        self,
        *,
        tool_name: str = "",
        session_id: str = "",
        group_id: int = 0,
        success: bool | None = None,
    ) -> int:
        clean_tool_name = str(tool_name or "").strip().lower()
        clean_session_id = str(session_id or "").strip().lower()
        clean_group_id = int(group_id or 0)
        removed_items: list[ToolCallLog] = []
        async with self._lock:
            kept: list[ToolCallLog] = []
            for item in self._tool_calls:
                matched = True
                if clean_tool_name and clean_tool_name not in str(item.tool_name or "").lower():
                    matched = False
                if matched and clean_session_id and clean_session_id not in str(item.session_id or "").lower():
                    matched = False
                if matched and clean_group_id > 0 and int(item.group_id or 0) != clean_group_id:
                    matched = False
                if matched and success is not None and bool(item.success) != bool(success):
                    matched = False
                if matched:
                    removed_items.append(item)
                    continue
                kept.append(item)
            if not removed_items:
                return 0
            self._tool_calls = kept
            for item in removed_items:
                self._tool_call_by_id.pop(int(item.id), None)
        await self._save_tool_call_store()
        return len(removed_items)

    async def get_tool_call_stats(self) -> dict[str, Any]:
        async with self._lock:
            rows = list(self._tool_calls)
        total = len(rows)
        success_count = sum(1 for item in rows if bool(item.success))
        fail_count = total - success_count
        last_item = rows[-1] if rows else None
        return {
            "total": total,
            "success": success_count,
            "failed": fail_count,
            "last_at": self._ensure_utc(last_item.created_at).isoformat() if last_item is not None else "",
            "max_entries": self._tool_call_max_entries,
        }

    async def list_memories(
        self,
        *,
        group_id: int = 0,
        mem_type: str = "",
        status: str = "",
        canonical_type: str = "",
        source_kind: str = "",
        keyword: str = "",
        page: int = 1,
        page_size: int = 20,
        sort: str = "updated",
        order: str = "desc",
    ) -> dict[str, Any]:
        clean_keyword = str(keyword or "").strip().lower()
        clean_mem_type = str(mem_type or "").strip().lower()
        clean_status = str(status or "").strip().lower()
        clean_canonical = str(canonical_type or "").strip().lower()
        clean_source = str(source_kind or "").strip().lower()

        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))

        async with self._lock:
            rows = list(self._memory_by_id.values())

        if group_id > 0:
            rows = [item for item in rows if int(item.group_id) == int(group_id)]
        if clean_mem_type:
            rows = [item for item in rows if str(item.mem_type) == clean_mem_type]
        if clean_status in {
            MEMORY_STATUS_ACTIVE,
            MEMORY_STATUS_CANDIDATE,
            MEMORY_STATUS_ARCHIVED,
            MEMORY_STATUS_LEGACY,
        }:
            rows = [item for item in rows if item.effective_status() == clean_status]
        if clean_canonical in {"fact", "episode", "preference", "constraint", "goal"}:
            rows = [item for item in rows if str(item.canonical_type) == clean_canonical]
        if clean_source in {"message", "summary", "topic", "manual", "migration"}:
            rows = [item for item in rows if str(item.source_kind) == clean_source]
        if clean_keyword:
            rows = [
                item
                for item in rows
                if clean_keyword in self._memory_search_text(item)
            ]

        rows = self._sort_memories(rows, sort=sort, order=order)
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size
        page_items = rows[start:end]
        return {
            "items": [self._memory_row(item) for item in page_items],
            "total": len(rows),
            "page": safe_page,
            "page_size": safe_page_size,
        }

    async def get_memory_detail(self, memory_id: int) -> dict[str, Any] | None:
        async with self._lock:
            item = self._memory_by_id.get(int(memory_id))
            if item is None:
                return None
            return self._memory_row(item)

    async def set_memory_status(self, *, memory_id: int, status: str) -> bool:
        clean_status = str(status or "").strip().lower()
        if clean_status not in {
            MEMORY_STATUS_ACTIVE,
            MEMORY_STATUS_CANDIDATE,
            MEMORY_STATUS_ARCHIVED,
            MEMORY_STATUS_LEGACY,
        }:
            return False

        item_for_index: MemoryItem | None = None
        changed = False
        async with self._lock:
            item = self._memory_by_id.get(int(memory_id))
            if item is None:
                return False
            if item.effective_status() != clean_status:
                item.status = clean_status  # type: ignore[assignment]
                item.updated_at = datetime.now(timezone.utc)
                if item.importance <= 0:
                    item.importance = self._importance_for_status(item.canonical_type, clean_status, item.evidence_count)
                changed = True
            if clean_status == MEMORY_STATUS_ACTIVE:
                item_for_index = item
            if changed and item.fact_key and clean_status == MEMORY_STATUS_ACTIVE:
                self._archive_conflicts_locked(item)

        if changed:
            await self._save_memory_store()
            if item_for_index is not None:
                await self._index_memory_if_eligible(item_for_index)
        return True

    async def delete_memory(self, memory_id: int) -> bool:
        removed = False
        async with self._lock:
            item = self._memory_by_id.pop(int(memory_id), None)
            if item is None:
                return False
            rows = self._memories.get(item.group_id, [])
            self._memories[item.group_id] = [row for row in rows if row.id != item.id]
            self._indexed_memory_ids.discard(item.id)
            removed = True

        if removed:
            await self._save_memory_store()
        return removed

    async def run_memory_convergence(self, *, reason: str = "manual") -> dict[str, int]:
        now = datetime.now(timezone.utc)
        changed_ids: set[int] = set()
        promoted = 0
        archived = 0
        async with self._lock:
            for item in list(self._memory_by_id.values()):
                if item.effective_status() != MEMORY_STATUS_CANDIDATE:
                    continue
                if self._can_promote_candidate(item, source_kind=str(item.source_kind), evidence_count=item.evidence_count):
                    item.status = MEMORY_STATUS_ACTIVE
                    item.updated_at = now
                    item.importance = self._importance_for_status(item.canonical_type, MEMORY_STATUS_ACTIVE, item.evidence_count)
                    promoted += 1
                    changed_ids.add(item.id)
                    if item.fact_key:
                        for archived_item in self._archive_conflicts_locked(item):
                            archived += 1
                            changed_ids.add(archived_item.id)
                    continue

                age = now - self._ensure_utc(item.updated_at)
                if age < timedelta(hours=self._candidate_grace_hours):
                    continue

                item.status = MEMORY_STATUS_ARCHIVED
                item.updated_at = now
                item.importance = self._importance_for_status(item.canonical_type, MEMORY_STATUS_ARCHIVED, item.evidence_count)
                archived += 1
                changed_ids.add(item.id)

        if changed_ids:
            await self._save_memory_store()
            async with self._lock:
                active_to_index = [self._memory_by_id[item_id] for item_id in changed_ids if item_id in self._memory_by_id and self._memory_by_id[item_id].effective_status() == MEMORY_STATUS_ACTIVE]
            for item in active_to_index:
                await self._index_memory_if_eligible(item)

        self._logger.info(
            "Memory convergence: reason=%s promoted=%s archived=%s changed=%s",
            reason,
            promoted,
            archived,
            len(changed_ids),
        )
        return {"promoted": promoted, "archived": archived, "changed": len(changed_ids)}

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
                        "Memory sediment success: session=%s memory_id=%s reason=%s",
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

    async def _run_convergence_worker(self) -> None:
        interval = max(1, int(self._convergence_interval_minutes))
        await asyncio.sleep(2.0)
        try:
            await self.run_memory_convergence(reason="startup")
        except Exception:
            self._logger.exception("Memory convergence startup run failed")
        while True:
            try:
                await asyncio.sleep(interval * 60)
                await self.run_memory_convergence(reason="timer")
            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.exception("Memory convergence timer run failed")

    async def _summarize_turns(self, task: SummaryTask) -> str:
        dialog_lines = []
        for idx, turn in enumerate(task.turns, start=1):
            timestamp = self._ensure_utc(turn.created_at).astimezone().strftime("%m-%d %H:%M:%S")
            dialog_lines.append(f"{idx}. [{timestamp}] {turn.speaker}({turn.role}): {turn.content}")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是记忆提炼器。请把对话压缩为长期记忆，必须输出纯文本，包含三段："
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
        item, _ = await self._upsert_memory_candidate(
            group_id=task.group_id,
            user_id=int(self.cfg.persona.qq or 0),
            content=summary,
            source_kind="summary",
            source_ref=f"summary:{task.session_id}:{int(datetime.now(timezone.utc).timestamp())}",
            mem_type_hint="conversation",
            canonical_type_hint="episode",
            force_status=MEMORY_STATUS_CANDIDATE,
            importance_override=self._importance_for_status("episode", MEMORY_STATUS_CANDIDATE, 1),
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
            "用户偏好：根据近几次对话继续保持当前风格回应。\n"
            "重要约定：" + " | ".join(previews)
        )

    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 120) -> list[str]:
        clean = str(text).strip()
        if not clean:
            return []
        chunk_size = max(200, int(chunk_size))
        overlap = max(0, min(int(overlap), chunk_size // 2))
        step = chunk_size - overlap
        chunks: list[str] = []
        start = 0
        total = len(clean)
        while start < total:
            end = min(total, start + chunk_size)
            chunk = clean[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= total:
                break
            start += step
        return chunks

    async def _rows_to_memory_lines(
        self,
        rows: list[VectorRecord],
        *,
        session_id: str,
        group_id: int | None,
        limit: int,
    ) -> list[str]:
        del session_id
        out: list[str] = []
        seen_ids: set[int] = set()
        async with self._lock:
            for row in rows:
                if len(out) >= limit:
                    break
                memory_id = self._read_memory_id(row)
                if memory_id is None:
                    text = self._format_row_text(row)
                    if text:
                        out.append(text)
                    continue
                item = self._memory_by_id.get(memory_id)
                if item is None:
                    continue
                if item.id in seen_ids:
                    continue
                if group_id is not None and int(item.group_id) != int(group_id):
                    continue
                if not item.recall_eligible():
                    continue
                item.access_count += 1
                seen_ids.add(item.id)
                out.append(self._memory_prompt_line(item, distance=row.distance))
        return out

    def _memory_prompt_line(self, item: MemoryItem, *, distance: float | None) -> str:
        label = f"[{item.canonical_type}/{item.mem_type}]"
        base = f"{label} {item.content}".strip()
        if distance is None:
            return base
        return f"{base} (distance={distance:.3f})"

    async def _upsert_memory_candidate(
        self,
        *,
        group_id: int,
        user_id: int,
        content: str,
        source_kind: MemorySourceKind,
        source_ref: str,
        mem_type_hint: MemoryType | None = None,
        canonical_type_hint: CanonicalMemoryType | None = None,
        force_status: MemoryStatus | None = None,
        importance_override: float | None = None,
    ) -> tuple[MemoryItem, str]:
        clean_text = self._normalize_text(content)
        if not clean_text:
            raise ValueError("memory content is empty")

        now = datetime.now(timezone.utc)
        canonical = canonical_type_hint or self._infer_canonical_type(clean_text)
        mem_type = mem_type_hint or self._infer_mem_type(canonical, user_id=user_id, group_id=group_id)
        slot_kind, slot_anchor = self._infer_slot_kind_and_anchor(canonical, clean_text)
        fact_key = ""
        if canonical in KEYED_CANONICAL_TYPES:
            fact_key = self._build_fact_key(
                group_id=group_id,
                user_id=user_id,
                mem_type=mem_type,
                slot_kind=slot_kind,
                slot_anchor=slot_anchor,
            )

        raw_status = force_status or MEMORY_STATUS_CANDIDATE
        status: MemoryStatus = raw_status if raw_status in {
            MEMORY_STATUS_ACTIVE,
            MEMORY_STATUS_CANDIDATE,
            MEMORY_STATUS_ARCHIVED,
            MEMORY_STATUS_LEGACY,
        } else MEMORY_STATUS_CANDIDATE

        to_index: list[MemoryItem] = []
        save_needed = False
        early_result: tuple[MemoryItem, str] | None = None

        async with self._lock:
            existing = self._find_latest_fact_match_locked(
                group_id=group_id,
                canonical_type=canonical,
                fact_key=fact_key,
            )
            if canonical == "episode" and source_ref:
                dup = self._find_duplicate_episode_locked(
                    group_id=group_id,
                    source_ref=source_ref,
                    content=clean_text,
                )
                if dup is not None:
                    old_ref = str(dup.source_ref)
                    dup.updated_at = now
                    dup.source_kind = source_kind
                    dup.source_ref = source_ref
                    dup.evidence_count = max(1, int(dup.evidence_count) + (1 if old_ref and old_ref != source_ref else 0))
                    dup.importance = self._importance_for_status(dup.canonical_type, dup.effective_status(), dup.evidence_count)
                    save_needed = True
                    result = dup
                    action = "deduplicated"
                    early_result = (result, action)

            if early_result is None and existing is not None and self._same_claim_value(existing.content, clean_text):
                old_ref = str(existing.source_ref)
                existing.updated_at = now
                existing.source_kind = source_kind
                existing.source_ref = source_ref
                action = "deduplicated"
                if source_ref and old_ref != source_ref:
                    existing.evidence_count = int(existing.evidence_count) + 1
                    action = "reinforced"

                if existing.effective_status() == MEMORY_STATUS_CANDIDATE and self._can_promote_candidate(
                    existing,
                    source_kind=str(source_kind),
                    evidence_count=existing.evidence_count,
                ):
                    existing.status = MEMORY_STATUS_ACTIVE
                    action = "promoted"
                    to_index.append(existing)
                    self._archive_conflicts_locked(existing)

                next_status = existing.effective_status()
                if force_status is not None:
                    existing.status = force_status
                    next_status = existing.effective_status()
                existing.importance = (
                    float(importance_override)
                    if importance_override is not None and importance_override > 0
                    else self._importance_for_status(existing.canonical_type, next_status, existing.evidence_count)
                )
                save_needed = True
                if existing.effective_status() == MEMORY_STATUS_ACTIVE:
                    to_index.append(existing)
                result = existing
            elif early_result is None:
                self._id += 1
                evidence_count = 1
                if status == MEMORY_STATUS_CANDIDATE and self._can_promote_candidate_dummy(
                    canonical_type=canonical,
                    source_kind=source_kind,
                    evidence_count=evidence_count,
                    fact_key=fact_key,
                ):
                    status = MEMORY_STATUS_ACTIVE
                importance = (
                    float(importance_override)
                    if importance_override is not None and importance_override > 0
                    else self._importance_for_status(canonical, status, evidence_count)
                )
                item = MemoryItem(
                    id=self._id,
                    group_id=int(group_id),
                    user_id=int(user_id or 0),
                    mem_type=mem_type,
                    content=clean_text,
                    canonical_type=canonical,
                    status=status,
                    evidence_count=evidence_count,
                    source_kind=source_kind,
                    source_ref=source_ref,
                    fact_key=fact_key,
                    importance=importance,
                    access_count=0,
                    created_at=now,
                    updated_at=now,
                )
                self._register_memory_locked(item)
                result = item
                action = "created"
                if existing is not None and fact_key:
                    action = "conflict-candidate"
                if result.effective_status() == MEMORY_STATUS_ACTIVE:
                    to_index.append(result)
                    self._archive_conflicts_locked(result)
                save_needed = True

        if save_needed:
            await self._save_memory_store()
        for row in to_index:
            await self._index_memory_if_eligible(row)
        if early_result is not None:
            return early_result
        return result, action

    def _register_memory_locked(self, item: MemoryItem) -> None:
        item.created_at = self._ensure_utc(item.created_at)
        item.updated_at = self._ensure_utc(item.updated_at)
        self._memory_by_id[item.id] = item
        rows = self._memories[item.group_id]
        rows.append(item)
        self._id = max(self._id, item.id)

    async def _index_memory_if_eligible(self, item: MemoryItem) -> None:
        if item.id <= 0:
            return
        if not item.recall_eligible():
            return
        if item.id in self._indexed_memory_ids:
            return

        await self._vector_storage.store(
            item.content,
            "user_memories",
            metadata={
                "memory_id": item.id,
                "group_id": int(item.group_id),
                "mem_type": str(item.mem_type),
                "importance": float(item.importance),
                "canonical_type": str(item.canonical_type),
                "status": str(item.status),
                "source_kind": str(item.source_kind),
                "source_ref": str(item.source_ref),
                "fact_key": str(item.fact_key),
                "created_at": self._ensure_utc(item.created_at).isoformat(),
                "updated_at": self._ensure_utc(item.updated_at).isoformat(),
            },
        )
        self._indexed_memory_ids.add(item.id)

    def _find_duplicate_episode_locked(self, *, group_id: int, source_ref: str, content: str) -> MemoryItem | None:
        for item in reversed(self._memories.get(group_id, [])):
            if item.canonical_type != "episode":
                continue
            if str(item.source_ref) != str(source_ref):
                continue
            if self._same_claim_value(item.content, content):
                return item
        return None

    def _find_latest_fact_match_locked(
        self,
        *,
        group_id: int,
        canonical_type: CanonicalMemoryType,
        fact_key: str,
    ) -> MemoryItem | None:
        if not fact_key:
            return None
        for item in reversed(self._memories.get(group_id, [])):
            if item.canonical_type != canonical_type:
                continue
            if item.fact_key != fact_key:
                continue
            if item.effective_status() == MEMORY_STATUS_ARCHIVED:
                continue
            return item
        return None

    def _archive_conflicts_locked(self, current: MemoryItem) -> list[MemoryItem]:
        if not current.fact_key:
            return []
        changed: list[MemoryItem] = []
        now = datetime.now(timezone.utc)
        for item in self._memories.get(current.group_id, []):
            if item.id == current.id:
                continue
            if item.canonical_type != current.canonical_type:
                continue
            if item.fact_key != current.fact_key:
                continue
            if item.effective_status() != MEMORY_STATUS_ACTIVE:
                continue
            item.status = MEMORY_STATUS_ARCHIVED
            item.updated_at = now
            item.importance = self._importance_for_status(item.canonical_type, MEMORY_STATUS_ARCHIVED, item.evidence_count)
            changed.append(item)
        return changed

    @staticmethod
    def _same_claim_value(existing: str, incoming: str) -> bool:
        left = MemoryManager._normalize_compare_text(existing)
        right = MemoryManager._normalize_compare_text(incoming)
        return bool(left) and left == right

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        return "".join(str(text or "").strip().lower().split())

    def _can_promote_candidate(
        self,
        item: MemoryItem,
        *,
        source_kind: str,
        evidence_count: int,
    ) -> bool:
        if item.canonical_type == "episode":
            return source_kind in self._SOURCE_PROMOTE_SET
        if not item.fact_key:
            return False
        if source_kind in self._SOURCE_PROMOTE_SET:
            return True
        return int(evidence_count) >= int(self._candidate_promote_evidence)

    def _can_promote_candidate_dummy(
        self,
        *,
        canonical_type: CanonicalMemoryType,
        source_kind: MemorySourceKind,
        evidence_count: int,
        fact_key: str,
    ) -> bool:
        if canonical_type == "episode":
            return str(source_kind) in self._SOURCE_PROMOTE_SET
        if not fact_key:
            return False
        if str(source_kind) in self._SOURCE_PROMOTE_SET:
            return True
        return int(evidence_count) >= int(self._candidate_promote_evidence)

    @staticmethod
    def _importance_for_status(
        canonical_type: CanonicalMemoryType,
        status: MemoryStatus,
        evidence_count: int,
    ) -> float:
        base = 0.45
        if canonical_type == "constraint":
            base = 0.82
        elif canonical_type == "goal":
            base = 0.74
        elif canonical_type == "preference":
            base = 0.62
        elif canonical_type == "episode":
            base = 0.58
        elif canonical_type == "fact":
            base = 0.68

        if status == MEMORY_STATUS_CANDIDATE:
            base -= 0.18
        elif status == MEMORY_STATUS_LEGACY:
            base -= 0.1
        elif status == MEMORY_STATUS_ARCHIVED:
            base -= 0.26

        if evidence_count > 1:
            base += float(evidence_count - 1) * 0.05

        if base < 0.1:
            return 0.1
        if base > 0.98:
            return 0.98
        return base

    @staticmethod
    def _infer_canonical_type(text: str) -> CanonicalMemoryType:
        clean = str(text or "").strip()
        lowered = clean.lower()
        if any(token in clean for token in ("喜欢", "讨厌", "偏好", "习惯", "爱好", "不喜欢")):
            return "preference"
        if any(token in clean for token in ("必须", "不要", "不能", "禁止", "约定", "规矩", "规则", "忌讳", "别")):
            return "constraint"
        if any(token in clean for token in ("目标", "计划", "打算", "准备", "截止", "里程碑", "推进", "完成")):
            return "goal"
        if any(token in clean for token in ("今天", "昨天", "刚刚", "刚才", "已经", "发生", "经历")) or any(
            token in lowered for token in ("today", "yesterday", "just", "happened")
        ):
            return "episode"
        return "fact"

    def _infer_mem_type(self, canonical_type: CanonicalMemoryType, *, user_id: int, group_id: int) -> MemoryType:
        bot_qq = int(getattr(self.cfg.persona, "qq", 0) or 0)
        if canonical_type == "episode":
            if bot_qq > 0 and int(user_id) == bot_qq:
                return "self_experience"
            return "conversation"
        if group_id > 0:
            return "group_fact"
        return "conversation"

    @staticmethod
    def _infer_slot_kind_and_anchor(canonical_type: CanonicalMemoryType, text: str) -> tuple[str, str]:
        clean = str(text or "").strip()
        if canonical_type == "preference":
            if any(token in clean for token in ("不喜欢", "讨厌", "反感", "排斥")):
                return "dislike", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("习惯", "通常", "经常", "总是")):
                return "habit", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("风格", "语气", "说话", "写法")):
                return "style", MemoryManager._slot_anchor(clean)
            return "like", MemoryManager._slot_anchor(clean)

        if canonical_type == "constraint":
            if any(token in clean for token in ("禁", "别", "不能", "不要")):
                return "taboo", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("边界", "范围", "权限")):
                return "boundary", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("避免", "绕开", "避开")):
                return "avoid", MemoryManager._slot_anchor(clean)
            return "rule", MemoryManager._slot_anchor(clean)

        if canonical_type == "goal":
            if any(token in clean for token in ("截止", "ddl", "期限", "到期")):
                return "deadline", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("里程碑", "阶段")):
                return "milestone", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("项目", "工程")):
                return "project", MemoryManager._slot_anchor(clean)
            return "task", MemoryManager._slot_anchor(clean)

        if canonical_type == "fact":
            if any(token in clean for token in ("身份", "是", "叫", "昵称")):
                return "identity", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("关系", "同学", "同事", "朋友")):
                return "relation", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("负责", "岗位", "角色")):
                return "role", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("安排", "排期", "时间", "周", "月")):
                return "schedule", MemoryManager._slot_anchor(clean)
            if any(token in clean for token in ("结论", "确认", "决定")):
                return "conclusion", MemoryManager._slot_anchor(clean)
            return "status", MemoryManager._slot_anchor(clean)

        return "", ""

    @staticmethod
    def _slot_anchor(text: str) -> str:
        tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,8}", str(text or "").lower())
        if not tokens:
            return ""
        stopwords = {
            "就是",
            "这个",
            "那个",
            "我们",
            "你们",
            "他们",
            "因为",
            "所以",
            "然后",
            "已经",
            "应该",
            "可以",
            "不能",
            "不要",
            "喜欢",
            "不喜欢",
            "讨厌",
            "计划",
            "目标",
            "规则",
            "约定",
        }
        picked: list[str] = []
        for token in tokens:
            if token in stopwords:
                continue
            if token.isdigit():
                continue
            picked.append(token)
            if len(picked) >= 4:
                break
        if not picked:
            picked = tokens[:2]
        anchor = "_".join(picked).strip("_")
        if len(anchor) > 40:
            anchor = anchor[:40].strip("_")
        return anchor

    def _build_fact_key(
        self,
        *,
        group_id: int,
        user_id: int,
        mem_type: MemoryType,
        slot_kind: str,
        slot_anchor: str,
    ) -> str:
        slot_kind = str(slot_kind or "").strip()
        slot_anchor = str(slot_anchor or "").strip()
        if not slot_kind or not slot_anchor:
            return ""

        scope_code = "cv"
        if mem_type == "group_fact":
            scope_code = "gf"
        elif mem_type == "self_experience":
            scope_code = "se"

        bot_qq = int(getattr(self.cfg.persona, "qq", 0) or 0)
        if bot_qq > 0 and int(user_id) == bot_qq:
            subject_token = f"self:{bot_qq}"
        elif user_id > 0:
            subject_token = f"u:{int(user_id)}"
        elif group_id > 0:
            subject_token = f"g:{int(group_id)}"
        else:
            subject_token = "unknown"

        short_hash = hashlib.sha1(slot_anchor.encode("utf-8")).hexdigest()[:10]
        return f"{scope_code}:{subject_token}:{slot_kind}:{short_hash}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        clean = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def _is_low_signal_text(text: str) -> bool:
        clean = str(text or "").strip()
        if not clean:
            return True
        compact = re.sub(r"[\W_]+", "", clean, flags=re.UNICODE)
        if len(compact) <= 2:
            return True
        if compact.isdigit() and len(compact) <= 5:
            return True
        return False

    def _memory_search_text(self, item: MemoryItem) -> str:
        return "\n".join(
            [
                str(item.content or "").lower(),
                str(item.mem_type or "").lower(),
                str(item.canonical_type or "").lower(),
                str(item.status or "").lower(),
                str(item.source_kind or "").lower(),
                str(item.source_ref or "").lower(),
                str(item.fact_key or "").lower(),
                str(item.user_id or "").lower(),
            ],
        )

    def _sort_memories(self, rows: list[MemoryItem], *, sort: str, order: str) -> list[MemoryItem]:
        clean_sort = str(sort or "").strip().lower()
        reverse = str(order or "").strip().lower() != "asc"

        def key_updated(item: MemoryItem) -> datetime:
            return self._ensure_utc(item.updated_at)

        def key_created(item: MemoryItem) -> datetime:
            return self._ensure_utc(item.created_at)

        def key_access(item: MemoryItem) -> int:
            return int(item.access_count)

        def key_importance(item: MemoryItem) -> float:
            return float(item.importance)

        def key_evidence(item: MemoryItem) -> int:
            return int(item.evidence_count)

        key_map = {
            "updated": key_updated,
            "created": key_created,
            "access": key_access,
            "importance": key_importance,
            "evidence": key_evidence,
        }
        key_func = key_map.get(clean_sort, key_updated)
        return sorted(rows, key=key_func, reverse=reverse)

    @staticmethod
    def _memory_row(item: MemoryItem) -> dict[str, Any]:
        return {
            "id": int(item.id),
            "group_id": int(item.group_id),
            "user_id": int(item.user_id or 0),
            "mem_type": str(item.mem_type),
            "canonical_type": str(item.canonical_type),
            "status": str(item.effective_status()),
            "evidence_count": int(item.evidence_count),
            "source_kind": str(item.source_kind),
            "source_ref": str(item.source_ref),
            "fact_key": str(item.fact_key),
            "importance": round(float(item.importance), 4),
            "access_count": int(item.access_count),
            "content": str(item.content),
            "created_at": MemoryManager._ensure_utc(item.created_at).isoformat(),
            "updated_at": MemoryManager._ensure_utc(item.updated_at).isoformat(),
        }

    async def _load_memory_store(self) -> None:
        if not self._store_path.exists():
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            return

        try:
            raw = self._store_path.read_text(encoding="utf-8")
        except OSError:
            self._logger.exception("Memory store read failed: path=%s", self._store_path)
            return
        if not raw.strip():
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._logger.warning("Memory store decode failed: path=%s err=%s", self._store_path, exc)
            return

        rows = payload.get("memories", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return

        loaded: list[MemoryItem] = []
        max_id = 0
        for row in rows:
            item = MemoryItem.from_dict(row)
            if item is None:
                continue
            item.created_at = self._ensure_utc(item.created_at)
            item.updated_at = self._ensure_utc(item.updated_at)
            loaded.append(item)
            max_id = max(max_id, int(item.id))

        async with self._lock:
            self._messages = defaultdict(list)
            self._memories = defaultdict(list)
            self._memory_by_id = {}
            self._indexed_memory_ids = set()
            self._id = max(self._id, int(payload.get("next_id", 0) or 0), max_id)
            for item in loaded:
                self._register_memory_locked(item)

        self._logger.info("Memory store loaded: path=%s count=%s", self._store_path, len(loaded))

    async def _save_memory_store(self) -> None:
        async with self._lock:
            rows = [item.to_dict() for item in sorted(self._memory_by_id.values(), key=lambda row: row.id)]
            payload = {
                "version": 1,
                "next_id": int(self._id),
                "memories": rows,
            }

        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._store_path.with_suffix(self._store_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temp_path.replace(self._store_path)

    @staticmethod
    def _sanitize_tool_payload(value: Any, *, depth: int = 0) -> Any:
        if depth >= 6:
            return str(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, str) and len(value) > 4000:
                return value[:4000]
            return value
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                out[str(key)] = MemoryManager._sanitize_tool_payload(item, depth=depth + 1)
            return out
        if isinstance(value, (list, tuple, set)):
            return [MemoryManager._sanitize_tool_payload(item, depth=depth + 1) for item in list(value)]
        return str(value)

    @staticmethod
    def _tool_payload_preview(value: Any, *, max_len: int = 200) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except TypeError:
            text = str(value)
        if len(text) <= max(40, int(max_len)):
            return text
        return text[: max(40, int(max_len))] + "..."

    def _tool_call_row(self, item: ToolCallLog) -> dict[str, Any]:
        return {
            "id": int(item.id),
            "session_id": str(item.session_id or ""),
            "message_type": str(item.message_type or ""),
            "group_id": int(item.group_id or 0),
            "user_id": int(item.user_id or 0),
            "speaker": str(item.speaker or ""),
            "step": int(item.step or 0),
            "tool_call_id": str(item.tool_call_id or ""),
            "tool_name": str(item.tool_name or ""),
            "success": bool(item.success),
            "error": str(item.error or ""),
            "arguments_preview": self._tool_payload_preview(item.arguments, max_len=160),
            "result_preview": self._tool_payload_preview(item.result, max_len=200),
            "created_at": self._ensure_utc(item.created_at).isoformat(),
        }

    def _trim_tool_calls_locked(self) -> None:
        overflow = len(self._tool_calls) - int(self._tool_call_max_entries)
        if overflow <= 0:
            return
        removed = self._tool_calls[:overflow]
        self._tool_calls = self._tool_calls[overflow:]
        for item in removed:
            self._tool_call_by_id.pop(int(item.id), None)

    async def _load_tool_call_store(self) -> None:
        if not self._tool_call_store_path.exists():
            self._tool_call_store_path.parent.mkdir(parents=True, exist_ok=True)
            return

        try:
            raw = self._tool_call_store_path.read_text(encoding="utf-8")
        except OSError:
            self._logger.exception("Tool call store read failed: path=%s", self._tool_call_store_path)
            return
        if not raw.strip():
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._logger.warning("Tool call store decode failed: path=%s err=%s", self._tool_call_store_path, exc)
            return

        rows = payload.get("tool_calls", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return

        loaded: list[ToolCallLog] = []
        max_id = 0
        for row in rows:
            item = ToolCallLog.from_dict(row if isinstance(row, dict) else {})
            if item is None:
                continue
            item.created_at = self._ensure_utc(item.created_at)
            item.arguments = self._sanitize_tool_payload(item.arguments) if isinstance(item.arguments, dict) else {}
            item.result = self._sanitize_tool_payload(item.result)
            loaded.append(item)
            max_id = max(max_id, int(item.id))

        loaded.sort(key=lambda row: int(row.id))
        if len(loaded) > self._tool_call_max_entries:
            loaded = loaded[-self._tool_call_max_entries :]

        async with self._lock:
            self._tool_calls = loaded
            self._tool_call_by_id = {int(item.id): item for item in loaded}
            self._tool_call_id = max(self._tool_call_id, int(payload.get("next_id", 0) or 0), max_id)

        self._logger.info("Tool call store loaded: path=%s count=%s", self._tool_call_store_path, len(loaded))

    async def _save_tool_call_store(self) -> None:
        async with self._lock:
            rows = [item.to_dict() for item in self._tool_calls]
            payload = {
                "version": 1,
                "next_id": int(self._tool_call_id),
                "tool_calls": rows,
            }

        self._tool_call_store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._tool_call_store_path.with_suffix(self._tool_call_store_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temp_path.replace(self._tool_call_store_path)

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

