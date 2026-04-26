from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from adapters.llm.embedding import EmbeddingAdapter
from internal.logger import get_logger

try:
    import chromadb
except Exception:  # pragma: no cover - runtime fallback when chromadb is not installed
    chromadb = None


CollectionType = Literal["user_memories", "external_knowledge"]


@dataclass(slots=True)
class VectorRecord:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


class ChromaVectorStorage:
    def __init__(self, *, persist_path: str, embedding_adapter: EmbeddingAdapter) -> None:
        self.persist_path = Path(persist_path)
        self.embedding_adapter = embedding_adapter
        self._logger = get_logger("ChromaVectorStorage")
        self._lock = asyncio.Lock()

        self._initialized = False
        self._use_chroma = False
        self._client: Any | None = None
        self._collections: dict[CollectionType, Any] = {}
        self._fallback_rows: dict[CollectionType, list[tuple[str, str, dict[str, Any], list[float]]]] = {
            "user_memories": [],
            "external_knowledge": [],
        }

    async def start(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            self.persist_path.mkdir(parents=True, exist_ok=True)
            if chromadb is None:
                self._use_chroma = False
                self._initialized = True
                self._logger.warning(
                    "ChromaDB is not installed; memory vector storage runs in fallback in-memory mode.",
                )
                return

            self._client = chromadb.PersistentClient(path=str(self.persist_path))
            self._collections["user_memories"] = self._client.get_or_create_collection(name="user_memories")
            self._collections["external_knowledge"] = self._client.get_or_create_collection(name="external_knowledge")
            self._use_chroma = True
            self._initialized = True
            self._logger.info(
                "Vector storage ready: mode=chromadb path=%s collections=%s",
                self.persist_path,
                ",".join(self._collections.keys()),
            )

    async def close(self) -> None:
        async with self._lock:
            self._initialized = False
            self._collections.clear()
            self._client = None

    async def store(
        self,
        text: str,
        collection_type: CollectionType,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        await self.start()
        clean_text = text.strip()
        if not clean_text:
            return ""

        embedding = await self.embedding_adapter.embed(clean_text)
        if not embedding:
            self._logger.warning("Vector store skipped: empty embedding collection=%s", collection_type)
            return ""

        row_id = str(uuid4())
        safe_metadata = self._sanitize_metadata(metadata or {})
        safe_metadata.setdefault("collection_type", collection_type)

        if self._use_chroma:
            collection = self._collections[collection_type]
            await asyncio.to_thread(
                collection.add,
                ids=[row_id],
                documents=[clean_text],
                embeddings=[embedding],
                metadatas=[safe_metadata],
            )
            return row_id

        self._fallback_rows[collection_type].append((row_id, clean_text, safe_metadata, embedding))
        return row_id

    async def query(
        self,
        text: str,
        collection_type: CollectionType,
        top_k: int = 5,
    ) -> list[VectorRecord]:
        await self.start()
        clean_text = text.strip()
        if not clean_text:
            return []

        embedding = await self.embedding_adapter.embed(clean_text)
        if not embedding:
            self._logger.warning("Vector query skipped: empty embedding collection=%s", collection_type)
            return []

        limit = max(1, int(top_k))
        if self._use_chroma:
            collection = self._collections[collection_type]
            payload = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            return self._parse_query_payload(payload)

        return self._fallback_query(collection_type, embedding, limit)

    async def clear_collection(self, collection_type: CollectionType) -> None:
        await self.start()
        if self._use_chroma:
            name = str(collection_type)
            try:
                await asyncio.to_thread(self._client.delete_collection, name=name)
            except Exception:
                # Keep idempotent semantics for reindex flows.
                pass
            self._collections[collection_type] = self._client.get_or_create_collection(name=name)
            return
        self._fallback_rows[collection_type] = []

    def _parse_query_payload(self, payload: dict[str, Any]) -> list[VectorRecord]:
        ids = payload.get("ids", [[]])
        docs = payload.get("documents", [[]])
        metas = payload.get("metadatas", [[]])
        distances = payload.get("distances", [[]])

        row_ids = ids[0] if ids else []
        row_docs = docs[0] if docs else []
        row_metas = metas[0] if metas else []
        row_distances = distances[0] if distances else []

        records: list[VectorRecord] = []
        for idx, row_id in enumerate(row_ids):
            text = row_docs[idx] if idx < len(row_docs) else ""
            metadata = row_metas[idx] if idx < len(row_metas) and isinstance(row_metas[idx], dict) else {}
            distance: float | None = None
            if idx < len(row_distances):
                try:
                    distance = float(row_distances[idx])
                except (TypeError, ValueError):
                    distance = None
            records.append(
                VectorRecord(
                    id=str(row_id),
                    text=str(text),
                    metadata=dict(metadata),
                    distance=distance,
                ),
            )
        return records

    def _fallback_query(
        self,
        collection_type: CollectionType,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorRecord]:
        rows = self._fallback_rows.get(collection_type, [])
        scored: list[tuple[float, str, str, dict[str, Any]]] = []
        for row_id, text, metadata, embedding in rows:
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append((1.0 - score, row_id, text, metadata))
        scored.sort(key=lambda item: item[0])
        out: list[VectorRecord] = []
        for distance, row_id, text, metadata in scored[:top_k]:
            out.append(VectorRecord(id=row_id, text=text, metadata=dict(metadata), distance=distance))
        return out

    @staticmethod
    def _sanitize_metadata(raw: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[str(key)] = value
                continue
            try:
                safe[str(key)] = json.dumps(value, ensure_ascii=False)
            except TypeError:
                safe[str(key)] = str(value)
        return safe

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
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
        return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
