from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MilvusAdapter:
    address: str
    vector_dim: int

    async def insert(self, memory_id: int, group_id: int, mem_type: str, embedding: list[float]) -> None:
        return None

    async def search(self, group_id: int, mem_type: str, embedding: list[float], top_k: int = 5) -> list[dict]:
        return []

    async def close(self) -> None:
        return None
