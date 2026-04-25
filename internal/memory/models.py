from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


MemoryType = Literal["group_fact", "self_experience", "conversation"]


@dataclass
class MessageLog:
    message_id: int
    group_id: int
    user_id: int
    nickname: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MemoryItem:
    id: int
    group_id: int
    mem_type: MemoryType
    content: str
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
