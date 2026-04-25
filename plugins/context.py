from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from adapters.onebot.client import OneBotAdapter
    from core.agent import AsyncReactAgent
    from internal.memory.manager import MemoryManager


SpeakCallback = Callable[[int, str, int | None, list[int] | None], Awaitable[int]]


@dataclass
class ToolContext:
    group_id: int
    memory_mgr: "MemoryManager"
    bot: "OneBotAdapter"
    agent: "AsyncReactAgent"
    speak_callback: Optional[SpeakCallback] = None
