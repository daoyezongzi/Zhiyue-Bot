from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from plugins.context import ToolContext


class BasePlugin(ABC):
    name: str
    description: str

    @abstractmethod
    async def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        raise NotImplementedError


@dataclass
class SimplePlugin(BasePlugin):
    name: str
    description: str
    handler: Callable[[ToolContext], Awaitable[Any]]

    async def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        return await self.handler(ctx, **kwargs)
