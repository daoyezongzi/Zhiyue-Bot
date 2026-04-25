from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def save_memory(ctx: ToolContext, content: str, mem_type: str = "conversation") -> dict:
    item = await ctx.memory_mgr.save_memory(ctx.group_id, content, mem_type=mem_type)
    return {"id": item.id, "content": item.content}


async def query_memory(ctx: ToolContext, query: str, limit: int = 5) -> list[dict]:
    items = await ctx.memory_mgr.query_memory(ctx.group_id, query, limit)
    return [{"id": item.id, "content": item.content} for item in items]


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("saveMemory", "保存长期记忆", save_memory),
        SimplePlugin("queryMemory", "检索长期记忆", query_memory),
    ]
