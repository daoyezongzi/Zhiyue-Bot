from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def search_jargon(ctx: ToolContext, keyword: str, limit: int = 10) -> list[dict]:
    manager = ctx.agent.jargon_mgr
    pairs = await manager.search(keyword, limit)
    return [{"term": k, "meaning": v} for k, v in pairs]


async def save_jargon(ctx: ToolContext, term: str, meaning: str) -> dict:
    await ctx.agent.jargon_mgr.add(term, meaning)
    return {"ok": True, "term": term}


async def get_unchecked_jargons(ctx: ToolContext, limit: int = 20) -> list[dict]:
    # skeleton: review queue is not implemented yet
    return []


async def review_jargon(ctx: ToolContext, ids: list[int], approve: bool) -> dict:
    return {"ok": True, "count": len(ids), "approve": approve}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("saveJargon", "保存黑话", save_jargon),
        SimplePlugin("searchJargon", "检索黑话", search_jargon),
        SimplePlugin("getUncheckedJargons", "查看待审黑话", get_unchecked_jargons),
        SimplePlugin("reviewJargon", "审核黑话", review_jargon),
    ]
