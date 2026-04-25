from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def save_style_card(ctx: ToolContext, intent: str, tone: str, example: str) -> dict:
    return {"ok": True, "intent": intent, "tone": tone, "example": example}


async def search_style_cards(ctx: ToolContext, keyword: str, limit: int = 3) -> list[dict]:
    return []


async def get_unchecked_style_cards(ctx: ToolContext, limit: int = 20) -> list[dict]:
    return []


async def review_style_card(ctx: ToolContext, ids: list[int], approve: bool) -> dict:
    return {"ok": True, "count": len(ids), "approve": approve}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("saveStyleCard", "保存风格卡片", save_style_card),
        SimplePlugin("searchStyleCards", "检索风格卡片", search_style_cards),
        SimplePlugin("getUncheckedStyleCards", "查看待审风格卡片", get_unchecked_style_cards),
        SimplePlugin("reviewStyleCard", "审核风格卡片", review_style_card),
    ]
