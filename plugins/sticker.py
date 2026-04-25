from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def search_stickers(ctx: ToolContext, keyword: str, limit: int = 8) -> list[dict]:
    return []


async def send_sticker(ctx: ToolContext, sticker_id: int) -> dict:
    return {"ok": True, "sticker_id": sticker_id}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("searchStickers", "搜索表情包", search_stickers),
        SimplePlugin("sendSticker", "发送表情包", send_sticker),
    ]
