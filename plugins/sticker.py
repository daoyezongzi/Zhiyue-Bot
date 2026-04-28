from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def search_stickers(ctx: ToolContext, keyword: str, limit: int = 8) -> list[dict]:
    collector = getattr(ctx.agent, "sticker_collector", None)
    if collector is None:
        return []
    # Prefer local stickers so search results can be sent directly by sendSticker.
    rows = await collector.search(keyword=keyword, limit=limit, storage_mode="local")
    return [row for row in rows if collector.is_sticker_item(row)]


async def send_sticker(ctx: ToolContext, sticker_id: int | str) -> dict:
    collector = getattr(ctx.agent, "sticker_collector", None)
    if collector is None:
        return {"ok": False, "error": "sticker collector is unavailable"}

    sticker_ref = str(sticker_id).strip()
    item = await collector.get_sticker(sticker_ref)
    if item is None:
        return {"ok": False, "error": "sticker not found"}
    if not collector.is_sticker_item(item):
        return {"ok": False, "error": "item is not a sticker"}

    if str(item.get("storage_mode", "local")).strip().lower() != "local":
        return {"ok": False, "error": "cloud sticker cannot be sent directly"}

    file_name = str(item.get("file_name", "")).strip()
    if not file_name:
        return {"ok": False, "error": "sticker file is missing"}
    try:
        if not collector.resolve_local_file_path(file_name).is_file():
            return {"ok": False, "error": "sticker file is missing"}
    except ValueError:
        return {"ok": False, "error": "invalid sticker file path"}

    mood = 50.0
    status_engine = getattr(ctx.agent, "status_engine", None)
    if status_engine is not None:
        try:
            status = await status_engine.get_snapshot()
            mood = float(status.energy)
        except Exception:
            pass
    decision = await collector.allow_sticker_for_reply(item=item, query=sticker_ref, mood=mood)
    if not decision.allowed:
        return {
            "ok": False,
            "error": "sticker rejected by persona filter",
            "reason": decision.reason,
            "source": decision.source,
        }

    content = collector.build_local_sticker_cq(file_name)
    if ctx.speak_callback is not None:
        message_id = await ctx.speak_callback(ctx.group_id, content, None, None)
    else:
        message_id = await ctx.bot.send_group_msg(ctx.group_id, content)
    return {"ok": True, "sticker_id": sticker_ref, "message_id": int(message_id)}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("searchStickers", "Search stickers", search_stickers),
        SimplePlugin("sendSticker", "Send sticker", send_sticker),
    ]
