from __future__ import annotations

import random
from typing import Any

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def search_stickers(ctx: ToolContext, keyword: str, limit: int = 8) -> list[dict]:
    collector = getattr(ctx.agent, "sticker_collector", None)
    if collector is None:
        return []
    # Prefer local stickers so search results can be sent directly by sendSticker.
    rows = await collector.search(keyword=keyword, limit=limit, storage_mode="local")
    return [row for row in rows if collector.is_sticker_item(row)]


def _is_random_request(text: str) -> bool:
    clean = str(text or "").strip().lower()
    if not clean:
        return False
    return any(token in clean for token in ("随机", "随便", "任意", "都行", "random", "any"))


def _select_local_sendable_item(collector: Any, rows: list[dict[str, Any]], *, require_sticker: bool) -> dict[str, Any] | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("storage_mode", "local")).strip().lower() != "local":
            continue
        if require_sticker and not collector.is_sticker_item(row):
            continue
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            continue
        try:
            if not collector.resolve_local_file_path(file_name).is_file():
                continue
        except ValueError:
            continue
        return dict(row)
    return None


async def _pick_random_local_item(collector: Any, *, require_sticker: bool) -> dict[str, Any] | None:
    rows = await collector.list_local_files()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if require_sticker and not collector.is_sticker_item(row):
            continue
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            continue
        try:
            if not collector.resolve_local_file_path(file_name).is_file():
                continue
        except ValueError:
            continue
        candidates.append(dict(row))
    if not candidates:
        return None
    return random.choice(candidates)


async def send_sticker(ctx: ToolContext, sticker_id: int | str) -> dict:
    collector = getattr(ctx.agent, "sticker_collector", None)
    if collector is None:
        return {"ok": False, "error": "sticker collector is unavailable"}

    sticker_ref = str(sticker_id).strip()
    item = None
    matched_by = "id"
    if sticker_ref:
        item = await collector.get_sticker(sticker_ref)

    if item is None and sticker_ref:
        rows = await collector.search(keyword=sticker_ref, limit=12, storage_mode="local")
        item = _select_local_sendable_item(collector, rows, require_sticker=True)
        if item is None:
            item = _select_local_sendable_item(collector, rows, require_sticker=False)
        if item is not None:
            matched_by = "keyword"

    random_mode = _is_random_request(sticker_ref)
    if item is None:
        item = await _pick_random_local_item(collector, require_sticker=True)
        if item is None:
            item = await _pick_random_local_item(collector, require_sticker=False)
        if item is not None:
            matched_by = "random" if random_mode or not sticker_ref else "fallback"

    if item is None:
        return {"ok": False, "error": "sticker not found"}

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

    file_name = str(item.get("file_name", "")).strip()
    if not file_name:
        return {"ok": False, "error": "sticker file is missing"}
    try:
        file_path = collector.resolve_local_file_path(file_name)
        if not file_path.is_file():
            return {"ok": False, "error": "sticker file is missing"}
    except ValueError:
        return {"ok": False, "error": "invalid sticker file path"}

    is_sticker = collector.is_sticker_item(item)
    if ctx.group_id > 0 and hasattr(ctx.bot, "send_group_image"):
        message_id = await ctx.bot.send_group_image(
            group_id=ctx.group_id,
            file_path=str(file_path),
            as_sticker=is_sticker,
        )
    else:
        content = collector.build_local_sticker_cq(file_name)
        if ctx.speak_callback is not None:
            message_id = await ctx.speak_callback(ctx.group_id, content, None, None)
        else:
            message_id = await ctx.bot.send_group_msg(ctx.group_id, content)

    resolved_id = str(item.get("id") or item.get("md5") or sticker_ref).strip()
    return {
        "ok": True,
        "sticker_id": resolved_id,
        "message_id": int(message_id),
        "matched_by": matched_by,
    }


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("searchStickers", "Search stickers", search_stickers),
        SimplePlugin("sendSticker", "Send a local sticker by id or keyword", send_sticker),
    ]
