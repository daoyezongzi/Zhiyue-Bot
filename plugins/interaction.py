from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def speak(ctx: ToolContext, content: str, reply_to: int | None = None, mentions: list[int] | None = None) -> dict:
    if ctx.speak_callback:
        msg_id = await ctx.speak_callback(ctx.group_id, content, reply_to, mentions)
    else:
        msg_id = await ctx.bot.send_group_message(ctx.group_id, content, reply_to, mentions)
    return {"ok": True, "message_id": msg_id}


async def stay_quiet(ctx: ToolContext, reason: str = "") -> dict:
    del ctx
    return {"ok": True, "action": "stayQuiet", "reason": str(reason or "").strip()}


async def poke(ctx: ToolContext, user_id: int) -> dict:
    return {"ok": True, "action": "poke", "user_id": user_id}


async def react_to_message(ctx: ToolContext, message_id: int, emoji_id: int) -> dict:
    return {"ok": True, "action": "reactToMessage", "message_id": message_id, "emoji_id": emoji_id}


async def recall_message(ctx: ToolContext, message_id: int) -> dict:
    return {"ok": True, "action": "recallMessage", "message_id": message_id}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("speak", "发送消息", speak),
        SimplePlugin("stayQuiet", "保持沉默", stay_quiet),
        SimplePlugin("poke", "戳一戳", poke),
        SimplePlugin("reactToMessage", "贴表情回应", react_to_message),
        SimplePlugin("recallMessage", "撤回消息", recall_message),
    ]
