from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def get_group_member_detail(ctx: ToolContext, user_id: int) -> dict:
    return {"user_id": user_id, "role": "member"}


async def get_recent_messages(ctx: ToolContext, limit: int = 20) -> list[dict]:
    rows = await ctx.memory_mgr.get_recent_messages(ctx.group_id, limit)
    return [{"user_id": r.user_id, "content": r.content} for r in rows]


async def get_group_notices(ctx: ToolContext) -> list[dict]:
    return []


async def get_essence_messages(ctx: ToolContext) -> list[dict]:
    return []


async def get_message_reactions(ctx: ToolContext, message_id: int) -> list[dict]:
    return []


async def get_forward_message_detail(ctx: ToolContext, forward_id: str) -> dict:
    return {"forward_id": forward_id, "messages": []}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("getGroupMemberDetail", "获取群成员详情", get_group_member_detail),
        SimplePlugin("getRecentMessages", "读取最近群消息", get_recent_messages),
        SimplePlugin("getGroupNotices", "读取群公告", get_group_notices),
        SimplePlugin("getEssenceMessages", "读取精华消息", get_essence_messages),
        SimplePlugin("getMessageReactions", "读取消息回应", get_message_reactions),
        SimplePlugin("getForwardMessageDetail", "读取合并转发详情", get_forward_message_detail),
    ]
