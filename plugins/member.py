from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def update_member_profile(ctx: ToolContext, user_id: int, summary: str) -> dict:
    return {"ok": True, "user_id": user_id, "summary": summary}


async def get_member_info(ctx: ToolContext, user_id: int) -> dict:
    return {"user_id": user_id, "nickname": "", "intimacy": 0.0, "activity": 0.0}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("updateMemberProfile", "更新群友画像", update_member_profile),
        SimplePlugin("getMemberInfo", "获取群友画像", get_member_info),
    ]
