from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def submit_style_classification(ctx: ToolContext, intent: str, tone: str) -> dict:
    return {"ok": True, "intent": intent, "tone": tone}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("submitStyleClassification", "提交风格分类", submit_style_classification),
    ]
