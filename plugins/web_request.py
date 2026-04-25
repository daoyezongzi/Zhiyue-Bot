from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def request_get(ctx: ToolContext, url: str) -> dict:
    return {"url": url, "content": ""}


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("request_get", "获取网页文本", request_get),
    ]
