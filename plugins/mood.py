from __future__ import annotations

from plugins.base import SimplePlugin
from plugins.context import ToolContext


async def update_mood(ctx: ToolContext, valence: float, energy: float, sociability: float) -> dict:
    return {
        "ok": True,
        "valence": valence,
        "energy": energy,
        "sociability": sociability,
    }


def build_plugins() -> list[SimplePlugin]:
    return [
        SimplePlugin("updateMood", "更新情绪状态", update_mood),
    ]
