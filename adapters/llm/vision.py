from __future__ import annotations

from internal.config.schema import VisionConfig


class VisionAdapter:
    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg

    async def describe_image(self, image_url: str) -> str:
        return f"[图片:{image_url}]"

    async def describe_video(self, video_url: str) -> str:
        return f"[视频:{video_url}]"
