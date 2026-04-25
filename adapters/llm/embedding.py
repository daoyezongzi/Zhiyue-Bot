from __future__ import annotations

from internal.config.schema import EmbeddingConfig


class EmbeddingAdapter:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg

    async def embed(self, text: str) -> list[float]:
        # placeholder: should call embedding API and return vector
        return [float(len(text) % 10)]
