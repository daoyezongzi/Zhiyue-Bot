from __future__ import annotations

import asyncio
import math
from typing import Any

import aiohttp

from internal.config.schema import EmbeddingConfig
from internal.logger import get_logger


class EmbeddingAdapter:
    def __init__(self, cfg: EmbeddingConfig, target_dim: int | None = None) -> None:
        self.cfg = cfg
        self.target_dim = target_dim
        self._logger = get_logger("EmbeddingAdapter")
        self._retry_count = 2
        self._request_timeout_sec = 45
        self._retry_backoff_sec = 0.6

    async def embed(self, text: str) -> list[float]:
        clean_text = text.strip()
        if not clean_text:
            return []

        if self.cfg.enabled and self.cfg.base_url.strip() and self.cfg.model.strip():
            vector = await self._request_embedding(clean_text)
            if vector:
                return self._fit_dim(vector)

        # fallback keeps the memory system available when embedding API is down or disabled
        return self._hash_embedding(clean_text, self.target_dim or 256)

    async def _request_embedding(self, text: str) -> list[float]:
        endpoint = self._build_endpoint(self.cfg.base_url)
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "input": text,
        }
        headers = {"Content-Type": "application/json"}
        api_key = (self.cfg.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = aiohttp.ClientTimeout(total=self._request_timeout_sec)
        last_error: str = ""
        for attempt in range(1, self._retry_count + 2):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(endpoint, json=payload, headers=headers) as resp:
                        raw = await resp.text()
                        if resp.status == 429:
                            retry_after = self._read_retry_after(resp.headers.get("Retry-After"))
                            wait_for = retry_after if retry_after is not None else self._retry_backoff_sec * attempt
                            self._logger.warning(
                                "Embedding request rate limited: status=429 attempt=%s wait=%.2fs",
                                attempt,
                                wait_for,
                            )
                            if attempt < self._retry_count + 1:
                                await asyncio.sleep(wait_for)
                                continue
                            last_error = "rate_limited"
                            break

                        if resp.status >= 500:
                            last_error = f"server_error_{resp.status}"
                            self._logger.warning(
                                "Embedding request server error: status=%s attempt=%s",
                                resp.status,
                                attempt,
                            )
                            if attempt < self._retry_count + 1:
                                await asyncio.sleep(self._retry_backoff_sec * attempt)
                                continue
                            break

                        if resp.status >= 400:
                            self._logger.warning(
                                "Embedding request failed: status=%s body=%s",
                                resp.status,
                                raw[:240],
                            )
                            return []

                        data = await resp.json(content_type=None)
                        vector = self._extract_vector(data)
                        if vector:
                            return vector
                        self._logger.warning("Embedding response has no vector: %s", data)
                        return []
            except asyncio.TimeoutError:
                last_error = "timeout"
                self._logger.warning("Embedding request timeout: attempt=%s", attempt)
                if attempt < self._retry_count + 1:
                    await asyncio.sleep(self._retry_backoff_sec * attempt)
                    continue
                break
            except aiohttp.ClientConnectionError as exc:
                last_error = f"connection_error={exc}"
                self._logger.warning("Embedding connection error: attempt=%s error=%s", attempt, exc)
                if attempt < self._retry_count + 1:
                    await asyncio.sleep(self._retry_backoff_sec * attempt)
                    continue
                break
            except aiohttp.ClientError as exc:
                last_error = f"client_error={exc}"
                self._logger.warning("Embedding client error: attempt=%s error=%s", attempt, exc)
                if attempt < self._retry_count + 1:
                    await asyncio.sleep(self._retry_backoff_sec * attempt)
                    continue
                break
            except Exception as exc:
                last_error = f"unexpected_error={exc}"
                self._logger.warning("Embedding unexpected error: attempt=%s error=%s", attempt, exc)
                break

        if last_error:
            self._logger.warning("Embedding fallback activated: reason=%s", last_error)
        return []

    @staticmethod
    def _build_endpoint(base_url: str) -> str:
        base = base_url.strip()
        if base.endswith("/embeddings"):
            return base
        return base.rstrip("/") + "/embeddings"

    @staticmethod
    def _read_retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    @staticmethod
    def _extract_vector(data: dict[str, Any]) -> list[float]:
        rows = data.get("data")
        if not isinstance(rows, list) or not rows:
            return []
        first = rows[0]
        if not isinstance(first, dict):
            return []
        emb = first.get("embedding")
        if not isinstance(emb, list):
            return []
        out: list[float] = []
        for item in emb:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                return []
        return out

    def _fit_dim(self, embedding: list[float]) -> list[float]:
        if self.target_dim is None or self.target_dim <= 0:
            return embedding
        dim = int(self.target_dim)
        if len(embedding) == dim:
            return embedding
        if len(embedding) > dim:
            return embedding[:dim]
        return embedding + [0.0] * (dim - len(embedding))

    @staticmethod
    def _hash_embedding(text: str, dim: int) -> list[float]:
        vec = [0.0 for _ in range(max(1, dim))]
        for idx, ch in enumerate(text):
            slot = (ord(ch) + idx * 131) % len(vec)
            vec[slot] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0:
            return vec
        return [v / norm for v in vec]
