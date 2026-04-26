from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import wraps
from typing import Any

import aiohttp

from internal.config.schema import LLMConfig
from internal.logger import get_logger


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class ModelRoute:
    name: str
    cfg: LLMConfig
    endpoint: str
    model: str


class RetryableLLMError(RuntimeError):
    pass


class FatalLLMError(RuntimeError):
    pass


def with_retry_and_fallback(func):
    @wraps(func)
    async def wrapper(
        self: "ChatLLMAdapter",
        messages: list[dict[str, Any]],
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._model_chain:
            self._logger.warning("LLM.Request: no available model route")
            return {}

        max_attempts = 1 + self._retry_count
        for route_index, route in enumerate(self._model_chain):
            for attempt in range(1, max_attempts + 1):
                self._logger.info(
                    "LLM.Request: stage=attempt route=%s model=%s attempt=%s/%s",
                    route.name,
                    route.model,
                    attempt,
                    max_attempts,
                )
                try:
                    data = await func(self, route, messages, extra_fields)
                    self._logger.info(
                        "LLM.Request: stage=success route=%s model=%s",
                        route.name,
                        route.model,
                    )
                    return data
                except RetryableLLMError as exc:
                    if attempt < max_attempts:
                        backoff = self._retry_backoff_sec * attempt
                        self._logger.warning(
                            "LLM.Request: stage=retry route=%s model=%s reason=%s backoff=%.2fs",
                            route.name,
                            route.model,
                            exc,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue

                    if route_index + 1 < len(self._model_chain):
                        next_route = self._model_chain[route_index + 1]
                        self._logger.warning(
                            "LLM.Request: stage=fallback from=%s/%s to=%s/%s reason=%s",
                            route.name,
                            route.model,
                            next_route.name,
                            next_route.model,
                            exc,
                        )
                        break

                    self._logger.warning(
                        "LLM.Request: stage=exhausted route=%s model=%s reason=%s",
                        route.name,
                        route.model,
                        exc,
                    )
                    return {}
                except FatalLLMError as exc:
                    self._logger.warning(
                        "LLM.Request: stage=fatal route=%s model=%s reason=%s",
                        route.name,
                        route.model,
                        exc,
                    )
                    return {}
                except Exception:
                    self._logger.exception(
                        "LLM.Request: stage=unexpected route=%s model=%s",
                        route.name,
                        route.model,
                    )
                    if route_index + 1 < len(self._model_chain):
                        next_route = self._model_chain[route_index + 1]
                        self._logger.warning(
                            "LLM.Request: stage=fallback from=%s/%s to=%s/%s reason=unexpected_error",
                            route.name,
                            route.model,
                            next_route.name,
                            next_route.model,
                        )
                        break
                    return {}

        return {}

    return wrapper


class ChatLLMAdapter:
    def __init__(self, cfg: LLMConfig, fallback_cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg
        self.fallback_cfg = fallback_cfg
        self._logger = get_logger("ChatLLMAdapter")
        self._retry_count = 1
        self._retry_backoff_sec = 0.35
        self._request_timeout_sec = 60
        self._model_chain = self._build_model_chain(cfg, fallback_cfg)
        if self._model_chain:
            route_desc = ", ".join(f"{item.name}:{item.model}" for item in self._model_chain)
            self._logger.info("LLM.Request: route chain ready: %s", route_desc)
        else:
            self._logger.warning("LLM.Request: model chain is empty")

    async def request_chat_completion(
        self,
        messages: list[dict[str, Any]],
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request_with_retry_and_fallback(messages=messages, extra_fields=extra_fields)

    @with_retry_and_fallback
    async def _request_with_retry_and_fallback(
        self,
        route: ModelRoute,
        messages: list[dict[str, Any]],
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": route.model,
            "messages": messages,
        }
        if extra_fields:
            payload.update(extra_fields)

        headers = {"Content-Type": "application/json"}
        api_key = (route.cfg.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = aiohttp.ClientTimeout(total=self._request_timeout_sec)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(route.endpoint, json=payload, headers=headers) as resp:
                    raw = await resp.text()
                    if resp.status in RETRYABLE_STATUS_CODES:
                        raise RetryableLLMError(f"status={resp.status} body={raw[:200]}")
                    if resp.status >= 400:
                        raise FatalLLMError(f"status={resp.status} body={raw[:200]}")
                    try:
                        return await resp.json(content_type=None)
                    except Exception as exc:
                        raise FatalLLMError(f"response_non_json={raw[:200]}") from exc
        except asyncio.TimeoutError as exc:
            raise RetryableLLMError("timeout") from exc
        except aiohttp.ClientConnectionError as exc:
            raise RetryableLLMError(f"connection_error={exc}") from exc
        except aiohttp.ClientError as exc:
            raise RetryableLLMError(f"client_error={exc}") from exc

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        return await self.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_fields=extra_fields,
        )

    async def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        data = await self.request_chat_completion(messages=messages, extra_fields=extra_fields)
        if not data:
            return ""

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            self._logger.warning("LLM response has empty choices: %s", data)
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return "".join(text_parts).strip()

        return ""

    def _build_model_chain(self, primary: LLMConfig, fallback: LLMConfig | None) -> list[ModelRoute]:
        chain: list[ModelRoute] = []

        primary_route = self._to_model_route("primary", primary)
        if primary_route is not None:
            chain.append(primary_route)

        fallback_route = self._to_model_route("fallback", fallback) if fallback is not None else None
        if fallback_route is not None and not self._is_same_route(primary_route, fallback_route):
            chain.append(fallback_route)

        return chain

    def _to_model_route(self, name: str, cfg: LLMConfig | None) -> ModelRoute | None:
        if cfg is None:
            return None

        endpoint = self._build_endpoint(cfg)
        model = str(cfg.model or "").strip()
        if not endpoint or not model:
            return None
        return ModelRoute(name=name, cfg=cfg, endpoint=endpoint, model=model)

    @staticmethod
    def _is_same_route(left: ModelRoute | None, right: ModelRoute | None) -> bool:
        if left is None or right is None:
            return False
        return left.endpoint == right.endpoint and left.model == right.model

    def _build_endpoint(self, cfg: LLMConfig) -> str:
        base = str(cfg.base_url or "").strip()
        if not base:
            return ""
        if base.endswith("/chat/completions"):
            return base
        return base.rstrip("/") + "/chat/completions"

    def get_loaded_models(self) -> list[dict[str, str]]:
        return [
            {
                "route": route.name,
                "model": route.model,
                "endpoint": route.endpoint,
            }
            for route in self._model_chain
        ]

    async def generate_with_history(
        self,
        system_prompt: str,
        conversation_lines: list[str],
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        history_text = "\n".join(conversation_lines[-30:])
        user_prompt = (
            "Here is the recent chat history. Reply naturally according to the role settings.\n"
            f"{history_text}\n"
            "Output only the final reply content."
        )
        return await self.generate(system_prompt, user_prompt, extra_fields)
