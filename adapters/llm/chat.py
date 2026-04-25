from __future__ import annotations

from typing import Any

import aiohttp

from internal.config.schema import LLMConfig
from internal.logger import get_logger


class ChatLLMAdapter:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._logger = get_logger("ChatLLMAdapter")

    async def request_chat_completion(
        self,
        messages: list[dict[str, Any]],
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.cfg.api_key:
            self._logger.debug("LLM API key is empty, skip request")
            return {}

        endpoint = self._build_endpoint()
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
        }
        if extra_fields:
            payload.update(extra_fields)

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=60)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload, headers=headers) as resp:
                    raw = await resp.text()
                    if resp.status >= 400:
                        self._logger.warning(
                            "LLM request failed: status=%s body=%s",
                            resp.status,
                            raw,
                        )
                        return {}

                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        self._logger.warning("LLM response is not JSON: %s", raw)
                        return {}
        except Exception as exc:
            self._logger.warning("LLM request error: %s", exc)
            return {}

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

    def _build_endpoint(self) -> str:
        base = self.cfg.base_url.strip()
        if base.endswith("/chat/completions"):
            return base
        return base.rstrip("/") + "/chat/completions"

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
