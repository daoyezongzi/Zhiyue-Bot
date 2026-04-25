from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Mapping

import websockets
from websockets.exceptions import ConnectionClosed

from internal.logger import get_logger


class OneBotClient:
    def __init__(
        self,
        ws_url: str = "ws://127.0.0.1:3001",
        access_token: str = "",
        reconnect_initial: float = 1.0,
        reconnect_max: float = 30.0,
    ) -> None:
        self.endpoint: str = ws_url or "ws://127.0.0.1:3001"
        # Backward-compatible alias used by existing startup logs.
        self.ws_url: str = self.endpoint
        self.access_token: str = access_token
        self.message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        self._logger = get_logger("OneBotClient")
        self._ws: Any | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._connected_event: asyncio.Event = asyncio.Event()
        self._send_lock: asyncio.Lock = asyncio.Lock()

        self._echo_seed: int = int(time.time() * 1000)
        self._reconnect_initial: float = reconnect_initial
        self._reconnect_max: float = reconnect_max

    @property
    def connected(self) -> bool:
        return self._connected_event.is_set()

    async def start(self) -> None:
        if self._runner_task and not self._runner_task.done():
            return
        self._stop_event.clear()
        self._runner_task = asyncio.create_task(self._run_forever(), name="onebot-connection-loop")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            await self._ws.close()

        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None

        self._connected_event.clear()
        self._logger.info("OneBot client stopped")

    async def _run_forever(self) -> None:
        backoff: float = max(0.1, self._reconnect_initial)
        while not self._stop_event.is_set():
            try:
                if self.access_token:
                    self._logger.warning(
                        "access_token is configured but ignored in websocket connect for compatibility",
                    )

                self._logger.info("Connecting to OneBot: %s", self.endpoint)
                async with websockets.connect(self.endpoint) as websocket:
                    self._ws = websocket
                    self._connected_event.set()
                    backoff = max(0.1, self._reconnect_initial)
                    self._logger.info("OneBot connected")
                    await self._receive_loop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected_event.clear()
                self._ws = None
                self._logger.warning(
                    "OneBot connection error: %s; reconnect in %.1fs",
                    exc,
                    backoff,
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, self._reconnect_max)
            finally:
                self._connected_event.clear()
                self._ws = None

    async def _receive_loop(self) -> None:
        if self._ws is None:
            return

        try:
            async for raw_message in self._ws:
                packet = self._parse_message(raw_message)
                if packet is None:
                    continue
                await self.message_queue.put(packet)
                self._logger.info(
                    "RX post_type=%s message_type=%s user_id=%s group_id=%s text=%s",
                    packet.get("post_type"),
                    packet.get("message_type"),
                    packet.get("user_id"),
                    packet.get("group_id"),
                    packet.get("text"),
                )
        except ConnectionClosed as exc:
            self._logger.warning("OneBot receive loop closed: %s", exc)
            raise

    def _parse_message(self, raw_message: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to parse OneBot JSON: %s", raw_message)
            return None

        if not isinstance(payload, Mapping):
            self._logger.warning("Received non-dict OneBot payload: %s", payload)
            return None

        data: dict[str, Any] = dict(payload)
        text = self._extract_text(data)

        structured: dict[str, Any] = {
            "post_type": str(data.get("post_type", "")),
            "message_type": str(data.get("message_type", "")),
            "sub_type": str(data.get("sub_type", "")),
            "time": data.get("time"),
            "self_id": data.get("self_id"),
            "message_id": data.get("message_id"),
            "user_id": data.get("user_id"),
            "group_id": data.get("group_id"),
            "raw_message": data.get("raw_message", ""),
            "message": data.get("message", ""),
            "text": text,
            "echo": data.get("echo"),
            "status": data.get("status"),
            "retcode": data.get("retcode"),
            "raw": data,
        }
        return structured

    @staticmethod
    def _extract_text(data: Mapping[str, Any]) -> str:
        message = data.get("message")
        if isinstance(message, str):
            return message.strip()

        if isinstance(message, list):
            text_parts: list[str] = []
            for segment in message:
                if not isinstance(segment, Mapping):
                    continue
                if segment.get("type") != "text":
                    continue
                segment_data = segment.get("data")
                if not isinstance(segment_data, Mapping):
                    continue
                text_value = segment_data.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
            return "".join(text_parts).strip()

        raw_message = data.get("raw_message")
        if isinstance(raw_message, str):
            return raw_message.strip()
        return ""

    async def send_private_msg(self, user_id: int, message: str) -> str:
        return await self._send_action(
            "send_private_msg",
            {
                "user_id": user_id,
                "message": message,
            },
        )

    async def send_group_msg(self, group_id: int, message: str) -> str:
        return await self._send_action(
            "send_group_msg",
            {
                "group_id": group_id,
                "message": message,
            },
        )

    async def send_group_message(
        self,
        group_id: int,
        content: str,
        reply_to: int | None = None,
        mentions: list[int] | None = None,
    ) -> int:
        del reply_to
        del mentions
        echo = await self.send_group_msg(group_id=group_id, message=content)
        return int(echo)

    async def send_private_message(self, user_id: int, content: str) -> int:
        echo = await self.send_private_msg(user_id=user_id, message=content)
        return int(echo)

    async def _send_action(self, action: str, params: dict[str, Any]) -> str:
        if not self.connected or self._ws is None:
            raise RuntimeError("OneBot is not connected")

        echo = self._next_echo()
        payload: dict[str, Any] = {
            "action": action,
            "params": params,
            "echo": echo,
        }

        async with self._send_lock:
            assert self._ws is not None
            await self._ws.send(json.dumps(payload, ensure_ascii=False))

        self._logger.info("TX action=%s echo=%s params=%s", action, echo, params)
        return echo

    def _next_echo(self) -> str:
        self._echo_seed += 1
        return str(self._echo_seed)


OneBotAdapter = OneBotClient

