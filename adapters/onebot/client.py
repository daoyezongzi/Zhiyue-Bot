from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.exceptions import ConnectionClosed

from internal.logger import get_logger


class OneBotClient:
    def __init__(
        self,
        ws_url: str = "ws://127.0.0.1:6199",
        ws_mode: str = "reverse",
        access_token: str = "",
        reconnect_initial: float = 1.0,
        reconnect_max: float = 30.0,
    ) -> None:
        self.endpoint: str = ws_url or "ws://127.0.0.1:6199"
        self.ws_mode: str = self._normalize_mode(ws_mode)
        # Backward-compatible alias used by existing startup logs.
        self.ws_url: str = self.endpoint
        self.access_token: str = (access_token or "").strip()
        self.message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        self._logger = get_logger("OneBotClient")
        self._ws: Any | None = None
        self._server: Any | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._connected_event: asyncio.Event = asyncio.Event()
        self._send_lock: asyncio.Lock = asyncio.Lock()
        self._connection_lock: asyncio.Lock = asyncio.Lock()

        self._echo_seed: int = int(time.time() * 1000)
        self._reconnect_initial: float = reconnect_initial
        self._reconnect_max: float = reconnect_max

        _, _, expected_path = self._parse_listen_endpoint(self.endpoint)
        self._expected_path: str = expected_path
        self._connect_header_kw: str = self._resolve_connect_header_kw()

    @property
    def connected(self) -> bool:
        return self._connected_event.is_set()

    async def start(self) -> None:
        if self._runner_task and not self._runner_task.done():
            return
        self._stop_event.clear()
        if self.ws_mode == "reverse":
            self._runner_task = asyncio.create_task(self._run_reverse_server(), name="onebot-reverse-server")
        else:
            self._runner_task = asyncio.create_task(self._run_forward_forever(), name="onebot-forward-loop")

    async def stop(self) -> None:
        self._stop_event.set()

        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None

        async with self._connection_lock:
            ws = self._ws
            self._ws = None
            self._connected_event.clear()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None

        self._logger.info("OneBot client stopped")

    async def _run_forward_forever(self) -> None:
        backoff: float = max(0.1, self._reconnect_initial)
        while not self._stop_event.is_set():
            try:
                connect_kwargs: dict[str, Any] = {}
                if self.access_token:
                    connect_kwargs[self._connect_header_kw] = {"Authorization": f"Bearer {self.access_token}"}

                self._logger.info("Connecting to OneBot (forward): %s", self.endpoint)
                async with websockets.connect(self.endpoint, **connect_kwargs) as websocket:
                    await self._attach_connection(websocket)
                    backoff = max(0.1, self._reconnect_initial)
                    self._logger.info("OneBot connected (forward)")
                    await self._receive_loop(websocket)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._detach_connection(None)
                self._logger.warning(
                    "OneBot forward connection error: %s; reconnect in %.1fs",
                    exc,
                    backoff,
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, self._reconnect_max)
            finally:
                await self._detach_connection(None)

    async def _run_reverse_server(self) -> None:
        host, port, path = self._parse_listen_endpoint(self.endpoint)
        self._expected_path = path
        self._logger.info("Starting OneBot reverse server: ws://%s:%s%s", host, port, path)

        async with websockets.serve(self._accept_reverse_connection, host=host, port=port) as server:
            self._server = server
            try:
                await self._stop_event.wait()
            finally:
                self._server = None

    async def _accept_reverse_connection(self, websocket: Any, path: str | None = None) -> None:
        raw_path = self._resolve_connection_path(websocket, path)
        path_only = self._path_only(raw_path)
        if self._expected_path != "/" and path_only != self._expected_path:
            self._logger.warning(
                "Rejected reverse connection due to path mismatch: got=%s expect=%s",
                path_only,
                self._expected_path,
            )
            await websocket.close(code=1008, reason="invalid path")
            return

        if not self._is_authorized_reverse_connection(websocket, raw_path):
            self._logger.warning("Rejected reverse connection: invalid access token")
            await websocket.close(code=1008, reason="unauthorized")
            return

        await self._attach_connection(websocket)
        self._logger.info("OneBot connected (reverse): path=%s", path_only)
        try:
            await self._receive_loop(websocket)
        except ConnectionClosed:
            pass
        finally:
            await self._detach_connection(websocket)
            self._logger.info("OneBot reverse connection closed")

    async def _attach_connection(self, websocket: Any) -> None:
        async with self._connection_lock:
            if self._ws is not None and self._ws is not websocket:
                try:
                    await self._ws.close(code=1012, reason="replaced by new connection")
                except Exception:
                    pass
            self._ws = websocket
            self._connected_event.set()

    async def _detach_connection(self, websocket: Any | None) -> None:
        async with self._connection_lock:
            if websocket is None or self._ws is websocket:
                self._ws = None
                self._connected_event.clear()

    async def _receive_loop(self, websocket: Any) -> None:
        try:
            async for raw_message in websocket:
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
        if not self.connected:
            raise RuntimeError("OneBot is not connected")

        echo = self._next_echo()
        payload: dict[str, Any] = {
            "action": action,
            "params": params,
            "echo": echo,
        }

        async with self._send_lock:
            websocket = self._ws
            if websocket is None:
                raise RuntimeError("OneBot is not connected")
            await websocket.send(json.dumps(payload, ensure_ascii=False))

        self._logger.info("TX action=%s echo=%s params=%s", action, echo, params)
        return echo

    def _is_authorized_reverse_connection(self, websocket: Any, raw_path: str) -> bool:
        expected = self.access_token.strip()
        if not expected:
            return True

        headers = self._extract_request_headers(websocket)
        token = self._extract_token_from_headers(headers)
        if token == expected:
            return True

        query = parse_qs(urlparse(raw_path).query)
        query_token = query.get("access_token", [""])[0].strip()
        return query_token == expected

    @staticmethod
    def _extract_request_headers(websocket: Any) -> Mapping[str, Any]:
        headers = getattr(websocket, "request_headers", None)
        if headers is not None:
            return headers
        request = getattr(websocket, "request", None)
        if request is not None:
            request_headers = getattr(request, "headers", None)
            if request_headers is not None:
                return request_headers
        return {}

    @staticmethod
    def _extract_token_from_headers(headers: Mapping[str, Any]) -> str:
        access_token = OneBotClient._get_header(headers, "x-access-token").strip()
        if access_token:
            return access_token

        auth = OneBotClient._get_header(headers, "authorization").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    @staticmethod
    def _get_header(headers: Mapping[str, Any], key: str) -> str:
        if hasattr(headers, "get"):
            value = headers.get(key)
            if value is None:
                value = headers.get(key.lower())  # type: ignore[arg-type]
            if value is not None:
                return str(value)

        for item_key, value in headers.items():
            if str(item_key).lower() == key.lower():
                return str(value)
        return ""

    @staticmethod
    def _resolve_connection_path(websocket: Any, path: str | None) -> str:
        if path is not None:
            return str(path)
        ws_path = getattr(websocket, "path", None)
        if ws_path is not None:
            return str(ws_path)
        request = getattr(websocket, "request", None)
        if request is not None:
            req_path = getattr(request, "path", None)
            if req_path is not None:
                return str(req_path)
        return "/"

    @staticmethod
    def _path_only(raw_path: str) -> str:
        parsed = urlparse(raw_path)
        return parsed.path or "/"

    @staticmethod
    def _parse_listen_endpoint(endpoint: str) -> tuple[str, int, str]:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "wss":
            port = 443
        else:
            port = 80
        path = parsed.path or "/"
        return host, port, path

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in {"forward", "reverse"}:
            return normalized
        return "reverse"

    @staticmethod
    def _resolve_connect_header_kw() -> str:
        try:
            params = inspect.signature(websockets.connect).parameters
        except (TypeError, ValueError):
            return "additional_headers"
        if "additional_headers" in params:
            return "additional_headers"
        if "extra_headers" in params:
            return "extra_headers"
        return "additional_headers"

    def _next_echo(self) -> str:
        self._echo_seed += 1
        return str(self._echo_seed)


OneBotAdapter = OneBotClient
