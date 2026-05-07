from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.exceptions import ConnectionClosed

from internal.logger import get_logger


class OneBotClient:
    def __init__(
        self,
        ws_url: str = "ws://127.0.0.1:18001/ws",
        ws_mode: str = "reverse",
        access_token: str = "",
        reconnect_initial: float = 1.0,
        reconnect_max: float = 30.0,
    ) -> None:
        self.endpoint: str = ws_url or "ws://127.0.0.1:18001/ws"
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
        self._pending_lock: asyncio.Lock = asyncio.Lock()
        self._pending_action_responses: dict[str, asyncio.Future[dict[str, Any]]] = {}

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

        await self._cancel_pending_action_responses("OneBot is stopped")

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
        should_cancel = False
        async with self._connection_lock:
            if websocket is None or self._ws is websocket:
                self._ws = None
                self._connected_event.clear()
                should_cancel = True
        if should_cancel:
            await self._cancel_pending_action_responses("OneBot connection closed")

    async def _receive_loop(self, websocket: Any) -> None:
        try:
            async for raw_message in websocket:
                payload = self._decode_payload(raw_message)
                if payload is None:
                    continue
                if await self._resolve_action_response(payload):
                    continue

                packet = self._build_event_packet(payload)
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

    def _decode_payload(self, raw_message: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self._logger.warning("Failed to parse OneBot JSON: %s", raw_message)
            return None

        if not isinstance(payload, Mapping):
            self._logger.warning("Received non-dict OneBot payload: %s", payload)
            return None
        return dict(payload)

    def _build_event_packet(self, data: Mapping[str, Any]) -> dict[str, Any] | None:
        post_type = str(data.get("post_type", "")).strip()
        if not post_type:
            return None

        text = self._extract_text(data)

        structured: dict[str, Any] = {
            "post_type": post_type,
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
            "raw": dict(data),
        }
        return structured

    def _parse_message(self, raw_message: str) -> dict[str, Any] | None:
        payload = self._decode_payload(raw_message)
        if payload is None:
            return None
        return self._build_event_packet(payload)

    async def _resolve_action_response(self, payload: Mapping[str, Any]) -> bool:
        if "echo" not in payload:
            return False
        echo = str(payload.get("echo", "")).strip()
        if not echo:
            return False

        async with self._pending_lock:
            future = self._pending_action_responses.pop(echo, None)
        if future is None:
            return False
        if not future.done():
            future.set_result(dict(payload))
        return True

    async def _cancel_pending_action_responses(self, reason: str) -> None:
        async with self._pending_lock:
            pending = list(self._pending_action_responses.values())
            self._pending_action_responses.clear()
        for future in pending:
            if future.done():
                continue
            future.set_exception(RuntimeError(reason))

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

    async def send_group_msg(self, group_id: int, message: Any) -> str:
        return await self._send_action(
            "send_group_msg",
            {
                "group_id": group_id,
                "message": message,
            },
        )

    async def send_group_image(self, group_id: int, file_path: str, *, as_sticker: bool = False) -> str:
        message = [
            {
                "type": "image",
                "data": {
                    "file": self._build_file_uri(file_path),
                    "sub_type": 1 if as_sticker else 0,
                },
            },
        ]
        return await self._send_action(
            "send_group_msg",
            {
                "group_id": int(group_id),
                "message": message,
            },
        )

    async def send_private_image(self, user_id: int, file_path: str, *, as_sticker: bool = False) -> str:
        message = [
            {
                "type": "image",
                "data": {
                    "file": self._build_file_uri(file_path),
                    "sub_type": 1 if as_sticker else 0,
                },
            },
        ]
        return await self._send_action(
            "send_private_msg",
            {
                "user_id": int(user_id),
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
        message: list[dict[str, Any]] = []

        try:
            reply_id = int(reply_to or 0)
        except (TypeError, ValueError):
            reply_id = 0
        if reply_id > 0:
            message.append(
                {
                    "type": "reply",
                    "data": {"id": str(reply_id)},
                },
            )

        for user_id in list(mentions or []):
            try:
                mention_id = int(user_id or 0)
            except (TypeError, ValueError):
                continue
            if mention_id <= 0:
                continue
            message.append(
                {
                    "type": "at",
                    "data": {"qq": str(mention_id)},
                },
            )
            message.append(
                {
                    "type": "text",
                    "data": {"text": " "},
                },
            )

        text = str(content or "")
        if text:
            message.append(
                {
                    "type": "text",
                    "data": {"text": text},
                },
            )

        payload: str | list[dict[str, Any]]
        if message:
            payload = message
        else:
            payload = text
        echo = await self.send_group_msg(group_id=group_id, message=payload)
        return int(echo)

    async def send_private_message(self, user_id: int, content: str) -> int:
        echo = await self.send_private_msg(user_id=user_id, message=content)
        return int(echo)

    async def call_action(self, action: str, params: Mapping[str, Any] | None = None) -> str:
        clean_action = str(action or "").strip()
        if not clean_action:
            raise ValueError("action is empty")
        payload = dict(params or {})
        return await self._send_action(clean_action, payload)

    async def call_action_with_response(
        self,
        action: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        clean_action = str(action or "").strip()
        if not clean_action:
            raise ValueError("action is empty")
        payload = dict(params or {})
        if not self.connected:
            raise RuntimeError("OneBot is not connected")

        echo = self._next_echo()
        wire_payload: dict[str, Any] = {
            "action": clean_action,
            "params": payload,
            "echo": echo,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        async with self._pending_lock:
            self._pending_action_responses[echo] = future

        try:
            async with self._send_lock:
                websocket = self._ws
                if websocket is None:
                    raise RuntimeError("OneBot is not connected")
                await websocket.send(json.dumps(wire_payload, ensure_ascii=False))
            self._logger.info("TX action=%s echo=%s params=%s wait_response=true", clean_action, echo, payload)
            return await asyncio.wait_for(future, timeout=max(0.5, float(timeout)))
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"OneBot action timeout: action={clean_action}") from exc
        finally:
            async with self._pending_lock:
                pending_future = self._pending_action_responses.get(echo)
                if pending_future is future:
                    self._pending_action_responses.pop(echo, None)

    async def get_group_member_info(
        self,
        *,
        group_id: int,
        user_id: int,
        no_cache: bool = False,
        timeout: float = 4.0,
    ) -> dict[str, Any]:
        response = await self.call_action_with_response(
            "get_group_member_info",
            {
                "group_id": int(group_id),
                "user_id": int(user_id),
                "no_cache": bool(no_cache),
            },
            timeout=timeout,
        )
        payload = response.get("data")
        if isinstance(payload, dict):
            return dict(payload)
        return {}

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

    @staticmethod
    def _build_file_uri(file_path: str) -> str:
        raw = str(file_path or "").strip()
        if not raw:
            raise ValueError("file path is empty")
        if raw.lower().startswith("file:///"):
            return raw
        path = Path(raw).expanduser()
        try:
            path = path.resolve(strict=False)
        except Exception:
            pass
        return f"file:///{path.as_posix()}"

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
