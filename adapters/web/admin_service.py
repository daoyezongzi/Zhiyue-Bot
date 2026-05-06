from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from urllib.parse import quote

import uvicorn
import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from internal.config.schema import GroupConfig
from internal.learning.user_profiling import (
    MEMBER_NAME_SOURCE_GROUP_CARD,
    latest_member_group_card,
    member_learned_aliases,
    member_names_for_admin,
)
from internal.management.log_stream import LogStreamHub
from plugins import RuntimePluginManager

if TYPE_CHECKING:
    from core.agent import ZhiyueAgent
    from internal.config.schema import Config


TEXT_FILE_EXTENSIONS = {
    "",
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".log",
}

STYLE_CARD_STATUSES = {"candidate", "active", "rejected"}
JARGON_SCOPES = {"user", "group", "public"}
JARGON_STATUSES = {"candidate", "active", "rejected"}


class ResetActionRequest(BaseModel):
    fill_energy: bool = False
    clear_session_id: str = ""


class UpdateConfigRequest(BaseModel):
    system_prompt: str = Field(default="", min_length=1)


class UpdateMasterRequest(BaseModel):
    master_id: int = Field(...)


class GroupSwitchRequest(BaseModel):
    enabled: bool = Field(...)


class GroupUpsertRequest(BaseModel):
    group_id: int = Field(...)
    enabled: bool = Field(default=True)
    group_name: str | None = Field(default=None)
    remark: str | None = Field(default=None)
    extra_prompt: str | None = Field(default=None)


class GroupMetaUpdateRequest(BaseModel):
    group_name: str | None = Field(default=None)
    remark: str | None = Field(default=None)


class MemoryCreateRequest(BaseModel):
    group_id: int = Field(...)
    content: str = Field(..., min_length=1)
    mem_type: str = Field(default="conversation")
    canonical_type: str = Field(default="fact")
    status: str = Field(default="candidate")
    source_kind: str = Field(default="manual")
    source_ref: str = Field(default="")
    user_id: int = Field(default=0)
    importance: float = Field(default=0.0)


class KnowledgeSaveRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str = Field(default="")


class StickerSettingsRequest(BaseModel):
    enabled: bool | None = None
    collection_rate: float | None = None
    storage_mode: str | None = None
    filter_keywords: list[str] | None = None
    user_weights: dict[str, float] | None = None
    allow_other_users_collection: bool | None = None
    enable_persona_filter: bool | None = None
    llm_filter_enabled: bool | None = None
    llm_filter_probability: float | None = None
    llm_filter_mood_threshold: float | None = None


class StickerUserWeightRequest(BaseModel):
    user_id: int = Field(...)
    weight: float = Field(...)


class StickerAuditRequest(BaseModel):
    force_llm: bool = Field(default=False)
    limit: int = Field(default=0, ge=0, le=5000)


class ToolCallClearRequest(BaseModel):
    tool_name: str = Field(default="")
    session_id: str = Field(default="")
    group_id: int = Field(default=0)
    success: bool | None = Field(default=None)


class StyleCardCreateRequest(BaseModel):
    group_id: int = Field(default=0)
    title: str = Field(default="", min_length=1)
    content: str = Field(default="", min_length=1)
    intent: str = Field(default="")
    tone: str = Field(default="")
    tags: list[str] | None = None
    source_kind: str = Field(default="manual")
    source_ref: str = Field(default="")
    status: str = Field(default="candidate")


class StyleCardUpdateRequest(BaseModel):
    group_id: int | None = None
    title: str | None = None
    content: str | None = None
    intent: str | None = None
    tone: str | None = None
    tags: list[str] | None = None
    source_kind: str | None = None
    source_ref: str | None = None


class StyleCardStatusRequest(BaseModel):
    status: str = Field(..., min_length=1)


class JargonCreateRequest(BaseModel):
    jargon: str = Field(..., min_length=1)
    standard: str = Field(..., min_length=1)
    meaning: str = Field(default="")
    confidence: float = Field(default=0.5)
    weight: float = Field(default=1.0)
    scope: str = Field(default="group")
    source_users: list[int] | None = None


class JargonStatusRequest(BaseModel):
    status: str = Field(..., min_length=1)
    scope: str = Field(default="group")


class UTF8JSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")


class AdminService:
    def __init__(
        self,
        *,
        cfg: "Config",
        agent: "ZhiyueAgent",
        config_path: str | Path,
        log_hub: LogStreamHub | None = None,
        plugin_manager: RuntimePluginManager | None = None,
        shutdown_handler: Callable[[], Awaitable[None]] | None = None,
        restart_handler: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._cfg = cfg
        self._agent = agent
        self._config_path = Path(config_path)
        self._persona_prompt_path = self._config_path.parent / "persona.prompt"
        if not self._persona_prompt_path.is_absolute():
            self._persona_prompt_path = (Path(__file__).resolve().parents[2] / self._persona_prompt_path).resolve()

        self._host = str(cfg.web.host or "127.0.0.1").strip() or "127.0.0.1"
        self._port = int(cfg.web.port or 18002)
        self._access_token = cfg.web.resolved_access_token()

        self._project_root = Path(__file__).resolve().parents[2]
        self._asset_root = self._project_root / "data" / "web_ui"
        self._upload_dir = self._asset_root / "uploads"
        self._upload_dir.mkdir(parents=True, exist_ok=True)

        knowledge_path = Path(getattr(cfg.paths, "knowledge_dir", "data/knowledge"))
        if not knowledge_path.is_absolute():
            knowledge_path = self._project_root / knowledge_path
        self._knowledge_dir = knowledge_path.resolve()
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)

        style_card_path = self._project_root / "data" / "style_cards.json"
        self._style_card_store_path = style_card_path.resolve()
        self._style_card_store_path.parent.mkdir(parents=True, exist_ok=True)

        jargon_store_path = Path(str(getattr(cfg.jargon, "lexicon_store_path", "data/jargon_lexicon.json") or ""))
        if not jargon_store_path.is_absolute():
            jargon_store_path = self._project_root / jargon_store_path
        self._jargon_store_path = jargon_store_path.resolve()
        self._jargon_store_path.parent.mkdir(parents=True, exist_ok=True)
        self._jargon_rejected_store_path = (
            self._jargon_store_path.parent / f"{self._jargon_store_path.stem}_rejected.json"
        )

        self._log_hub = log_hub or LogStreamHub()
        self._plugin_manager = plugin_manager or RuntimePluginManager(self._project_root / "plugins")
        self._shutdown_handler = shutdown_handler
        self._restart_handler = restart_handler
        self._shutdown_requested = False
        self._shutdown_lock = asyncio.Lock()
        self._restart_requested = False
        self._restart_lock = asyncio.Lock()
        self._config_lock = asyncio.Lock()
        self._style_card_lock = asyncio.Lock()
        self._jargon_store_lock = asyncio.Lock()

        self._app = FastAPI(
            title="Zhiyue Unified Dashboard",
            docs_url="/docs",
            redoc_url="/redoc",
            default_response_class=UTF8JSONResponse,
        )
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=self._build_cors_origins(),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._app.mount("/assets", StaticFiles(directory=str(self._asset_root)), name="assets")
        self._bind_routes()

        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None

    @property
    def app(self) -> FastAPI:
        return self._app

    async def start(self) -> None:
        if self._serve_task is not None:
            return

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            access_log=False,
            loop="asyncio",
            lifespan="on",
        )
        self._server = uvicorn.Server(config=config)
        self._serve_task = asyncio.create_task(self._server.serve(), name="admin-service")
        await self._wait_started()

    async def stop(self) -> None:
        if self._serve_task is None:
            return
        if self._server is not None:
            self._server.should_exit = True
        try:
            await self._serve_task
        except asyncio.CancelledError:
            pass
        finally:
            self._serve_task = None
            self._server = None

    async def _wait_started(self) -> None:
        for _ in range(120):
            if self._server is not None and getattr(self._server, "started", False):
                return
            if self._serve_task is None:
                return
            if self._serve_task.done():
                break
            await asyncio.sleep(0.05)

        if self._serve_task is not None and self._serve_task.done():
            exc = self._serve_task.exception()
            if exc is not None:
                raise exc

    def _bind_routes(self) -> None:
        async def _stream_logs_ws(websocket: WebSocket) -> None:
            if not await self._authorize_websocket(websocket):
                return
            await websocket.accept()
            queue = await self._log_hub.subscribe(with_backlog=True)
            try:
                while True:
                    event = await queue.get()
                    await websocket.send_json(event.as_dict())
                    queue.task_done()
            except WebSocketDisconnect:
                return
            finally:
                await self._log_hub.unsubscribe(queue)

        @self._app.websocket("/ws")
        async def ws_logs_root(websocket: WebSocket) -> None:
            await _stream_logs_ws(websocket)

        @self._app.websocket("/ws/logs")
        async def ws_logs(websocket: WebSocket) -> None:
            await _stream_logs_ws(websocket)

        @self._app.get("/health", include_in_schema=False)
        async def health() -> dict[str, Any]:
            return self._health_payload()

        @self._app.get("/api/health")
        async def api_health() -> dict[str, Any]:
            return self._health_payload()

        @self._app.get("/api/status")
        async def api_status(_: None = Depends(self._require_token),) -> dict[str, Any]:
            status_data = await self._agent.get_admin_status()
            status_data["runtime_config"] = self._runtime_config_payload()
            return status_data

        @self._app.get("/api/config/runtime")
        async def api_runtime_config(_: None = Depends(self._require_token),) -> dict[str, Any]:
            return self._runtime_config_payload()

        @self._app.get("/api/tool-calls")
        async def api_tool_calls(
            keyword: str = Query(default=""),
            tool_name: str = Query(default=""),
            session_id: str = Query(default=""),
            group_id: int = Query(default=0),
            success: bool | None = Query(default=None),
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=50, ge=1, le=200),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            data = await self._agent.memory_mgr.list_tool_calls(
                keyword=str(keyword or ""),
                tool_name=str(tool_name or ""),
                session_id=str(session_id or ""),
                group_id=int(group_id or 0),
                success=success,
                page=int(page),
                page_size=int(page_size),
            )
            data["query"] = {
                "keyword": str(keyword or ""),
                "tool_name": str(tool_name or ""),
                "session_id": str(session_id or ""),
                "group_id": int(group_id or 0),
                "success": success,
            }
            return data

        @self._app.get("/api/tool-calls/{tool_call_id}")
        async def api_tool_call_detail(
            tool_call_id: int,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            detail = await self._agent.memory_mgr.get_tool_call_detail(int(tool_call_id))
            if detail is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="tool call log not found")
            return {"ok": True, "item": detail}

        @self._app.delete("/api/tool-calls/{tool_call_id}")
        async def api_tool_call_delete(
            tool_call_id: int,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            ok = await self._agent.memory_mgr.delete_tool_call(int(tool_call_id))
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="tool call log not found")
            return {"ok": True, "deleted_id": int(tool_call_id)}

        @self._app.post("/api/tool-calls/clear")
        async def api_tool_call_clear(
            payload: ToolCallClearRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            removed = await self._agent.memory_mgr.clear_tool_calls(
                tool_name=str(payload.tool_name or ""),
                session_id=str(payload.session_id or ""),
                group_id=int(payload.group_id or 0),
                success=payload.success,
            )
            return {
                "ok": True,
                "removed": int(removed),
                "filters": {
                    "tool_name": str(payload.tool_name or ""),
                    "session_id": str(payload.session_id or ""),
                    "group_id": int(payload.group_id or 0),
                    "success": payload.success,
                },
            }

        @self._app.get("/api/topics")
        async def api_topics(
            group_id: int = Query(default=0),
            status: str = Query(default=""),
            keyword: str = Query(default=""),
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=20, ge=1, le=200),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            data = await self._agent.topic_mgr.list_topics(
                group_id=int(group_id),
                status=str(status or ""),
                keyword=str(keyword or ""),
                page=int(page),
                page_size=int(page_size),
            )
            data["query"] = {
                "group_id": int(group_id),
                "status": str(status or ""),
                "keyword": str(keyword or ""),
            }
            return data

        @self._app.get("/api/topics/{topic_id}")
        async def api_topic_detail(
            topic_id: int,
            message_limit: int = Query(default=80, ge=1, le=500),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            detail = await self._agent.topic_mgr.get_topic_detail(
                topic_id=int(topic_id),
                message_limit=int(message_limit),
            )
            if detail is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="topic not found")
            return {"ok": True, "item": detail}

        @self._app.post("/api/topics/{topic_id}/archive")
        async def api_topic_archive(topic_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            ok = await self._agent.topic_mgr.set_topic_status(topic_id=int(topic_id), status="archived")
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="topic not found")
            return {"ok": True, "topic_id": int(topic_id), "status": "archived"}

        @self._app.post("/api/topics/{topic_id}/activate")
        async def api_topic_activate(topic_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            ok = await self._agent.topic_mgr.set_topic_status(topic_id=int(topic_id), status="active")
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="topic not found")
            return {"ok": True, "topic_id": int(topic_id), "status": "active"}

        @self._app.post("/api/topics/{topic_id}/summary/refresh")
        async def api_topic_refresh_summary(topic_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            ok = await self._agent.topic_mgr.refresh_topic_summary(topic_id=int(topic_id), reason="manual")
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="topic not found")
            detail = await self._agent.topic_mgr.get_topic_detail(topic_id=int(topic_id), message_limit=40)
            return {"ok": True, "topic_id": int(topic_id), "item": detail}

        @self._app.get("/api/style-cards")
        async def api_style_cards(
            group_id: int = Query(default=0),
            status_filter: str = Query(default="", alias="status"),
            keyword: str = Query(default=""),
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=20, ge=1, le=200),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            return await self._list_style_cards(
                group_id=int(group_id),
                status_filter=str(status_filter or ""),
                keyword=str(keyword or ""),
                page=int(page),
                page_size=int(page_size),
            )

        @self._app.get("/api/style-cards/{style_card_id}")
        async def api_style_card_detail(style_card_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            row = await self._get_style_card(int(style_card_id))
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="style card not found")
            return {"ok": True, "item": row}

        @self._app.post("/api/style-cards")
        async def api_style_card_create(
            payload: StyleCardCreateRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                row = await self._create_style_card(payload)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            return {"ok": True, "item": row}

        @self._app.post("/api/style-cards/{style_card_id}")
        async def api_style_card_update(
            style_card_id: int,
            payload: StyleCardUpdateRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                row = await self._update_style_card(int(style_card_id), payload)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="style card not found")
            return {"ok": True, "item": row}

        @self._app.post("/api/style-cards/{style_card_id}/status")
        async def api_style_card_status(
            style_card_id: int,
            payload: StyleCardStatusRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                row = await self._set_style_card_status(int(style_card_id), payload.status)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="style card not found")
            return {"ok": True, "item": row}

        @self._app.delete("/api/style-cards/{style_card_id}")
        async def api_style_card_delete(style_card_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            deleted = await self._delete_style_card(int(style_card_id))
            if not deleted:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="style card not found")
            return {"ok": True, "style_card_id": int(style_card_id)}

        @self._app.get("/api/jargons")
        async def api_jargons(
            status_filter: str = Query(default="", alias="status"),
            scope: str = Query(default=""),
            keyword: str = Query(default=""),
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=20, ge=1, le=200),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            return await self._list_jargons(
                status_filter=str(status_filter or ""),
                scope=str(scope or ""),
                keyword=str(keyword or ""),
                page=int(page),
                page_size=int(page_size),
            )

        @self._app.get("/api/jargons/{jargon_id}")
        async def api_jargon_detail(jargon_id: str, _: None = Depends(self._require_token),) -> dict[str, Any]:
            row = await self._get_jargon(jargon_id)
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="jargon not found")
            return {"ok": True, "item": row}

        @self._app.post("/api/jargons")
        async def api_jargon_create(
            payload: JargonCreateRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                row = await self._create_jargon(payload)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            return {"ok": True, "item": row}

        @self._app.post("/api/jargons/{jargon_id}/status")
        async def api_jargon_status(
            jargon_id: str,
            payload: JargonStatusRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                row = await self._set_jargon_status(jargon_id, payload.status, payload.scope)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="jargon not found")
            return {"ok": True, "item": row}

        @self._app.delete("/api/jargons/{jargon_id}")
        async def api_jargon_delete(jargon_id: str, _: None = Depends(self._require_token),) -> dict[str, Any]:
            deleted = await self._delete_jargon(jargon_id)
            if not deleted:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="jargon not found")
            return {"ok": True, "jargon_id": jargon_id}

        @self._app.get("/api/memories")
        async def api_memories(
            group_id: int = Query(default=0),
            mem_type: str = Query(default=""),
            status_filter: str = Query(default="", alias="status"),
            canonical_type: str = Query(default=""),
            source_kind: str = Query(default=""),
            keyword: str = Query(default=""),
            sort: str = Query(default="updated"),
            order: str = Query(default="desc"),
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=20, ge=1, le=200),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            data = await self._agent.memory_mgr.list_memories(
                group_id=int(group_id),
                mem_type=str(mem_type or ""),
                status=str(status_filter or ""),
                canonical_type=str(canonical_type or ""),
                source_kind=str(source_kind or ""),
                keyword=str(keyword or ""),
                page=int(page),
                page_size=int(page_size),
                sort=str(sort or "updated"),
                order=str(order or "desc"),
            )
            data["query"] = {
                "group_id": int(group_id),
                "mem_type": str(mem_type or ""),
                "status": str(status_filter or ""),
                "canonical_type": str(canonical_type or ""),
                "source_kind": str(source_kind or ""),
                "keyword": str(keyword or ""),
                "sort": str(sort or "updated"),
                "order": str(order or "desc"),
            }
            return data

        @self._app.post("/api/memories")
        async def api_memory_create(
            payload: MemoryCreateRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                item = await self._agent.memory_mgr.save_governed_memory(
                    group_id=int(payload.group_id),
                    user_id=int(payload.user_id),
                    content=str(payload.content or ""),
                    mem_type=str(payload.mem_type or "conversation"),  # type: ignore[arg-type]
                    canonical_type=str(payload.canonical_type or "fact"),  # type: ignore[arg-type]
                    status=str(payload.status or "candidate"),  # type: ignore[arg-type]
                    source_kind=str(payload.source_kind or "manual"),  # type: ignore[arg-type]
                    source_ref=str(payload.source_ref or ""),
                    importance=float(payload.importance or 0.0),
                )
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            detail = await self._agent.memory_mgr.get_memory_detail(int(item.id))
            return {"ok": True, "item": detail}

        @self._app.post("/api/memories/convergence")
        async def api_memory_convergence(_: None = Depends(self._require_token),) -> dict[str, Any]:
            stats = await self._agent.memory_mgr.run_memory_convergence(reason="api")
            snapshot = await self._agent.memory_mgr.get_runtime_snapshot()
            return {"ok": True, "stats": stats, "snapshot": snapshot}

        @self._app.get("/api/memories/{memory_id}")
        async def api_memory_detail(memory_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            detail = await self._agent.memory_mgr.get_memory_detail(int(memory_id))
            if detail is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
            return {"ok": True, "item": detail}

        @self._app.post("/api/memories/{memory_id}/archive")
        async def api_memory_archive(memory_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            ok = await self._agent.memory_mgr.set_memory_status(memory_id=int(memory_id), status="archived")
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
            detail = await self._agent.memory_mgr.get_memory_detail(int(memory_id))
            return {"ok": True, "memory_id": int(memory_id), "item": detail}

        @self._app.post("/api/memories/{memory_id}/activate")
        async def api_memory_activate(memory_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            ok = await self._agent.memory_mgr.set_memory_status(memory_id=int(memory_id), status="active")
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
            detail = await self._agent.memory_mgr.get_memory_detail(int(memory_id))
            return {"ok": True, "memory_id": int(memory_id), "item": detail}

        @self._app.post("/api/memories/{memory_id}/candidate")
        async def api_memory_candidate(memory_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            ok = await self._agent.memory_mgr.set_memory_status(memory_id=int(memory_id), status="candidate")
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
            detail = await self._agent.memory_mgr.get_memory_detail(int(memory_id))
            return {"ok": True, "memory_id": int(memory_id), "item": detail}

        @self._app.delete("/api/memories/{memory_id}")
        async def api_memory_delete(memory_id: int, _: None = Depends(self._require_token),) -> dict[str, Any]:
            ok = await self._agent.memory_mgr.delete_memory(int(memory_id))
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
            return {"ok": True, "memory_id": int(memory_id)}

        @self._app.post("/api/config/master")
        async def api_config_master(
            payload: UpdateMasterRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            master_id = int(payload.master_id)
            if master_id < 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="master_id must be >= 0")
            self._cfg.persona.master_id = master_id
            await self._persist_config(self._set_master_id, master_id)
            return {"ok": True, "master_id": master_id}

        @self._app.get("/api/groups")
        async def api_groups(_: None = Depends(self._require_token),) -> dict[str, Any]:
            groups = await self._hydrate_group_names(self._serialize_groups())
            return {"groups": groups}

        @self._app.get("/api/groups/{group_id}/name")
        async def api_group_name(
            group_id: int,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_group_id = int(group_id)
            if clean_group_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="group_id must be > 0")

            configured = self._cfg.get_group(clean_group_id)
            configured_name = str(configured.group_name if configured is not None else "").strip()
            fetched_name = await self._fetch_group_name_from_onebot(clean_group_id)
            final_name = fetched_name or configured_name
            return {
                "group_id": clean_group_id,
                "group_name": final_name,
                "configured_name": configured_name,
                "fetched": bool(fetched_name),
            }

        @self._app.post("/api/groups")
        async def api_groups_upsert(
            payload: GroupUpsertRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_group_id = int(payload.group_id)
            if clean_group_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="group_id must be > 0")

            enabled = bool(payload.enabled)
            group_name_raw = payload.group_name.strip() if isinstance(payload.group_name, str) else ""
            group_name = group_name_raw if group_name_raw else None
            if group_name is None:
                group_name = await self._fetch_group_name_from_onebot(clean_group_id) or None
            remark = payload.remark.strip() if isinstance(payload.remark, str) else None
            extra_prompt = payload.extra_prompt.strip() if isinstance(payload.extra_prompt, str) else None
            created = self._upsert_runtime_group(
                group_id=clean_group_id,
                enabled=enabled,
                group_name=group_name,
                remark=remark,
                extra_prompt=extra_prompt,
            )
            await self._persist_config(
                self._upsert_group_config,
                {
                    "group_id": clean_group_id,
                    "enabled": enabled,
                    "group_name": group_name,
                    "remark": remark,
                    "extra_prompt": extra_prompt,
                },
            )
            group = self._cfg.get_group(clean_group_id)
            return {
                "ok": True,
                "group_id": clean_group_id,
                "enabled": enabled,
                "group_name": str(group.group_name if group is not None else group_name or ""),
                "remark": str(group.remark if group is not None else remark or ""),
                "created": created,
            }

        @self._app.post("/api/groups/{group_id}")
        async def api_groups_update(
            group_id: int,
            payload: GroupSwitchRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_group_id = int(group_id)
            if clean_group_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="group_id must be > 0")
            enabled = bool(payload.enabled)
            created = self._upsert_runtime_group(
                group_id=clean_group_id,
                enabled=enabled,
                group_name=None,
                remark=None,
                extra_prompt=None,
            )
            await self._persist_config(
                self._upsert_group_config,
                {
                    "group_id": clean_group_id,
                    "enabled": enabled,
                    "group_name": None,
                    "remark": None,
                    "extra_prompt": None,
                },
            )
            return {"ok": True, "group_id": clean_group_id, "enabled": enabled, "created": created}

        @self._app.post("/api/groups/{group_id}/meta")
        async def api_groups_meta_update(
            group_id: int,
            payload: GroupMetaUpdateRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_group_id = int(group_id)
            if clean_group_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="group_id must be > 0")

            group_name = payload.group_name.strip() if isinstance(payload.group_name, str) else None
            remark = payload.remark.strip() if isinstance(payload.remark, str) else None
            existing = self._cfg.get_group(clean_group_id)
            enabled = bool(existing.enabled) if existing is not None else True
            created = self._upsert_runtime_group(
                group_id=clean_group_id,
                enabled=enabled,
                group_name=group_name,
                remark=remark,
                extra_prompt=None,
            )
            await self._persist_config(
                self._upsert_group_config,
                {
                    "group_id": clean_group_id,
                    "enabled": enabled,
                    "group_name": group_name,
                    "remark": remark,
                    "extra_prompt": None,
                },
            )
            group = self._cfg.get_group(clean_group_id)
            return {
                "ok": True,
                "group_id": clean_group_id,
                "enabled": enabled,
                "group_name": str(group.group_name if group is not None else group_name or ""),
                "remark": str(group.remark if group is not None else remark or ""),
                "created": created,
            }

        @self._app.get("/api/members")
        async def api_members(
            keyword: str = Query(default=""),
            limit: int = Query(default=200, ge=1, le=1000),
            group_id: int = Query(default=0),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            profiles = await self._agent.user_profiler.list_user_profiles(
                keyword=keyword,
                limit=limit,
            )
            target_group_id = int(group_id) if int(group_id) > 0 else None
            items = [self._member_profile_payload(item, target_group_id=target_group_id) for item in profiles]
            return {
                "keyword": str(keyword or ""),
                "group_id": target_group_id or 0,
                "count": len(items),
                "items": items,
            }

        @self._app.get("/api/members/{user_id}")
        async def api_member_detail(
            user_id: int,
            group_id: int = Query(default=0),
            refresh_identity: bool = Query(default=True),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_user_id = int(user_id)
            if clean_user_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id must be > 0")
            target_group_id = int(group_id) if int(group_id) > 0 else None

            refreshed = False
            if refresh_identity and target_group_id is not None:
                refreshed = await self._refresh_member_identity_from_onebot(
                    user_id=clean_user_id,
                    group_id=target_group_id,
                )

            profile = await self._agent.user_profiler.get_user_profile(clean_user_id)
            payload = self._member_profile_payload(profile, target_group_id=target_group_id) if profile else {
                "user_id": clean_user_id,
                "nickname": "",
                "display_name": str(clean_user_id),
                "current_group_card": "",
                "group_cards": [],
                "learned_aliases": [],
                "names": [],
                "tags": [],
                "affinity": 0.0,
                "interaction_style": "",
                "updated_at": "",
            }
            return {
                "ok": True,
                "refreshed_identity": refreshed,
                "item": payload,
            }

        @self._app.get("/api/stickers/settings")
        async def api_sticker_settings(_: None = Depends(self._require_token),) -> dict[str, Any]:
            return await self._agent.sticker_collector.runtime_settings()

        @self._app.post("/api/stickers/settings")
        async def api_sticker_settings_update(
            payload: StickerSettingsRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            updates: dict[str, Any] = {}

            if payload.enabled is not None:
                self._cfg.sticker.enabled = bool(payload.enabled)
                updates["enabled"] = bool(payload.enabled)

            if payload.collection_rate is not None:
                clean_rate = self._clamp_probability(payload.collection_rate)
                self._cfg.sticker.collection_rate = clean_rate
                updates["collection_rate"] = clean_rate

            if payload.storage_mode is not None:
                clean_mode = self._normalize_storage_mode(payload.storage_mode)
                self._cfg.sticker.storage_mode = clean_mode
                updates["storage_mode"] = clean_mode

            if payload.filter_keywords is not None:
                clean_keywords = self._normalize_filter_keywords(payload.filter_keywords)
                self._cfg.sticker.filter_keywords = clean_keywords
                updates["filter_keywords"] = clean_keywords

            if payload.user_weights is not None:
                clean_weights = self._normalize_user_weights(payload.user_weights)
                self._cfg.sticker.user_weights = clean_weights
                updates["user_weights"] = clean_weights

            if payload.allow_other_users_collection is not None:
                clean_allow_others = bool(payload.allow_other_users_collection)
                self._cfg.sticker.allow_other_users_collection = clean_allow_others
                updates["allow_other_users_collection"] = clean_allow_others

            if payload.enable_persona_filter is not None:
                clean_persona_filter = bool(payload.enable_persona_filter)
                self._cfg.sticker.enable_persona_filter = clean_persona_filter
                updates["enable_persona_filter"] = clean_persona_filter

            if payload.llm_filter_enabled is not None:
                clean_llm_filter = bool(payload.llm_filter_enabled)
                self._cfg.sticker.llm_filter_enabled = clean_llm_filter
                updates["llm_filter_enabled"] = clean_llm_filter

            if payload.llm_filter_probability is not None:
                clean_llm_probability = self._clamp_probability(payload.llm_filter_probability)
                self._cfg.sticker.llm_filter_probability = clean_llm_probability
                updates["llm_filter_probability"] = clean_llm_probability

            if payload.llm_filter_mood_threshold is not None:
                clean_llm_mood_threshold = float(payload.llm_filter_mood_threshold)
                self._cfg.sticker.llm_filter_mood_threshold = clean_llm_mood_threshold
                updates["llm_filter_mood_threshold"] = clean_llm_mood_threshold

            if updates:
                await self._persist_config(self._set_sticker_settings, updates)

            return await self._agent.sticker_collector.runtime_settings()

        @self._app.get("/api/stickers/user-weights")
        async def api_sticker_user_weights(_: None = Depends(self._require_token),) -> dict[str, Any]:
            runtime = await self._agent.sticker_collector.runtime_settings()
            return {"user_weights": runtime.get("user_weights", {})}

        @self._app.post("/api/stickers/user-weights")
        async def api_sticker_user_weight_upsert(
            payload: StickerUserWeightRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            user_id = int(payload.user_id)
            weight = float(payload.weight)
            if user_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id must be > 0")
            self._cfg.sticker.user_weights[str(user_id)] = weight
            self._cfg.sticker.user_weights = self._normalize_user_weights(self._cfg.sticker.user_weights)
            await self._persist_config(
                self._set_sticker_user_weight,
                {"user_id": user_id, "weight": weight},
            )
            return await self._agent.sticker_collector.runtime_settings()

        @self._app.delete("/api/stickers/user-weights/{user_id}")
        async def api_sticker_user_weight_delete(
            user_id: int,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_id = int(user_id)
            if clean_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id must be > 0")
            self._cfg.sticker.user_weights.pop(str(clean_id), None)
            self._cfg.sticker.user_weights = self._normalize_user_weights(self._cfg.sticker.user_weights)
            await self._persist_config(self._delete_sticker_user_weight, clean_id)
            return await self._agent.sticker_collector.runtime_settings()

        @self._app.get("/api/stickers/files")
        async def api_sticker_files(_: None = Depends(self._require_token),) -> dict[str, Any]:
            files = await self._agent.sticker_collector.list_local_files()
            hydrated_files: list[dict[str, Any]] = []
            for item in files:
                row = dict(item)
                file_name = str(row.get("file_name", "")).strip()
                row["content_url"] = self._sticker_content_url(file_name) if file_name else ""
                hydrated_files.append(row)
            return {
                "root": str(self._agent.sticker_collector.local_dir),
                "files": hydrated_files,
            }

        @self._app.get("/api/stickers/pending/files")
        async def api_sticker_pending_files(_: None = Depends(self._require_token),) -> dict[str, Any]:
            files = await self._agent.sticker_collector.list_pending_files()
            hydrated_files: list[dict[str, Any]] = []
            for item in files:
                row = dict(item)
                file_name = str(row.get("file_name", "")).strip()
                row["content_url"] = self._sticker_pending_content_url(file_name) if file_name else ""
                hydrated_files.append(row)
            return {
                "root": str(self._agent.sticker_collector.pending_dir),
                "files": hydrated_files,
            }

        @self._app.post("/api/stickers/pending/files/{file_name}/approve")
        async def api_sticker_pending_file_approve(
            file_name: str,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                result = await self._agent.sticker_collector.approve_pending_file(file_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

            if not bool(result.get("ok")):
                detail = str(result.get("error") or "failed to approve pending sticker")
                code = status.HTTP_404_NOT_FOUND if "not found" in detail.lower() else status.HTTP_400_BAD_REQUEST
                raise HTTPException(status_code=code, detail=detail)
            return result

        @self._app.delete("/api/stickers/pending/files/{file_name}")
        async def api_sticker_pending_file_delete(
            file_name: str,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                deleted = await self._agent.sticker_collector.reject_pending_file(file_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

            if not deleted:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pending sticker file not found")
            return {"ok": True, "file_name": file_name}

        @self._app.delete("/api/stickers/files/{file_name}")
        async def api_sticker_file_delete(
            file_name: str,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            try:
                deleted = await self._agent.sticker_collector.delete_local_file(file_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

            if not deleted:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sticker file not found")
            return {"ok": True, "file_name": file_name}

        @self._app.get("/api/stickers/files/{file_name}/content")
        async def api_sticker_file_content(
            file_name: str,
            download: bool = Query(default=False),
            _: None = Depends(self._require_token),
        ) -> FileResponse:
            try:
                target = self._agent.sticker_collector.resolve_local_file_path(file_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            if not target.exists() or not target.is_file():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sticker file not found")

            media_type, _ = mimetypes.guess_type(str(target))
            if download:
                return FileResponse(path=str(target), media_type=media_type or "application/octet-stream", filename=target.name)
            return FileResponse(path=str(target), media_type=media_type or "application/octet-stream")

        @self._app.get("/api/stickers/pending/files/{file_name}/content")
        async def api_sticker_pending_file_content(
            file_name: str,
            download: bool = Query(default=False),
            _: None = Depends(self._require_token),
        ) -> FileResponse:
            try:
                target = self._agent.sticker_collector.resolve_pending_file_path(file_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            if not target.exists() or not target.is_file():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pending sticker file not found")

            media_type, _ = mimetypes.guess_type(str(target))
            if download:
                return FileResponse(path=str(target), media_type=media_type or "application/octet-stream", filename=target.name)
            return FileResponse(path=str(target), media_type=media_type or "application/octet-stream")

        @self._app.get("/api/stickers/library")
        async def api_sticker_library(
            keyword: str = Query(default=""),
            limit: int = Query(default=200, ge=1, le=500),
            storage_mode: str = Query(default="local"),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_mode = self._normalize_sticker_library_mode(storage_mode)
            mode_for_search = clean_mode if clean_mode in {"local", "cloud"} else "all"
            items = await self._agent.sticker_collector.search(
                keyword=keyword,
                limit=limit,
                storage_mode=mode_for_search,
            )
            hydrated_items: list[dict[str, Any]] = []
            for item in items:
                row = dict(item)
                file_name = str(row.get("file_name", "")).strip()
                row["content_url"] = (
                    self._sticker_content_url(file_name)
                    if str(row.get("storage_mode", "")).lower() == "local" and file_name
                    else ""
                )
                hydrated_items.append(row)
            return {
                "root": str(self._agent.sticker_collector.local_dir),
                "storage_mode": clean_mode,
                "count": len(hydrated_items),
                "items": hydrated_items,
            }

        @self._app.post("/api/stickers/library/audit")
        async def api_sticker_library_audit(
            payload: StickerAuditRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            result = await self._agent.sticker_collector.audit_and_tag_library(
                force_llm=bool(payload.force_llm),
                limit=int(payload.limit),
            )
            return result

        @self._app.post("/api/stickers/open-dir")
        async def api_sticker_open_dir(_: None = Depends(self._require_token),) -> dict[str, Any]:
            sticker_dir = self._agent.sticker_collector.local_dir
            try:
                self._open_local_directory(sticker_dir)
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
            return {"ok": True, "path": str(sticker_dir)}

        @self._app.post("/api/stickers/open-pending-dir")
        async def api_sticker_open_pending_dir(_: None = Depends(self._require_token),) -> dict[str, Any]:
            pending_dir = self._agent.sticker_collector.pending_dir
            try:
                self._open_local_directory(pending_dir)
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
            return {"ok": True, "path": str(pending_dir)}

        @self._app.post("/api/action/reset")
        async def api_reset(
            payload: ResetActionRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            return await self._agent.reset_runtime_state(
                fill_energy=payload.fill_energy,
                session_id=payload.clear_session_id.strip() or None,
            )

        @self._app.post("/api/action/shutdown")
        async def api_shutdown(_: None = Depends(self._require_token),) -> dict[str, Any]:
            if self._shutdown_handler is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="shutdown handler is not available",
                )

            async with self._shutdown_lock:
                if self._shutdown_requested:
                    return {"ok": True, "scheduled": True, "message": "shutdown already requested"}
                self._shutdown_requested = True

            await self._log_hub.publish(
                "system",
                "[system] shutdown requested from dashboard",
                channel="system",
            )
            asyncio.create_task(self._schedule_shutdown(), name="dashboard-shutdown")
            return {"ok": True, "scheduled": True, "message": "shutdown scheduled"}

        @self._app.post("/api/action/restart")
        async def api_restart(_: None = Depends(self._require_token),) -> dict[str, Any]:
            if self._restart_handler is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="restart handler is not available",
                )

            async with self._restart_lock:
                if self._restart_requested:
                    return {"ok": True, "scheduled": True, "message": "restart already requested"}
                self._restart_requested = True

            await self._log_hub.publish(
                "system",
                "[system] restart requested from dashboard",
                channel="system",
            )
            asyncio.create_task(self._schedule_restart(), name="dashboard-restart")
            return {"ok": True, "scheduled": True, "message": "restart scheduled"}

        @self._app.post("/api/config/update")
        async def api_config_update(
            payload: UpdateConfigRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean = payload.system_prompt.strip()
            if not clean:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="system_prompt is empty")

            self._agent.update_system_prompt(clean)
            self._cfg.persona.system_prompt = clean
            await self._save_persona_prompt_template(clean)
            return {"ok": True, "system_prompt_length": len(clean)}

        @self._app.post("/api/ui/background")
        async def api_ui_background(
            request: Request,
            background_url: str | None = Form(default=None),
            file: UploadFile | None = File(default=None),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_url = (background_url or "").strip()
            content_type = request.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                try:
                    payload = await request.json()
                except ValueError as exc:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid json body") from exc
                if isinstance(payload, dict):
                    clean_url = str(payload.get("background_url", "")).strip()

            if file is not None:
                clean_url = await self._save_uploaded_background(file)

            if not clean_url:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="background_url or file is required")
            self._validate_background_url(clean_url)

            self._cfg.web.ui_settings.background_url = clean_url
            await self._persist_config(self._set_background_url, clean_url)
            return {"ok": True, "background_url": clean_url}

        @self._app.get("/api/knowledge/files")
        async def api_knowledge_files(_: None = Depends(self._require_token),) -> dict[str, Any]:
            return {
                "root": str(self._knowledge_dir),
                "files": self._list_knowledge_files(),
            }

        @self._app.get("/api/knowledge/file")
        async def api_knowledge_file(
            path: str = Query(..., min_length=1),
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            target = self._resolve_knowledge_path(path)
            if not target.exists() or not target.is_file():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="knowledge file not found")
            content = target.read_text(encoding="utf-8", errors="replace")
            return {
                "path": str(target.relative_to(self._knowledge_dir)).replace("\\", "/"),
                "content": content,
            }

        @self._app.post("/api/knowledge/file")
        async def api_knowledge_save(
            payload: KnowledgeSaveRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            target = self._resolve_knowledge_path(payload.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload.content, encoding="utf-8")
            return {
                "ok": True,
                "path": str(target.relative_to(self._knowledge_dir)).replace("\\", "/"),
                "size": target.stat().st_size,
            }

        @self._app.post("/api/knowledge/reindex")
        async def api_knowledge_reindex(_: None = Depends(self._require_token),) -> dict[str, Any]:
            result = await self._agent.memory_mgr.reindex_external_knowledge(self._knowledge_dir)
            return {"ok": True, "result": result}

        @self._app.get("/api/plugins")
        async def api_plugins(_: None = Depends(self._require_token),) -> dict[str, Any]:
            states = await self._plugin_manager.list_states()
            return {
                "modules": states,
                "loaded_plugins": self._plugin_manager.list_loaded_plugins(),
            }

        @self._app.post("/api/plugins/{module_name}/load")
        async def api_plugin_load(module_name: str, _: None = Depends(self._require_token),) -> dict[str, Any]:
            try:
                result = await self._plugin_manager.load_module(module_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            return {"ok": True, "state": result}

        @self._app.post("/api/plugins/{module_name}/unload")
        async def api_plugin_unload(module_name: str, _: None = Depends(self._require_token),) -> dict[str, Any]:
            try:
                result = await self._plugin_manager.unload_module(module_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            return {"ok": True, "state": result}

        @self._app.post("/api/plugins/{module_name}/reload")
        async def api_plugin_reload(module_name: str, _: None = Depends(self._require_token),) -> dict[str, Any]:
            try:
                result = await self._plugin_manager.reload_module(module_name)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            return {"ok": True, "state": result}

    async def _schedule_shutdown(self) -> None:
        # Let the HTTP response flush first, then stop the whole runtime.
        await asyncio.sleep(0.25)
        handler = self._shutdown_handler
        if handler is None:
            return
        try:
            await handler()
        except Exception:
            async with self._shutdown_lock:
                self._shutdown_requested = False

    async def _schedule_restart(self) -> None:
        # Let the HTTP response flush first, then restart the whole runtime.
        await asyncio.sleep(0.25)
        handler = self._restart_handler
        if handler is None:
            return
        try:
            await handler()
        except Exception:
            async with self._restart_lock:
                self._restart_requested = False

    async def _save_uploaded_background(self, file: UploadFile) -> str:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            suffix = ".png"

        random_name = f"{secrets.token_hex(16)}{suffix}"
        target = self._upload_dir / random_name
        content = await file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty")
        target.write_bytes(content)
        return f"/assets/uploads/{random_name}"

    def _runtime_config_payload(self) -> dict[str, Any]:
        return {
            "master_id": int(self._cfg.persona.master_id or 0),
            "persona_prompt": str(self._cfg.persona.system_prompt or ""),
            "groups": self._serialize_groups(),
            "sticker": {
                "enabled": bool(self._cfg.sticker.enabled),
                "collection_rate": float(self._cfg.sticker.collection_rate),
                "storage_mode": self._normalize_storage_mode(self._cfg.sticker.storage_mode),
                "local_dir": str(self._agent.sticker_collector.local_dir),
                "pending_local_dir": str(self._agent.sticker_collector.pending_dir),
                "filter_keywords": self._normalize_filter_keywords(self._cfg.sticker.filter_keywords),
                "user_weights": self._normalize_user_weights(self._cfg.sticker.user_weights),
                "allow_other_users_collection": bool(self._cfg.sticker.allow_other_users_collection),
                "enable_persona_filter": bool(self._cfg.sticker.enable_persona_filter),
                "llm_filter_enabled": bool(self._cfg.sticker.llm_filter_enabled),
                "llm_filter_probability": self._clamp_probability(self._cfg.sticker.llm_filter_probability),
                "llm_filter_mood_threshold": float(self._cfg.sticker.llm_filter_mood_threshold),
            },
            "memory": {
                "memory_store_path": str(self._cfg.memory.memory_store_path),
                "tool_call_store_path": str(self._cfg.memory.tool_call_store_path),
                "tool_call_max_entries": int(self._cfg.memory.tool_call_max_entries),
                "memory_auto_ingest_enabled": bool(self._cfg.memory.memory_auto_ingest_enabled),
                "memory_convergence_interval_minutes": int(self._cfg.memory.memory_convergence_interval_minutes),
                "memory_candidate_grace_hours": int(self._cfg.memory.memory_candidate_grace_hours),
                "memory_candidate_promote_evidence": int(self._cfg.memory.memory_candidate_promote_evidence),
            },
            "ops": {
                "style_card_store_path": str(self._style_card_store_path),
                "jargon_store_path": str(self._jargon_store_path),
                "jargon_rejected_store_path": str(self._jargon_rejected_store_path),
            },
        }

    def _serialize_groups(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._cfg.groups:
            out.append(
                {
                    "group_id": int(item.group_id),
                    "enabled": bool(item.enabled),
                    "group_name": str(item.group_name or ""),
                    "remark": str(item.remark or ""),
                    "extra_prompt": str(item.extra_prompt or ""),
                },
            )
        out.sort(key=lambda row: int(row["group_id"]))
        return out

    async def _hydrate_group_names(self, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        changed_rows: list[dict[str, Any]] = []

        for row in groups:
            item = dict(row)
            group_id = int(item.get("group_id", 0) or 0)
            if group_id <= 0:
                hydrated.append(item)
                continue

            fetched_name = await self._fetch_group_name_from_onebot(group_id)
            if fetched_name:
                item["group_name"] = fetched_name
                current = self._cfg.get_group(group_id)
                if current is not None and str(current.group_name or "").strip() != fetched_name:
                    current.group_name = fetched_name
                    changed_rows.append(item)

            hydrated.append(item)

        for item in changed_rows:
            await self._persist_config(
                self._upsert_group_config,
                {
                    "group_id": int(item.get("group_id", 0) or 0),
                    "enabled": bool(item.get("enabled", True)),
                    "group_name": str(item.get("group_name", "") or ""),
                    "remark": None,
                    "extra_prompt": None,
                },
            )

        return hydrated

    async def _fetch_group_name_from_onebot(self, group_id: int) -> str:
        if int(group_id) <= 0:
            return ""
        bot_client = getattr(self._agent, "bot_client", None)
        if bot_client is None:
            return ""

        try:
            response = await bot_client.call_action_with_response(
                "get_group_info",
                {"group_id": int(group_id), "no_cache": False},
                timeout=4.0,
            )
        except Exception:
            return ""

        if not isinstance(response, dict):
            return ""

        status_raw = str(response.get("status", "") or "").strip().lower()
        if status_raw and status_raw != "ok":
            return ""

        retcode_raw = response.get("retcode")
        if isinstance(retcode_raw, int) and retcode_raw != 0:
            return ""

        data = response.get("data")
        if not isinstance(data, dict):
            return ""

        return str(data.get("group_name", "") or "").strip()

    def _member_profile_payload(self, profile: Any, *, target_group_id: int | None) -> dict[str, Any]:
        if profile is None:
            return {}

        group_cards = [
            {
                "group_id": int(item.group_id or 0),
                "card": str(item.content or "").strip(),
                "updated_at": str(item.updated_at or "").strip(),
            }
            for item in getattr(profile, "name_records", ())
            if str(getattr(item, "source", "")).strip() == MEMBER_NAME_SOURCE_GROUP_CARD
            and str(getattr(item, "content", "")).strip()
        ]
        learned_aliases = member_learned_aliases(getattr(profile, "name_records", ()))
        names = member_names_for_admin(getattr(profile, "name_records", ()), str(getattr(profile, "nickname", "") or ""))
        current_group_card = (
            latest_member_group_card(getattr(profile, "name_records", ()), target_group_id)
            if target_group_id is not None
            else ""
        )
        display_name = str(current_group_card or getattr(profile, "nickname", "") or "").strip()
        if not display_name and learned_aliases:
            display_name = learned_aliases[0]
        if not display_name:
            display_name = str(int(getattr(profile, "user_id", 0) or 0))

        return {
            "user_id": int(getattr(profile, "user_id", 0) or 0),
            "nickname": str(getattr(profile, "nickname", "") or "").strip(),
            "display_name": display_name,
            "current_group_card": current_group_card,
            "group_cards": group_cards,
            "learned_aliases": learned_aliases,
            "names": names,
            "tags": list(getattr(profile, "tags", ()) or ()),
            "affinity": float(getattr(profile, "affinity", 0.0) or 0.0),
            "interaction_style": str(getattr(profile, "interaction_style", "") or "").strip(),
            "updated_at": str(getattr(profile, "updated_at", "") or "").strip(),
        }

    async def _refresh_member_identity_from_onebot(self, *, user_id: int, group_id: int) -> bool:
        if int(user_id) <= 0 or int(group_id) <= 0:
            return False
        bot_client = getattr(self._agent, "bot_client", None)
        if bot_client is None:
            return False

        try:
            info = await bot_client.get_group_member_info(
                group_id=int(group_id),
                user_id=int(user_id),
                no_cache=False,
                timeout=4.0,
            )
        except Exception:
            return False

        if not isinstance(info, dict):
            return False
        latest_nickname = str(info.get("nickname", "") or "").strip()
        latest_group_card = str(info.get("card", "") or "").strip()
        if not latest_nickname and not latest_group_card:
            return False

        if latest_nickname:
            await self._agent.user_profiler.replace_profile_nickname(
                user_id=int(user_id),
                nickname=latest_nickname,
            )
        await self._agent.user_profiler.sync_member_identity(
            user_id=int(user_id),
            nickname=latest_nickname,
            group_id=int(group_id),
            group_card=latest_group_card,
        )
        return True

    @staticmethod
    def _clamp_probability(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _normalize_storage_mode(raw_mode: str) -> str:
        mode = str(raw_mode or "").strip().lower()
        return mode if mode in {"local", "cloud"} else "local"

    @staticmethod
    def _normalize_sticker_library_mode(raw_mode: str) -> str:
        mode = str(raw_mode or "").strip().lower()
        if mode in {"all", "*"}:
            return "all"
        if mode in {"local", "cloud"}:
            return mode
        return "local"

    @staticmethod
    def _sticker_content_url(file_name: str) -> str:
        clean_name = str(file_name or "").strip()
        if not clean_name:
            return ""
        return f"/api/stickers/files/{quote(clean_name, safe='')}/content"

    @staticmethod
    def _sticker_pending_content_url(file_name: str) -> str:
        clean_name = str(file_name or "").strip()
        if not clean_name:
            return ""
        return f"/api/stickers/pending/files/{quote(clean_name, safe='')}/content"

    @staticmethod
    def _normalize_filter_keywords(raw_keywords: Any) -> list[str]:
        if isinstance(raw_keywords, str):
            candidates = [item.strip() for item in raw_keywords.split(",")]
        elif isinstance(raw_keywords, list):
            candidates = [str(item).strip() for item in raw_keywords]
        else:
            candidates = []
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if not item:
                continue
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(item)
        return out

    @staticmethod
    def _normalize_user_weights(raw_weights: Any) -> dict[str, float]:
        if not isinstance(raw_weights, dict):
            return {}
        out: dict[str, float] = {}
        for raw_key, raw_value in raw_weights.items():
            key = str(raw_key).strip()
            if not key:
                continue
            try:
                out[key] = float(raw_value)
            except (TypeError, ValueError):
                continue
        return dict(sorted(out.items(), key=lambda item: item[0]))

    @staticmethod
    def _open_local_directory(target_dir: Path) -> None:
        path = target_dir.resolve()
        if os.name == "nt":
            subprocess.Popen(["explorer", str(path)])
            return
        if os.name == "posix":
            subprocess.Popen(["xdg-open", str(path)])
            return
        raise RuntimeError("open directory is not supported on this platform")

    def _list_knowledge_files(self) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for file_path in sorted(self._knowledge_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            if not self._is_text_file(file_path):
                continue
            relative = str(file_path.relative_to(self._knowledge_dir)).replace("\\", "/")
            stat = file_path.stat()
            files.append(
                {
                    "path": relative,
                    "size": int(stat.st_size),
                    "updated_at": int(stat.st_mtime),
                },
            )
        return files

    def _resolve_knowledge_path(self, raw_relative: str) -> Path:
        clean_relative = str(raw_relative or "").strip().replace("\\", "/")
        if not clean_relative:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="knowledge path is empty")
        relative_path = Path(clean_relative)
        if relative_path.is_absolute():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="absolute path is not allowed")
        if any(part in {"..", ""} for part in relative_path.parts):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid knowledge path")

        target = (self._knowledge_dir / relative_path).resolve()
        if self._knowledge_dir not in target.parents and target != self._knowledge_dir:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid knowledge path")
        if not self._is_text_file(target):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="only text files are allowed")
        return target

    @staticmethod
    def _is_text_file(file_path: Path) -> bool:
        return file_path.suffix.lower() in TEXT_FILE_EXTENSIONS

    def _validate_background_url(self, url: str) -> None:
        clean = url.strip()
        if not clean:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="background_url is empty")
        if clean.startswith("/assets/"):
            return
        if clean.startswith("http://") or clean.startswith("https://"):
            return
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="background_url must start with http://, https://, or /assets/",
        )

    def _build_cors_origins(self) -> list[str]:
        # Allow local frontend dev servers by default.
        local_hosts = {"localhost", "127.0.0.1"}
        local_ports = {3000, 5000, 8000, 18002, int(self._port)}
        origins: set[str] = set()

        for scheme in ("http", "https"):
            for host in local_hosts:
                for port in local_ports:
                    origins.add(f"{scheme}://{host}:{int(port)}")

        clean_host = self._host.strip()
        if clean_host and clean_host not in {"0.0.0.0", "::", "[::]"}:
            for scheme in ("http", "https"):
                origins.add(f"{scheme}://{clean_host}:{int(self._port)}")

        return sorted(origins)

    async def _require_token(self, request: Request) -> None:
        expected = self._access_token.strip()
        if not expected:
            # No-token mode: open access for local dashboard usage.
            return

        provided = self._extract_token(request)
        if provided != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid access token")

    async def _authorize_websocket(self, websocket: WebSocket) -> bool:
        expected = self._access_token.strip()
        if not expected:
            # No-token mode: open websocket access.
            return True
        provided = self._extract_ws_token(websocket)
        if provided != expected:
            await websocket.close(code=1008, reason="invalid access token")
            return False
        return True

    @staticmethod
    def _extract_token(request: Request) -> str:
        header = str(request.headers.get("x-access-token", "")).strip()
        if header:
            return header

        auth = str(request.headers.get("authorization", "")).strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    @staticmethod
    def _extract_ws_token(websocket: WebSocket) -> str:
        query_token = str(websocket.query_params.get("token", "")).strip()
        if query_token:
            return query_token
        header_token = str(websocket.headers.get("x-access-token", "")).strip()
        if header_token:
            return header_token
        auth = str(websocket.headers.get("authorization", "")).strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    def _health_payload(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        started_at_raw = getattr(self._agent, "_started_at_utc", None)
        started_at = started_at_raw.isoformat() if isinstance(started_at_raw, datetime) else ""
        connected = bool(getattr(getattr(self._agent, "bot_client", None), "connected", False))
        return {
            "status": "ok",
            "name": "zhiyue-bot",
            "time": now,
            "connected": connected,
            "uptime_seconds": int(getattr(self._agent, "uptime_seconds", lambda: 0)() or 0),
            "started_at": started_at,
        }

    async def _list_style_cards(
        self,
        *,
        group_id: int,
        status_filter: str,
        keyword: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))
        clean_status = self._normalize_style_card_status(status_filter, allow_empty=True)
        clean_keyword = str(keyword or "").strip().lower()
        clean_group_id = int(group_id)

        async with self._style_card_lock:
            store = self._load_style_card_store_locked()
            rows = [dict(item) for item in store.get("items", []) if isinstance(item, dict)]

        if clean_group_id > 0:
            rows = [item for item in rows if int(item.get("group_id", 0) or 0) == clean_group_id]
        if clean_status:
            rows = [item for item in rows if str(item.get("status", "")).strip().lower() == clean_status]
        if clean_keyword:
            rows = [item for item in rows if clean_keyword in self._style_card_search_text(item)]

        rows.sort(key=lambda item: str(item.get("updated_at", "") or ""), reverse=True)
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size
        items = [self._style_card_row(item) for item in rows[start:end]]
        return {
            "items": items,
            "total": len(rows),
            "page": safe_page,
            "page_size": safe_page_size,
            "query": {
                "group_id": clean_group_id,
                "status": clean_status,
                "keyword": str(keyword or ""),
            },
        }

    async def _get_style_card(self, style_card_id: int) -> dict[str, Any] | None:
        clean_id = int(style_card_id)
        if clean_id <= 0:
            return None
        async with self._style_card_lock:
            store = self._load_style_card_store_locked()
            for item in store.get("items", []):
                if not isinstance(item, dict):
                    continue
                if int(item.get("id", 0) or 0) != clean_id:
                    continue
                return self._style_card_row(item)
        return None

    async def _create_style_card(self, payload: StyleCardCreateRequest) -> dict[str, Any]:
        title = str(payload.title or "").strip()
        content = str(payload.content or "").strip()
        if not title:
            raise ValueError("title is empty")
        if not content:
            raise ValueError("content is empty")

        clean_status = self._normalize_style_card_status(payload.status, allow_empty=False)
        if not clean_status:
            raise ValueError("invalid style card status")
        now = datetime.now(timezone.utc).isoformat()
        async with self._style_card_lock:
            store = self._load_style_card_store_locked()
            card_id = int(store.get("next_id", 1) or 1)
            store["next_id"] = card_id + 1
            item = {
                "id": card_id,
                "group_id": max(0, int(payload.group_id or 0)),
                "title": title,
                "content": content,
                "intent": str(payload.intent or "").strip(),
                "tone": str(payload.tone or "").strip(),
                "tags": self._normalize_tags(payload.tags),
                "status": clean_status,
                "source_kind": str(payload.source_kind or "manual").strip() or "manual",
                "source_ref": str(payload.source_ref or "").strip(),
                "use_count": 0,
                "evidence_count": 0,
                "created_at": now,
                "updated_at": now,
            }
            store.setdefault("items", []).append(item)
            self._save_style_card_store_locked(store)
            return self._style_card_row(item)

    async def _update_style_card(self, style_card_id: int, payload: StyleCardUpdateRequest) -> dict[str, Any] | None:
        clean_id = int(style_card_id)
        if clean_id <= 0:
            return None
        async with self._style_card_lock:
            store = self._load_style_card_store_locked()
            for item in store.get("items", []):
                if not isinstance(item, dict):
                    continue
                if int(item.get("id", 0) or 0) != clean_id:
                    continue
                if payload.group_id is not None:
                    item["group_id"] = max(0, int(payload.group_id or 0))
                if payload.title is not None:
                    title = str(payload.title or "").strip()
                    if not title:
                        raise ValueError("title is empty")
                    item["title"] = title
                if payload.content is not None:
                    content = str(payload.content or "").strip()
                    if not content:
                        raise ValueError("content is empty")
                    item["content"] = content
                if payload.intent is not None:
                    item["intent"] = str(payload.intent or "").strip()
                if payload.tone is not None:
                    item["tone"] = str(payload.tone or "").strip()
                if payload.tags is not None:
                    item["tags"] = self._normalize_tags(payload.tags)
                if payload.source_kind is not None:
                    item["source_kind"] = str(payload.source_kind or "").strip() or "manual"
                if payload.source_ref is not None:
                    item["source_ref"] = str(payload.source_ref or "").strip()
                item["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_style_card_store_locked(store)
                return self._style_card_row(item)
        return None

    async def _set_style_card_status(self, style_card_id: int, raw_status: str) -> dict[str, Any] | None:
        clean_id = int(style_card_id)
        if clean_id <= 0:
            return None
        clean_status = self._normalize_style_card_status(raw_status, allow_empty=False)
        if not clean_status:
            raise ValueError("invalid style card status")
        async with self._style_card_lock:
            store = self._load_style_card_store_locked()
            for item in store.get("items", []):
                if not isinstance(item, dict):
                    continue
                if int(item.get("id", 0) or 0) != clean_id:
                    continue
                item["status"] = clean_status
                item["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_style_card_store_locked(store)
                return self._style_card_row(item)
        return None

    async def _delete_style_card(self, style_card_id: int) -> bool:
        clean_id = int(style_card_id)
        if clean_id <= 0:
            return False
        async with self._style_card_lock:
            store = self._load_style_card_store_locked()
            items = [item for item in store.get("items", []) if isinstance(item, dict)]
            next_items = [item for item in items if int(item.get("id", 0) or 0) != clean_id]
            if len(next_items) == len(items):
                return False
            store["items"] = next_items
            self._save_style_card_store_locked(store)
            return True

    def _load_style_card_store_locked(self) -> dict[str, Any]:
        default = {"version": 1, "next_id": 1, "items": []}
        if not self._style_card_store_path.exists():
            return dict(default)
        try:
            payload = json.loads(self._style_card_store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return dict(default)
        if not isinstance(payload, dict):
            return dict(default)
        raw_items = payload.get("items", [])
        items: list[dict[str, Any]] = []
        max_id = 0
        if isinstance(raw_items, list):
            for row in raw_items:
                if not isinstance(row, dict):
                    continue
                style_id = int(row.get("id", 0) or 0)
                if style_id <= 0:
                    continue
                max_id = max(max_id, style_id)
                normalized = {
                    "id": style_id,
                    "group_id": max(0, int(row.get("group_id", 0) or 0)),
                    "title": str(row.get("title", "") or "").strip(),
                    "content": str(row.get("content", "") or "").strip(),
                    "intent": str(row.get("intent", "") or "").strip(),
                    "tone": str(row.get("tone", "") or "").strip(),
                    "tags": self._normalize_tags(row.get("tags", [])),
                    "status": self._normalize_style_card_status(str(row.get("status", "") or ""), allow_empty=False)
                    or "candidate",
                    "source_kind": str(row.get("source_kind", "manual") or "manual").strip() or "manual",
                    "source_ref": str(row.get("source_ref", "") or "").strip(),
                    "use_count": max(0, int(row.get("use_count", 0) or 0)),
                    "evidence_count": max(0, int(row.get("evidence_count", 0) or 0)),
                    "created_at": str(row.get("created_at", "") or "").strip() or datetime.now(timezone.utc).isoformat(),
                    "updated_at": str(row.get("updated_at", "") or "").strip() or datetime.now(timezone.utc).isoformat(),
                }
                if normalized["title"] and normalized["content"]:
                    items.append(normalized)
        next_id_raw = int(payload.get("next_id", 0) or 0)
        next_id = next_id_raw if next_id_raw > max_id else (max_id + 1 if max_id > 0 else 1)
        return {"version": 1, "next_id": next_id, "items": items}

    def _save_style_card_store_locked(self, payload: dict[str, Any]) -> None:
        self._style_card_store_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._style_card_store_path.with_suffix(self._style_card_store_path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self._style_card_store_path)

    @staticmethod
    def _normalize_style_card_status(raw_status: str, *, allow_empty: bool) -> str:
        clean = str(raw_status or "").strip().lower()
        if not clean:
            return "" if allow_empty else "candidate"
        if clean not in STYLE_CARD_STATUSES:
            return ""
        return clean

    @staticmethod
    def _style_card_search_text(item: dict[str, Any]) -> str:
        parts = [
            str(item.get("title", "") or ""),
            str(item.get("content", "") or ""),
            str(item.get("intent", "") or ""),
            str(item.get("tone", "") or ""),
            " ".join(str(tag or "").strip() for tag in item.get("tags", []) if str(tag or "").strip()),
        ]
        return "\n".join(parts).lower()

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> list[str]:
        if not isinstance(raw_tags, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw_tags:
            clean = str(item or "").strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(clean[:32])
            if len(out) >= 24:
                break
        return out

    @staticmethod
    def _style_card_row(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(item.get("id", 0) or 0),
            "group_id": int(item.get("group_id", 0) or 0),
            "title": str(item.get("title", "") or ""),
            "content": str(item.get("content", "") or ""),
            "intent": str(item.get("intent", "") or ""),
            "tone": str(item.get("tone", "") or ""),
            "tags": list(item.get("tags", []) if isinstance(item.get("tags"), list) else []),
            "status": str(item.get("status", "candidate") or "candidate"),
            "source_kind": str(item.get("source_kind", "manual") or "manual"),
            "source_ref": str(item.get("source_ref", "") or ""),
            "use_count": int(item.get("use_count", 0) or 0),
            "evidence_count": int(item.get("evidence_count", 0) or 0),
            "created_at": str(item.get("created_at", "") or ""),
            "updated_at": str(item.get("updated_at", "") or ""),
        }

    async def _list_jargons(
        self,
        *,
        status_filter: str,
        scope: str,
        keyword: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        clean_status = self._normalize_jargon_status(status_filter, allow_empty=True)
        clean_scope = self._normalize_jargon_scope(scope, allow_empty=True)
        clean_keyword = str(keyword or "").strip().lower()
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))

        async with self._jargon_store_lock:
            _, managed, _ = self._load_jargon_payload_locked()
            rejected = self._load_rejected_jargons_locked()
            items = self._collect_jargon_rows_locked(managed, rejected)

        if clean_status:
            items = [row for row in items if str(row.get("status", "")) == clean_status]
        if clean_scope:
            items = [row for row in items if str(row.get("scope", "")) == clean_scope]
        if clean_keyword:
            items = [row for row in items if clean_keyword in self._jargon_search_text(row)]

        items.sort(key=lambda row: str(row.get("updated_at", "") or ""), reverse=True)
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size
        return {
            "items": items[start:end],
            "total": len(items),
            "page": safe_page,
            "page_size": safe_page_size,
            "query": {
                "status": clean_status,
                "scope": clean_scope,
                "keyword": str(keyword or ""),
            },
        }

    async def _get_jargon(self, jargon_id: str) -> dict[str, Any] | None:
        clean_id = str(jargon_id or "").strip()
        if not clean_id:
            return None
        async with self._jargon_store_lock:
            _, managed, _ = self._load_jargon_payload_locked()
            rejected = self._load_rejected_jargons_locked()
            for row in self._collect_jargon_rows_locked(managed, rejected):
                if str(row.get("id", "")) == clean_id:
                    return row
        return None

    async def _create_jargon(self, payload: JargonCreateRequest) -> dict[str, Any]:
        jargon = str(payload.jargon or "").strip()
        standard = str(payload.standard or "").strip()
        if not jargon:
            raise ValueError("jargon is empty")
        if not standard:
            raise ValueError("standard is empty")
        target_scope = self._normalize_jargon_scope(payload.scope, allow_empty=False)
        if not target_scope:
            raise ValueError("invalid scope")
        now = datetime.now(timezone.utc).isoformat()
        new_entry = self._normalize_jargon_entry(
            {
                "jargon": jargon,
                "standard": standard,
                "meaning": str(payload.meaning or "").strip(),
                "confidence": payload.confidence,
                "weight": payload.weight,
                "source_users": payload.source_users or [],
                "updated_at": now,
            }
        )
        if new_entry is None:
            raise ValueError("invalid jargon entry")
        key = self._compose_jargon_key(new_entry["standard"], new_entry["jargon"])

        async with self._jargon_store_lock:
            payload_data, managed, extras = self._load_jargon_payload_locked()
            rejected = self._load_rejected_jargons_locked()
            for scope_name in JARGON_SCOPES:
                managed.setdefault(scope_name, {}).pop(key, None)
            managed.setdefault(target_scope, {})[key] = new_entry
            self._save_jargon_payload_locked(payload_data, managed, extras)
            rejected = [row for row in rejected if self._compose_jargon_key(row["standard"], row["jargon"]) != key]
            self._save_rejected_jargons_locked(rejected)

        await self._reload_jargon_runtime()
        status_text = "active" if target_scope == "public" else "candidate"
        return self._build_jargon_row(
            scope=target_scope,
            status=status_text,
            key=key,
            entry=new_entry,
        )

    async def _set_jargon_status(self, jargon_id: str, raw_status: str, raw_scope: str) -> dict[str, Any] | None:
        clean_id = str(jargon_id or "").strip()
        if not clean_id:
            return None
        target_status = self._normalize_jargon_status(raw_status, allow_empty=False)
        if not target_status:
            raise ValueError("invalid jargon status")
        target_candidate_scope = self._normalize_jargon_scope(raw_scope, allow_empty=False) or "group"
        if target_candidate_scope == "public":
            target_candidate_scope = "group"

        changed_live = False
        changed_rejected = False
        final_row: dict[str, Any] | None = None
        async with self._jargon_store_lock:
            payload_data, managed, extras = self._load_jargon_payload_locked()
            rejected = self._load_rejected_jargons_locked()

            found_scope = ""
            found_key = ""
            found_entry: dict[str, Any] | None = None
            for scope_name, bucket in managed.items():
                for key, entry in bucket.items():
                    if self._jargon_id(scope_name, key, "active" if scope_name == "public" else "candidate") != clean_id:
                        continue
                    found_scope = scope_name
                    found_key = key
                    found_entry = dict(entry)
                    break
                if found_entry is not None:
                    break

            found_rejected_idx = -1
            found_rejected: dict[str, Any] | None = None
            if found_entry is None:
                for idx, row in enumerate(rejected):
                    if str(row.get("id", "")) == clean_id:
                        found_rejected_idx = idx
                        found_rejected = dict(row)
                        break

            if found_entry is None and found_rejected is None:
                return None

            if found_entry is not None:
                if target_status == "rejected":
                    managed[found_scope].pop(found_key, None)
                    changed_live = True
                    rejected_row = self._build_rejected_jargon_row(
                        scope=found_scope,
                        key=found_key,
                        entry=found_entry,
                    )
                    rejected = [row for row in rejected if str(row.get("id", "")) != str(rejected_row.get("id", ""))]
                    rejected.append(rejected_row)
                    changed_rejected = True
                    final_row = rejected_row
                else:
                    target_scope = "public" if target_status == "active" else target_candidate_scope
                    managed[found_scope].pop(found_key, None)
                    found_entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                    target_key = self._compose_jargon_key(found_entry["standard"], found_entry["jargon"])
                    existing = managed.setdefault(target_scope, {}).get(target_key)
                    if isinstance(existing, dict):
                        found_entry = self._merge_jargon_entries(existing, found_entry)
                    managed[target_scope][target_key] = found_entry
                    changed_live = True
                    status_text = "active" if target_scope == "public" else "candidate"
                    final_row = self._build_jargon_row(
                        scope=target_scope,
                        status=status_text,
                        key=target_key,
                        entry=found_entry,
                    )
            elif found_rejected is not None:
                if target_status == "rejected":
                    final_row = found_rejected
                else:
                    restored_scope = "public" if target_status == "active" else target_candidate_scope
                    restored_entry = self._normalize_jargon_entry(found_rejected)
                    if restored_entry is None:
                        raise ValueError("invalid rejected jargon row")
                    restored_entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                    restored_key = self._compose_jargon_key(restored_entry["standard"], restored_entry["jargon"])
                    existing = managed.setdefault(restored_scope, {}).get(restored_key)
                    if isinstance(existing, dict):
                        restored_entry = self._merge_jargon_entries(existing, restored_entry)
                    managed[restored_scope][restored_key] = restored_entry
                    changed_live = True
                    if found_rejected_idx >= 0:
                        del rejected[found_rejected_idx]
                        changed_rejected = True
                    status_text = "active" if restored_scope == "public" else "candidate"
                    final_row = self._build_jargon_row(
                        scope=restored_scope,
                        status=status_text,
                        key=restored_key,
                        entry=restored_entry,
                    )

            if changed_live:
                self._save_jargon_payload_locked(payload_data, managed, extras)
            if changed_rejected:
                self._save_rejected_jargons_locked(rejected)

        if changed_live:
            await self._reload_jargon_runtime()
        return final_row

    async def _delete_jargon(self, jargon_id: str) -> bool:
        clean_id = str(jargon_id or "").strip()
        if not clean_id:
            return False
        changed_live = False
        changed_rejected = False
        deleted = False
        async with self._jargon_store_lock:
            payload_data, managed, extras = self._load_jargon_payload_locked()
            rejected = self._load_rejected_jargons_locked()

            for scope_name in JARGON_SCOPES:
                bucket = managed.get(scope_name, {})
                remove_key = ""
                for key in bucket.keys():
                    status_text = "active" if scope_name == "public" else "candidate"
                    if self._jargon_id(scope_name, key, status_text) == clean_id:
                        remove_key = key
                        break
                if remove_key:
                    bucket.pop(remove_key, None)
                    changed_live = True
                    deleted = True
                    break

            if not deleted:
                next_rejected = [row for row in rejected if str(row.get("id", "")) != clean_id]
                if len(next_rejected) != len(rejected):
                    rejected = next_rejected
                    changed_rejected = True
                    deleted = True

            if changed_live:
                self._save_jargon_payload_locked(payload_data, managed, extras)
            if changed_rejected:
                self._save_rejected_jargons_locked(rejected)

        if changed_live:
            await self._reload_jargon_runtime()
        return deleted

    async def _reload_jargon_runtime(self) -> None:
        async with self._jargon_store_lock:
            _, managed, _ = self._load_jargon_payload_locked()
            live_rows = self._collect_live_jargon_entries_locked(managed)
        mapping: dict[str, str] = {}
        for row in live_rows:
            term = str(row.get("jargon", "") or "").strip()
            if not term:
                continue
            meaning = str(row.get("meaning", "") or "").strip() or str(row.get("standard", "") or "").strip()
            mapping[term] = meaning
        await self._agent.jargon_mgr.reload(mapping)
        try:
            await self._agent.jargon_engine.reload_automaton()
        except Exception:
            # Keep admin operations available even when automaton reload fails.
            return

    def _load_jargon_payload_locked(self) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
        payload: dict[str, Any] = {}
        if self._jargon_store_path.exists():
            try:
                raw = json.loads(self._jargon_store_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    payload = raw
            except (json.JSONDecodeError, OSError):
                payload = {}

        raw_spaces = payload.get("spaces")
        if not isinstance(raw_spaces, dict):
            raw_spaces = {}
        managed: dict[str, dict[str, dict[str, Any]]] = {scope: {} for scope in JARGON_SCOPES}
        extras: dict[str, Any] = {}

        for scope_name, bucket in raw_spaces.items():
            if scope_name not in JARGON_SCOPES:
                extras[str(scope_name)] = bucket
                continue
            if not isinstance(bucket, dict):
                continue
            normalized_bucket: dict[str, dict[str, Any]] = {}
            for raw_key, raw_entry in bucket.items():
                if not isinstance(raw_entry, dict):
                    continue
                entry = self._normalize_jargon_entry(raw_entry)
                if entry is None:
                    continue
                key = str(raw_key or "").strip() or self._compose_jargon_key(entry["standard"], entry["jargon"])
                normalized_bucket[key] = entry
            managed[scope_name] = normalized_bucket

        return payload, managed, extras

    def _save_jargon_payload_locked(
        self,
        payload: dict[str, Any],
        managed: dict[str, dict[str, dict[str, Any]]],
        extras: dict[str, Any],
    ) -> None:
        next_payload = dict(payload) if isinstance(payload, dict) else {}
        spaces: dict[str, Any] = {}
        for scope_name in sorted(JARGON_SCOPES):
            bucket = managed.get(scope_name, {})
            safe_bucket: dict[str, Any] = {}
            for key, entry in bucket.items():
                normalized = self._normalize_jargon_entry(entry)
                if normalized is None:
                    continue
                safe_bucket[str(key)] = normalized
            spaces[scope_name] = safe_bucket
        for scope_name, value in extras.items():
            if scope_name in spaces:
                continue
            spaces[scope_name] = value
        next_payload["version"] = 1
        next_payload["spaces"] = spaces

        self._jargon_store_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._jargon_store_path.with_suffix(self._jargon_store_path.suffix + ".tmp")
        temp.write_text(json.dumps(next_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self._jargon_store_path)

    def _load_rejected_jargons_locked(self) -> list[dict[str, Any]]:
        if not self._jargon_rejected_store_path.exists():
            return []
        try:
            payload = json.loads(self._jargon_rejected_store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, dict):
            return []
        rows = payload.get("items", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            entry = self._normalize_jargon_entry(row)
            if entry is None:
                continue
            scope_name = self._normalize_jargon_scope(str(row.get("scope", "") or ""), allow_empty=False) or "group"
            key = str(row.get("key", "") or "").strip() or self._compose_jargon_key(entry["standard"], entry["jargon"])
            rejected_id = str(row.get("id", "") or "").strip()
            if not rejected_id:
                rejected_id = self._jargon_id(scope_name, key, "rejected")
            out.append(
                {
                    "id": rejected_id,
                    "scope": scope_name,
                    "status": "rejected",
                    "key": key,
                    "jargon": entry["jargon"],
                    "standard": entry["standard"],
                    "meaning": entry["meaning"],
                    "confidence": entry["confidence"],
                    "weight": entry["weight"],
                    "source_users": entry["source_users"],
                    "updated_at": str(row.get("updated_at", "") or entry["updated_at"]),
                    "rejected_at": str(row.get("rejected_at", "") or entry["updated_at"]),
                }
            )
        return out

    def _save_rejected_jargons_locked(self, rows: list[dict[str, Any]]) -> None:
        payload = {
            "version": 1,
            "items": rows,
        }
        self._jargon_rejected_store_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._jargon_rejected_store_path.with_suffix(self._jargon_rejected_store_path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self._jargon_rejected_store_path)

    def _collect_live_jargon_entries_locked(self, managed: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scope_name, bucket in managed.items():
            for entry in bucket.values():
                if not isinstance(entry, dict):
                    continue
                normalized = self._normalize_jargon_entry(entry)
                if normalized is None:
                    continue
                rows.append(normalized)
        return rows

    def _collect_jargon_rows_locked(
        self,
        managed: dict[str, dict[str, dict[str, Any]]],
        rejected: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scope_name, bucket in managed.items():
            for key, entry in bucket.items():
                status_text = "active" if scope_name == "public" else "candidate"
                rows.append(self._build_jargon_row(scope=scope_name, status=status_text, key=key, entry=entry))
        rows.extend(dict(item) for item in rejected)
        return rows

    def _build_jargon_row(self, *, scope: str, status: str, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_jargon_entry(entry) or {}
        jargon = str(normalized.get("jargon", "") or "").strip()
        standard = str(normalized.get("standard", "") or "").strip()
        meaning = str(normalized.get("meaning", "") or "").strip()
        confidence = float(normalized.get("confidence", 0.5) or 0.5)
        weight = float(normalized.get("weight", 1.0) or 1.0)
        source_users = list(normalized.get("source_users", []) if isinstance(normalized.get("source_users"), list) else [])
        updated_at = str(normalized.get("updated_at", "") or "")
        return {
            "id": self._jargon_id(scope, key, status),
            "scope": scope,
            "status": status,
            "key": key,
            "jargon": jargon,
            "standard": standard,
            "meaning": meaning,
            "confidence": confidence,
            "weight": weight,
            "source_users": source_users,
            "updated_at": updated_at,
        }

    def _build_rejected_jargon_row(self, *, scope: str, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        row = self._build_jargon_row(scope=scope, status="rejected", key=key, entry=entry)
        row["rejected_at"] = datetime.now(timezone.utc).isoformat()
        return row

    def _normalize_jargon_entry(self, raw_entry: Any) -> dict[str, Any] | None:
        if not isinstance(raw_entry, dict):
            return None
        jargon = str(raw_entry.get("jargon", "") or "").strip()
        standard = str(raw_entry.get("standard", "") or "").strip()
        if not jargon or not standard:
            return None
        meaning = str(raw_entry.get("meaning", "") or "").strip()
        try:
            confidence = float(raw_entry.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        try:
            weight = float(raw_entry.get("weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(0.05, min(100.0, weight))
        source_users: list[int] = []
        raw_users = raw_entry.get("source_users", [])
        if isinstance(raw_users, list):
            for item in raw_users:
                try:
                    uid = int(item)
                except (TypeError, ValueError):
                    continue
                if uid <= 0 or uid in source_users:
                    continue
                source_users.append(uid)
        updated_at = str(raw_entry.get("updated_at", "") or "").strip() or datetime.now(timezone.utc).isoformat()
        return {
            "jargon": jargon,
            "standard": standard,
            "meaning": meaning,
            "confidence": confidence,
            "weight": weight,
            "source_users": source_users,
            "updated_at": updated_at,
        }

    @staticmethod
    def _merge_jargon_entries(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        if str(incoming.get("meaning", "") or "").strip():
            if not str(merged.get("meaning", "") or "").strip():
                merged["meaning"] = str(incoming.get("meaning", "")).strip()
            else:
                merged["meaning"] = str(incoming.get("meaning", "")).strip()
        merged["confidence"] = max(
            0.0,
            min(
                1.0,
                (float(merged.get("confidence", 0.5) or 0.5) * 0.7)
                + (float(incoming.get("confidence", 0.5) or 0.5) * 0.3),
            ),
        )
        merged["weight"] = max(
            0.05,
            min(
                100.0,
                (float(merged.get("weight", 1.0) or 1.0) * 0.75)
                + (float(incoming.get("weight", 1.0) or 1.0) * 0.4),
            ),
        )
        merged["updated_at"] = datetime.now(timezone.utc).isoformat()
        users: list[int] = []
        for raw_list in (merged.get("source_users", []), incoming.get("source_users", [])):
            if not isinstance(raw_list, list):
                continue
            for item in raw_list:
                try:
                    uid = int(item)
                except (TypeError, ValueError):
                    continue
                if uid <= 0 or uid in users:
                    continue
                users.append(uid)
        merged["source_users"] = users
        merged["jargon"] = str(incoming.get("jargon", merged.get("jargon", "")) or "").strip()
        merged["standard"] = str(incoming.get("standard", merged.get("standard", "")) or "").strip()
        return merged

    @staticmethod
    def _compose_jargon_key(standard: str, jargon: str) -> str:
        return f"{str(standard or '').strip().lower()}\t{str(jargon or '').strip().lower()}"

    @staticmethod
    def _normalize_jargon_scope(raw_scope: str, *, allow_empty: bool) -> str:
        clean = str(raw_scope or "").strip().lower()
        if not clean:
            return "" if allow_empty else "group"
        if clean not in JARGON_SCOPES and clean != "rejected":
            return ""
        return clean

    @staticmethod
    def _normalize_jargon_status(raw_status: str, *, allow_empty: bool) -> str:
        clean = str(raw_status or "").strip().lower()
        if not clean:
            return "" if allow_empty else "candidate"
        if clean not in JARGON_STATUSES:
            return ""
        return clean

    @staticmethod
    def _jargon_search_text(item: dict[str, Any]) -> str:
        return (
            f"{item.get('jargon', '')}\n"
            f"{item.get('standard', '')}\n"
            f"{item.get('meaning', '')}\n"
            f"{item.get('scope', '')}\n"
            f"{item.get('status', '')}"
        ).lower()

    @staticmethod
    def _jargon_id(scope: str, key: str, status: str) -> str:
        digest = hashlib.sha1(f"{scope}\n{status}\n{key}".encode("utf-8", errors="ignore")).hexdigest()
        return digest[:24]

    async def _persist_config(
        self,
        mutator: Callable[[dict[str, Any], Any], None],
        value: Any,
    ) -> None:
        async with self._config_lock:
            config_data: dict[str, Any] = {}
            if self._config_path.exists():
                raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    config_data = raw

            mutator(config_data, value)

            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
            temp_file.write_text(
                yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            temp_file.replace(self._config_path)

    async def _save_persona_prompt_template(self, content: str) -> None:
        async with self._config_lock:
            self._persona_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self._persona_prompt_path.with_suffix(self._persona_prompt_path.suffix + ".tmp")
            temp_file.write_text(str(content or "").strip() + "\n", encoding="utf-8")
            temp_file.replace(self._persona_prompt_path)

    def _upsert_runtime_group(
        self,
        *,
        group_id: int,
        enabled: bool,
        group_name: str | None,
        remark: str | None,
        extra_prompt: str | None,
    ) -> bool:
        group = self._cfg.get_group(int(group_id))
        if group is None:
            self._cfg.groups.append(
                GroupConfig(
                    group_id=int(group_id),
                    enabled=bool(enabled),
                    group_name=str(group_name or ""),
                    remark=str(remark or ""),
                    extra_prompt=str(extra_prompt or ""),
                )
            )
            return True

        group.enabled = bool(enabled)
        if group_name is not None:
            group.group_name = str(group_name)
        if remark is not None:
            group.remark = str(remark)
        if extra_prompt is not None:
            group.extra_prompt = str(extra_prompt)
        return False

    @staticmethod
    def _set_background_url(config_data: dict[str, Any], value: str) -> None:
        web = config_data.setdefault("web", {})
        if not isinstance(web, dict):
            web = {}
            config_data["web"] = web

        ui_settings = web.setdefault("ui_settings", {})
        if not isinstance(ui_settings, dict):
            ui_settings = {}
            web["ui_settings"] = ui_settings
        ui_settings["background_url"] = value

    @staticmethod
    def _set_master_id(config_data: dict[str, Any], value: int) -> None:
        persona = config_data.setdefault("persona", {})
        if not isinstance(persona, dict):
            persona = {}
            config_data["persona"] = persona
        persona["master_id"] = int(value)

    @staticmethod
    def _set_sticker_settings(config_data: dict[str, Any], value: dict[str, Any]) -> None:
        sticker = config_data.setdefault("sticker", {})
        if not isinstance(sticker, dict):
            sticker = {}
            config_data["sticker"] = sticker

        for key in (
            "enabled",
            "collection_rate",
            "storage_mode",
            "filter_keywords",
            "user_weights",
            "allow_other_users_collection",
            "enable_persona_filter",
            "llm_filter_enabled",
            "llm_filter_probability",
            "llm_filter_mood_threshold",
        ):
            if key not in value:
                continue
            sticker[key] = value[key]

    @staticmethod
    def _set_sticker_user_weight(config_data: dict[str, Any], value: dict[str, Any]) -> None:
        sticker = config_data.setdefault("sticker", {})
        if not isinstance(sticker, dict):
            sticker = {}
            config_data["sticker"] = sticker

        weights = sticker.setdefault("user_weights", {})
        if not isinstance(weights, dict):
            weights = {}
            sticker["user_weights"] = weights

        user_id = str(value.get("user_id", "")).strip()
        if not user_id:
            return
        weights[user_id] = float(value.get("weight", 1.0))

    @staticmethod
    def _delete_sticker_user_weight(config_data: dict[str, Any], value: int) -> None:
        sticker = config_data.get("sticker")
        if not isinstance(sticker, dict):
            return
        weights = sticker.get("user_weights")
        if not isinstance(weights, dict):
            return
        weights.pop(str(int(value)), None)

    @staticmethod
    def _upsert_group_config(config_data: dict[str, Any], value: dict[str, Any]) -> None:
        group_id = int(value.get("group_id"))
        enabled = bool(value.get("enabled"))
        group_name_raw = value.get("group_name")
        group_name = str(group_name_raw) if group_name_raw is not None else None
        remark_raw = value.get("remark")
        remark = str(remark_raw) if remark_raw is not None else None
        extra_prompt_raw = value.get("extra_prompt")
        extra_prompt = str(extra_prompt_raw) if extra_prompt_raw is not None else None

        groups = config_data.setdefault("groups", [])
        if not isinstance(groups, list):
            groups = []
            config_data["groups"] = groups

        for row in groups:
            if not isinstance(row, dict):
                continue
            raw_group = row.get("group_id")
            try:
                if int(raw_group) != group_id:
                    continue
            except (TypeError, ValueError):
                continue
            row["enabled"] = enabled
            if group_name is not None:
                row["group_name"] = group_name
            if remark is not None:
                row["remark"] = remark
            if extra_prompt is not None:
                row["extra_prompt"] = extra_prompt
            return

        groups.append(
            {
                "group_id": group_id,
                "enabled": enabled,
                "group_name": group_name or "",
                "remark": remark or "",
                "extra_prompt": extra_prompt or "",
            },
        )
