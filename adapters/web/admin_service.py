from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import secrets
import subprocess
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
    extra_prompt: str | None = Field(default=None)


class KnowledgeSaveRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str = Field(default="")


class StickerSettingsRequest(BaseModel):
    enabled: bool | None = None
    collection_rate: float | None = None
    storage_mode: str | None = None
    filter_keywords: list[str] | None = None
    user_weights: dict[str, float] | None = None
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

        self._log_hub = log_hub or LogStreamHub()
        self._plugin_manager = plugin_manager or RuntimePluginManager(self._project_root / "plugins")
        self._shutdown_handler = shutdown_handler
        self._restart_handler = restart_handler
        self._shutdown_requested = False
        self._shutdown_lock = asyncio.Lock()
        self._restart_requested = False
        self._restart_lock = asyncio.Lock()
        self._config_lock = asyncio.Lock()

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

        @self._app.get("/api/status")
        async def api_status(_: None = Depends(self._require_token),) -> dict[str, Any]:
            status_data = await self._agent.get_admin_status()
            status_data["runtime_config"] = self._runtime_config_payload()
            return status_data

        @self._app.get("/api/config/runtime")
        async def api_runtime_config(_: None = Depends(self._require_token),) -> dict[str, Any]:
            return self._runtime_config_payload()

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
            return {"groups": self._serialize_groups()}

        @self._app.post("/api/groups")
        async def api_groups_upsert(
            payload: GroupUpsertRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_group_id = int(payload.group_id)
            if clean_group_id <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="group_id must be > 0")

            enabled = bool(payload.enabled)
            extra_prompt = payload.extra_prompt.strip() if isinstance(payload.extra_prompt, str) else None
            created = self._upsert_runtime_group(
                group_id=clean_group_id,
                enabled=enabled,
                extra_prompt=extra_prompt,
            )
            await self._persist_config(
                self._upsert_group_config,
                {
                    "group_id": clean_group_id,
                    "enabled": enabled,
                    "extra_prompt": extra_prompt,
                },
            )
            return {
                "ok": True,
                "group_id": clean_group_id,
                "enabled": enabled,
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
                extra_prompt=None,
            )
            await self._persist_config(
                self._upsert_group_config,
                {
                    "group_id": clean_group_id,
                    "enabled": enabled,
                    "extra_prompt": None,
                },
            )
            return {"ok": True, "group_id": clean_group_id, "enabled": enabled, "created": created}

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
            await self._persist_config(self._set_system_prompt, clean)
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
            "groups": self._serialize_groups(),
            "sticker": {
                "enabled": bool(self._cfg.sticker.enabled),
                "collection_rate": float(self._cfg.sticker.collection_rate),
                "storage_mode": self._normalize_storage_mode(self._cfg.sticker.storage_mode),
                "local_dir": str(self._agent.sticker_collector.local_dir),
                "filter_keywords": self._normalize_filter_keywords(self._cfg.sticker.filter_keywords),
                "user_weights": self._normalize_user_weights(self._cfg.sticker.user_weights),
                "enable_persona_filter": bool(self._cfg.sticker.enable_persona_filter),
                "llm_filter_enabled": bool(self._cfg.sticker.llm_filter_enabled),
                "llm_filter_probability": self._clamp_probability(self._cfg.sticker.llm_filter_probability),
                "llm_filter_mood_threshold": float(self._cfg.sticker.llm_filter_mood_threshold),
            },
        }

    def _serialize_groups(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._cfg.groups:
            out.append(
                {
                    "group_id": int(item.group_id),
                    "enabled": bool(item.enabled),
                    "extra_prompt": str(item.extra_prompt or ""),
                },
            )
        out.sort(key=lambda row: int(row["group_id"]))
        return out

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

    def _upsert_runtime_group(
        self,
        *,
        group_id: int,
        enabled: bool,
        extra_prompt: str | None,
    ) -> bool:
        group = self._cfg.get_group(int(group_id))
        if group is None:
            self._cfg.groups.append(
                GroupConfig(
                    group_id=int(group_id),
                    enabled=bool(enabled),
                    extra_prompt=str(extra_prompt or ""),
                )
            )
            return True

        group.enabled = bool(enabled)
        if extra_prompt is not None:
            group.extra_prompt = str(extra_prompt)
        return False

    @staticmethod
    def _set_system_prompt(config_data: dict[str, Any], value: str) -> None:
        persona = config_data.setdefault("persona", {})
        if not isinstance(persona, dict):
            persona = {}
            config_data["persona"] = persona
        persona["system_prompt"] = value

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
            if extra_prompt is not None:
                row["extra_prompt"] = extra_prompt
            return

        groups.append(
            {
                "group_id": group_id,
                "enabled": enabled,
                "extra_prompt": extra_prompt or "",
            },
        )
