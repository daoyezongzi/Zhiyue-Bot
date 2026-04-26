from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import uvicorn
import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
    reset_mood: bool = False
    clear_session_id: str = ""


class UpdateConfigRequest(BaseModel):
    system_prompt: str = Field(default="", min_length=1)


class UpdateMasterRequest(BaseModel):
    master_id: int = Field(...)


class GroupSwitchRequest(BaseModel):
    enabled: bool = Field(...)


class KnowledgeSaveRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str = Field(default="")


class AdminService:
    def __init__(
        self,
        *,
        cfg: "Config",
        agent: "ZhiyueAgent",
        config_path: str | Path,
        log_hub: LogStreamHub | None = None,
        plugin_manager: RuntimePluginManager | None = None,
    ) -> None:
        self._cfg = cfg
        self._agent = agent
        self._config_path = Path(config_path)

        self._host = str(cfg.web.host or "127.0.0.1").strip() or "127.0.0.1"
        self._port = int(cfg.web.port or 8080)
        self._access_token = cfg.web.resolved_access_token()

        self._project_root = Path(__file__).resolve().parents[2]
        self._template_dir = self._project_root / "web_ui"
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
        self._config_lock = asyncio.Lock()

        self._app = FastAPI(title="Zhiyue Unified Dashboard", docs_url="/docs", redoc_url="/redoc")
        self._templates = Jinja2Templates(directory=str(self._template_dir))
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
        @self._app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request) -> HTMLResponse:
            return self._templates.TemplateResponse(
                request=request,
                name="dashboard.html",
                context={
                    "background_url": self._cfg.web.ui_settings.background_url or "",
                    "host": self._host,
                    "port": self._port,
                    "token_ready": bool(self._access_token),
                    "knowledge_dir": str(self._knowledge_dir),
                },
            )

        @self._app.websocket("/ws/logs")
        async def ws_logs(websocket: WebSocket) -> None:
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

        @self._app.post("/api/groups/{group_id}")
        async def api_groups_update(
            group_id: int,
            payload: GroupSwitchRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            clean_group_id = int(group_id)
            enabled = bool(payload.enabled)
            group = self._cfg.get_group(clean_group_id)
            if group is None:
                group = GroupConfig(group_id=clean_group_id, enabled=enabled, extra_prompt="")
                self._cfg.groups.append(group)
            else:
                group.enabled = enabled
            await self._persist_config(
                self._set_group_enabled,
                {
                    "group_id": clean_group_id,
                    "enabled": enabled,
                },
            )
            return {"ok": True, "group_id": clean_group_id, "enabled": enabled}

        @self._app.post("/api/action/reset")
        async def api_reset(
            payload: ResetActionRequest,
            _: None = Depends(self._require_token),
        ) -> dict[str, Any]:
            return await self._agent.reset_runtime_state(
                fill_energy=payload.fill_energy,
                reset_mood=payload.reset_mood,
                session_id=payload.clear_session_id.strip() or None,
            )

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

    async def _require_token(self, request: Request) -> None:
        expected = self._access_token.strip()
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="admin access token is not configured",
            )

        provided = self._extract_token(request)
        if provided != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid access token")

    async def _authorize_websocket(self, websocket: WebSocket) -> bool:
        expected = self._access_token.strip()
        if not expected:
            await websocket.close(code=1013, reason="admin access token is not configured")
            return False
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
    def _set_group_enabled(config_data: dict[str, Any], value: dict[str, Any]) -> None:
        group_id = int(value.get("group_id"))
        enabled = bool(value.get("enabled"))

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
            return

        groups.append(
            {
                "group_id": group_id,
                "enabled": enabled,
                "extra_prompt": "",
            },
        )
