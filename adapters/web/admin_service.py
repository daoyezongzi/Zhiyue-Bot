from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.agent import ZhiyueAgent
    from internal.config.schema import Config


class ResetActionRequest(BaseModel):
    fill_energy: bool = False
    reset_mood: bool = False
    clear_session_id: str = ""


class UpdateConfigRequest(BaseModel):
    system_prompt: str = Field(default="", min_length=1)


class AdminService:
    def __init__(
        self,
        *,
        cfg: "Config",
        agent: "ZhiyueAgent",
        config_path: str | Path,
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

        self._app = FastAPI(title="Zhiyue Admin Service", docs_url="/docs", redoc_url="/redoc")
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
                },
            )

        @self._app.get("/api/status")
        async def api_status(_: None = Depends(self._require_token),) -> dict[str, Any]:
            return await self._agent.get_admin_status()

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
            self._persist_config(self._set_system_prompt, clean)
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
            self._persist_config(self._set_background_url, clean_url)
            return {"ok": True, "background_url": clean_url}

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

    @staticmethod
    def _extract_token(request: Request) -> str:
        header = str(request.headers.get("x-access-token", "")).strip()
        if header:
            return header

        auth = str(request.headers.get("authorization", "")).strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    def _persist_config(self, mutator: Any, value: str) -> None:
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
