from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from adapters.llm.chat import ChatLLMAdapter
from adapters.onebot import OneBotClient
from plugins import RuntimePluginManager
from core.agent import ZhiyueAgent
from internal.config import Config, load_config
from internal.logger import get_logger, init_logger
from internal.management import BotLogCapture, LogStreamHub, ProcessSupervisor

INVALID_LLM_API_KEYS = {
    "",
    "sk-...",
    "your_api_key_here",
    "your-api-key-here",
    "<your_api_key>",
    "replace_with_your_api_key",
}
DETECTED_NAPCAT_BOOTMAIN_EXE = Path(
    "C:\\0_Storage\\qqbot\u76f8\u5173\\NapCat.Shell.Windows.OneKey\\bootmain\\NapCatWinBootMain.exe"
)

def _is_invalid_llm_api_key(api_key: str) -> bool:
    normalized = api_key.strip().lower()
    if not normalized:
        return True
    return normalized in INVALID_LLM_API_KEYS


class UTF8StaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict[str, Any]) -> Any:
        response = await super().get_response(path, scope)
        content_type = str(response.headers.get("content-type", ""))
        lowered = content_type.lower()
        if content_type and "charset=" not in lowered:
            if (
                lowered.startswith("text/")
                or lowered.startswith("application/javascript")
                or lowered.startswith("application/json")
                or lowered.startswith("application/xml")
            ):
                response.headers["content-type"] = f"{content_type}; charset=utf-8"
        return response


class BotApp:
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path: str = config_path
        self.cfg, self.config_file = self._load_config(config_path)

        init_logger(self.cfg.app.log_level, self.cfg.app.debug)
        self.logger = get_logger("BotApp")
        self._validate_llm_api_key()

        ws_url = self.cfg.onebot.ws_url or "ws://127.0.0.1:18001/ws"
        self.onebot_client = OneBotClient(
            ws_url=ws_url,
            ws_mode=self.cfg.onebot.ws_mode,
            access_token=self.cfg.onebot.access_token,
            reconnect_initial=float(max(1, self.cfg.onebot.reconnect_interval)),
        )
        self.llm = ChatLLMAdapter(self.cfg.llm, self.cfg.auxiliary_model)
        self.agent = ZhiyueAgent(
            self.onebot_client,
            self.cfg,
            self.llm,
            config_path=self.config_file,
            shutdown_handler=self.stop,
        )
        self.log_hub = LogStreamHub()
        self.log_capture = BotLogCapture(self.log_hub)
        self.process_supervisor = ProcessSupervisor(self.log_hub)
        self.plugin_manager = RuntimePluginManager(Path(__file__).resolve().parent / "plugins")
        self.admin_service: Any | None = None
        if self.cfg.web.enabled:
            from adapters.web import AdminService

            self.admin_service = AdminService(
                cfg=self.cfg,
                agent=self.agent,
                config_path=self.config_file,
                log_hub=self.log_hub,
                plugin_manager=self.plugin_manager,
                shutdown_handler=self.stop,
                restart_handler=self.restart,
            )
            web_ui_dir = Path(__file__).resolve().parent / "web_ui"
            dashboard_file = web_ui_dir / "dashboard.html"

            def _read_dashboard_html() -> str:
                return dashboard_file.read_text(encoding="utf-8", errors="replace")

            @self.admin_service.app.get("/", response_class=HTMLResponse, include_in_schema=False)
            async def dashboard_root() -> HTMLResponse:
                return HTMLResponse(
                    content=_read_dashboard_html(),
                    media_type="text/html; charset=utf-8",
                )

            @self.admin_service.app.get("/dashboard.html", response_class=HTMLResponse, include_in_schema=False)
            async def dashboard_html() -> HTMLResponse:
                return HTMLResponse(
                    content=_read_dashboard_html(),
                    media_type="text/html; charset=utf-8",
                )

            self.admin_service.app.mount(
                "/web_ui",
                UTF8StaticFiles(directory=str(web_ui_dir), html=True),
                name="web_ui",
            )

        self._stop_event: asyncio.Event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self.logger.info("Starting BotApp")
        self.log_capture.install(asyncio.get_running_loop())
        await self.plugin_manager.start()
        await self._start_managed_onebot()
        startup_tasks = [self.onebot_client.start(), self.agent.start()]
        if self.admin_service is not None:
            startup_tasks.append(self.admin_service.start())
        await asyncio.gather(*startup_tasks)

        processor = asyncio.create_task(
            self.process_messages(),
            name="process-messages",
        )
        self._tasks.append(processor)
        self.logger.info("BotApp started; OneBot endpoint=%s", self.onebot_client.ws_url)

    async def stop(self) -> None:
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        self.logger.info("Stopping BotApp")

        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        shutdown_tasks = [self.agent.stop(), self.onebot_client.stop()]
        if self.admin_service is not None:
            shutdown_tasks.append(self.admin_service.stop())
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        await asyncio.gather(
            self.process_supervisor.stop_all(),
            self.plugin_manager.stop(),
            return_exceptions=True,
        )
        self.log_capture.restore()
        self.logger.info("BotApp stopped")

    async def restart(self) -> None:
        if self._stop_event.is_set():
            return

        script_path = Path(__file__).resolve()
        command = [sys.executable, str(script_path)]
        creationflags = 0
        if os.name == "nt":
            if hasattr(subprocess, "DETACHED_PROCESS"):
                creationflags |= int(getattr(subprocess, "DETACHED_PROCESS"))
            if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP"))
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                creationflags |= int(getattr(subprocess, "CREATE_NO_WINDOW"))

        subprocess.Popen(
            command,
            cwd=str(script_path.parent),
            env=dict(os.environ),
            creationflags=creationflags,
        )
        await self.log_hub.publish(
            "system",
            f"[system] restart child spawned: {' '.join(command)}",
            channel="system",
        )
        await self.stop()

    async def wait_closed(self) -> None:
        await self._stop_event.wait()

    async def process_messages(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = await self.onebot_client.message_queue.get()
            except asyncio.CancelledError:
                return

            try:
                await self.agent.handle_message(packet)
            except Exception:
                self.logger.exception("Message processing failed")
            finally:
                self.onebot_client.message_queue.task_done()

    @staticmethod
    def _load_config(config_path: str) -> tuple[Config, Path]:
        file_path = Path(config_path)
        if file_path.exists():
            return load_config(file_path), file_path

        fallback = Path("config/config.yaml.example")
        if fallback.exists():
            return load_config(fallback), fallback

        raise FileNotFoundError(
            "Cannot find config/config.yaml or config/config.yaml.example",
        )

    def _validate_llm_api_key(self) -> None:
        api_key = self.cfg.llm.api_key or ""
        if _is_invalid_llm_api_key(api_key):
            self.logger.error(
                "Startup aborted: invalid llm.api_key. Configure LLM_API_KEY "
                "(or set LLM_PROVIDER + <PROVIDER>_API_KEY, e.g. DEEPSEEK_API_KEY), "
                "or set llm.api_key in config/config.yaml. Placeholder values "
                "like 'sk-...' are not allowed.",
            )
            raise SystemExit(1)

    async def _start_managed_onebot(self) -> None:
        skip_managed = str(os.getenv("SKIP_MANAGED_NAPCAT", "")).strip().lower()
        if skip_managed in {"1", "true", "yes", "on"}:
            self.logger.info("Skip managed NapCat startup because SKIP_MANAGED_NAPCAT is enabled")
            await self.log_hub.publish(
                "system",
                "[napcat] managed startup skipped (SKIP_MANAGED_NAPCAT enabled)",
                channel="system",
            )
            return

        executable = self._resolve_napcat_executable()
        if not executable:
            detected = self._resolve_detected_napcat_fallback("")
            if detected:
                executable = detected
                await self.log_hub.publish(
                    "system",
                    f"[napcat] primary path is empty, use detected fallback path: {detected}",
                    channel="system",
                )
            else:
                hint = (
                    "[napcat] managed startup disabled: NAPCAT_PATH is empty. "
                    "Set NAPCAT_PATH in .env (full path to NapCatWinBootMain.exe)."
                )
                self.logger.warning(hint)
                await self.log_hub.publish("system", hint, channel="system")
                return

        raw_args = [str(item) for item in list(getattr(self.cfg.paths, "napcat_args", []) or [])]
        args = self._normalize_napcat_args(raw_args)
        if args != raw_args:
            self.logger.info("Normalize NapCat args from %s to %s", raw_args, args)
            await self.log_hub.publish(
                "system",
                f"[napcat] normalize args from {raw_args} to {args}",
                channel="system",
            )

        bot_qq = self._read_env_bot_qq()
        if bot_qq and not self._has_napcat_login_arg(args):
            args = [bot_qq, *args]
            self.logger.info("Inject NapCat quickLogin parameter from BOT_QQ in .env")
            await self.log_hub.publish(
                "system",
                f"[napcat] inject quickLogin parameter from .env BOT_QQ={bot_qq}",
                channel="system",
            )
        elif not bot_qq and not self._has_napcat_login_arg(args):
            warn = "BOT_QQ is not set in .env and NapCat args has no quickLogin parameter"
            self.logger.warning(warn)
            await self.log_hub.publish("system", f"[napcat] {warn}", channel="system")

        executable_path = Path(os.path.expandvars(executable)).expanduser()
        if executable_path.suffix.lower() in {".bat", ".cmd"}:
            warn = (
                "[napcat] NAPCAT_PATH points to .bat/.cmd. This launcher may not expose "
                "runtime stdout/stderr to the web console. Prefer NapCatWinBootMain.exe."
            )
            self.logger.warning(warn)
            await self.log_hub.publish("system", warn, channel="system")

        await self.log_hub.publish(
            "system",
            f"[napcat] starting managed process: executable={executable} args={args}",
            channel="system",
        )
        started = await self._try_start_napcat_process(
            executable=executable,
            args=args,
            label="primary",
        )
        if started:
            return

        fallback_executable = self._resolve_detected_napcat_fallback(executable)
        if not fallback_executable:
            return

        await self.log_hub.publish(
            "system",
            (
                "[napcat] primary startup failed, retry with detected fallback path: "
                f"{fallback_executable}"
            ),
            channel="system",
        )
        await self._try_start_napcat_process(
            executable=fallback_executable,
            args=args,
            label="fallback",
        )

    def _resolve_napcat_executable(self) -> str:
        configured = str(getattr(self.cfg.paths, "napcat_path", "")).strip()
        if configured:
            resolved_configured = self._resolve_candidate_napcat_executable(configured)
            if resolved_configured:
                if resolved_configured != configured:
                    self.logger.info(
                        "Resolve NAPCAT_PATH from %s to %s",
                        configured,
                        resolved_configured,
                    )
                return resolved_configured
            self.logger.warning("Configured NAPCAT_PATH is unavailable: %s", configured)
            return ""

        local_default = Path(__file__).resolve().parent / "NapCatWinBootMain.exe"
        local_resolved = self._resolve_candidate_napcat_executable(local_default)
        if local_resolved:
            self.logger.info("Use local NapCat executable: %s", local_resolved)
            return local_resolved
        return ""

    @staticmethod
    def _resolve_detected_napcat_fallback(current_executable: str) -> str:
        fallback = BotApp._resolve_candidate_napcat_executable(DETECTED_NAPCAT_BOOTMAIN_EXE)
        if not fallback:
            return ""

        if not current_executable:
            return fallback

        try:
            current = Path(os.path.expandvars(current_executable)).expanduser()
            if not current.is_absolute():
                current = Path.cwd() / current
            if current.resolve() == Path(fallback).resolve():
                return ""
        except Exception:
            pass
        return fallback

    @staticmethod
    def _resolve_candidate_napcat_executable(raw_executable: str | Path) -> str:
        raw = str(raw_executable or "").strip()
        if not raw:
            return ""

        candidate = Path(os.path.expandvars(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            candidate = candidate.resolve()
        except Exception:
            pass

        if candidate.suffix.lower() in {".bat", ".cmd"}:
            candidate = candidate.parent / "NapCatWinBootMain.exe"

        selected = BotApp._select_napcat_bootmain(candidate)
        if selected is None:
            return ""
        return str(selected)

    @staticmethod
    def _select_napcat_bootmain(candidate: Path) -> Path | None:
        if not candidate.exists():
            return None

        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate

        if resolved.name.lower() != "napcatwinbootmain.exe":
            return resolved

        if (resolved.parent / "QQ.exe").exists():
            return resolved

        for root in (resolved.parent, resolved.parent.parent):
            discovered = BotApp._find_bootmain_with_qq(root)
            if discovered is not None:
                return discovered

        return resolved

    @staticmethod
    def _find_bootmain_with_qq(root: Path) -> Path | None:
        try:
            base = root.resolve()
        except Exception:
            base = root
        if not base.exists() or not base.is_dir():
            return None

        direct = base / "NapCatWinBootMain.exe"
        if direct.exists() and (direct.parent / "QQ.exe").exists():
            try:
                return direct.resolve()
            except Exception:
                return direct

        for pattern in ("*/NapCatWinBootMain.exe", "*/*/NapCatWinBootMain.exe"):
            for item in base.glob(pattern):
                if not item.exists():
                    continue
                if not (item.parent / "QQ.exe").exists():
                    continue
                try:
                    return item.resolve()
                except Exception:
                    return item
        return None

    async def _try_start_napcat_process(
        self,
        *,
        executable: str,
        args: list[str],
        label: str,
    ) -> bool:
        try:
            await self.process_supervisor.start_napcat(
                executable=executable,
                args=args,
            )
            self.logger.info("Managed NapCat started (%s): %s", label, executable)
            await self.log_hub.publish(
                "system",
                f"[napcat] managed process started ({label}): {executable}",
                channel="system",
            )
            return True
        except Exception as exc:
            self.logger.exception("Failed to start managed NapCat process (%s): %s", label, executable)
            await self.log_hub.publish(
                "system",
                f"[napcat] startup failed ({label}): {exc}",
                channel="system",
            )
            return False

    @staticmethod
    def _read_env_bot_qq() -> str:
        raw = str(os.getenv("BOT_QQ", "")).strip()
        if raw.isdigit():
            return raw
        return ""

    @staticmethod
    def _has_napcat_login_arg(args: list[str]) -> bool:
        for index, raw_arg in enumerate(args):
            arg = str(raw_arg).strip()
            if not arg:
                continue
            lower_arg = arg.lower()
            if lower_arg == "-q" and index + 1 < len(args) and str(args[index + 1]).strip():
                return True
            if lower_arg.startswith("-q="):
                return True
            if not arg.startswith("-"):
                return True
        return False

    @staticmethod
    def _normalize_napcat_args(args: list[str]) -> list[str]:
        normalized: list[str] = []
        extracted_login = ""
        index = 0
        total = len(args)

        while index < total:
            raw_arg = str(args[index]).strip()
            index += 1
            if not raw_arg:
                continue

            lower_arg = raw_arg.lower()
            if lower_arg == "-q":
                if index < total:
                    candidate = str(args[index]).strip()
                    index += 1
                    if candidate and not extracted_login:
                        extracted_login = candidate
                continue
            if lower_arg.startswith("-q="):
                candidate = raw_arg[3:].strip()
                if candidate and not extracted_login:
                    extracted_login = candidate
                continue

            normalized.append(raw_arg)

        if extracted_login and not any(str(item).strip() == extracted_login for item in normalized):
            normalized.insert(0, extracted_login)
        return normalized


async def _register_signals(app: BotApp) -> None:
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        asyncio.create_task(app.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _request_stop())


async def main() -> None:
    app = BotApp()
    await _register_signals(app)
    try:
        await app.start()
        await app.wait_closed()
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
