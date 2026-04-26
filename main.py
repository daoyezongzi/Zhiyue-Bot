from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

from adapters.llm.chat import ChatLLMAdapter
from adapters.onebot import OneBotClient
from core.agent import ZhiyueAgent
from internal.config import Config, load_config
from internal.logger import get_logger, init_logger

INVALID_LLM_API_KEYS = {
    "",
    "sk-...",
    "your_api_key_here",
    "your-api-key-here",
    "<your_api_key>",
    "replace_with_your_api_key",
}


def _is_invalid_llm_api_key(api_key: str) -> bool:
    normalized = api_key.strip().lower()
    if not normalized:
        return True
    return normalized in INVALID_LLM_API_KEYS


class BotApp:
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path: str = config_path
        self.cfg, self.config_file = self._load_config(config_path)

        init_logger(self.cfg.app.log_level, self.cfg.app.debug)
        self.logger = get_logger("BotApp")
        self._validate_llm_api_key()

        ws_url = self.cfg.onebot.ws_url or "ws://127.0.0.1:6199"
        self.onebot_client = OneBotClient(
            ws_url=ws_url,
            ws_mode=self.cfg.onebot.ws_mode,
            access_token=self.cfg.onebot.access_token,
            reconnect_initial=float(max(1, self.cfg.onebot.reconnect_interval)),
        )
        self.llm = ChatLLMAdapter(self.cfg.llm, self.cfg.auxiliary_model)
        self.agent = ZhiyueAgent(self.onebot_client, self.cfg, self.llm)
        self.admin_service: Any | None = None
        if self.cfg.web.enabled:
            from adapters.web import AdminService

            self.admin_service = AdminService(
                cfg=self.cfg,
                agent=self.agent,
                config_path=self.config_file,
            )

        self._stop_event: asyncio.Event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self.logger.info("Starting BotApp")
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
        self.logger.info("BotApp stopped")

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
    await app.start()
    await app.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
