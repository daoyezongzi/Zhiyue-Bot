from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.llm.chat import ChatLLMAdapter
from adapters.onebot import OneBotClient
from core.agent import ZhiyueAgent
from internal.config import Config, load_config
from internal.logger import get_logger, init_logger


@dataclass
class Runtime:
    cfg: Config
    onebot_client: OneBotClient
    agent: ZhiyueAgent
    admin_service: Any | None = None
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _processor: asyncio.Task[None] | None = None

    async def start(self) -> None:
        tasks = [self.onebot_client.start(), self.agent.start()]
        if self.admin_service is not None:
            tasks.append(self.admin_service.start())
        await asyncio.gather(*tasks)
        self._processor = asyncio.create_task(self._process_messages(), name="runtime-process-messages")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._processor is not None:
            self._processor.cancel()
            try:
                await self._processor
            except asyncio.CancelledError:
                pass
            self._processor = None

        tasks = [self.agent.stop(), self.onebot_client.stop()]
        if self.admin_service is not None:
            tasks.append(self.admin_service.stop())
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_messages(self) -> None:
        logger = get_logger("Runtime")
        while not self._stop_event.is_set():
            try:
                packet = await self.onebot_client.message_queue.get()
            except asyncio.CancelledError:
                return

            try:
                await self.agent.handle_message(packet)
            except Exception:
                logger.exception("Runtime message dispatch failed")
            finally:
                self.onebot_client.message_queue.task_done()


async def build_runtime(config_path: str | Path = "config/config.yaml") -> Runtime:
    config_file = Path(config_path)
    cfg = load_config(config_file)
    init_logger(cfg.app.log_level, cfg.app.debug)
    logger = get_logger("Runtime")
    logger.info("Config loaded: %s", config_file)

    onebot_client = OneBotClient(
        ws_url=cfg.onebot.ws_url or "ws://127.0.0.1:6199",
        ws_mode=cfg.onebot.ws_mode,
        access_token=cfg.onebot.access_token,
        reconnect_initial=float(max(1, cfg.onebot.reconnect_interval)),
    )
    llm = ChatLLMAdapter(cfg.llm, cfg.auxiliary_model)
    agent = ZhiyueAgent(onebot_client, cfg, llm)
    admin_service = None
    if cfg.web.enabled:
        from adapters.web import AdminService

        admin_service = AdminService(cfg=cfg, agent=agent, config_path=config_file)
    return Runtime(cfg=cfg, onebot_client=onebot_client, agent=agent, admin_service=admin_service)
