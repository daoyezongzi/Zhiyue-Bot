from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdminWebAdapter:
    port: int
    admin_key: str

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None
