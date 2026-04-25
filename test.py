from __future__ import annotations

import asyncio

from core.runtime import build_runtime


async def smoke_test() -> None:
    runtime = await build_runtime("config/config.yaml")
    await runtime.start()
    try:
        if runtime.cfg.groups:
            gid = runtime.cfg.groups[0].group_id
            await runtime.agent.emit_debug_message(gid, "@Zhiyue 今天天气怎么样")
            await asyncio.sleep(0.1)
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(smoke_test())
