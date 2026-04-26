from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from internal.logger import get_logger
from internal.management.log_stream import LogStreamHub


@dataclass(slots=True)
class ManagedProcess:
    name: str
    source: str
    command: list[str]
    process: asyncio.subprocess.Process
    stdout_task: asyncio.Task[None] | None
    stderr_task: asyncio.Task[None] | None
    wait_task: asyncio.Task[None] | None


class ProcessSupervisor:
    def __init__(self, log_hub: LogStreamHub) -> None:
        self._logger = get_logger("ProcessSupervisor")
        self._log_hub = log_hub
        self._lock = asyncio.Lock()
        self._processes: dict[str, ManagedProcess] = {}

    async def start_napcat(
        self,
        *,
        executable: str,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
    ) -> bool:
        raw_executable = os.path.expandvars(str(executable or "").strip())
        if not raw_executable:
            self._logger.info("NapCat executable is empty; skip subprocess startup")
            return False

        napcat_path = Path(raw_executable).expanduser()
        if not napcat_path.is_absolute():
            napcat_path = Path.cwd() / napcat_path
        napcat_path = napcat_path.resolve()
        if not napcat_path.exists():
            raise FileNotFoundError(f"NapCat executable not found: {napcat_path}")

        command = self._build_napcat_command(napcat_path, args or [])
        workdir = Path(cwd).resolve() if cwd else napcat_path.parent
        await self.start_process(
            name="napcat",
            source="onebot",
            command=command,
            cwd=workdir,
            hide_window=True,
        )
        return True

    async def start_process(
        self,
        *,
        name: str,
        source: str,
        command: list[str],
        cwd: str | Path | None = None,
        hide_window: bool = False,
    ) -> None:
        clean_name = str(name).strip().lower()
        if not clean_name:
            raise ValueError("process name is empty")
        if not command:
            raise ValueError("process command is empty")

        await self.stop(clean_name)

        creationflags = 0
        if hide_window and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW"))

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
        )

        stdout_task = asyncio.create_task(
            self._pump_stream(clean_name, source, process.stdout, stream_kind="stdout"),
            name=f"{clean_name}-stdout",
        )
        stderr_task = asyncio.create_task(
            self._pump_stream(clean_name, source, process.stderr, stream_kind="stderr"),
            name=f"{clean_name}-stderr",
        )
        entry = ManagedProcess(
            name=clean_name,
            source=source,
            command=list(command),
            process=process,
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            wait_task=None,
        )
        wait_task = asyncio.create_task(
            self._wait_for_exit(entry),
            name=f"{clean_name}-wait",
        )
        entry.wait_task = wait_task

        async with self._lock:
            self._processes[clean_name] = entry

        self._logger.info("Managed process started: %s pid=%s cmd=%s", clean_name, process.pid, command)
        await self._log_hub.publish(
            source,
            f"[{clean_name}] started pid={process.pid}",
        )

    async def stop(self, name: str) -> None:
        clean_name = str(name).strip().lower()
        if not clean_name:
            return
        async with self._lock:
            entry = self._processes.pop(clean_name, None)
        if entry is None:
            return
        await self._terminate_entry(entry, reason="manual stop")

    async def stop_all(self) -> None:
        async with self._lock:
            entries = list(self._processes.values())
            self._processes.clear()
        if not entries:
            return
        await asyncio.gather(
            *(self._terminate_entry(entry, reason="shutdown") for entry in entries),
            return_exceptions=True,
        )

    async def _wait_for_exit(self, entry: ManagedProcess) -> None:
        return_code = await entry.process.wait()
        await self._log_hub.publish(
            entry.source,
            f"[{entry.name}] exited with code {return_code}",
        )
        self._logger.info("Managed process exited: %s code=%s", entry.name, return_code)
        async with self._lock:
            current = self._processes.get(entry.name)
            if current is entry:
                self._processes.pop(entry.name, None)

    async def _terminate_entry(self, entry: ManagedProcess, *, reason: str) -> None:
        process = entry.process
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._logger.warning("Managed process kill timeout: %s", entry.name)

        for task in (entry.stdout_task, entry.stderr_task, entry.wait_task):
            if task is None:
                continue
            task.cancel()
        await asyncio.gather(
            *(task for task in (entry.stdout_task, entry.stderr_task, entry.wait_task) if task is not None),
            return_exceptions=True,
        )

        await self._log_hub.publish(
            entry.source,
            f"[{entry.name}] stopped ({reason})",
        )
        self._logger.info(
            "Managed process stopped: %s reason=%s code=%s",
            entry.name,
            reason,
            process.returncode,
        )

    async def _pump_stream(
        self,
        name: str,
        source: str,
        stream: asyncio.StreamReader | None,
        *,
        stream_kind: str,
    ) -> None:
        if stream is None:
            return
        while True:
            try:
                raw = await stream.readline()
            except asyncio.CancelledError:
                return
            if not raw:
                return
            message = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not message:
                continue
            await self._log_hub.publish(source, message)
            self._logger.info("ProcessLog[%s:%s] %s", name, stream_kind, message)

    @staticmethod
    def _build_napcat_command(executable: Path, args: list[str]) -> list[str]:
        suffix = executable.suffix.lower()
        if suffix in {".bat", ".cmd"}:
            command = ["cmd", "/c", str(executable)]
        else:
            command = [str(executable)]
        command.extend(str(item) for item in args)
        return command
