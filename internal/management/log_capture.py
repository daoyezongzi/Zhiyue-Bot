from __future__ import annotations

import asyncio
import io
import logging
import sys
import threading
from typing import Any

from internal.management.log_stream import LogStreamHub


class _LineMirroringStream(io.TextIOBase):
    def __init__(
        self,
        *,
        original: Any,
        source: str,
        hub: LogStreamHub,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._original = original
        self._source = source
        self._hub = hub
        self._loop = loop
        self._lock = threading.Lock()
        self._buffer = ""

    @property
    def encoding(self) -> str:
        return str(getattr(self._original, "encoding", "utf-8"))

    @property
    def errors(self) -> str:
        return str(getattr(self._original, "errors", "replace"))

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        text = str(s)
        written = self._original.write(text)

        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit_line(line.rstrip("\r"))

        return int(written) if isinstance(written, int) else len(text)

    def flush(self) -> None:
        if hasattr(self._original, "flush"):
            self._original.flush()

    def fileno(self) -> int:
        if hasattr(self._original, "fileno"):
            return int(self._original.fileno())
        raise OSError("stream has no fileno")

    def isatty(self) -> bool:
        if hasattr(self._original, "isatty"):
            return bool(self._original.isatty())
        return False

    def flush_remainder(self) -> None:
        with self._lock:
            remaining = self._buffer.rstrip("\r\n")
            self._buffer = ""
        if remaining:
            self._emit_line(remaining)

    def _emit_line(self, message: str) -> None:
        clean = message.strip()
        if not clean:
            return
        if self._loop.is_closed():
            return

        def _publisher() -> None:
            asyncio.create_task(self._hub.publish(self._source, clean))

        try:
            self._loop.call_soon_threadsafe(_publisher)
        except RuntimeError:
            return


class _HubLoggingHandler(logging.Handler):
    def __init__(
        self,
        *,
        hub: LogStreamHub,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(level=logging.NOTSET)
        self._hub = hub
        self._loop = loop
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if self._loop.is_closed():
            return
        try:
            message = self.format(record)
        except Exception:
            message = str(record.getMessage())
        clean = str(message).strip()
        if not clean:
            return

        source = str(getattr(record, "name", "") or "system")
        channel = self._channel_for_logger_name(source)

        def _publisher() -> None:
            asyncio.create_task(self._hub.publish(source, clean, channel=channel))

        try:
            self._loop.call_soon_threadsafe(_publisher)
        except RuntimeError:
            return

    @staticmethod
    def _channel_for_logger_name(logger_name: str) -> str | None:
        name = str(logger_name or "").strip().lower()
        if not name:
            return None
        if "agent" in name or "onebot" in name or "plugin" in name:
            return "action"
        if "napcat" in name:
            return "napcat"
        return None


class BotLogCapture:
    def __init__(self, hub: LogStreamHub) -> None:
        self._hub = hub
        self._installed = False
        self._old_stdout: Any | None = None
        self._old_stderr: Any | None = None
        self._stdout_proxy: _LineMirroringStream | None = None
        self._stderr_proxy: _LineMirroringStream | None = None
        self._logging_handler: _HubLoggingHandler | None = None

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._installed:
            return

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._stdout_proxy = _LineMirroringStream(
            original=self._old_stdout,
            source="system",
            hub=self._hub,
            loop=loop,
        )
        self._stderr_proxy = _LineMirroringStream(
            original=self._old_stderr,
            source="system",
            hub=self._hub,
            loop=loop,
        )
        sys.stdout = self._stdout_proxy
        sys.stderr = self._stderr_proxy

        root_logger = logging.getLogger()
        self._logging_handler = _HubLoggingHandler(hub=self._hub, loop=loop)
        root_logger.addHandler(self._logging_handler)
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return

        if self._stdout_proxy is not None:
            self._stdout_proxy.flush_remainder()
        if self._stderr_proxy is not None:
            self._stderr_proxy.flush_remainder()

        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr

        if self._logging_handler is not None:
            try:
                logging.getLogger().removeHandler(self._logging_handler)
            except Exception:
                pass

        self._stdout_proxy = None
        self._stderr_proxy = None
        self._old_stdout = None
        self._old_stderr = None
        self._logging_handler = None
        self._installed = False
