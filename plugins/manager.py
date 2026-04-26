from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from internal.logger import get_logger
from plugins.registry import PluginRegistry


IGNORED_PLUGIN_MODULES = {
    "__init__",
    "base",
    "context",
    "hooks",
    "registry",
    "manager",
}


@dataclass(slots=True)
class PluginModuleState:
    module: str
    file_path: str
    enabled: bool = True
    loaded: bool = False
    plugin_names: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "file_path": self.file_path,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "plugin_count": len(self.plugin_names),
            "plugin_names": list(self.plugin_names),
            "error": self.error,
        }


class RuntimePluginManager:
    def __init__(self, plugin_dir: str | Path) -> None:
        self._logger = get_logger("RuntimePluginManager")
        self._plugin_dir = Path(plugin_dir).resolve()
        self._registry = PluginRegistry()
        self._states: dict[str, PluginModuleState] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self.refresh_modules()
        states = await self.list_states()
        for state in states:
            if state["enabled"]:
                await self.load_module(str(state["module"]))

    async def stop(self) -> None:
        async with self._lock:
            self._registry.clear()
            for state in self._states.values():
                state.loaded = False
                state.plugin_names = []
                state.error = ""
            for module_name in list(sys.modules.keys()):
                if module_name.startswith("plugins."):
                    short_name = module_name.split(".", 1)[1]
                    if short_name in self._states:
                        sys.modules.pop(module_name, None)

    async def refresh_modules(self) -> list[PluginModuleState]:
        discovered = self._discover_module_files()
        async with self._lock:
            previous = dict(self._states)
            removed_modules = set(previous.keys()) - set(discovered.keys())
            for module_name in removed_modules:
                self._registry.unregister_by_owner(module_name)
                sys.modules.pop(f"plugins.{module_name}", None)
            self._states = {}
            for module_name, file_path in discovered.items():
                if module_name in previous:
                    state = previous[module_name]
                    state.file_path = str(file_path)
                else:
                    state = PluginModuleState(
                        module=module_name,
                        file_path=str(file_path),
                        enabled=True,
                    )
                self._states[module_name] = state
            return list(self._states.values())

    async def list_states(self) -> list[dict[str, Any]]:
        await self.refresh_modules()
        async with self._lock:
            return [self._states[name].as_dict() for name in sorted(self._states.keys())]

    async def load_module(self, module_name: str) -> dict[str, Any]:
        clean_name = self._normalize_module_name(module_name)
        await self.refresh_modules()
        async with self._lock:
            state = self._states.get(clean_name)
            if state is None:
                raise ValueError(f"plugin module not found: {clean_name}")
            self._registry.unregister_by_owner(clean_name)
            state.loaded = False
            state.plugin_names = []
            state.error = ""
            state.enabled = True

        try:
            importlib.invalidate_caches()
            import_target = f"plugins.{clean_name}"
            if import_target in sys.modules:
                module = importlib.reload(sys.modules[import_target])
            else:
                module = importlib.import_module(import_target)

            builder = getattr(module, "build_plugins", None)
            if not callable(builder):
                raise RuntimeError("build_plugins() is missing")
            plugins = list(builder())
            plugin_names = [str(item.name) for item in plugins]

            async with self._lock:
                for item in plugins:
                    self._registry.register(item, owner=clean_name)
                state = self._states[clean_name]
                state.loaded = True
                state.plugin_names = plugin_names
                state.error = ""
                result = state.as_dict()

            self._logger.info("Plugin module loaded: %s plugins=%s", clean_name, plugin_names)
            return result
        except Exception as exc:
            async with self._lock:
                state = self._states[clean_name]
                state.loaded = False
                state.plugin_names = []
                state.error = str(exc)
                result = state.as_dict()
            self._logger.exception("Plugin load failed: %s", clean_name)
            return result

    async def unload_module(self, module_name: str) -> dict[str, Any]:
        clean_name = self._normalize_module_name(module_name)
        await self.refresh_modules()
        import_target = f"plugins.{clean_name}"
        async with self._lock:
            state = self._states.get(clean_name)
            if state is None:
                raise ValueError(f"plugin module not found: {clean_name}")
            self._registry.unregister_by_owner(clean_name)
            sys.modules.pop(import_target, None)
            state.enabled = False
            state.loaded = False
            state.plugin_names = []
            state.error = ""
            result = state.as_dict()
        self._logger.info("Plugin module unloaded: %s", clean_name)
        return result

    async def reload_module(self, module_name: str) -> dict[str, Any]:
        clean_name = self._normalize_module_name(module_name)
        await self.unload_module(clean_name)
        return await self.load_module(clean_name)

    def list_loaded_plugins(self) -> list[dict[str, str]]:
        return self._registry.list_plugin_details()

    def _discover_module_files(self) -> dict[str, Path]:
        discovered: dict[str, Path] = {}
        if not self._plugin_dir.exists():
            return discovered
        for file in sorted(self._plugin_dir.glob("*.py")):
            module_name = file.stem
            if module_name.startswith("_") or module_name in IGNORED_PLUGIN_MODULES:
                continue
            discovered[module_name] = file.resolve()
        return discovered

    @staticmethod
    def _normalize_module_name(module_name: str) -> str:
        return str(module_name or "").strip().replace(".py", "")
