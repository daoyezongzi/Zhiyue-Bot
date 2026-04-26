from __future__ import annotations

from typing import Dict

from plugins.base import BasePlugin


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, BasePlugin] = {}
        self._owners: Dict[str, str] = {}

    def register(self, plugin: BasePlugin, owner: str = "default") -> None:
        self._plugins[plugin.name] = plugin
        self._owners[plugin.name] = owner

    def unregister(self, name: str) -> None:
        self._plugins.pop(name, None)
        self._owners.pop(name, None)

    def unregister_by_owner(self, owner: str) -> None:
        clean_owner = str(owner).strip()
        names = [name for name, item_owner in self._owners.items() if item_owner == clean_owner]
        for name in names:
            self.unregister(name)

    def clear(self) -> None:
        self._plugins.clear()
        self._owners.clear()

    def register_defaults(self) -> None:
        from plugins.expression import build_plugins as build_expression_plugins
        from plugins.group_info import build_plugins as build_group_plugins
        from plugins.interaction import build_plugins as build_interaction_plugins
        from plugins.jargon import build_plugins as build_jargon_plugins
        from plugins.memory import build_plugins as build_memory_plugins
        from plugins.mood import build_plugins as build_mood_plugins
        from plugins.sticker import build_plugins as build_sticker_plugins
        from plugins.style_classification import build_plugins as build_style_plugins
        from plugins.web_request import build_plugins as build_web_plugins

        builders = [
            build_memory_plugins,
            build_jargon_plugins,
            build_expression_plugins,
            build_interaction_plugins,
            build_group_plugins,
            build_sticker_plugins,
            build_mood_plugins,
            build_style_plugins,
            build_web_plugins,
        ]
        for build in builders:
            for plugin in build():
                self.register(plugin, owner="default")

    def get(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        return sorted(self._plugins.keys())

    def list_plugin_details(self) -> list[dict[str, str]]:
        details: list[dict[str, str]] = []
        for name in sorted(self._plugins.keys()):
            plugin = self._plugins[name]
            details.append(
                {
                    "name": plugin.name,
                    "description": plugin.description,
                    "owner": self._owners.get(name, ""),
                },
            )
        return details
