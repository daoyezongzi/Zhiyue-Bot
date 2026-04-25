from __future__ import annotations

from typing import Dict

from plugins.base import BasePlugin
from plugins.expression import build_plugins as build_expression_plugins
from plugins.group_info import build_plugins as build_group_plugins
from plugins.interaction import build_plugins as build_interaction_plugins
from plugins.jargon import build_plugins as build_jargon_plugins
from plugins.memory import build_plugins as build_memory_plugins
from plugins.mood import build_plugins as build_mood_plugins
from plugins.sticker import build_plugins as build_sticker_plugins
from plugins.style_classification import build_plugins as build_style_plugins
from plugins.web_request import build_plugins as build_web_plugins


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, BasePlugin] = {}

    def register(self, plugin: BasePlugin) -> None:
        self._plugins[plugin.name] = plugin

    def register_defaults(self) -> None:
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
                self.register(plugin)

    def get(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        return sorted(self._plugins.keys())
