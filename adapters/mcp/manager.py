from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class MCPTool:
    name: str
    source: str


@dataclass
class MCPManager:
    tools: List[MCPTool] = field(default_factory=list)

    async def load_from_config(self, path: str = "config/mcp.json") -> None:
        file_path = Path(path)
        if not file_path.exists():
            return
        data = json.loads(file_path.read_text(encoding="utf-8"))
        tools: list[MCPTool] = []
        for server in data.get("servers", []):
            if not server.get("enabled"):
                continue
            names = server.get("tool_name_list") or []
            for name in names:
                tools.append(MCPTool(name=name, source=server.get("name", "mcp")))
        self.tools = tools

    def tool_names(self) -> list[str]:
        return [item.name for item in self.tools]
