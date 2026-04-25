from __future__ import annotations

import json
from typing import Set


class ToolDedup:
    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def mark_seen(self, name: str, arguments: dict) -> bool:
        key = f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
        if key in self._seen:
            return True
        self._seen.add(key)
        return False
