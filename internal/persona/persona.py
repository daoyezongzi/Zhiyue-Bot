from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from internal.config.schema import PersonaConfig


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True, frozen=True)
class MoodInfo:
    valence: float
    energy: float
    sociability: float


@dataclass(slots=True)
class Persona:
    cfg: PersonaConfig
    energy: float = 100.0
    mood: float = 50.0

    def is_mentioned(self, text: str) -> bool:
        names: Iterable[str] = [self.cfg.name, *self.cfg.alias_names]
        lowered = text.lower()
        return any(name and name.lower() in lowered for name in names)

    def get_name(self) -> str:
        return self.cfg.name

    def get_system_prompt(self) -> str:
        style = self.cfg.speaking_style.strip() or "自然、口语化、简洁。"
        custom_prompt = self.cfg.system_prompt.strip()
        base_prompt = (
            f"你是{self.cfg.name}，QQ群里的一个普通群友。\n"
            f"说话风格：{style}\n"
            "用纯文字回复，不要使用 Markdown。"
        )
        if not custom_prompt:
            return base_prompt
        return f"{custom_prompt}\n\n{base_prompt}"

    def get_mood_prompt(self, mood: MoodInfo) -> str:
        return (
            "## 情绪状态\n"
            f"心情={mood.valence:.2f} 精力={mood.energy:.2f} 社交意愿={mood.sociability:.2f}\n"
            f"生理状态：energy={self.energy:.0f}/100 mood={self.mood:.0f}/100"
        )

    def get_think_prompt(
        self,
        chat_context: str,
        group_extra: str = "",
        mood: MoodInfo | None = None,
    ) -> str:
        extra = f"\n群附加设定：{group_extra}" if group_extra else ""
        mood_text = f"{self.get_mood_prompt(mood)}\n\n" if mood else ""
        return f"{mood_text}最近群聊如下：\n{chat_context}{extra}\n请决定发言或保持沉默。"

    def set_physical_state(self, *, energy: float | None = None, mood: float | None = None) -> None:
        if energy is not None:
            self.energy = _clamp(float(energy), 0.0, 100.0)
        if mood is not None:
            self.mood = _clamp(float(mood), 0.0, 100.0)
