from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from internal.config.schema import PersonaConfig


@dataclass(slots=True, frozen=True)
class MoodInfo:
    valence: float
    energy: float
    sociability: float


@dataclass
class Persona:
    cfg: PersonaConfig

    def is_mentioned(self, text: str) -> bool:
        names: Iterable[str] = [self.cfg.name, *self.cfg.alias_names]
        lowered = text.lower()
        return any(name and name.lower() in lowered for name in names)

    def get_name(self) -> str:
        return self.cfg.name

    def get_system_prompt(self) -> str:
        return (
            f"你是{self.cfg.name}，QQ群里的普通群友。"
            f"说话风格：{self.cfg.speaking_style}。"
            "不用 Markdown，保持口语自然。"
        )

    def get_mood_prompt(self, mood: MoodInfo) -> str:
        lines: list[str] = [
            "## 情绪状态",
            "你有一个持续存在的情绪状态，会随着对话和时间自然变化。",
            (
                f"当前状态：心情={mood.valence:.2f}  "
                f"精力={mood.energy:.2f}  社交意愿={mood.sociability:.2f}"
            ),
            "",
            "【心情】" + self._describe_valence(mood.valence),
            "【精力】" + self._describe_energy(mood.energy),
            "【社交意愿】" + self._describe_sociability(mood.sociability),
            "",
            "【情绪调整】",
            "- 你可以根据对话内容，使用 updateMood 工具调整情绪",
            "- 情绪会自然衰减回归平静，不需要强行干预",
        ]
        return "\n".join(lines)

    def get_think_prompt(
        self,
        chat_context: str,
        group_extra: str = "",
        mood: MoodInfo | None = None,
    ) -> str:
        extra = f"\n群附加设定：{group_extra}" if group_extra else ""
        mood_text = f"{self.get_mood_prompt(mood)}\n\n" if mood else ""
        return f"{mood_text}最近群聊如下：\n{chat_context}{extra}\n请决定发言或保持沉默。"

    @staticmethod
    def _describe_valence(value: float) -> str:
        if value >= 0.5:
            return "非常好"
        if value >= 0.2:
            return "还不错"
        if value >= -0.2:
            return "一般般"
        if value >= -0.5:
            return "有点烦"
        return "很差"

    @staticmethod
    def _describe_energy(value: float) -> str:
        if value >= 0.7:
            return "很有精神"
        if value >= 0.4:
            return "正常状态"
        return "有点累"

    @staticmethod
    def _describe_sociability(value: float) -> str:
        if value >= 0.7:
            return "很想聊天"
        if value >= 0.4:
            return "正常状态"
        return "不太想说话"
