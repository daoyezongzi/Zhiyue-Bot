from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

_MAJOR_ARCANA_INDEX_BY_CARD_ID = {
    "major_00_the_fool": "0",
    "major_01_the_magician": "I",
    "major_02_the_high_priestess": "II",
    "major_03_the_empress": "III",
    "major_04_the_emperor": "IV",
    "major_05_the_hierophant": "V",
    "major_06_the_lovers": "VI",
    "major_07_the_chariot": "VII",
    "major_08_strength": "VIII",
    "major_09_the_hermit": "IX",
    "major_10_wheel_of_fortune": "X",
    "major_11_justice": "XI",
    "major_12_the_hanged_man": "XII",
    "major_13_death": "XIII",
    "major_14_temperance": "XIV",
    "major_15_the_devil": "XV",
    "major_16_the_tower": "XVI",
    "major_17_the_star": "XVII",
    "major_18_the_moon": "XVIII",
    "major_19_the_sun": "XIX",
    "major_20_judgement": "XX",
    "major_21_the_world": "XXI",
}


@dataclass(slots=True, frozen=True)
class TarotCard:
    card_id: str
    name_cn: str
    name_en: str
    arcana: str
    suit: str
    rank: str
    upright_meaning: str
    reversed_meaning: str

    @property
    def display_name_cn(self) -> str:
        if self.arcana != "major":
            return self.name_cn
        index = _MAJOR_ARCANA_INDEX_BY_CARD_ID.get(self.card_id, "")
        if not index:
            return self.name_cn
        if self.name_cn.endswith(index):
            return self.name_cn
        return f"{self.name_cn}{index}"


@dataclass(slots=True, frozen=True)
class TarotDraw:
    card: TarotCard
    orientation_key: str
    orientation_label: str
    meaning: str


class TarotKnowledgeBase:
    """Local tarot deck repository backed by JSON file."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = Path(file_path)
        self._cards: list[TarotCard] = []
        self._load_error: str | None = None
        try:
            self._cards = self._load_cards()
        except RuntimeError as exc:
            self._load_error = str(exc)

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def card_count(self) -> int:
        return len(self._cards)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def draw(self, rng: random.Random | None = None) -> TarotDraw:
        if self._load_error:
            raise RuntimeError(self._load_error)
        if not self._cards:
            raise RuntimeError("塔罗知识库为空，没有可抽取的牌。")

        chooser = rng or random.Random()
        card = chooser.choice(self._cards)
        candidates: list[tuple[str, str, str]] = []
        if card.upright_meaning:
            candidates.append(("upright", "正位", card.upright_meaning))
        if card.reversed_meaning:
            candidates.append(("reversed", "逆位", card.reversed_meaning))
        if not candidates:
            raise RuntimeError(f"塔罗牌缺少解释：{card.card_id}")

        orientation_key, orientation_label, meaning = chooser.choice(candidates)
        return TarotDraw(
            card=card,
            orientation_key=orientation_key,
            orientation_label=orientation_label,
            meaning=meaning,
        )

    def _load_cards(self) -> list[TarotCard]:
        if not self._file_path.exists():
            raise RuntimeError(f"塔罗知识库文件不存在：{self._file_path}")

        try:
            raw = json.loads(self._file_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"读取塔罗知识库失败：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"塔罗知识库 JSON 格式错误：{exc}") from exc

        if not isinstance(raw, dict):
            raise RuntimeError("塔罗知识库顶层必须是 JSON 对象。")

        source_cards = raw.get("cards")
        if not isinstance(source_cards, list):
            raise RuntimeError("塔罗知识库缺少 cards 数组。")

        parsed: list[TarotCard] = []
        for item in source_cards:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("id") or "").strip()
            name_cn = str(item.get("name_cn") or "").strip()
            name_en = str(item.get("name_en") or "").strip()
            arcana = str(item.get("arcana") or "").strip().lower() or "minor"
            suit = str(item.get("suit") or "").strip().lower()
            rank = str(item.get("rank") or "").strip()
            upright_meaning = str(item.get("upright") or "").strip()
            reversed_meaning = str(item.get("reversed") or "").strip()

            if not card_id or not name_cn:
                continue
            if not upright_meaning and not reversed_meaning:
                continue

            parsed.append(
                TarotCard(
                    card_id=card_id,
                    name_cn=name_cn,
                    name_en=name_en,
                    arcana=arcana,
                    suit=suit,
                    rank=rank,
                    upright_meaning=upright_meaning,
                    reversed_meaning=reversed_meaning,
                )
            )

        if not parsed:
            raise RuntimeError("塔罗知识库没有有效卡牌数据。")
        return parsed
