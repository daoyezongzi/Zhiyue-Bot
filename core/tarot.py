from __future__ import annotations

import hashlib
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
    image_ref: str = ""

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

    _SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

    def __init__(self, file_path: Path, image_dir: Path | None = None) -> None:
        self._file_path = Path(file_path)
        if image_dir is None:
            raw_image_dir = self._file_path.parent / "tarot_images"
        else:
            raw_image_dir = Path(image_dir)
        self._image_dir = raw_image_dir.resolve()
        self._generated_dir = self._image_dir / "_generated"
        self._cards: list[TarotCard] = []
        self._load_error: str | None = None
        self._prepare_image_dirs()
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

    @property
    def image_dir(self) -> Path:
        return self._image_dir

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

    def resolve_draw_image_path(self, draw: TarotDraw) -> Path | None:
        source = self.resolve_card_image_path(draw.card)
        if source is None:
            return None
        if draw.orientation_key != "reversed":
            return source
        return self._build_reversed_image(source, draw.card.card_id)

    def resolve_card_image_path(self, card: TarotCard) -> Path | None:
        image_ref = str(card.image_ref or "").strip()
        if image_ref:
            resolved = self._resolve_image_ref(image_ref)
            if resolved is not None:
                return resolved

        for suffix in self._SUPPORTED_IMAGE_SUFFIXES:
            candidate = (self._image_dir / f"{card.card_id}{suffix}").resolve()
            if candidate.is_file():
                return candidate
        return None

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
            image_ref = str(item.get("image") or item.get("image_file") or "").strip()

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
                    image_ref=image_ref,
                )
            )

        if not parsed:
            raise RuntimeError("塔罗知识库没有有效卡牌数据。")
        return parsed

    def _prepare_image_dirs(self) -> None:
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._generated_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_image_ref(self, image_ref: str) -> Path | None:
        candidate = Path(image_ref)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
            return None

        resolved = (self._image_dir / candidate).resolve()
        if not self._is_inside_dir(resolved, self._image_dir):
            return None
        if resolved.is_file():
            return resolved
        return None

    def _build_reversed_image(self, source: Path, card_id: str) -> Path:
        cache_path = self._reversed_cache_path(source, card_id)
        try:
            source_mtime = source.stat().st_mtime
            if cache_path.is_file() and cache_path.stat().st_mtime >= source_mtime:
                return cache_path
        except OSError:
            return source

        try:
            from PIL import Image  # type: ignore[import-not-found]
        except Exception:
            return source

        try:
            with Image.open(source) as image:
                rotated = image.rotate(180, expand=True)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                rotated.save(cache_path, format="PNG")
        except Exception:
            return source
        return cache_path

    def _reversed_cache_path(self, source: Path, card_id: str) -> Path:
        safe_card_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(card_id or "card"))
        safe_card_id = safe_card_id.strip("_") or "card"
        digest = hashlib.sha1(str(source.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:12]
        return self._generated_dir / f"{safe_card_id}_{digest}_r180.png"

    @staticmethod
    def _is_inside_dir(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
