from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Dict, Iterable, Pattern

try:
    import ahocorasick
except ImportError:  # pragma: no cover - fallback path
    ahocorasick = None

from internal.config.schema import JargonConfig


@dataclass(slots=True, frozen=True)
class StyleClassification:
    intent: str
    tone: str
    tone_key: str


LoaderType = Callable[[], Iterable[Any] | Awaitable[Iterable[Any]]]


class JargonManager:
    def __init__(
        self,
        cfg: JargonConfig | None = None,
        *,
        loader: LoaderType | None = None,
    ) -> None:
        self.cfg = cfg or JargonConfig()
        self._loader = loader
        self._jargons: Dict[str, str] = {}

        self._lock = asyncio.Lock()
        self._automaton: Any | None = None
        self._pattern_keys: list[str] = []
        self._pattern_lowers: list[str] = []
        self._meanings: list[str] = []

        self._keyword_rules = self._compile_keyword_aliases(self.cfg.keyword_aliases)
        self._style_rules = self._compile_style_rules(self.cfg.style_rules)

    async def reload(self, rows: Iterable[Any] | dict[str, str] | None = None) -> int:
        source = rows
        if source is None and self._loader is not None:
            loaded = self._loader()
            source = await loaded if isawaitable(loaded) else loaded

        if source is None:
            source = self._jargons.items()

        normalized = self._normalize_reload_rows(source)
        async with self._lock:
            self._jargons = normalized
            self._rebuild_automaton_locked()
            return len(self._pattern_keys)

    async def add(self, term: str, meaning: str) -> None:
        clean_term = term.strip()
        clean_meaning = meaning.strip()
        if not clean_term:
            return

        async with self._lock:
            existing_key = self._find_key_case_insensitive_locked(clean_term)
            if existing_key is not None:
                self._jargons[existing_key] = clean_meaning
            else:
                self._jargons[clean_term] = clean_meaning
            self._rebuild_automaton_locked()

    async def match(self, text: str) -> dict[str, str]:
        content = text.strip()
        if not content:
            return {}

        async with self._lock:
            automaton = self._automaton
            pattern_keys = list(self._pattern_keys)
            pattern_lowers = list(self._pattern_lowers)
            meanings = list(self._meanings)

        if automaton is None or not pattern_keys:
            return self._naive_match(content)

        lowered = content.lower()
        candidates: list[tuple[int, int, int]] = []
        for end_idx, pattern_idx in automaton.iter(lowered):
            if pattern_idx < 0 or pattern_idx >= len(pattern_lowers):
                continue
            token = pattern_lowers[pattern_idx]
            start_idx = end_idx - len(token) + 1
            candidates.append((start_idx, end_idx, pattern_idx))

        if not candidates:
            return {}

        picked = self._leftmost_longest(candidates)
        result: dict[str, str] = {}
        for _, _, pattern_idx in picked:
            if 0 <= pattern_idx < len(pattern_keys):
                result[pattern_keys[pattern_idx]] = meanings[pattern_idx]
        return result

    async def search(self, keyword: str, limit: int = 10) -> list[tuple[str, str]]:
        key = keyword.strip().lower()
        async with self._lock:
            pairs = [(k, v) for k, v in self._jargons.items() if key in k.lower() or key in v.lower()]
        return pairs[:limit]

    def classify_style(self, mood: float, energy: float, speaker_is_master: bool = False) -> StyleClassification:
        if mood <= self.cfg.low_mood_threshold:
            return StyleClassification(intent="安抚缓和", tone="克制", tone_key="restrained")

        if mood >= self.cfg.high_mood_threshold:
            if energy >= 0.65:
                return StyleClassification(intent="轻快回应", tone="舒展", tone_key="exaggerate")
            return StyleClassification(intent="自然应答", tone="轻和", tone_key="light")

        if energy <= 0.35:
            return StyleClassification(intent="询问推进", tone="直接", tone_key="direct")

        if speaker_is_master:
            return StyleClassification(intent="自然应答", tone="轻和", tone_key="light")

        if mood >= 0:
            return StyleClassification(intent="自然应答", tone="轻和", tone_key="light")

        return StyleClassification(intent="询问推进", tone="直接", tone_key="direct")

    def build_style_prompt(self, mood: float, energy: float, speaker_is_master: bool = False) -> str:
        style = self.classify_style(mood, energy, speaker_is_master=speaker_is_master)
        return (
            "风格建议："
            f"intent={style.intent}，tone={style.tone}。"
            "表达以自然、克制、略带文艺为主；可保留少量古典意象，不主动堆叠黑话或语气助词。"
        )

    def apply_post_process(
        self,
        text: str,
        *,
        mood: float,
        energy: float,
        speaker_is_master: bool = False,
    ) -> str:
        content = text.strip()
        if not content or not self.cfg.enabled:
            return content

        style = self.classify_style(mood, energy, speaker_is_master=speaker_is_master)
        content = self._apply_rules(content, self._keyword_rules)
        content = self._apply_rules(content, self._style_rules.get(style.tone_key, ()))
        content = self._inject_particle(content, style.tone_key)
        return content.strip()

    def _rebuild_automaton_locked(self) -> None:
        if ahocorasick is None:
            self._automaton = None
            self._pattern_keys = list(self._jargons.keys())
            self._pattern_lowers = [item.lower() for item in self._pattern_keys]
            self._meanings = [self._jargons[item] for item in self._pattern_keys]
            return

        pattern_keys: list[str] = []
        pattern_lowers: list[str] = []
        meanings: list[str] = []
        automaton = ahocorasick.Automaton()

        for pattern, meaning in self._jargons.items():
            clean_pattern = str(pattern).strip()
            if not clean_pattern:
                continue
            clean_meaning = str(meaning).strip()
            idx = len(pattern_keys)
            lowered = clean_pattern.lower()
            automaton.add_word(lowered, idx)
            pattern_keys.append(clean_pattern)
            pattern_lowers.append(lowered)
            meanings.append(clean_meaning)

        if pattern_keys:
            automaton.make_automaton()
            self._automaton = automaton
        else:
            self._automaton = None

        self._pattern_keys = pattern_keys
        self._pattern_lowers = pattern_lowers
        self._meanings = meanings

    @staticmethod
    def _normalize_reload_rows(rows: Iterable[Any] | dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}

        if isinstance(rows, dict):
            for term, meaning in rows.items():
                clean_term = str(term).strip()
                if not clean_term:
                    continue
                normalized[clean_term] = str(meaning).strip()
            return normalized

        for row in rows:
            term = ""
            meaning = ""

            if isinstance(row, tuple) and len(row) >= 2:
                term = str(row[0]).strip()
                meaning = str(row[1]).strip()
            elif isinstance(row, dict):
                term = str(row.get("content") or row.get("term") or "").strip()
                meaning = str(row.get("meaning") or "").strip()
            else:
                term = str(getattr(row, "content", "") or getattr(row, "term", "")).strip()
                meaning = str(getattr(row, "meaning", "")).strip()

            if not term:
                continue
            normalized[term] = meaning

        return normalized

    def _find_key_case_insensitive_locked(self, term: str) -> str | None:
        lowered = term.lower()
        for key in self._jargons.keys():
            if key.lower() == lowered:
                return key
        return None

    @staticmethod
    def _leftmost_longest(matches: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if not matches:
            return []

        ordered = sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        picked: list[tuple[int, int, int]] = []
        cursor = 0
        idx = 0

        while idx < len(ordered):
            while idx < len(ordered) and ordered[idx][1] < cursor:
                idx += 1
            if idx >= len(ordered):
                break

            probe = idx
            while probe < len(ordered) and ordered[probe][0] < cursor:
                probe += 1
            if probe >= len(ordered):
                break

            start = ordered[probe][0]
            best = ordered[probe]
            probe += 1
            while probe < len(ordered) and ordered[probe][0] == start:
                current = ordered[probe]
                if (current[1] - current[0]) > (best[1] - best[0]):
                    best = current
                probe += 1

            picked.append(best)
            cursor = best[1] + 1
            idx = probe

        return picked

    def _naive_match(self, text: str) -> dict[str, str]:
        lowered = text.lower()
        return {term: meaning for term, meaning in self._jargons.items() if term.lower() in lowered}

    @staticmethod
    def _compile_keyword_aliases(aliases: Dict[str, str]) -> list[tuple[Pattern[str], str]]:
        rules: list[tuple[Pattern[str], str]] = []
        for src, dst in aliases.items():
            src = src.strip()
            if not src:
                continue
            rules.append((re.compile(re.escape(src), re.IGNORECASE), dst))
        return rules

    @staticmethod
    def _compile_style_rules(raw: Dict[str, list]) -> dict[str, list[tuple[Pattern[str], str]]]:
        result: dict[str, list[tuple[Pattern[str], str]]] = {}
        for tone_key, rows in raw.items():
            compiled: list[tuple[Pattern[str], str]] = []
            for row in rows:
                pattern = str(getattr(row, "pattern", "")).strip()
                replacement = str(getattr(row, "replacement", "")).strip()
                if not pattern:
                    continue
                try:
                    compiled.append((re.compile(pattern, re.IGNORECASE), replacement))
                except re.error:
                    continue
            if compiled:
                result[tone_key] = compiled
        return result

    @staticmethod
    def _apply_rules(text: str, rules: Iterable[tuple[Pattern[str], str]]) -> str:
        updated = text
        for pattern, replacement in rules:
            updated = pattern.sub(replacement, updated)
        return updated

    def _inject_particle(self, text: str, tone_key: str) -> str:
        particles = self.cfg.tone_particles.get(tone_key) or []
        if not particles:
            return text

        particle = particles[abs(hash(text)) % len(particles)].strip()
        if not particle:
            return text

        if particle in text[-4:]:
            return text

        if re.search(r"[。！？!?~～…]+$", text):
            if particle in {"。", "！", "!", "？", "?", "？！", "!?"}:
                return text
            return re.sub(r"([。！？!?~～…]+)$", rf"{particle}\1", text, count=1)

        return text + particle
