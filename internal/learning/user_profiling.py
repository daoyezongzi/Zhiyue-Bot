from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from adapters.llm.chat import ChatLLMAdapter
from internal.logger import get_logger

ProfileScope = Literal["user", "group", "public"]
MemberNameSource = Literal["group_card", "learned_alias"]
MEMBER_NAME_SOURCE_GROUP_CARD: MemberNameSource = "group_card"
MEMBER_NAME_SOURCE_LEARNED_ALIAS: MemberNameSource = "learned_alias"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None

    def _decode(candidate: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
        return None

    direct = _decode(text)
    if direct is not None:
        return direct

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", text, re.IGNORECASE)
    if fenced:
        parsed = _decode(fenced.group(1))
        if parsed is not None:
            return parsed

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        parsed = _decode(text[first_obj : last_obj + 1])
        if parsed is not None:
            return parsed
    return None


def _split_style_items(style: str) -> list[str]:
    items = re.split(r"[,;\uFF0C\u3001\uFF1B\n]+", style)
    result: list[str] = []
    for raw in items:
        text = raw.strip()
        if text:
            result.append(text)
    return result


def _parse_iso_datetime(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_alias_items(values: Iterable[str] | None, *, limit: int = 8) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max(1, limit):
            break
    return out


@dataclass(slots=True, frozen=True)
class MemberNameRecord:
    content: str
    source: MemberNameSource
    group_id: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "group_id": self.group_id,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MemberNameRecord | None:
        content = str(raw.get("content", "")).strip()
        source = str(raw.get("source", "")).strip()
        if not content:
            return None
        if source not in {MEMBER_NAME_SOURCE_GROUP_CARD, MEMBER_NAME_SOURCE_LEARNED_ALIAS}:
            return None
        group_id = 0
        try:
            group_id = int(raw.get("group_id", 0) or 0)
        except (TypeError, ValueError):
            group_id = 0
        updated_at = str(raw.get("updated_at", "") or "").strip()
        return cls(
            content=content,
            source=source,  # type: ignore[arg-type]
            group_id=group_id,
            updated_at=updated_at,
        )


def normalize_member_name_records(records: Iterable[MemberNameRecord] | None) -> list[MemberNameRecord]:
    if not records:
        return []

    latest_by_key: dict[tuple[str, str, int], MemberNameRecord] = {}
    for row in records:
        content = str(row.content or "").strip()
        source = str(row.source or "").strip()
        group_id = int(row.group_id or 0)
        if not content:
            continue

        if source == MEMBER_NAME_SOURCE_GROUP_CARD:
            if group_id <= 0:
                continue
        elif source == MEMBER_NAME_SOURCE_LEARNED_ALIAS:
            group_id = 0
        else:
            continue

        clean = MemberNameRecord(
            content=content,
            source=source,  # type: ignore[arg-type]
            group_id=group_id,
            updated_at=str(row.updated_at or "").strip(),
        )
        key = (clean.content.lower(), clean.source, clean.group_id)
        existing = latest_by_key.get(key)
        if existing is None:
            latest_by_key[key] = clean
            continue
        if _parse_iso_datetime(existing.updated_at) >= _parse_iso_datetime(clean.updated_at):
            continue
        latest_by_key[key] = clean

    rows = list(latest_by_key.values())
    rows.sort(
        key=lambda item: (
            _parse_iso_datetime(item.updated_at),
            item.source,
            item.group_id,
            item.content.lower(),
        ),
        reverse=True,
    )
    return rows


def upsert_member_group_card(
    records: Iterable[MemberNameRecord] | None,
    *,
    group_id: int | None,
    card: str,
    updated_at: str,
) -> list[MemberNameRecord]:
    clean_card = str(card or "").strip()
    clean_group = int(group_id or 0)
    if clean_group <= 0 or not clean_card:
        return normalize_member_name_records(records)
    merged = list(records or [])
    merged.append(
        MemberNameRecord(
            content=clean_card,
            source=MEMBER_NAME_SOURCE_GROUP_CARD,
            group_id=clean_group,
            updated_at=str(updated_at or "").strip() or _now_iso(),
        )
    )
    return normalize_member_name_records(merged)


def upsert_member_learned_aliases(
    records: Iterable[MemberNameRecord] | None,
    aliases: Iterable[str] | None,
    *,
    updated_at: str,
) -> list[MemberNameRecord]:
    merged = list(records or [])
    for alias in _normalize_alias_items(aliases):
        merged.append(
            MemberNameRecord(
                content=alias,
                source=MEMBER_NAME_SOURCE_LEARNED_ALIAS,
                group_id=0,
                updated_at=str(updated_at or "").strip() or _now_iso(),
            )
        )
    return normalize_member_name_records(merged)


def latest_member_group_card(records: Iterable[MemberNameRecord] | None, group_id: int | None) -> str:
    clean_group = int(group_id or 0)
    if clean_group <= 0:
        return ""
    for row in normalize_member_name_records(records):
        if row.source == MEMBER_NAME_SOURCE_GROUP_CARD and int(row.group_id or 0) == clean_group:
            return str(row.content or "").strip()
    return ""


def member_learned_aliases(records: Iterable[MemberNameRecord] | None) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for row in normalize_member_name_records(records):
        if row.source != MEMBER_NAME_SOURCE_LEARNED_ALIAS:
            continue
        alias = str(row.content or "").strip()
        if not alias:
            continue
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        aliases.append(alias)
    return aliases


def member_names_for_admin(records: Iterable[MemberNameRecord] | None, nickname: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(text)

    _push(nickname)
    for row in normalize_member_name_records(records):
        _push(row.content)
    return names


def member_names_search_text(records: Iterable[MemberNameRecord] | None, nickname: str) -> str:
    return " ".join(member_names_for_admin(records, nickname))


@dataclass(slots=True, frozen=True)
class UserProfile:
    user_id: int
    nickname: str = ""
    name_records: tuple[MemberNameRecord, ...] = ()
    tags: tuple[str, ...] = ()
    affinity: float = 0.0
    interaction_style: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class UserProfileDelta:
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    affinity_delta: float = 0.0
    interaction_style: str = ""
    confidence: float = 0.5

    @classmethod
    def from_llm_json(cls, payload: dict[str, Any]) -> UserProfileDelta:
        raw_tags = payload.get("tags", [])
        tags: list[str] = []
        if isinstance(raw_tags, list):
            for item in raw_tags:
                text = str(item).strip()
                if text and text not in tags:
                    tags.append(text)

        aliases = _normalize_alias_items(payload.get("aliases", []), limit=8)

        affinity_delta = 0.0
        try:
            affinity_delta = float(payload.get("affinity_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            affinity_delta = 0.0
        affinity_delta = _clamp(affinity_delta, -0.4, 0.4)

        interaction_style = str(payload.get("interaction_style", "") or "").strip()
        confidence = 0.5
        try:
            confidence = float(payload.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = _clamp(confidence, 0.0, 1.0)

        return cls(
            tags=tags[:8],
            aliases=aliases,
            affinity_delta=affinity_delta,
            interaction_style=interaction_style[:160],
            confidence=confidence,
        )

    def is_effective(self) -> bool:
        return bool(self.tags or self.aliases or self.interaction_style or abs(self.affinity_delta) > 1e-6)


@dataclass(slots=True)
class _ProfileRecord:
    nickname: str = ""
    name_records: list[MemberNameRecord] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    affinity: float = 0.0
    interaction_style: str = ""
    updated_at: str = ""
    tag_scores: dict[str, float] = field(default_factory=dict)
    style_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "name_records": [item.to_dict() for item in normalize_member_name_records(self.name_records)],
            "tags": self.tags,
            "affinity": self.affinity,
            "interaction_style": self.interaction_style,
            "updated_at": self.updated_at,
            "tag_scores": self.tag_scores,
            "style_scores": self.style_scores,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> _ProfileRecord:
        nickname = str(raw.get("nickname", "") or "").strip()

        name_records: list[MemberNameRecord] = []
        raw_name_records = raw.get("name_records", [])
        if isinstance(raw_name_records, list):
            for item in raw_name_records:
                if not isinstance(item, dict):
                    continue
                parsed = MemberNameRecord.from_dict(item)
                if parsed is not None:
                    name_records.append(parsed)
        name_records = normalize_member_name_records(name_records)

        tags: list[str] = []
        raw_tags = raw.get("tags", [])
        if isinstance(raw_tags, list):
            for item in raw_tags:
                text = str(item).strip()
                if text:
                    tags.append(text)

        affinity = 0.0
        try:
            affinity = float(raw.get("affinity", 0.0) or 0.0)
        except (TypeError, ValueError):
            affinity = 0.0

        tag_scores: dict[str, float] = {}
        raw_tag_scores = raw.get("tag_scores", {})
        if isinstance(raw_tag_scores, dict):
            for key, value in raw_tag_scores.items():
                tag = str(key).strip()
                if not tag:
                    continue
                try:
                    tag_scores[tag] = float(value)
                except (TypeError, ValueError):
                    continue

        style_scores: dict[str, float] = {}
        raw_style_scores = raw.get("style_scores", {})
        if isinstance(raw_style_scores, dict):
            for key, value in raw_style_scores.items():
                style = str(key).strip()
                if not style:
                    continue
                try:
                    style_scores[style] = float(value)
                except (TypeError, ValueError):
                    continue

        return cls(
            nickname=nickname,
            name_records=name_records,
            tags=tags,
            affinity=_clamp(affinity, -1.0, 1.0),
            interaction_style=str(raw.get("interaction_style", "") or "").strip(),
            updated_at=str(raw.get("updated_at", "") or "").strip(),
            tag_scores=tag_scores,
            style_scores=style_scores,
        )

    def to_user_profile(self, user_id: int) -> UserProfile:
        return UserProfile(
            user_id=user_id,
            nickname=self.nickname,
            name_records=tuple(normalize_member_name_records(self.name_records)),
            tags=tuple(self.tags),
            affinity=self.affinity,
            interaction_style=self.interaction_style,
            updated_at=self.updated_at,
        )


class UserProfileStore:
    def __init__(self, path: str | Path) -> None:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        self._path = file_path
        self._lock = asyncio.Lock()
        self._loaded = False
        self._spaces: dict[ProfileScope, dict[str, _ProfileRecord]] = {
            "user": {},
            "group": {},
            "public": {},
        }
        self._logger = get_logger("UserProfileStore")

    async def load(self) -> None:
        async with self._lock:
            if self._loaded:
                return

            self._spaces = {
                "user": {},
                "group": {},
                "public": {},
            }
            if not self._path.exists():
                self._loaded = True
                return

            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._logger.exception("Failed to load user profiles: %s", self._path)
                self._loaded = True
                return

            raw_spaces = payload.get("spaces", {})
            if isinstance(raw_spaces, dict):
                for scope in ("user", "group", "public"):
                    raw_rows = raw_spaces.get(scope, {})
                    if not isinstance(raw_rows, dict):
                        continue
                    parsed: dict[str, _ProfileRecord] = {}
                    for key, value in raw_rows.items():
                        if isinstance(value, dict):
                            parsed[str(key)] = _ProfileRecord.from_dict(value)
                    self._spaces[scope] = parsed
            self._loaded = True

    async def get_user_profile(self, user_id: int) -> UserProfile:
        return await self.get_profile("user", str(user_id))

    async def get_profile(self, scope: ProfileScope, key: str) -> UserProfile:
        await self.load()
        async with self._lock:
            record = self._spaces.get(scope, {}).get(str(key))
            if record is None:
                uid = int(key) if key.isdigit() else 0
                return UserProfile(user_id=uid)
            uid = int(key) if key.isdigit() else 0
            return record.to_user_profile(uid)

    async def upsert_member_identity(
        self,
        *,
        user_id: int,
        nickname: str = "",
        group_id: int | None = None,
        group_card: str = "",
        updated_at: str = "",
    ) -> UserProfile:
        await self.load()
        clean_uid = int(user_id or 0)
        if clean_uid <= 0:
            return UserProfile(user_id=0)

        key = str(clean_uid)
        ts = str(updated_at or "").strip() or _now_iso()
        clean_nickname = str(nickname or "").strip()
        clean_group_card = str(group_card or "").strip()
        clean_group_id = int(group_id or 0)

        async with self._lock:
            bucket = self._spaces["user"]
            record = bucket.get(key)
            if record is None:
                record = _ProfileRecord()
                bucket[key] = record

            changed = False
            if clean_nickname and clean_nickname != record.nickname:
                record.nickname = clean_nickname
                changed = True

            if clean_group_id > 0 and clean_group_card:
                next_records = upsert_member_group_card(
                    record.name_records,
                    group_id=clean_group_id,
                    card=clean_group_card,
                    updated_at=ts,
                )
                if next_records != normalize_member_name_records(record.name_records):
                    record.name_records = next_records
                    changed = True

            if changed:
                record.updated_at = ts
                await self._persist_locked()

            return record.to_user_profile(clean_uid)

    async def replace_user_nickname(self, user_id: int, nickname: str) -> bool:
        await self.load()
        clean_uid = int(user_id or 0)
        if clean_uid <= 0:
            return False
        clean_name = str(nickname or "").strip()
        if not clean_name:
            return False

        key = str(clean_uid)
        async with self._lock:
            bucket = self._spaces["user"]
            record = bucket.get(key)
            if record is None:
                record = _ProfileRecord()
                bucket[key] = record
            if record.nickname == clean_name:
                return False
            record.nickname = clean_name
            record.updated_at = _now_iso()
            await self._persist_locked()
            return True

    async def list_user_profiles(self, *, keyword: str = "", limit: int = 200) -> list[UserProfile]:
        await self.load()
        clean_keyword = str(keyword or "").strip().lower()
        max_limit = max(1, min(int(limit or 200), 1000))

        async with self._lock:
            rows = list(self._spaces["user"].items())

        matched: list[tuple[int, UserProfile]] = []
        for key, record in rows:
            user_id = int(key) if key.isdigit() else 0
            profile = record.to_user_profile(user_id)
            if clean_keyword:
                search_blocks = [
                    str(user_id),
                    profile.nickname,
                    member_names_search_text(profile.name_records, profile.nickname),
                    " ".join(profile.tags),
                    profile.interaction_style,
                ]
                haystack = " ".join(item for item in search_blocks if item).lower()
                if clean_keyword not in haystack:
                    continue
            matched.append((user_id, profile))

        matched.sort(
            key=lambda item: (
                _parse_iso_datetime(item[1].updated_at),
                item[0],
            ),
            reverse=True,
        )
        return [item[1] for item in matched[:max_limit]]

    async def merge_user_profile(
        self,
        user_id: int,
        delta: UserProfileDelta,
        *,
        max_tags: int = 12,
    ) -> UserProfile:
        await self.load()
        key = str(user_id)
        async with self._lock:
            bucket = self._spaces["user"]
            record = bucket.get(key)
            if record is None:
                record = _ProfileRecord()
                bucket[key] = record

            self._merge_record(record, delta, max_tags=max_tags)
            await self._persist_locked()
            return record.to_user_profile(user_id)

    async def _persist_locked(self) -> None:
        payload = {
            "version": 1,
            "spaces": {
                scope: {key: row.to_dict() for key, row in bucket.items()} for scope, bucket in self._spaces.items()
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_suffix(self._path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self._path)

    @staticmethod
    def _merge_record(record: _ProfileRecord, delta: UserProfileDelta, *, max_tags: int) -> None:
        confidence = _clamp(delta.confidence, 0.0, 1.0)
        now_iso = _now_iso()
        record.name_records = normalize_member_name_records(record.name_records)

        for tag in record.tags:
            record.tag_scores.setdefault(tag, 1.0)
        for style in _split_style_items(record.interaction_style):
            record.style_scores.setdefault(style, 1.0)

        for tag in delta.tags:
            clean = tag.strip()
            if not clean:
                continue
            record.tag_scores[clean] = record.tag_scores.get(clean, 0.0) + (0.8 + confidence)

        for key in list(record.tag_scores.keys()):
            next_score = record.tag_scores[key] * 0.985
            if next_score < 0.18:
                del record.tag_scores[key]
            else:
                record.tag_scores[key] = next_score

        ranked_tags = sorted(record.tag_scores.items(), key=lambda item: item[1], reverse=True)
        record.tags = [item[0] for item in ranked_tags[: max(1, max_tags)]]

        target_affinity = _clamp(record.affinity + delta.affinity_delta, -1.0, 1.0)
        record.affinity = _clamp((record.affinity * 0.78) + (target_affinity * 0.22), -1.0, 1.0)

        for style in _split_style_items(delta.interaction_style):
            record.style_scores[style] = record.style_scores.get(style, 0.0) + (0.7 + confidence)

        for key in list(record.style_scores.keys()):
            next_score = record.style_scores[key] * 0.99
            if next_score < 0.15:
                del record.style_scores[key]
            else:
                record.style_scores[key] = next_score

        ranked_styles = sorted(record.style_scores.items(), key=lambda item: item[1], reverse=True)
        top_styles = [item[0] for item in ranked_styles[:3]]
        record.interaction_style = "\u3001".join(top_styles)

        if delta.aliases:
            record.name_records = upsert_member_learned_aliases(
                record.name_records,
                delta.aliases,
                updated_at=now_iso,
            )

        record.updated_at = now_iso


class UserProfilingEngine:
    def __init__(
        self,
        *,
        llm: ChatLLMAdapter,
        store: UserProfileStore,
        trigger_message_count: int = 20,
        context_limit: int = 40,
        max_tags: int = 12,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._store = store
        self._enabled = enabled
        self._trigger = max(1, trigger_message_count)
        self._context_limit = max(self._trigger, context_limit)
        self._max_tags = max(4, max_tags)

        self._lock = asyncio.Lock()
        self._buffers: dict[int, list[dict[str, Any]]] = {}
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._needs_rerun: set[int] = set()
        self._logger = get_logger("UserProfilingEngine")

    async def start(self) -> None:
        await self._store.load()

    async def stop(self) -> None:
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_user_profile(self, user_id: int | None) -> UserProfile | None:
        if user_id is None or user_id <= 0:
            return None
        return await self._store.get_user_profile(user_id)

    async def list_user_profiles(self, *, keyword: str = "", limit: int = 200) -> list[UserProfile]:
        return await self._store.list_user_profiles(keyword=keyword, limit=limit)

    async def sync_member_identity(
        self,
        *,
        user_id: int | None,
        nickname: str = "",
        group_id: int | None = None,
        group_card: str = "",
        updated_at: str = "",
    ) -> UserProfile | None:
        uid = int(user_id or 0)
        if uid <= 0:
            return None
        return await self._store.upsert_member_identity(
            user_id=uid,
            nickname=nickname,
            group_id=group_id,
            group_card=group_card,
            updated_at=updated_at,
        )

    async def replace_profile_nickname(self, *, user_id: int | None, nickname: str) -> bool:
        uid = int(user_id or 0)
        if uid <= 0:
            return False
        return await self._store.replace_user_nickname(uid, nickname)

    @staticmethod
    def display_name_for_group(
        profile: UserProfile | None,
        *,
        group_id: int | None,
        current_group_card: str = "",
        current_nickname: str = "",
    ) -> str:
        card = str(current_group_card or "").strip()
        if card:
            return card
        if profile is not None:
            history_group_card = latest_member_group_card(profile.name_records, group_id)
            if history_group_card:
                return history_group_card
            if profile.nickname:
                return profile.nickname
        return str(current_nickname or "").strip()

    async def build_social_background(self, user_id: int | None, speaker: str) -> str:
        profile = await self.get_user_profile(user_id)
        if profile is None:
            return ""

        tags = ", ".join(profile.tags[:6]) if profile.tags else "no stable tags yet"
        style = profile.interaction_style or "no clear preference"
        affinity = profile.affinity
        if affinity >= 0.45:
            attitude = "be warmer and allow light teasing."
        elif affinity <= -0.35:
            attitude = "stay restrained and avoid provocative tone."
        else:
            attitude = "keep it natural and adjust by feedback."

        return (
            f"Current speaker: {speaker or 'unknown'} (uid={user_id})\n"
            f"- Profile tags: {tags}\n"
            f"- Affinity: {affinity:.2f} (-1 to 1)\n"
            f"- Interaction style: {style}\n"
            f"- Suggested attitude: {attitude}\n"
            "Note: this is incremental long-term cognition; do not overreact to a single message."
        )

    async def observe_user_message(
        self,
        *,
        user_id: int | None,
        speaker: str,
        nickname: str,
        group_card: str,
        text: str,
        session_id: str,
        group_id: int | None,
    ) -> None:
        await self._observe(
            user_id=user_id,
            role="user",
            speaker=speaker,
            nickname=nickname,
            group_card=group_card,
            text=text,
            session_id=session_id,
            group_id=group_id,
        )

    async def observe_bot_reply(
        self,
        *,
        user_id: int | None,
        text: str,
        session_id: str,
        group_id: int | None,
    ) -> None:
        await self._observe(
            user_id=user_id,
            role="assistant",
            speaker="Zhiyue",
            nickname="",
            group_card="",
            text=text,
            session_id=session_id,
            group_id=group_id,
        )

    async def _observe(
        self,
        *,
        user_id: int | None,
        role: str,
        speaker: str,
        nickname: str,
        group_card: str,
        text: str,
        session_id: str,
        group_id: int | None,
    ) -> None:
        uid = int(user_id or 0)
        if uid <= 0:
            return

        if not self._enabled:
            return

        content = str(text).strip()
        if not content:
            return

        event = {
            "role": role,
            "speaker": speaker,
            "text": content,
            "session_id": session_id,
            "group_id": group_id,
            "at": _now_iso(),
        }

        async with self._lock:
            buffer = self._buffers.setdefault(uid, [])
            buffer.append(event)
            if len(buffer) > (self._context_limit * 4):
                del buffer[: len(buffer) - (self._context_limit * 4)]

            if len(buffer) < self._trigger:
                return

            if uid in self._tasks:
                self._needs_rerun.add(uid)
                return

            self._start_task_locked(uid)

    def _start_task_locked(self, user_id: int) -> None:
        task = asyncio.create_task(self._run_profile_update(user_id), name=f"profile-update-{user_id}")
        self._tasks[user_id] = task
        task.add_done_callback(lambda done: asyncio.create_task(self._on_task_done(user_id, done)))

    async def _on_task_done(self, user_id: int, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            self._logger.exception("User profile update failed: user=%s", user_id)

        async with self._lock:
            self._tasks.pop(user_id, None)
            should_rerun = user_id in self._needs_rerun
            if should_rerun:
                self._needs_rerun.discard(user_id)
            buffer_len = len(self._buffers.get(user_id, []))
            if should_rerun and buffer_len >= self._trigger:
                self._start_task_locked(user_id)

    async def _run_profile_update(self, user_id: int) -> None:
        profile = await self._store.get_user_profile(user_id)
        async with self._lock:
            rows = list(self._buffers.get(user_id, []))
        if len(rows) < self._trigger:
            return

        clip = rows[-self._context_limit :]
        transcript = "\n".join(
            f"[{row['role']}] {row['speaker']}: {row['text']}" for row in clip if str(row.get("text", "")).strip()
        )
        if not transcript:
            return

        prompt = (
            "You are a user profiling analyzer. Extract incremental long-term profile signals from dialogue."
            " Output JSON only.\n\n"
            "Output format:\n"
            "{\n"
            '  "tags": ["tag1", "tag2"],\n'
            '  "aliases": ["stable alias 1"],\n'
            '  "affinity_delta": 0.0,\n'
            '  "interaction_style": "one-line interaction preference",\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            "Rules:\n"
            "1. Keep only stable and evidence-backed tags (max 6).\n"
            "2. aliases must be stable, repeatedly-used names backed by explicit evidence; avoid one-off jokes.\n"
            "3. affinity_delta must be in [-0.3, 0.3].\n"
            "4. interaction_style should be one concise sentence.\n"
            "5. If evidence is weak, return empty tags/aliases and affinity_delta=0.\n\n"
            f"Current profile: tags={list(profile.tags)}, affinity={profile.affinity:.3f}, "
            f"aliases={member_learned_aliases(profile.name_records)}, "
            f"interaction_style={profile.interaction_style or 'none'}\n\n"
            f"Dialogue transcript:\n{transcript}"
        )

        messages = [
            {"role": "system", "content": "Return a valid JSON object only."},
            {"role": "user", "content": prompt},
        ]
        raw = await self._llm.generate_from_messages(messages, extra_fields={"temperature": 0.2})
        payload = _extract_json_object(raw)
        if payload is None:
            self._logger.warning("Profile LLM result is not JSON: user=%s", user_id)
            return

        delta = UserProfileDelta.from_llm_json(payload)
        if not delta.is_effective():
            await self._trim_buffer(user_id)
            return

        await self._store.merge_user_profile(user_id, delta, max_tags=self._max_tags)
        await self._trim_buffer(user_id)

    async def _trim_buffer(self, user_id: int) -> None:
        keep = max(8, self._trigger // 2)
        async with self._lock:
            rows = self._buffers.get(user_id, [])
            if len(rows) > keep:
                self._buffers[user_id] = rows[-keep:]
