from __future__ import annotations

import asyncio
import hashlib
import html
import json
import mimetypes
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import aiohttp

from adapters.llm.chat import ChatLLMAdapter
from adapters.onebot.client import OneBotClient
from internal.config.schema import Config
from internal.logger import get_logger


@dataclass(slots=True)
class StickerSegment:
    segment_type: str
    url: str
    file: str
    summary: str
    sub_type: int


@dataclass(slots=True)
class PersonaFilterDecision:
    allowed: bool
    reason: str
    source: str


class StickerCollector:
    INDEX_FILE_NAME = "index.json"
    INDEX_VERSION = 2
    MAX_PERSONA_CACHE_ITEMS = 2048
    _CQ_IMAGE_PATTERN = re.compile(r"\[CQ:(?:image|mface),([^\]]+)\]", re.IGNORECASE)
    _VOLATILE_HEX_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)
    _VOLATILE_DIGIT_TOKEN_PATTERN = re.compile(r"^\d{6,}$")
    _VOLATILE_MIXED_ID_TOKEN_PATTERN = re.compile(r"^(?=.*\d)[a-z0-9]{12,}$", re.IGNORECASE)
    _CACHE_TOKEN_EDGE_PATTERN = re.compile(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$")
    _DEFAULT_CLOUD_ACTIONS = (
        "set_msg_favorite",
        "nc_set_msg_favorite",
        "mark_msg_as_favorite",
    )
    _STICKER_SUMMARY_HINTS = (
        "表情",
        "动画",
        "emoji",
        "sticker",
        "meme",
    )
    _STICKER_SOURCE_HINTS = (
        "gxh.vip.qq.com/club/item/parcel/item/",
        "/club/item/parcel/item/",
        "multimedia.nt.qq.com.cn/download",
        "appid=1407",
        "sticker",
        "mface",
    )
    _TAG_TOKEN_TABLE = (
        ("开心", ("笑", "得意", "好耶", "开心")),
        ("难过", ("哭", "委屈", "伤心")),
        ("生气", ("怒", "生气", "火大")),
        ("无语", ("无语", "汗", "尴尬")),
        ("可爱", ("猫", "狗", "熊猫", "萌", "可爱")),
    )

    def __init__(
        self,
        *,
        cfg: Config,
        bot_client: OneBotClient,
        llm: ChatLLMAdapter | None = None,
    ) -> None:
        self._cfg = cfg
        self._bot = bot_client
        self._llm = llm
        self._rng = random.Random()
        self._logger = get_logger("StickerCollector")
        self._project_root = Path(__file__).resolve().parents[2]
        self._index_lock = asyncio.Lock()
        self._persona_cache_lock = asyncio.Lock()
        self._persona_cache: dict[str, tuple[float, PersonaFilterDecision]] = {}

    @property
    def local_dir(self) -> Path:
        raw = str(self._cfg.sticker.local_dir or "data/stickers").strip() or "data/stickers"
        folder = Path(raw)
        if not folder.is_absolute():
            folder = self._project_root / folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder.resolve()

    def resolve_local_file_path(self, file_name: str) -> Path:
        clean_name = self._sanitize_file_name(file_name)
        target = (self.local_dir / clean_name).resolve()
        if target.parent != self.local_dir:
            raise ValueError("invalid sticker file path")
        return target

    async def runtime_settings(self) -> dict[str, Any]:
        cfg = self._cfg.sticker
        return {
            "enabled": bool(cfg.enabled),
            "collection_rate": round(self._clamp_probability(cfg.collection_rate), 4),
            "storage_mode": self._normalize_storage_mode(cfg.storage_mode),
            "local_dir": str(self.local_dir),
            "filter_keywords": self._normalize_keywords(cfg.filter_keywords),
            "user_weights": self._normalize_user_weights(cfg.user_weights),
            "enable_persona_filter": bool(cfg.enable_persona_filter),
            "llm_filter_enabled": bool(cfg.llm_filter_enabled),
            "llm_filter_probability": round(self._clamp_probability(cfg.llm_filter_probability), 4),
            "llm_filter_mood_threshold": float(cfg.llm_filter_mood_threshold or 0.0),
        }

    async def audit_and_tag_library(self, *, force_llm: bool = False, limit: int = 0) -> dict[str, Any]:
        max_items = max(0, int(limit or 0))
        reviewed = 0
        approved = 0
        rejected = 0
        changed = 0

        async with self._index_lock:
            index = self._load_index_nolock()
            items = index.get("items", [])
            if not isinstance(items, list):
                items = []
            normalized_items: list[dict[str, Any]] = []
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                row = dict(raw)
                row, _ = self._normalize_index_item(row)
                if max_items > 0 and reviewed >= max_items:
                    normalized_items.append(row)
                    continue

                if not bool(row.get("is_sticker")):
                    decision = PersonaFilterDecision(
                        allowed=False,
                        reason="not_sticker",
                        source="classifier",
                    )
                    status = "pending"
                else:
                    filter_text = self._build_reply_filter_text(item=row, query="")
                    decision = await self._run_persona_filter(
                        filter_text=filter_text,
                        mood=100.0,
                        skip_filter=False,
                        force_llm=force_llm,
                    )
                    status = "approved" if decision.allowed else "rejected"
                    if status == "approved":
                        approved += 1
                    else:
                        rejected += 1
                reviewed += 1

                original_status = str(row.get("review_status", "")).strip().lower()
                original_reason = str(row.get("review_reason", "")).strip()
                original_source = str(row.get("review_source", "")).strip()
                original_tags = self._normalize_tags(row.get("tags", []))
                new_tags = self.derive_sticker_tags(row)

                row["review_status"] = status
                row["review_reason"] = str(decision.reason or "")
                row["review_source"] = str(decision.source or "")
                row["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                row["tags"] = new_tags

                if (
                    original_status != row["review_status"]
                    or original_reason != row["review_reason"]
                    or original_source != row["review_source"]
                    or original_tags != new_tags
                ):
                    changed += 1

                normalized_items.append(row)

            index["items"] = normalized_items
            if changed > 0:
                self._save_index_nolock(index)

        return {
            "ok": True,
            "reviewed": reviewed,
            "approved": approved,
            "rejected": rejected,
            "changed": changed,
            "force_llm": bool(force_llm),
        }

    async def observe_group_message(
        self,
        *,
        message: Mapping[str, Any],
        group_id: int | None,
        sender_id: int | None,
        speaker: str,
        mood: float,
        is_admin: bool,
    ) -> None:
        cfg = self._cfg.sticker
        if not cfg.enabled:
            return
        if group_id is None:
            return

        segments = self._extract_sticker_segments(message)
        if not segments:
            return

        message_id = self._to_int(message.get("message_id"))
        base_rate = self._clamp_probability(cfg.collection_rate)
        weight = self._resolve_user_weight(sender_id)
        if not is_admin and weight <= 0.0:
            self._logger.info(
                "Sticker.Collect skipped: reason=blacklist user_id=%s group_id=%s",
                sender_id,
                group_id,
            )
            return

        final_rate = 1.0
        roll = 0.0
        if not is_admin:
            mood_factor = self._clamp_probability(float(mood) / 100.0)
            # Keep collection_rate as the baseline and only slightly adjust by mood.
            mood_adjustment = 0.8 + (0.4 * mood_factor)  # 0.8 ~ 1.2
            final_rate = self._clamp_probability(base_rate * weight * mood_adjustment)
            if final_rate <= 0.0:
                return
            roll = self._rng.random()
            if roll > final_rate:
                self._logger.debug(
                    (
                        "Sticker.Collect sampled-out: user_id=%s group_id=%s "
                        "base_rate=%.3f weight=%.3f mood=%.1f mood_adj=%.3f final_rate=%.3f roll=%.3f"
                    ),
                    sender_id,
                    group_id,
                    base_rate,
                    weight,
                    mood,
                    mood_adjustment,
                    final_rate,
                    roll,
                )
                return

        for segment in segments:
            try:
                await self._collect_segment(
                    segment=segment,
                    group_id=group_id,
                    sender_id=sender_id,
                    speaker=speaker,
                    mood=mood,
                    message_id=message_id,
                    is_admin=is_admin,
                )
            except Exception:
                self._logger.exception(
                    "Sticker.Collect failed: user_id=%s group_id=%s message_id=%s",
                    sender_id,
                    group_id,
                    message_id,
                )

        if is_admin:
            self._logger.info(
                "Sticker.Collect forced by admin: user_id=%s group_id=%s segments=%s",
                sender_id,
                group_id,
                len(segments),
            )
            return
        self._logger.info(
            (
                "Sticker.Collect sampled-in: user_id=%s group_id=%s segments=%s "
                "final_rate=%.3f roll=%.3f"
            ),
            sender_id,
            group_id,
            len(segments),
            final_rate,
            roll,
        )

    async def list_local_files(self) -> list[dict[str, Any]]:
        root = self.local_dir
        metadata_by_name: dict[str, dict[str, Any]] = {}

        async with self._index_lock:
            index = self._load_index_nolock()
        for item in index.get("items", []):
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file_name", "")).strip()
            if not file_name:
                continue
            metadata_by_name[file_name] = item

        files: list[dict[str, Any]] = []
        for path in root.iterdir():
            if not path.is_file():
                continue
            if path.name == self.INDEX_FILE_NAME:
                continue
            stat = path.stat()
            info = metadata_by_name.get(path.name, {})
            info_with_file = dict(info)
            info_with_file.setdefault("file_name", path.name)
            is_sticker = self.is_sticker_item(info_with_file)
            media_kind = "sticker" if is_sticker else "image"
            files.append(
                {
                    "file_name": path.name,
                    "size": int(stat.st_size),
                    "updated_at": int(stat.st_mtime),
                    "md5": str(info.get("md5") or path.stem),
                    "summary": self._normalize_summary_text(info.get("summary", "")),
                    "media_kind": media_kind,
                    "is_sticker": is_sticker,
                    "tags": self._normalize_tags(info.get("tags", [])),
                    "review_status": str(info.get("review_status", "")).strip().lower(),
                    "review_reason": str(info.get("review_reason", "")).strip(),
                    "review_source": str(info.get("review_source", "")).strip(),
                    "reviewed_at": str(info.get("reviewed_at", "")).strip(),
                    "sender_id": self._to_int(info.get("sender_id")),
                    "group_id": self._to_int(info.get("group_id")),
                    "source_message_id": self._to_int(info.get("source_message_id")),
                    "created_at": str(info.get("created_at", "")),
                },
            )
        files.sort(key=lambda row: int(row.get("updated_at") or 0), reverse=True)
        return files

    async def delete_local_file(self, file_name: str) -> bool:
        clean_name = self._sanitize_file_name(file_name)
        target = self.resolve_local_file_path(clean_name)

        deleted = False
        if target.exists() and target.is_file():
            target.unlink()
            deleted = True

        async with self._index_lock:
            index = self._load_index_nolock()
            items = index.get("items", [])
            if not isinstance(items, list):
                items = []
            remaining = [
                item
                for item in items
                if not (isinstance(item, dict) and str(item.get("file_name", "")).strip() == clean_name)
            ]
            if len(remaining) != len(items):
                index["items"] = remaining
                self._save_index_nolock(index)
                deleted = True

        return deleted

    async def search(
        self,
        keyword: str,
        limit: int = 8,
        storage_mode: str | None = "local",
    ) -> list[dict[str, Any]]:
        clean_keyword = self._normalize_summary_text(keyword).lower()
        try:
            parsed_limit = int(limit)
        except (TypeError, ValueError):
            parsed_limit = 8
        max_items = max(1, min(200, parsed_limit))
        wanted_mode = self._normalize_search_storage_mode(storage_mode)

        async with self._index_lock:
            index = self._load_index_nolock()
        items = index.get("items", [])
        if not isinstance(items, list):
            return []

        scored: list[dict[str, Any]] = []
        for row in items:
            if not isinstance(row, dict):
                continue
            row_mode = self._normalize_storage_mode(str(row.get("storage_mode", "local")))
            if wanted_mode != "all" and row_mode != wanted_mode:
                continue

            summary = self._normalize_summary_text(row.get("summary", ""))
            file_name = str(row.get("file_name", "")).strip()
            md5_hex = str(row.get("md5", "")).strip()
            source_file = str(row.get("source_file", "")).strip()
            tags = self._normalize_tags(row.get("tags", []))
            haystack_parts = [
                summary.lower(),
                file_name.lower(),
                md5_hex.lower(),
                source_file.lower(),
                " ".join(tags).lower(),
            ]
            if clean_keyword and not any(clean_keyword in part for part in haystack_parts if part):
                continue

            local_exists = False
            if row_mode == "local" and file_name:
                try:
                    local_exists = self.resolve_local_file_path(file_name).is_file()
                except ValueError:
                    local_exists = False

            score = 0
            if clean_keyword:
                if clean_keyword in summary.lower():
                    score += 4
                if clean_keyword in file_name.lower():
                    score += 2
                if clean_keyword in source_file.lower():
                    score += 1
                if clean_keyword in md5_hex.lower():
                    score += 1
            if row_mode == "local":
                score += 3
            if local_exists:
                score += 2
            is_sticker = self.is_sticker_item(row)
            media_kind = "sticker" if is_sticker else "image"

            scored.append(
                {
                    "id": str(row.get("id") or row.get("md5") or ""),
                    "summary": summary,
                    "file_name": file_name,
                    "storage_mode": row_mode,
                    "media_kind": media_kind,
                    "is_sticker": is_sticker,
                    "tags": tags,
                    "review_status": str(row.get("review_status", "")).strip().lower(),
                    "review_reason": str(row.get("review_reason", "")).strip(),
                    "review_source": str(row.get("review_source", "")).strip(),
                    "reviewed_at": str(row.get("reviewed_at", "")).strip(),
                    "md5": md5_hex,
                    "created_at": str(row.get("created_at", "")),
                    "sender_id": self._to_int(row.get("sender_id")),
                    "group_id": self._to_int(row.get("group_id")),
                    "source_message_id": self._to_int(row.get("source_message_id")),
                    "local_exists": local_exists if row_mode == "local" else False,
                    "_score": score,
                },
            )

        scored.sort(
            key=lambda row: (
                int(row.get("_score", 0)),
                row.get("created_at") or "",
            ),
            reverse=True,
        )
        out = scored[:max_items]
        for row in out:
            row.pop("_score", None)
        return out

    async def get_sticker(self, sticker_id: str) -> dict[str, Any] | None:
        target = str(sticker_id or "").strip()
        if not target:
            return None
        async with self._index_lock:
            index = self._load_index_nolock()
        items = index.get("items", [])
        if not isinstance(items, list):
            return None
        for row in items:
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id") or row.get("md5") or "").strip()
            if row_id == target:
                row = dict(row)
                row["summary"] = self._normalize_summary_text(row.get("summary", ""))
                row["storage_mode"] = self._normalize_storage_mode(str(row.get("storage_mode", "local")))
                row["is_sticker"] = self.is_sticker_item(row)
                row["media_kind"] = "sticker" if bool(row["is_sticker"]) else "image"
                row["tags"] = self._normalize_tags(row.get("tags", []))
                row["review_status"] = str(row.get("review_status", "")).strip().lower()
                row["review_reason"] = str(row.get("review_reason", "")).strip()
                row["review_source"] = str(row.get("review_source", "")).strip()
                row["reviewed_at"] = str(row.get("reviewed_at", "")).strip()
                return row
        return None

    def build_local_sticker_cq(self, file_name: str) -> str:
        path = self.resolve_local_file_path(file_name)
        return f"[CQ:image,file=file:///{path.as_posix()}]"

    def is_sticker_item(self, item: Mapping[str, Any]) -> bool:
        media_kind = str(item.get("media_kind", "")).strip().lower()
        if media_kind in {"sticker", "image"}:
            return media_kind == "sticker"
        return self._infer_is_sticker_item(item)

    def _infer_is_sticker_item(self, item: Mapping[str, Any]) -> bool:
        segment_type = str(item.get("segment_type", "")).strip().lower()
        sub_type = self._to_int(item.get("sub_type"))
        summary = str(item.get("summary", ""))
        source_url = str(item.get("source_url", ""))
        source_file = str(item.get("source_file", ""))
        file_name = str(item.get("file_name", ""))
        return self._is_sticker_segment(
            segment_type=segment_type,
            sub_type=sub_type,
            summary=summary,
            url=source_url,
            file_ref=source_file or file_name,
        )

    async def allow_sticker_for_reply(
        self,
        *,
        item: Mapping[str, Any],
        query: str,
        mood: float,
    ) -> PersonaFilterDecision:
        review_status = str(item.get("review_status", "")).strip().lower()
        if review_status == "rejected":
            return PersonaFilterDecision(allowed=False, reason="pre_review_rejected", source="index")
        filter_text = self._build_reply_filter_text(item=item, query=query)
        return await self._run_persona_filter(
            filter_text=filter_text,
            mood=mood,
            skip_filter=False,
        )

    def derive_sticker_tags(self, item: Mapping[str, Any]) -> list[str]:
        tags: list[str] = []
        is_sticker = self.is_sticker_item(item)
        tags.append("表情包" if is_sticker else "图片")

        file_name = str(item.get("file_name", "")).strip().lower()
        suffix = Path(file_name).suffix.lower()
        if suffix in {".gif", ".apng"}:
            tags.append("动图")
        elif suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            tags.append("静态")

        source_url = str(item.get("source_url", "")).strip().lower()
        if "multimedia.nt.qq.com.cn/download" in source_url:
            tags.append("qqnt")
        if "gxh.vip.qq.com/club/item/parcel/item/" in source_url:
            tags.append("qq商城")

        summary = self._normalize_summary_text(item.get("summary", "")).lower()
        source_file = str(item.get("source_file", "")).strip().lower()
        file_tokens = str(file_name or source_file).lower()
        haystack = f"{summary} {file_tokens}"
        for tag, tokens in self._TAG_TOKEN_TABLE:
            if any(token in haystack for token in tokens):
                tags.append(tag)

        return self._normalize_tags(tags)

    def review_sticker_item(self, item: Mapping[str, Any]) -> dict[str, str]:
        filter_text = self._build_reply_filter_text(item=item, query="")
        keywords = self._normalize_keywords(self._cfg.sticker.filter_keywords)
        lowered = filter_text.lower()
        for keyword in keywords:
            clean = str(keyword or "").strip().lower()
            if clean and clean in lowered:
                return {
                    "review_status": "rejected",
                    "review_reason": f"keyword:{keyword}",
                    "review_source": "keyword",
                }
        if self.is_sticker_item(item):
            return {
                "review_status": "approved",
                "review_reason": "heuristic_allow",
                "review_source": "heuristic",
            }
        return {
            "review_status": "pending",
            "review_reason": "not_sticker",
            "review_source": "classifier",
        }

    async def _collect_segment(
        self,
        *,
        segment: StickerSegment,
        group_id: int,
        sender_id: int | None,
        speaker: str,
        mood: float,
        message_id: int | None,
        is_admin: bool,
    ) -> None:
        mode = self._normalize_storage_mode(self._cfg.sticker.storage_mode)
        filter_text = self._build_filter_text(segment, speaker=speaker)
        summary = self._normalize_summary_text(segment.summary)
        if not summary:
            summary = self._normalize_summary_text(self._extract_file_name(segment.url, segment.file))
        if not summary:
            summary = self._normalize_summary_text(filter_text)

        decision = await self._run_persona_filter(
            filter_text=filter_text,
            mood=mood,
            skip_filter=False,
        )
        if not decision.allowed:
            self._logger.info(
                (
                    "Sticker.Collect rejected: source=%s reason=%s user_id=%s "
                    "group_id=%s segment_type=%s"
                ),
                decision.source,
                decision.reason,
                sender_id,
                group_id,
                segment.segment_type,
            )
            return

        if mode == "cloud":
            ok, note = await self._collect_cloud(
                segment=segment,
                group_id=group_id,
                sender_id=sender_id,
                message_id=message_id,
                summary=summary,
                decision=decision,
            )
        else:
            ok, note = await self._collect_local(
                segment=segment,
                group_id=group_id,
                sender_id=sender_id,
                message_id=message_id,
                summary=summary,
                storage_mode=mode,
                decision=decision,
            )

        if ok:
            self._logger.info(
                "Sticker.Collect success: mode=%s user_id=%s group_id=%s note=%s",
                mode,
                sender_id,
                group_id,
                note,
            )
            return
        self._logger.debug(
            "Sticker.Collect skipped: mode=%s user_id=%s group_id=%s reason=%s",
            mode,
            sender_id,
            group_id,
            note,
        )

    async def _collect_local(
        self,
        *,
        segment: StickerSegment,
        group_id: int,
        sender_id: int | None,
        message_id: int | None,
        summary: str,
        storage_mode: str,
        decision: PersonaFilterDecision,
    ) -> tuple[bool, str]:
        image_data, content_type = await self._download_image_bytes(segment)
        if not image_data:
            return False, "empty_image"

        md5_hex = hashlib.md5(image_data).hexdigest()
        suffix = self._guess_suffix(segment=segment, content_type=content_type)
        file_name = f"{md5_hex}{suffix}"
        target = self.local_dir / file_name

        async with self._index_lock:
            index = self._load_index_nolock()
            items = index.get("items", [])
            if not isinstance(items, list):
                items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("md5", "")).strip() == md5_hex:
                    return False, "duplicate_md5"

            if not target.exists():
                target.write_bytes(image_data)

            entry = {
                "id": md5_hex,
                "md5": md5_hex,
                "file_name": file_name,
                "summary": self._normalize_summary_text(summary),
                "media_kind": "sticker",
                "storage_mode": storage_mode,
                "segment_type": str(segment.segment_type or "").strip().lower(),
                "sub_type": int(segment.sub_type),
                "sender_id": sender_id,
                "group_id": group_id,
                "source_url": segment.url,
                "source_file": segment.file,
                "source_message_id": message_id,
                "review_status": "approved",
                "review_reason": str(decision.reason or "allow"),
                "review_source": str(decision.source or "filter"),
                "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            entry["tags"] = self.derive_sticker_tags(entry)
            items.append(entry)
            index["items"] = items
            self._save_index_nolock(index)
        return True, f"saved:{file_name}"

    async def _collect_cloud(
        self,
        *,
        segment: StickerSegment,
        group_id: int,
        sender_id: int | None,
        message_id: int | None,
        summary: str,
        decision: PersonaFilterDecision,
    ) -> tuple[bool, str]:
        if message_id is None or message_id <= 0:
            return False, "missing_message_id"

        dedupe_key = f"{message_id}|{segment.url}|{segment.file}"
        item_id = hashlib.md5(dedupe_key.encode("utf-8")).hexdigest()

        async with self._index_lock:
            index = self._load_index_nolock()
            items = index.get("items", [])
            if not isinstance(items, list):
                items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id", "")).strip() == item_id:
                    return False, "duplicate_cloud_item"

        actions = [str(item).strip() for item in self._cfg.sticker.cloud_actions if str(item).strip()]
        if not actions:
            actions = list(self._DEFAULT_CLOUD_ACTIONS)

        used_action = ""
        last_error = ""
        payloads = [
            {"message_id": int(message_id)},
            {"id": int(message_id)},
            {"message_id": int(message_id), "group_id": int(group_id)},
            {"id": int(message_id), "group_id": int(group_id)},
        ]
        for action in actions:
            for params in payloads:
                try:
                    await self._bot.call_action(action, params)
                    used_action = action
                    break
                except Exception as exc:
                    last_error = str(exc)
                    continue
            if used_action:
                break

        if not used_action:
            return False, f"cloud_action_failed:{last_error or 'no_action'}"

        async with self._index_lock:
            index = self._load_index_nolock()
            items = index.get("items", [])
            if not isinstance(items, list):
                items = []
            entry = {
                "id": item_id,
                "md5": "",
                "file_name": "",
                "summary": self._normalize_summary_text(summary),
                "media_kind": "sticker",
                "storage_mode": "cloud",
                "segment_type": str(segment.segment_type or "").strip().lower(),
                "sub_type": int(segment.sub_type),
                "sender_id": sender_id,
                "group_id": group_id,
                "source_url": segment.url,
                "source_file": segment.file,
                "source_message_id": message_id,
                "cloud_action": used_action,
                "review_status": "approved",
                "review_reason": str(decision.reason or "allow"),
                "review_source": str(decision.source or "filter"),
                "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            entry["tags"] = self.derive_sticker_tags(entry)
            items.append(entry)
            index["items"] = items
            self._save_index_nolock(index)
        return True, f"cloud_action:{used_action}"

    async def _run_persona_filter(
        self,
        *,
        filter_text: str,
        mood: float,
        skip_filter: bool,
        force_llm: bool = False,
    ) -> PersonaFilterDecision:
        if skip_filter:
            return PersonaFilterDecision(allowed=True, reason="admin_bypass", source="admin")

        cfg = self._cfg.sticker
        if not cfg.enable_persona_filter:
            return PersonaFilterDecision(allowed=True, reason="disabled", source="config")

        keywords = self._normalize_keywords(cfg.filter_keywords)
        lowered = filter_text.lower()
        for keyword in keywords:
            if keyword.lower() and keyword.lower() in lowered:
                return PersonaFilterDecision(allowed=False, reason=f"keyword:{keyword}", source="keyword")

        if not cfg.llm_filter_enabled or self._llm is None:
            return PersonaFilterDecision(allowed=True, reason="llm_disabled", source="heuristic")

        if not force_llm:
            mood_threshold = float(cfg.llm_filter_mood_threshold or 0.0)
            if float(mood) <= mood_threshold:
                return PersonaFilterDecision(allowed=True, reason="mood_below_threshold", source="heuristic")

            llm_probability = self._clamp_probability(cfg.llm_filter_probability)
            if llm_probability <= 0.0:
                return PersonaFilterDecision(allowed=True, reason="llm_probability_zero", source="heuristic")
            if self._rng.random() > llm_probability:
                return PersonaFilterDecision(allowed=True, reason="llm_skipped_by_sampling", source="heuristic")

        cache_key = self._build_persona_cache_key(filter_text=filter_text, keywords=keywords)
        cached = await self._get_cached_persona_decision(cache_key)
        if cached is not None:
            self._logger.debug("Sticker.PersonaCache hit: key=%s", cache_key[:12])
            return cached

        self._logger.debug("Sticker.PersonaCache miss: key=%s", cache_key[:12])
        decision = await self._llm_persona_judge(filter_text, keywords)
        await self._set_cached_persona_decision(cache_key, decision)
        return decision

    async def _llm_persona_judge(self, filter_text: str, keywords: list[str]) -> PersonaFilterDecision:
        if self._llm is None:
            return PersonaFilterDecision(allowed=True, reason="no_llm", source="heuristic")

        max_tokens = self._cfg.sticker.llm_filter_max_tokens
        max_tokens = max(1, min(10, int(max_tokens if max_tokens else 10)))
        messages = [
            {
                "role": "system",
                "content": (
                    "你是纸月的表情包过滤器。"
                    "纸月风格是冷淡、静谧、克制。"
                    "你只能输出 allow 或 reject，不要解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "如果内容明显浮夸、低俗、吵闹、恶臭或攻击性过强则输出 reject，否则输出 allow。\n"
                    f"重点屏蔽词：{', '.join(keywords) if keywords else '无'}\n"
                    f"待判定文本：{filter_text[:300]}"
                ),
            },
        ]

        reply = await self._llm.generate_from_messages(
            messages=messages,
            extra_fields={"max_tokens": max_tokens},
        )
        normalized = str(reply or "").strip().lower()
        if "reject" in normalized or "拒绝" in normalized:
            return PersonaFilterDecision(allowed=False, reason=f"llm:{normalized[:20]}", source="llm")
        return PersonaFilterDecision(allowed=True, reason=f"llm:{normalized[:20] or 'allow'}", source="llm")

    async def _get_cached_persona_decision(self, key: str) -> PersonaFilterDecision | None:
        now_ts = time.time()
        async with self._persona_cache_lock:
            cached = self._persona_cache.get(key)
            if cached is None:
                return None
            expires_at, decision = cached
            if expires_at < now_ts:
                self._persona_cache.pop(key, None)
                return None
            return decision

    async def _set_cached_persona_decision(self, key: str, decision: PersonaFilterDecision) -> None:
        ttl_seconds = max(60, int(self._cfg.sticker.llm_filter_cache_ttl_seconds or 86400))
        expires_at = time.time() + ttl_seconds
        async with self._persona_cache_lock:
            self._persona_cache[key] = (expires_at, decision)
            if len(self._persona_cache) <= self.MAX_PERSONA_CACHE_ITEMS:
                return
            overflow = len(self._persona_cache) - self.MAX_PERSONA_CACHE_ITEMS
            for cache_key in list(self._persona_cache.keys())[:overflow]:
                self._persona_cache.pop(cache_key, None)

    def _build_persona_cache_key(self, *, filter_text: str, keywords: list[str]) -> str:
        stable_text = self._normalize_filter_text_for_cache(filter_text)
        stable_keywords = self._normalize_keywords_for_cache(keywords)
        payload = f"{stable_text}\n{'|'.join(stable_keywords)}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    @classmethod
    def _normalize_filter_text_for_cache(cls, filter_text: str) -> str:
        raw = str(filter_text or "").strip().lower()
        if not raw:
            return ""
        tokens = re.split(r"\s+", raw)
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            clean = cls._normalize_cache_token(token)
            if not clean:
                continue
            if clean.startswith("sender:"):
                continue
            if cls._is_volatile_cache_token(clean):
                continue
            if clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        if deduped:
            return " ".join(deduped[:24])
        return ""

    @classmethod
    def _normalize_cache_token(cls, token: str) -> str:
        clean = str(token or "").strip().lower()
        if not clean:
            return ""
        if clean.startswith(("http://", "https://", "file://")):
            return ""
        clean = cls._CACHE_TOKEN_EDGE_PATTERN.sub("", clean)
        if not clean:
            return ""
        return clean[:80]

    @classmethod
    def _is_volatile_cache_token(cls, token: str) -> bool:
        compact = token.replace("-", "").replace("_", "").replace(".", "")
        if len(compact) < 6:
            return False
        if cls._VOLATILE_HEX_TOKEN_PATTERN.fullmatch(compact):
            return True
        if cls._VOLATILE_DIGIT_TOKEN_PATTERN.fullmatch(compact):
            return True
        if cls._VOLATILE_MIXED_ID_TOKEN_PATTERN.fullmatch(compact):
            return True
        return False

    def _normalize_keywords_for_cache(self, keywords: list[str]) -> list[str]:
        normalized = self._normalize_keywords(keywords)
        lowered: list[str] = []
        seen: set[str] = set()
        for keyword in normalized:
            clean = str(keyword or "").strip().lower()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            lowered.append(clean)
        lowered.sort()
        return lowered

    async def _download_image_bytes(self, segment: StickerSegment) -> tuple[bytes, str]:
        if segment.url.startswith("http://") or segment.url.startswith("https://"):
            timeout = aiohttp.ClientTimeout(total=max(3.0, float(self._cfg.sticker.download_timeout_sec or 12.0)))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(segment.url) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"download_status={resp.status}")
                    return await resp.read(), str(resp.headers.get("content-type", ""))

        for candidate in (segment.url, segment.file):
            data = self._read_local_candidate(candidate)
            if data is not None:
                return data, ""

        raise RuntimeError("sticker_source_unavailable")

    def _read_local_candidate(self, value: str) -> bytes | None:
        raw = str(value or "").strip()
        if not raw:
            return None

        path = Path(raw)
        if raw.startswith("file://"):
            parsed = urlparse(raw)
            local_path = unquote(parsed.path or "")
            if os.name == "nt" and re.match(r"^/[a-zA-Z]:/", local_path):
                local_path = local_path[1:]
            path = Path(local_path)

        if not path.is_absolute():
            path = (self._project_root / path).resolve()
        if not path.exists() or not path.is_file():
            return None
        return path.read_bytes()

    def _build_filter_text(self, segment: StickerSegment, *, speaker: str) -> str:
        file_name = self._extract_file_name(segment.url, segment.file)
        tokens: list[str] = []
        if file_name:
            tokens.append(file_name)
            tokens.extend(self._extract_name_tokens(file_name))
        if segment.summary:
            tokens.append(self._normalize_summary_text(segment.summary))
        if speaker:
            tokens.append(f"sender:{speaker}")
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            clean = str(token or "").strip()
            if not clean:
                continue
            lowered = clean.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(clean)
        return " ".join(deduped)[:400]

    def _build_reply_filter_text(self, *, item: Mapping[str, Any], query: str) -> str:
        tokens: list[str] = []
        summary = self._normalize_summary_text(item.get("summary", ""))
        if summary:
            tokens.append(summary)

        file_name = str(item.get("file_name", "")).strip()
        source_file = str(item.get("source_file", "")).strip()
        source_url = str(item.get("source_url", "")).strip()
        for raw_name in (file_name, source_file, self._extract_file_name(source_url, "")):
            clean = str(raw_name or "").strip()
            if not clean:
                continue
            tokens.append(clean)
            tokens.extend(self._extract_name_tokens(clean))

        clean_query = self._normalize_summary_text(query)
        if clean_query:
            tokens.append(f"query:{clean_query}")

        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            clean = str(token or "").strip()
            if not clean:
                continue
            lowered = clean.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(clean)
        return " ".join(deduped)[:400]

    @classmethod
    def _is_sticker_segment(
        cls,
        *,
        segment_type: str,
        sub_type: int | None,
        summary: str,
        url: str,
        file_ref: str,
    ) -> bool:
        seg_type = str(segment_type or "").strip().lower()
        if seg_type == "mface":
            return True
        if sub_type is not None and int(sub_type) > 0:
            return True

        summary_text = cls._normalize_summary_text(summary).lower()
        if summary_text:
            if any(hint in summary_text for hint in cls._STICKER_SUMMARY_HINTS):
                return True
            if re.fullmatch(r"\[[^\]]{1,20}\]", summary_text):
                return True

        source_text = f"{str(url or '').lower()} {str(file_ref or '').lower()}"
        if any(hint in source_text for hint in cls._STICKER_SOURCE_HINTS):
            return True
        return False

    @classmethod
    def _extract_name_tokens(cls, file_name: str) -> list[str]:
        stem = Path(file_name).stem
        parts = re.split(r"[_\-\s\.\|]+", stem)
        tokens: list[str] = []
        for part in parts:
            clean = part.strip()
            if not clean:
                continue
            if len(clean) < 2:
                continue
            tokens.append(clean)
        return tokens

    @staticmethod
    def _extract_file_name(url: str, file_ref: str) -> str:
        for raw in (file_ref, url):
            clean = str(raw or "").strip()
            if not clean:
                continue
            if clean.startswith("http://") or clean.startswith("https://") or clean.startswith("file://"):
                parsed = urlparse(clean)
                name = Path(unquote(parsed.path or "")).name
            else:
                name = Path(clean).name
            if name:
                return name
        return ""

    def _guess_suffix(self, *, segment: StickerSegment, content_type: str) -> str:
        for candidate in (segment.file, segment.url):
            name = self._extract_file_name(candidate, "")
            suffix = Path(name).suffix.lower()
            if suffix and re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                return suffix
        content_main = str(content_type or "").split(";", 1)[0].strip().lower()
        guessed = mimetypes.guess_extension(content_main) if content_main else None
        if guessed and re.fullmatch(r"\.[a-z0-9]{1,8}", guessed):
            return guessed
        return ".png"

    def _extract_sticker_segments(self, message: Mapping[str, Any]) -> list[StickerSegment]:
        raw_segments = message.get("message")
        if isinstance(raw_segments, list):
            parsed = self._extract_from_segment_list(raw_segments)
            if parsed:
                return parsed

        raw_message = str(message.get("raw_message") or "")
        if raw_message:
            return self._extract_from_cq(raw_message)
        return []

    @classmethod
    def _extract_from_segment_list(cls, segments: list[Any]) -> list[StickerSegment]:
        out: list[StickerSegment] = []
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            seg_type = str(segment.get("type", "")).strip().lower()
            if seg_type not in {"image", "mface"}:
                continue
            data = segment.get("data")
            if not isinstance(data, Mapping):
                continue
            url = str(data.get("url", "")).strip()
            file_ref = str(data.get("file", "")).strip()
            if not url and not file_ref:
                continue
            summary = cls._normalize_summary_text(data.get("summary", ""))
            sub_type = cls._to_int(data.get("sub_type"))
            if sub_type is None:
                sub_type = cls._to_int(data.get("subType"))
            if sub_type is None:
                sub_type = 1 if seg_type == "mface" else 0
            if not cls._is_sticker_segment(
                segment_type=seg_type,
                sub_type=sub_type,
                summary=summary,
                url=url,
                file_ref=file_ref,
            ):
                continue
            out.append(
                StickerSegment(
                    segment_type=seg_type,
                    url=url,
                    file=file_ref,
                    summary=summary,
                    sub_type=sub_type,
                ),
            )
        return out

    @classmethod
    def _extract_from_cq(cls, raw_message: str) -> list[StickerSegment]:
        out: list[StickerSegment] = []
        for match in cls._CQ_IMAGE_PATTERN.finditer(raw_message):
            data = cls._parse_cq_data(match.group(1))
            url = str(data.get("url", "")).strip()
            file_ref = str(data.get("file", "")).strip()
            if not url and not file_ref:
                continue
            summary = cls._normalize_summary_text(data.get("summary", ""))
            sub_type = cls._to_int(data.get("sub_type")) or 0
            if not cls._is_sticker_segment(
                segment_type="image",
                sub_type=sub_type,
                summary=summary,
                url=url,
                file_ref=file_ref,
            ):
                continue
            out.append(
                StickerSegment(
                    segment_type="image",
                    url=url,
                    file=file_ref,
                    summary=summary,
                    sub_type=sub_type,
                ),
            )
        return out

    @staticmethod
    def _parse_cq_data(raw_data: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in str(raw_data or "").split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            out[key.strip()] = html.unescape(value.strip())
        return out

    def _resolve_user_weight(self, sender_id: int | None) -> float:
        if sender_id is None:
            return 1.0
        key = str(sender_id)
        raw = self._cfg.sticker.user_weights.get(key, 1.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0

    def _index_path(self) -> Path:
        return self.local_dir / self.INDEX_FILE_NAME

    def _load_index_nolock(self) -> dict[str, Any]:
        path = self._index_path()
        if not path.exists():
            return {"version": self.INDEX_VERSION, "items": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": self.INDEX_VERSION, "items": []}
        if not isinstance(payload, dict):
            return {"version": self.INDEX_VERSION, "items": []}
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []

        normalized_items: list[dict[str, Any]] = []
        changed = False
        for raw in raw_items:
            if not isinstance(raw, dict):
                changed = True
                continue
            normalized, item_changed = self._normalize_index_item(raw)
            normalized_items.append(normalized)
            if item_changed:
                changed = True

        raw_version = int(payload.get("version") or self.INDEX_VERSION)
        if raw_version != self.INDEX_VERSION:
            changed = True

        normalized_payload = {
            "version": self.INDEX_VERSION,
            "items": normalized_items,
        }
        if changed:
            self._save_index_nolock(normalized_payload)
        return normalized_payload

    def _normalize_index_item(self, raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        row = dict(raw)
        changed = False

        summary = self._normalize_summary_text(row.get("summary", ""))
        if summary != str(row.get("summary", "")).strip():
            changed = True
        row["summary"] = summary

        row["storage_mode"] = self._normalize_storage_mode(str(row.get("storage_mode", "local")))
        is_sticker = self._infer_is_sticker_item(row)
        media_kind = "sticker" if is_sticker else "image"
        if str(row.get("media_kind", "")).strip().lower() != media_kind:
            changed = True
        row["media_kind"] = media_kind
        if bool(row.get("is_sticker")) != is_sticker:
            changed = True
        row["is_sticker"] = is_sticker

        tags = self.derive_sticker_tags(row)
        old_tags = self._normalize_tags(row.get("tags", []))
        if old_tags != tags:
            changed = True
        row["tags"] = tags

        current_status = str(row.get("review_status", "")).strip().lower()
        current_reason = str(row.get("review_reason", "")).strip()
        current_source = str(row.get("review_source", "")).strip()
        has_review = bool(current_status and current_reason and current_source)
        if has_review:
            normalized_status = current_status if current_status in {"approved", "rejected", "pending"} else "pending"
            if normalized_status != current_status:
                changed = True
            row["review_status"] = normalized_status
            row["review_reason"] = current_reason
            row["review_source"] = current_source
        else:
            reviewed = self.review_sticker_item(row)
            for key, value in reviewed.items():
                if str(row.get(key, "")).strip().lower() != str(value).strip().lower():
                    changed = True
                row[key] = value

        reviewed_at = str(row.get("reviewed_at", "")).strip()
        if not reviewed_at:
            reviewed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            changed = True
        row["reviewed_at"] = reviewed_at
        return row, changed

    def _save_index_nolock(self, payload: dict[str, Any]) -> None:
        path = self._index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=2),
            encoding="utf-8",
        )
        temp.replace(path)

    @staticmethod
    def _normalize_storage_mode(raw_mode: str) -> str:
        mode = str(raw_mode or "").strip().lower()
        return mode if mode in {"local", "cloud"} else "local"

    @staticmethod
    def _normalize_search_storage_mode(raw_mode: str | None) -> str:
        mode = str(raw_mode or "").strip().lower()
        if mode in {"all", "*"}:
            return "all"
        if mode in {"local", "cloud"}:
            return mode
        return "local"

    @staticmethod
    def _normalize_summary_text(raw_summary: Any) -> str:
        clean = str(raw_summary or "").strip()
        if not clean:
            return ""
        clean = html.unescape(clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean[:240]

    @staticmethod
    def _normalize_keywords(raw_keywords: Any) -> list[str]:
        if isinstance(raw_keywords, str):
            candidates = re.split(r"[,\n\r;|]+", raw_keywords)
        elif isinstance(raw_keywords, list):
            candidates = [str(item) for item in raw_keywords]
        else:
            candidates = []
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            clean = str(item or "").strip()
            if not clean:
                continue
            lowered = clean.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(clean)
        return out

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> list[str]:
        if isinstance(raw_tags, str):
            candidates = re.split(r"[,\n\r;|]+", raw_tags)
        elif isinstance(raw_tags, list):
            candidates = [str(item) for item in raw_tags]
        else:
            candidates = []
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            clean = str(item or "").strip()
            if not clean:
                continue
            lowered = clean.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(clean[:20])
        return out[:12]

    @staticmethod
    def _normalize_user_weights(raw_weights: Any) -> dict[str, float]:
        if not isinstance(raw_weights, Mapping):
            return {}
        out: dict[str, float] = {}
        for raw_key, raw_value in raw_weights.items():
            key = str(raw_key).strip()
            if not key:
                continue
            try:
                weight = float(raw_value)
            except (TypeError, ValueError):
                continue
            out[key] = weight
        return dict(sorted(out.items(), key=lambda item: item[0]))

    @staticmethod
    def _sanitize_file_name(file_name: str) -> str:
        clean = Path(str(file_name or "").strip()).name
        if not clean or clean in {".", ".."}:
            raise ValueError("invalid sticker file name")
        return clean

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp_probability(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

