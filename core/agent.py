from __future__ import annotations

import asyncio
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from adapters.llm.chat import ChatLLMAdapter
from adapters.onebot.client import OneBotClient
from core.memory import ConversationMessage, MessageHistory
from core.tarot import TarotKnowledgeBase
from internal.config.schema import Config, GroupConfig
from internal.jargon import JargonManager, StyleClassification
from internal.jargon.jargon_engine import JargonEvolutionEngine, JargonLexiconStore
from internal.learning.user_profiling import UserProfileStore, UserProfilingEngine
from internal.logger import get_logger
from internal.memory import MemoryManager, MessageLog
from internal.persona import MoodInfo, PersonalityManager, PromptContext, StatusEngine, StatusSnapshot
from internal.sticker import StickerCollector


_PROMPT_INLINE_SPACE_PATTERN = re.compile(r"[ \t\f\v]+")
_PROMPT_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_PROMPT_PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "、": ",",
        "…": "...",
        "—": "-",
        "～": "~",
    }
)


def _normalize_prompt_input(text: Any) -> str:
    clean = unicodedata.normalize("NFKC", str(text or ""))
    clean = clean.replace("\u00A0", " ").replace("\u3000", " ")
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.translate(_PROMPT_PUNCT_TRANSLATION)
    clean = _PROMPT_INLINE_SPACE_PATTERN.sub(" ", clean)
    clean = re.sub(r"[ \t]*\n[ \t]*", "\n", clean)
    clean = _PROMPT_MULTI_NEWLINE_PATTERN.sub("\n\n", clean)
    return clean.strip()


@dataclass(slots=True)
class ThinkContext:
    session_id: str
    message_type: str
    user_id: int | None
    group_id: int | None
    speaker: str
    source_text: str
    is_master: bool
    mood: MoodInfo
    style: StyleClassification
    prompt_context: PromptContext
    llm_messages: list[dict[str, str]]
    status: StatusSnapshot | None = None
    social_background: str = ""
    style_source: str = "heuristic"
    planned_tools: list[str] | None = None


@dataclass(slots=True)
class ContextBuildResult:
    mood: MoodInfo
    style: StyleClassification
    prompt_context: PromptContext
    llm_messages: list[dict[str, str]]


@dataclass(slots=True)
class DebounceEntry:
    packet: dict[str, Any]
    generation: int
    merged_count: int
    mentioned: bool
    explicit_at: bool
    timer_task: asyncio.Task[None]


@dataclass(slots=True, frozen=True)
class SenderIdentity:
    nickname: str = ""
    group_card: str = ""


class ContextBuilder:
    def __init__(
        self,
        *,
        cfg: Config,
        history: MessageHistory,
        personality: PersonalityManager,
        jargon_mgr: JargonManager,
    ) -> None:
        self._cfg = cfg
        self._history = history
        self._personality = personality
        self._jargon_mgr = jargon_mgr

    async def build(
        self,
        *,
        session_id: str,
        message: dict[str, Any],
        speaker: str,
        user_id: int | None,
        is_master: bool,
        mentioned_in_window: bool = False,
        history_background: list[str] | None = None,
        related_knowledge: list[str] | None = None,
        social_background: str = "",
    ) -> ContextBuildResult:
        mood = self._personality.get_current_mood()
        style = self._jargon_mgr.classify_style(
            mood.valence,
            mood.energy,
            speaker_is_master=is_master,
        )

        current_text = _normalize_prompt_input(message.get("text", ""))
        jargon_matches = await self._jargon_mgr.match(current_text)

        context_messages = self._history.get_structured_messages(session_id)
        chat_context_lines: list[str] = []
        for item in context_messages:
            normalized = _normalize_prompt_input(item.get("content", ""))
            if normalized:
                chat_context_lines.append(normalized)
        chat_context = "\n".join(chat_context_lines)
        if not chat_context:
            chat_context = f"[{speaker}] {current_text}"

        group_id = self._to_int(message.get("group_id"))
        prompt_ctx = PromptContext(
            group_id=group_id,
            mood_state=mood,
            jargon_matches=jargon_matches,
        )

        group_extra = ""
        if group_id is not None:
            group_cfg = self._cfg.get_group(group_id)
            if group_cfg and group_cfg.extra_prompt.strip():
                group_extra = _normalize_prompt_input(group_cfg.extra_prompt)

        summary_hint = _normalize_prompt_input(self._history.summary_prompt(session_id))
        if summary_hint:
            prompt_ctx.related_memories.append(summary_hint)
        elif self._history.should_refresh_summary(session_id):
            prompt_ctx.related_memories.append("历史上下文窗口已裁剪，暂无可用总结。")
        if history_background:
            for item in history_background[:5]:
                normalized = _normalize_prompt_input(item)
                if normalized:
                    prompt_ctx.related_memories.append(normalized)
        if related_knowledge:
            for item in related_knowledge[:5]:
                normalized = _normalize_prompt_input(item)
                if normalized:
                    prompt_ctx.cross_group_experiences.append(normalized)
        social_hint = _normalize_prompt_input(social_background)
        if social_hint:
            prompt_ctx.style_hints.append(social_hint)

        recent_people = _normalize_prompt_input(self._build_recent_people(session_id))

        # Keep system prompt as stable as possible for model-side prefix caching.
        system_prompt = self._personality.get_system_prompt(
            is_master=is_master,
            model_name=str(getattr(self._cfg.llm, "model", "") or ""),
            is_group_chat=(str(message.get("message_type", "")).strip() == "group"),
        )
        think_prompt = self._personality.get_think_prompt(
            prompt_ctx,
            chat_context,
            group_extra,
            recent_people,
        )

        if mentioned_in_window or self._personality.is_mentioned(current_text):
            think_prompt += "\n注意：有人提到了你，可能在找你说话，你可以视情况回复。\n"

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": think_prompt},
        ]
        return ContextBuildResult(
            mood=mood,
            style=style,
            prompt_context=prompt_ctx,
            llm_messages=llm_messages,
        )

    @staticmethod
    def _build_retrieval_block(
        history_background: list[str] | None,
        related_knowledge: list[str] | None,
    ) -> str:
        history_lines = [item.strip() for item in (history_background or []) if item and item.strip()]
        knowledge_lines = [item.strip() for item in (related_knowledge or []) if item and item.strip()]
        if not history_lines and not knowledge_lines:
            return ""

        history_text = "\n".join(f"- {line}" for line in history_lines) if history_lines else "无"
        knowledge_text = "\n".join(f"- {line}" for line in knowledge_lines) if knowledge_lines else "无"
        return f"## 历史背景\n{history_text}\n## 相关知识\n{knowledge_text}"

    def _build_hobbies_hint(self, jargon_matches: dict[str, str]) -> list[str]:
        hobbies: list[str] = []
        raw_hobbies = getattr(self._cfg.persona, "hobbies", None)
        if isinstance(raw_hobbies, list):
            hobbies.extend(str(item).strip() for item in raw_hobbies if str(item).strip())

        raw_interests = getattr(self._cfg.persona, "interests", None)
        if isinstance(raw_interests, list):
            hobbies.extend(str(item).strip() for item in raw_interests if str(item).strip())

        for term in list(jargon_matches.keys())[:2]:
            marker = f"当前话题包含梗：{term}"
            hobbies.append(marker)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in hobbies:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _build_recent_people(self, session_id: str, limit: int = 10) -> str:
        rows = self._history.get_recent(session_id, limit=limit)
        if not rows:
            return ""

        names: list[str] = []
        seen: set[str] = set()
        self_qq = int(getattr(self._cfg.persona, "qq", 0) or 0)
        for row in reversed(rows):
            uid = int(row.user_id or 0) if row.user_id is not None else 0
            if self_qq > 0 and uid == self_qq:
                continue
            name = str(row.speaker).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)

        names.reverse()
        return "、".join(names)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class ZhiyueAgent:
    _CQ_AT_PATTERN = re.compile(r"\[CQ:at,[^\]]*(?:qq|uid|user_id)=(\d+)[^\]]*\]", re.IGNORECASE)
    _STICKER_MARKER_PATTERN = re.compile(
        r"\[\[\s*(?:sticker|表情包)\s*[:：]\s*(?P<query>[^\]\r\n]{1,120})\s*\]\]",
        re.IGNORECASE,
    )
    _STICKER_FUNC_CALL_PATTERN = re.compile(
        r"sendSticker\s*[\(（]\s*(?P<query>[^)\）\r\n]{1,120})\s*[\)）]",
        re.IGNORECASE,
    )
    _STICKER_INTENT_PATTERN = re.compile(
        (
            r"(?:sendSticker\s*[\(（])|"
            r"(?:\[\[\s*(?:sticker|表情包)\s*[:：])|"
            r"(?:发|来|整|给|甩|丢|回|用).{0,6}(?:表情包|斗图)|"
            r"(?:表情包|斗图).{0,6}(?:发|来|整|给|甩|丢|回|用)|"
            r"(?:\bsend\b.{0,8}\bsticker\b)|"
            r"(?:\bshow\b.{0,8}\bmeme\b)"
        ),
        re.IGNORECASE,
    )
    _FAKE_STICKER_TEXT_PATTERN = re.compile(
        r"[（(]?\s*(?:发送|发|甩|丢|整|来个|来一张)[^)\n]{0,20}表情包[^)\n]*[）)]?",
        re.IGNORECASE,
    )
    _SILENCE_PLACEHOLDER_PATTERN = re.compile(
        (
            r"^\s*[（(]?\s*"
            r"(?:沉默观察|保持沉默|保持安静即可|保持安静|安静即可|先观察|继续观察|无必要回应|无需回应|无需回复|暂不回应|不作回应|没有新内容需要回应|先潜水|先潜水了|先潜水啦)"
            r"(?:\s*[，,、；;]\s*"
            r"(?:沉默观察|保持沉默|保持安静即可|保持安静|安静即可|先观察|继续观察|无必要回应|无需回应|无需回复|暂不回应|不作回应|没有新内容需要回应|先潜水|先潜水了|先潜水啦))*"
            r"\s*[）)]?\s*$"
        ),
        re.IGNORECASE,
    )
    _STAY_QUIET_PLACEHOLDER_PATTERN = re.compile(
        (
            r"^\s*(?:"
            r"stayQuiet|stay_quiet|保持沉默|保持安静即可|保持安静|安静即可|不回复|无需回复|无必要回应|"
            r"\{.*?(?:stayQuiet|stay_quiet).*?\}"
            r")\s*$"
        ),
        re.IGNORECASE | re.DOTALL,
    )
    _BRACKETED_SILENCE_PLACEHOLDER_PATTERN = re.compile(
        r"^\s*\[{1,2}\s*(?P<token>[^\[\]\r\n]{1,40})\s*\]{1,2}\s*$",
        re.IGNORECASE,
    )
    _TRAILING_PLACEHOLDER_PUNCT_PATTERN = re.compile(r"[。.!！?？~～…]+\s*$")
    _BRACKETED_SILENCE_TOKENS = {
        "quiet",
        "silence",
        "silent",
        "stayquiet",
        "noreply",
        "保持沉默",
        "保持安静",
        "保持安静即可",
        "安静即可",
        "沉默观察",
        "不回复",
        "无需回复",
        "无必要回应",
    }
    _PASSIVE_GROUP_REPLY_CUE_PATTERN = re.compile(
        r"[?？]|怎么|如何|为啥|为什么|是否|能不能|可不可以|有没有|要不要|谁|哪|几|啥|吗|呢|求|帮|请教|建议|怎么看",
        re.IGNORECASE,
    )
    _LOW_SIGNAL_GROUP_TEXT_PATTERN = re.compile(
        (
            r"^(?:"
            r"[哈呵嘿啊嗯哦呃诶欸]+|"
            r"6{1,6}|"
            r"ok+|kk+|emm+|hhh+|lol+|"
            r"收到|好的?|行吧?|可以|可|知道了|懂了|确实|是的|对的?|支持|顶|\+1|1|nb|牛逼?|卧槽|草|笑死"
            r")$"
        ),
        re.IGNORECASE,
    )
    _REPLY_LINE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)+")
    _REPLY_SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|$)")
    _LOG_STYLE_SELF_PREFIX_PATTERN = re.compile(
        r"^\s*(?:>\s*)?(?:(?:\[[^\]\r\n]+\]\s*)+(?:\(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\)\s*)?|\(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\)\s*)(?P<speaker>[^:\r\n]{1,32})\s*[:：]\s*(?P<body>.*)$",
        re.DOTALL,
    )
    _LEADING_QUOTES = ("\"", "'", "“", "‘", "「", "『")
    _LOW_ENERGY_THRESHOLD = 30.0
    _LOW_ENERGY_GROUP_BASE_REPLY_PROBABILITY = 0.001
    _LOW_ENERGY_GROUP_REPLY_ENERGY_SPAN = 0.009
    _LOW_ENERGY_DIRECT_BASE_REPLY_PROBABILITY = 0.02
    _LOW_ENERGY_DIRECT_REPLY_ENERGY_SPAN = 0.08
    _LOW_ENERGY_GROUP_COOLDOWN_SEC = 360
    _LOW_ENERGY_DIRECT_COOLDOWN_SEC = 120
    _LOW_ENERGY_REPLIES = ("嗯。", "收到。", "知道了。", "行。")
    _LOW_ENERGY_AT_ONLY_REPLY = "我累了，先休息一下。"
    _AUTO_STICKER_SEND_PROBABILITY = 0.22
    _AUTO_STICKER_GROUP_COOLDOWN_SEC = 300
    _AUTO_STICKER_DIRECT_COOLDOWN_SEC = 180
    _TAROT_DRAW_COOLDOWN_SEC = 60
    _SUPPORTED_ADMIN_ACTIONS = {"toggle_group_chat", "join_group_chat", "shutdown"}
    _ADMIN_HELP_SUBCOMMANDS = {"帮助", "列表", "list", "help", "帮助/list", "help/list"}
    _ADMIN_ACTION_LABELS = {
        "toggle_group_chat": "开关群聊",
        "join_group_chat": "加入群聊",
        "shutdown": "关闭程序",
    }

    def __init__(
        self,
        bot_client: OneBotClient,
        cfg: Config,
        llm: ChatLLMAdapter,
        *,
        config_path: str | Path | None = None,
        shutdown_handler: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.bot_client = bot_client
        self.cfg = cfg
        self.llm = llm

        self._logger = get_logger("ZhiyueAgent")
        self._started_at_utc = datetime.now(timezone.utc)
        self._history = MessageHistory(self.cfg.agent.context_window_size)

        self.personality = PersonalityManager(self.cfg.persona, self.cfg.personality)
        self.status_engine = StatusEngine(
            initial_energy=float(self.cfg.personality.energy) * 100.0,
            heartbeat_interval_sec=600,
            idle_threshold_sec=180,
            recovery_step=8.0,
            reply_cost_per_turn=2.0,
            timezone_offset_hours=self.cfg.personality.energy_timezone_offset_hours,
            active_start_hour=self.cfg.personality.energy_active_start_hour,
            active_end_hour=self.cfg.personality.energy_active_end_hour,
            active_recovery_multiplier=self.cfg.personality.energy_active_recovery_multiplier,
            active_reply_cost_multiplier=self.cfg.personality.energy_active_reply_cost_multiplier,
            rest_recovery_multiplier=self.cfg.personality.energy_rest_recovery_multiplier,
            rest_reply_cost_multiplier=self.cfg.personality.energy_rest_reply_cost_multiplier,
        )
        self.jargon_mgr = JargonManager(self.cfg.jargon)
        self.memory_mgr = MemoryManager(cfg=self.cfg, llm=self.llm, on_summary=self._on_memory_summary)
        self._evolution_llm = ChatLLMAdapter(self.cfg.auxiliary_model, self.cfg.llm)
        self.sticker_collector = StickerCollector(
            cfg=self.cfg,
            bot_client=self.bot_client,
            llm=self._evolution_llm,
        )
        self.user_profile_store = UserProfileStore(self.cfg.learning.profile_store_path)
        self.user_profiler = UserProfilingEngine(
            llm=self._evolution_llm,
            store=self.user_profile_store,
            trigger_message_count=self.cfg.learning.profile_trigger_count,
            context_limit=self.cfg.learning.profile_context_limit,
            max_tags=self.cfg.learning.profile_max_tags,
            enabled=self.cfg.learning.enabled,
        )
        self.jargon_lexicon_store = JargonLexiconStore(self.cfg.jargon.lexicon_store_path)
        self.jargon_engine = JargonEvolutionEngine(
            llm=self._evolution_llm,
            store=self.jargon_lexicon_store,
            conversion_rate=self.cfg.jargon.conversion_rate,
            trigger_message_count=self.cfg.jargon.learn_trigger_count,
            context_limit=self.cfg.jargon.learn_context_limit,
            enabled=self.cfg.jargon.enabled,
            on_learned=self._sync_learned_jargon,
        )
        self.context_builder = ContextBuilder(
            cfg=self.cfg,
            history=self._history,
            personality=self.personality,
            jargon_mgr=self.jargon_mgr,
        )
        self._inbound_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._dispatch_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._debounce_entries: dict[str, DebounceEntry] = {}
        self._debounce_lock = asyncio.Lock()
        self._inbound_worker: asyncio.Task[None] | None = None
        self._dispatch_worker: asyncio.Task[None] | None = None
        self._prompt_cache_heartbeat_task: asyncio.Task[None] | None = None
        self._debounce_window_sec = 0.5
        self._low_energy_last_reply_at: dict[str, datetime] = {}
        self._low_energy_last_reply_text: dict[str, str] = {}
        self._sticker_last_sent_at: dict[str, datetime] = {}
        self._tarot_last_draw_at: dict[str, datetime] = {}
        self._rng = random.Random()
        project_root = Path(__file__).resolve().parents[1]
        knowledge_dir = Path(str(getattr(self.cfg.paths, "knowledge_dir", "data/knowledge") or "data/knowledge"))
        if not knowledge_dir.is_absolute():
            knowledge_dir = project_root / knowledge_dir
        self._tarot_knowledge = TarotKnowledgeBase(
            knowledge_dir / "tarot_cards.json",
            image_dir=knowledge_dir / "tarot_images",
        )
        self._config_path = Path(config_path).resolve() if config_path is not None else None
        self._config_lock = asyncio.Lock()
        self._shutdown_handler = shutdown_handler
        self._shutdown_lock = asyncio.Lock()
        self._shutdown_requested = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started_at_utc = datetime.now(timezone.utc)
        await self.jargon_mgr.reload()
        await self.memory_mgr.start()
        await self.user_profiler.start()
        await self.jargon_engine.start()
        await self.status_engine.start()
        restored_status = await self._sync_personality_with_status_engine()
        self._inbound_worker = asyncio.create_task(self._inbound_loop(), name="zhiyue-inbound-worker")
        self._dispatch_worker = asyncio.create_task(self._dispatch_loop(), name="zhiyue-dispatch-worker")
        self._started = True
        if self.cfg.agent.prompt_cache_heartbeat_enabled:
            self._prompt_cache_heartbeat_task = asyncio.create_task(
                self._prompt_cache_heartbeat_loop(),
                name="zhiyue-prompt-cache-heartbeat",
            )
        self._logger.info(
            "Agent started at %s (energy=%.1f tier=%s)",
            self._started_at_utc.isoformat(),
            restored_status.energy,
            restored_status.energy_tier,
        )
        if self._tarot_knowledge.load_error:
            self._logger.error(
                "Tarot knowledge load failed: path=%s err=%s",
                self._tarot_knowledge.file_path,
                self._tarot_knowledge.load_error,
            )
        else:
            self._logger.info(
                "Tarot knowledge loaded: path=%s cards=%s image_dir=%s",
                self._tarot_knowledge.file_path,
                self._tarot_knowledge.card_count,
                self._tarot_knowledge.image_dir,
            )

    async def stop(self) -> None:
        self._started = False
        if self._prompt_cache_heartbeat_task is not None:
            self._prompt_cache_heartbeat_task.cancel()
            try:
                await self._prompt_cache_heartbeat_task
            except asyncio.CancelledError:
                pass
            self._prompt_cache_heartbeat_task = None

        if self._inbound_worker is not None:
            self._inbound_worker.cancel()
            try:
                await self._inbound_worker
            except asyncio.CancelledError:
                pass
            self._inbound_worker = None

        await self._clear_debounce_entries()

        if self._dispatch_worker is not None:
            self._dispatch_worker.cancel()
            try:
                await self._dispatch_worker
            except asyncio.CancelledError:
                pass
            self._dispatch_worker = None
        await self.user_profiler.stop()
        await self.jargon_engine.stop()
        await self.memory_mgr.close()
        await self.status_engine.stop()
        self._logger.info("Agent stopped")

    async def handle_message(self, message: dict[str, Any]) -> None:
        packet = dict(message)
        debounce_key = self._build_debounce_key(packet)
        if not self._started:
            self._logger.warning("Queue.Enqueue while agent is not started: key=%s", debounce_key)
        await self._inbound_queue.put(packet)
        self._logger.info(
            "Queue.Enqueue: key=%s message_id=%s user_id=%s group_id=%s size=%s",
            debounce_key,
            packet.get("message_id"),
            packet.get("user_id"),
            packet.get("group_id"),
            self._inbound_queue.qsize(),
        )

    async def _inbound_loop(self) -> None:
        while True:
            try:
                packet = await self._inbound_queue.get()
            except asyncio.CancelledError:
                return

            try:
                await self._debounce(packet)
            except Exception:
                self._logger.exception("Queue.Inbound processing failed")
            finally:
                self._inbound_queue.task_done()

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                packet = await self._dispatch_queue.get()
            except asyncio.CancelledError:
                return

            try:
                await self._process_message(packet)
            except Exception:
                self._logger.exception("Queue.Dispatch processing failed")
            finally:
                self._dispatch_queue.task_done()

    async def _prompt_cache_heartbeat_loop(self) -> None:
        interval = max(30, int(self.cfg.agent.prompt_cache_heartbeat_interval_sec))
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return

            if not self._started:
                return
            if self._inbound_queue.qsize() > 0 or self._dispatch_queue.qsize() > 0:
                continue

            await self._warm_prompt_cache()

    async def _warm_prompt_cache(self) -> None:
        try:
            status = await self.status_engine.get_snapshot()
            status_block = self._build_status_prompt(status)
            user_prompt = self.personality.get_think_prompt(
                None,
                "[system] prompt cache heartbeat",
                "",
                "",
            )
            user_prompt = self._append_prompt_block(user_prompt, status_block)
            user_prompt = self._append_prompt_block(
                user_prompt,
                "## 心跳请求(动态)\n这是缓存保温请求,无需真实对话。若必须回复,只输出 OK。",
            )
            messages = [
                {
                    "role": "system",
                    "content": self.personality.get_system_prompt(
                        is_master=False,
                        model_name=str(getattr(self.cfg.llm, "model", "") or ""),
                        is_group_chat=False,
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            extra_fields: dict[str, Any] = dict(self.cfg.llm.extra_fields)
            self._set_response_token_cap(extra_fields, 8)
            if "temperature" not in extra_fields:
                extra_fields["temperature"] = 0
            await self.llm.generate_from_messages(messages, extra_fields)
            self._logger.debug("PromptCache.Heartbeat: warmed")
        except Exception:
            self._logger.debug("PromptCache.Heartbeat: failed", exc_info=True)

    async def _debounce(self, packet: dict[str, Any]) -> None:
        key = self._build_debounce_key(packet)
        mentioned = self._is_packet_mentioned(packet)
        explicit_at = self._is_packet_explicit_at(packet)
        async with self._debounce_lock:
            previous = self._debounce_entries.get(key)
            if previous is not None:
                previous.timer_task.cancel()
                generation = previous.generation + 1
                merged_count = previous.merged_count + 1
                mentioned = mentioned or previous.mentioned
                explicit_at = explicit_at or previous.explicit_at
                timer_task = asyncio.create_task(
                    self._flush_debounce_after(key, generation),
                    name=f"debounce-{key}",
                )
                self._debounce_entries[key] = DebounceEntry(
                    packet=packet,
                    generation=generation,
                    merged_count=merged_count,
                    mentioned=mentioned,
                    explicit_at=explicit_at,
                    timer_task=timer_task,
                )
                self._logger.info(
                    "Queue.Debounce: key=%s mode=update merged=%s message_id=%s",
                    key,
                    merged_count,
                    packet.get("message_id"),
                )
                return

            timer_task = asyncio.create_task(
                self._flush_debounce_after(key, 1),
                name=f"debounce-{key}",
            )
            self._debounce_entries[key] = DebounceEntry(
                packet=packet,
                generation=1,
                merged_count=1,
                mentioned=mentioned,
                explicit_at=explicit_at,
                timer_task=timer_task,
            )
            self._logger.info(
                "Queue.Debounce: key=%s mode=start message_id=%s",
                key,
                packet.get("message_id"),
            )

    async def _flush_debounce_after(self, key: str, generation: int) -> None:
        try:
            await asyncio.sleep(self._debounce_window_sec)
        except asyncio.CancelledError:
            return
        await self._flush_debounce(key, generation)

    async def _flush_debounce(self, key: str, generation: int) -> None:
        packet: dict[str, Any] | None = None
        merged_count = 1
        mentioned = False
        explicit_at = False
        async with self._debounce_lock:
            entry = self._debounce_entries.get(key)
            if entry is None or entry.generation != generation:
                return
            packet = dict(entry.packet)
            merged_count = entry.merged_count
            mentioned = entry.mentioned
            explicit_at = entry.explicit_at
            del self._debounce_entries[key]

        assert packet is not None
        packet["_debounced_mention"] = mentioned
        packet["_debounced_explicit_at"] = explicit_at
        packet["_debounced_count"] = merged_count
        await self._dispatch_queue.put(packet)
        self._logger.info(
            "Queue.Debounce: key=%s mode=flush merged=%s message_id=%s dispatch_size=%s",
            key,
            merged_count,
            packet.get("message_id"),
            self._dispatch_queue.qsize(),
        )

    async def _clear_debounce_entries(self) -> None:
        async with self._debounce_lock:
            entries = list(self._debounce_entries.values())
            self._debounce_entries.clear()
        for entry in entries:
            entry.timer_task.cancel()

    async def _process_message(self, message: dict[str, Any]) -> None:
        if str(message.get("post_type", "")).strip() != "message":
            return

        message_type = str(message.get("message_type", "")).strip() or "private"
        group_id = self._to_int(message.get("group_id"))
        if message_type == "group":
            if group_id is None:
                self._logger.warning("Skip group message: invalid group_id raw=%s", message.get("group_id"))
                return

        mentioned_in_window = bool(message.get("_debounced_mention"))
        explicit_at_in_window = bool(message.get("_debounced_explicit_at"))
        user_id = self._to_int(message.get("user_id"))
        self_id = self._to_int(message.get("self_id"))
        if user_id is not None and self_id is not None and user_id == self_id:
            return

        sender_identity = self._extract_sender_identity(message)
        speaker = await self._resolve_chat_speaker(
            message=message,
            user_id=user_id,
            group_id=group_id,
            identity=sender_identity,
        )
        await self.user_profiler.sync_member_identity(
            user_id=user_id,
            nickname=sender_identity.nickname,
            group_id=group_id,
            group_card=sender_identity.group_card,
        )
        is_master = self._is_master(user_id)
        is_admin_sender = self._is_admin_sender(user_id=user_id, speaker=speaker)

        raw_text = str(message.get("text", "")).strip()
        if raw_text:
            command_handled = await self._try_handle_admin_command(
                message=message,
                text=raw_text,
                user_id=user_id,
                speaker=speaker,
                group_id=group_id,
                is_admin_sender=is_admin_sender,
            )
            if command_handled:
                return

        if message_type == "group" and group_id is not None and not self.cfg.is_group_enabled(group_id):
            self._logger.info("Skip group message: group not enabled group_id=%s", group_id)
            return

        if raw_text:
            tarot_handled = await self._try_handle_tarot_command(message=message, text=raw_text)
            if tarot_handled:
                return

        if message_type == "group":
            status_snapshot = await self.status_engine.get_snapshot()
            await self.sticker_collector.observe_group_message(
                message=message,
                group_id=group_id,
                sender_id=user_id,
                speaker=speaker,
                mood=status_snapshot.energy,
                is_admin=is_admin_sender,
            )

        if not raw_text:
            if message_type == "group" and mentioned_in_window:
                raw_text = "[用户仅@了你]"
            else:
                return
        text = _normalize_prompt_input(raw_text)
        if not text:
            return
        message["text"] = text

        now_utc = datetime.now(timezone.utc)
        session_id = self._build_session_id(message)
        merged_count = int(message.get("_debounced_count", 1) or 1)
        memory_group_id = group_id if group_id is not None else 0
        sticker_intent = self._is_sticker_request(text)
        prefer_llm_route = bool(message_type != "group" or mentioned_in_window or sticker_intent)

        self._logger.info(
            "Queue.Dispatch: session=%s message_id=%s merged=%s mentioned=%s",
            session_id,
            message.get("message_id"),
            merged_count,
            mentioned_in_window,
        )

        status_after_user = await self.status_engine.apply_user_message(text)

        self._history.append(
            session_id,
            ConversationMessage(
                role="user",
                content=text,
                message_type=message_type,
                speaker=speaker,
                user_id=user_id,
                is_master=is_master,
                created_at=now_utc,
            ),
        )
        await self.memory_mgr.add_message(
            MessageLog(
                message_id=self._to_int(message.get("message_id")) or 0,
                group_id=memory_group_id,
                user_id=user_id or 0,
                nickname=speaker,
                content=text,
                created_at=now_utc,
            ),
        )
        await self.memory_mgr.record_conversation_turn(
            session_id=session_id,
            group_id=memory_group_id,
            role="user",
            content=text,
            speaker=speaker,
            user_id=user_id,
            created_at=now_utc,
        )
        await self._run_background_observers(
            self.user_profiler.observe_user_message(
                user_id=user_id,
                speaker=speaker,
                nickname=sender_identity.nickname,
                group_card=sender_identity.group_card,
                text=text,
                session_id=session_id,
                group_id=group_id,
            ),
            self.jargon_engine.observe_user_message(
                user_id=user_id,
                speaker=speaker,
                text=text,
                session_id=session_id,
                group_id=group_id,
            ),
            stage="user_observe",
        )

        self._logger.info(
            (
                "HandleMessage: session=%s type=%s user_id=%s group_id=%s master=%s "
                "status_energy=%.1f status_tier=%s fatigue=%s"
            ),
            session_id,
            message_type,
            user_id,
            message.get("group_id"),
            is_master,
            status_after_user.energy,
            status_after_user.energy_tier,
            status_after_user.fatigue_mode,
        )

        low_energy_mode = bool(status_after_user.fatigue_mode)
        final_reply = ""
        forced_rest = False

        if low_energy_mode:
            if not explicit_at_in_window:
                self._logger.info(
                    "Queue.SkipReply: session=%s reason=low_energy_no_explicit_at status_energy=%.1f",
                    session_id,
                    status_after_user.energy,
                )
                return

            final_reply = self._LOW_ENERGY_AT_ONLY_REPLY
            forced_rest = True
            self._logger.info(
                "Queue.LowEnergyReply: session=%s status_energy=%.1f fatigue=%s forced_rest=%s mode=explicit_at_only",
                session_id,
                status_after_user.energy,
                status_after_user.fatigue_mode,
                forced_rest,
            )

        if not final_reply:
            llm_route_probability = self._llm_route_probability(
                message_type=message_type,
                mentioned_in_window=mentioned_in_window,
                status_energy=status_after_user.energy,
                is_master=is_master,
                is_admin_sender=is_admin_sender,
                sticker_intent=sticker_intent,
                source_text=text,
            )
            if llm_route_probability < 1.0:
                roll = self._rng.random()
                if roll >= llm_route_probability:
                    self._logger.info(
                        (
                            "Queue.SkipReply: session=%s reason=llm_route_probability "
                            "roll=%.4f threshold=%.4f status_energy=%.1f"
                        ),
                        session_id,
                        roll,
                        llm_route_probability,
                        status_after_user.energy,
                    )
                    return

            retrieval = await self.memory_mgr.retrieve_for_prompt(
                text=text,
                session_id=session_id,
                group_id=group_id,
                top_k=self.cfg.memory.rag_top_k,
            )
            social_background = await self.user_profiler.build_social_background(user_id, speaker)
            ctx = await self._build_context(
                session_id=session_id,
                message=message,
                speaker=speaker,
                user_id=user_id,
                is_master=is_master,
                mentioned_in_window=mentioned_in_window,
                history_background=retrieval.history_background,
                related_knowledge=retrieval.related_knowledge,
                social_background=social_background,
            )
            ctx = await self._before_llm_think(ctx)
            reply = await self._llm_think(ctx)
            reply = await self._after_llm_think(ctx, reply)
            if not reply:
                return

            final_reply = await self._response_post_process(reply, ctx)
            if not final_reply:
                return

            final_reply, forced_rest = await self.status_engine.apply_reply_policy(final_reply)
            if not final_reply:
                return

        await self._reply(message, final_reply)
        if low_energy_mode:
            self._mark_low_energy_reply(session_id=session_id, reply=final_reply)

        now_after_reply = datetime.now(timezone.utc)
        status_after_reply = await self.status_engine.consume_reply(final_reply)
        self._history.append(
            session_id,
            ConversationMessage(
                role="assistant",
                content=final_reply,
                message_type=message_type,
                speaker=self.cfg.persona.name,
                user_id=self.cfg.persona.qq,
                is_master=False,
                created_at=now_after_reply,
            ),
        )
        await self.memory_mgr.record_conversation_turn(
            session_id=session_id,
            group_id=memory_group_id,
            role="assistant",
            content=final_reply,
            speaker=self.cfg.persona.name,
            user_id=self.cfg.persona.qq,
            created_at=now_after_reply,
        )
        await self._run_background_observers(
            self.user_profiler.observe_bot_reply(
                user_id=user_id,
                text=final_reply,
                session_id=session_id,
                group_id=group_id,
            ),
            stage="assistant_observe",
        )
        self._logger.info(
            "ReplyStatus: session=%s forced_rest=%s status_energy=%.1f status_tier=%s fatigue=%s",
            session_id,
            forced_rest,
            status_after_reply.energy,
            status_after_reply.energy_tier,
            status_after_reply.fatigue_mode,
        )

    async def _build_context(
        self,
        *,
        session_id: str,
        message: dict[str, Any],
        speaker: str,
        user_id: int | None,
        is_master: bool,
        mentioned_in_window: bool,
        history_background: list[str],
        related_knowledge: list[str],
        social_background: str,
    ) -> ThinkContext:
        status = await self.status_engine.get_snapshot()
        current = self.personality.get_current_mood()
        self.personality.set_mood(
            valence=current.valence,
            energy=status.energy / 100.0,
            sociability=current.sociability,
        )
        built = await self.context_builder.build(
            session_id=session_id,
            message=message,
            speaker=speaker,
            user_id=user_id,
            is_master=is_master,
            mentioned_in_window=mentioned_in_window,
            history_background=history_background,
            related_knowledge=related_knowledge,
            social_background=social_background,
        )
        status_block = self._build_status_prompt(status)
        llm_messages = [dict(item) for item in built.llm_messages]
        status_injected = False
        for message_item in reversed(llm_messages):
            if str(message_item.get("role", "")).strip() != "user":
                continue
            message_item["content"] = self._append_prompt_block(message_item.get("content", ""), status_block)
            status_injected = True
            break
        if not status_injected:
            llm_messages.append({"role": "user", "content": status_block})

        return ThinkContext(
            session_id=session_id,
            message_type=str(message.get("message_type", "")).strip() or "private",
            user_id=user_id,
            group_id=self._to_int(message.get("group_id")),
            speaker=speaker,
            source_text=str(message.get("text", "")).strip(),
            is_master=is_master,
            mood=built.mood,
            style=built.style,
            prompt_context=built.prompt_context,
            llm_messages=llm_messages,
            status=status,
            social_background=social_background,
        )

    async def _before_llm_think(self, ctx: ThinkContext) -> ThinkContext:
        # 濡澘瀚弳鈧柨娑欒壘椤曨喗顬?ReAct 婵炵繝鑳堕埢鍏肩▔椤撶姵鐣遍柍銉︾矊娴兼劙宕楅悿顖ｆ綈闁?+ 濡炲瀛╅悧鎼佸礆閸℃瑨顫﹂柍銉︾箞濡礁鈻撻悙鍏夊亾?
        ctx.planned_tools = await self._plan_tool_calls(ctx)
        ctx.style = await self._classify_style_context(ctx)
        return ctx

    async def _llm_think(
        self,
        ctx: ThinkContext,
        *,
        max_tokens_override: int | None = None,
        temp_system_prompt: str = "",
    ) -> str:
        extra_fields: dict[str, Any] = dict(self.cfg.llm.extra_fields)
        if self.cfg.llm.max_response_tokens > 0:
            if "max_tokens" not in extra_fields and "max_completion_tokens" not in extra_fields:
                extra_fields["max_tokens"] = self.cfg.llm.max_response_tokens

        if max_tokens_override is not None and max_tokens_override > 0:
            self._set_response_token_cap(extra_fields, int(max_tokens_override))

        llm_messages = ctx.llm_messages
        hint = _normalize_prompt_input(temp_system_prompt)
        if hint:
            llm_messages = [dict(item) for item in ctx.llm_messages]
            injected = False
            hint_block = f"## 临时补充规则(动态)\n{hint}"
            for message in reversed(llm_messages):
                if str(message.get("role", "")).strip() != "user":
                    continue
                message["content"] = self._append_prompt_block(message.get("content", ""), hint_block)
                injected = True
                break
            if not injected:
                llm_messages.append({"role": "user", "content": hint_block})

        reply = await self.llm.generate_from_messages(llm_messages, extra_fields)
        content = reply.strip()
        self._logger.info(
            "LLM.Think done: session=%s tone=%s has_reply=%s preview=%s",
            ctx.session_id,
            ctx.style.tone,
            bool(content),
            content[:120],
        )
        return content

    async def _after_llm_think(self, ctx: ThinkContext, reply: str) -> str:
        tool_result = await self._apply_tool_results(ctx, reply)
        clean_reply = str(tool_result.get("clean_reply", reply) or "").strip()
        if self._is_silence_placeholder_reply(clean_reply):
            self._logger.info("LLM.Reply suppressed: session=%s reason=silence_placeholder", ctx.session_id)
            return ""
        return clean_reply

    async def _plan_tool_calls(self, ctx: ThinkContext) -> list[str]:
        del ctx
        return []

    async def _classify_style_context(self, ctx: ThinkContext) -> StyleClassification:
        energy_ratio = self._clamp_probability(
            (ctx.status.energy / 100.0) if ctx.status is not None else ctx.mood.energy,
        )
        return self.jargon_mgr.classify_style(
            0.0,
            energy_ratio,
            speaker_is_master=ctx.is_master,
        )

    async def _apply_tool_results(self, ctx: ThinkContext, reply: str) -> dict[str, Any]:
        raw_reply = str(reply or "")
        clean_reply = raw_reply
        sticker_queries = self._extract_sticker_queries(raw_reply)

        if not sticker_queries and self._FAKE_STICKER_TEXT_PATTERN.search(raw_reply):
            fallback_query = self._extract_sticker_fallback_query(raw_reply, ctx.source_text)
            if fallback_query:
                sticker_queries.append(fallback_query)

        sticker_sent = False
        used_query = ""
        explicit_sticker_request = self._is_sticker_request(ctx.source_text)
        auto_sticker_blocked = False
        allow_sticker_send = True
        if sticker_queries and not explicit_sticker_request:
            allow_sticker_send = self._allow_auto_sticker_send(ctx=ctx)
            auto_sticker_blocked = not allow_sticker_send

        if allow_sticker_send:
            for query in sticker_queries[:2]:
                if await self._send_sticker_from_library(ctx, query):
                    sticker_sent = True
                    used_query = query
                    self._mark_sticker_reply(session_id=ctx.session_id)
                    break

        clean_reply = self._STICKER_MARKER_PATTERN.sub("", clean_reply)
        clean_reply = self._STICKER_FUNC_CALL_PATTERN.sub("", clean_reply)
        if sticker_sent:
            clean_reply = self._FAKE_STICKER_TEXT_PATTERN.sub("", clean_reply)
            clean_reply = self._normalize_reply_spaces(clean_reply)
            self._logger.info(
                "Sticker.Reply executed: session=%s group_id=%s query=%s",
                ctx.session_id,
                ctx.group_id,
                used_query,
            )
        elif sticker_queries:
            clean_reply = self._normalize_reply_spaces(clean_reply)
            if not clean_reply and not auto_sticker_blocked:
                fallback_query = ""
                if sticker_queries:
                    fallback_query = str(sticker_queries[0] or "").strip()
                if not fallback_query:
                    fallback_query = self._extract_sticker_fallback_query(raw_reply, ctx.source_text)
                clean_reply = self._build_sticker_action_text(fallback_query)
            self._logger.info(
                "Sticker.Reply skipped: session=%s group_id=%s queries=%s reason=%s",
                ctx.session_id,
                ctx.group_id,
                sticker_queries,
                "auto_rate_limit" if auto_sticker_blocked else "send_failed",
            )

        return {
            "clean_reply": clean_reply.strip(),
            "sticker_sent": sticker_sent,
            "queries": sticker_queries,
        }

    def _extract_sticker_queries(self, reply: str) -> list[str]:
        clean_reply = str(reply or "")
        if not clean_reply:
            return []

        out: list[str] = []
        seen: set[str] = set()

        def _append(raw_query: str) -> None:
            query = str(raw_query or "").strip().strip("\"'`“”‘’")
            if not query:
                return
            lowered = query.lower()
            if lowered in seen:
                return
            seen.add(lowered)
            out.append(query)

        for match in self._STICKER_MARKER_PATTERN.finditer(clean_reply):
            _append(match.group("query"))
        for match in self._STICKER_FUNC_CALL_PATTERN.finditer(clean_reply):
            _append(match.group("query"))
        return out

    @staticmethod
    def _normalize_reply_spaces(text: str) -> str:
        clean = str(text or "")
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        clean = re.sub(r"^[\s，。,\.!！?？;；:：\-~]+", "", clean)
        return clean.strip()

    @staticmethod
    def _append_prompt_block(base: Any, block: Any) -> str:
        base_text = str(base or "").strip()
        block_text = str(block or "").strip()
        if not base_text:
            return block_text
        if not block_text:
            return base_text
        return f"{base_text}\n\n{block_text}"

    @staticmethod
    def _extract_sticker_fallback_query(reply: str, source_text: str) -> str:
        clean_reply = str(reply or "")
        for pattern in (
            r"(?:发了?|来|整|甩|丢|找)(?:一个|一张|个|张)?([^，。！？\s]{1,20})(?:的)?表情包",
            r"表情包\s*[:：]\s*([^\s，。！？,;；]{1,20})",
            r"[“\"「『]([^”\"」』]{1,20})[”\"」』]",
        ):
            match = re.search(pattern, clean_reply)
            if match is not None:
                query = str(match.group(1) or "").strip()
                if query:
                    return query
        source = str(source_text or "").strip()
        if source:
            return source[:20]
        return ""

    @staticmethod
    def _build_sticker_query_candidates(query: str, source_text: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        def _append(value: str) -> None:
            clean = str(value or "").strip().strip("\"'`“”‘’")
            clean = re.sub(r"\s+", "", clean)
            if not clean:
                return
            lowered = clean.lower()
            if lowered in seen:
                return
            seen.add(lowered)
            out.append(clean)

        raw_query = str(query or "").strip()
        _append(raw_query)

        for text in (raw_query, str(source_text or "").strip()):
            clean = str(text or "").strip()
            if not clean:
                continue
            for pattern in (
                r"(?:发了?|来|整|甩|丢|找|回)(?:一个|一张|个|张)?([^，。！？\s]{1,20})(?:的)?表情包",
                r"表情包\s*[:：]\s*([^\s，。！？,;；]{1,20})",
            ):
                match = re.search(pattern, clean)
                if match is not None:
                    _append(str(match.group(1) or ""))

            stripped = re.sub(r"(表情包|斗图|sticker|meme)", "", clean, flags=re.IGNORECASE)
            stripped = re.sub(r"(发|来|整|给|甩|丢|回|找|从|库|里|面|一个|一张|发出|出来|随机|随便|纸月|小纸月)", "", stripped)
            _append(stripped[:20])

        return out[:6]

    @staticmethod
    def _build_sticker_action_text(query: str) -> str:
        clean_query = str(query or "").strip().strip("\"'`“”‘’")
        clean_query = re.sub(r"\s+", "", clean_query)
        if ZhiyueAgent._is_random_pick_request(clean_query):
            return "（发了一个随机表情包）"
        clean_query = re.sub(r"(表情包|斗图|sticker|meme)", "", clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r"(发|来|整|给|甩|丢|回|找|从|库|里|面|一个|一张|发出|出来|随机|随便)", "", clean_query)
        clean_query = clean_query[:12]
        if clean_query:
            return f"（发了一个{clean_query}的表情包）"
        return "（发了一个表情包）"

    async def _send_sticker_from_library(self, ctx: ThinkContext, query: str) -> bool:
        group_id = ctx.group_id if (ctx.group_id is not None and ctx.group_id > 0) else None
        user_id = ctx.user_id if (ctx.user_id is not None and ctx.user_id > 0) else None
        if group_id is None and user_id is None:
            return False

        clean_query = str(query or "").strip().strip("\"'`“”‘’")
        if not clean_query:
            return False

        query_candidates = self._build_sticker_query_candidates(clean_query, ctx.source_text)
        if not query_candidates:
            query_candidates = [clean_query]
        random_pick = any(self._is_random_pick_request(item) for item in query_candidates) or self._is_random_pick_request(ctx.source_text)
        reply_mood = float(ctx.status.energy) if ctx.status is not None else 50.0
        candidate_items: list[dict[str, Any]] = []
        seen_files: set[str] = set()

        def _append_candidate(item: Any, *, require_sticker: bool) -> None:
            if not isinstance(item, dict):
                return
            if str(item.get("storage_mode", "local")).strip().lower() != "local":
                return
            if require_sticker and not self.sticker_collector.is_sticker_item(item):
                return
            clean_name = str(item.get("file_name", "")).strip()
            if not clean_name:
                return
            lowered = clean_name.lower()
            if lowered in seen_files:
                return
            seen_files.add(lowered)
            candidate_items.append(dict(item))

        for candidate in query_candidates:
            if candidate.isdigit():
                item = await self.sticker_collector.get_sticker(candidate)
                if item is not None:
                    _append_candidate(item, require_sticker=True)

            rows = await self.sticker_collector.search(candidate, limit=12, storage_mode="local")
            for row in rows:
                _append_candidate(row, require_sticker=True)

            if not candidate_items:
                for row in rows:
                    _append_candidate(row, require_sticker=False)
                if candidate_items:
                    self._logger.info(
                        "Sticker.Reply compatibility fallback: use_image_as_sticker query=%s target=%s count=%s",
                        candidate,
                        group_id if group_id is not None else user_id,
                        len(candidate_items),
                    )
            if candidate_items:
                break
        if not candidate_items:
            random_item = await self._pick_random_local_sticker_item(require_sticker=True)
            if random_item is None:
                random_item = await self._pick_random_local_sticker_item(require_sticker=False)
            if random_item is not None:
                _append_candidate(random_item, require_sticker=False)
                self._logger.info(
                    "Sticker.Reply fallback: random_pick_for_empty_result query=%s group_id=%s",
                    clean_query,
                    group_id if group_id is not None else user_id,
                )

        if random_pick and not candidate_items:
            random_item = await self._pick_random_local_sticker_item(require_sticker=True)
            if random_item is None:
                random_item = await self._pick_random_local_sticker_item(require_sticker=False)
            if random_item is not None:
                _append_candidate(random_item, require_sticker=False)

        for item in candidate_items:
            file_name = str(item.get("file_name", "")).strip()
            try:
                if not self.sticker_collector.resolve_local_file_path(file_name).is_file():
                    continue
            except ValueError:
                continue

            decision = await self.sticker_collector.allow_sticker_for_reply(
                item=item,
                query=clean_query,
                mood=reply_mood,
            )
            if not decision.allowed:
                self._logger.info(
                    "Sticker.Reply rejected by persona filter: target=%s file=%s reason=%s source=%s",
                    group_id if group_id is not None else user_id,
                    file_name,
                    decision.reason,
                    decision.source,
                )
                continue

            content = self.sticker_collector.build_local_sticker_cq(file_name)
            if group_id is not None:
                await self.bot_client.send_group_msg(group_id=group_id, message=content)
            else:
                assert user_id is not None
                await self.bot_client.send_private_msg(user_id=user_id, message=content)
            return True

        if random_pick:
            random_item = await self._pick_random_local_sticker_item(require_sticker=True)
            if random_item is None:
                random_item = await self._pick_random_local_sticker_item(require_sticker=False)
            if random_item is not None:
                decision = await self.sticker_collector.allow_sticker_for_reply(
                    item=random_item,
                    query=clean_query,
                    mood=reply_mood,
                )
                if decision.allowed:
                    file_name = str(random_item.get("file_name", "")).strip()
                    content = self.sticker_collector.build_local_sticker_cq(file_name)
                    if group_id is not None:
                        await self.bot_client.send_group_msg(group_id=group_id, message=content)
                    else:
                        assert user_id is not None
                        await self.bot_client.send_private_msg(user_id=user_id, message=content)
                    return True

        return False

    async def _pick_random_local_sticker_item(self, *, require_sticker: bool) -> dict[str, Any] | None:
        files = await self.sticker_collector.list_local_files()
        candidates: list[dict[str, Any]] = []
        for row in files:
            if require_sticker and not self.sticker_collector.is_sticker_item(row):
                continue
            file_name = str(row.get("file_name", "")).strip()
            if not file_name:
                continue
            try:
                if not self.sticker_collector.resolve_local_file_path(file_name).is_file():
                    continue
            except ValueError:
                continue
            candidates.append(dict(row))
        if not candidates:
            return None
        return dict(self._rng.choice(candidates))

    @staticmethod
    def _is_random_pick_request(text: str) -> bool:
        clean_text = str(text or "").strip().lower()
        if not clean_text:
            return False
        if any(token in clean_text for token in ("随机", "随便", "任意", "都行", "库里", "库中", "库内")):
            return True
        if re.search(r"(来|找|挑|发).{0,6}(一|1)?(个|张)", clean_text):
            return True
        return False

    async def _response_post_process(self, reply: str, ctx: ThinkContext) -> str:
        energy_ratio = self._clamp_probability(
            (ctx.status.energy / 100.0) if ctx.status is not None else ctx.mood.energy,
        )
        styled = self.jargon_mgr.apply_post_process(
            reply,
            mood=0.0,
            energy=energy_ratio,
            speaker_is_master=ctx.is_master,
        )
        processed = await self.jargon_engine.apply_to_reply(styled)
        clean = self._strip_self_prefix(processed)
        if self._is_silence_placeholder_reply(clean):
            self._logger.info("LLM.Reply suppressed: session=%s reason=silence_placeholder_post", ctx.session_id)
            return ""
        return clean

    @classmethod
    def _is_silence_placeholder_reply(cls, text: str) -> bool:
        clean = str(text or "").strip()
        if not clean:
            return False
        candidates = [clean]
        trimmed = cls._TRAILING_PLACEHOLDER_PUNCT_PATTERN.sub("", clean).strip()
        if trimmed and trimmed != clean:
            candidates.append(trimmed)

        for candidate in candidates:
            if cls._STAY_QUIET_PLACEHOLDER_PATTERN.fullmatch(candidate):
                return True
            bracketed = cls._BRACKETED_SILENCE_PLACEHOLDER_PATTERN.fullmatch(candidate)
            if bracketed is None:
                continue
            token = str(bracketed.group("token") or "").strip()
            normalized_token = re.sub(r"[\s_\-]+", "", token).lower()
            if normalized_token in cls._BRACKETED_SILENCE_TOKENS:
                return True

        for candidate in candidates:
            if len(candidate) > 48:
                continue
            if cls._SILENCE_PLACEHOLDER_PATTERN.fullmatch(candidate):
                return True
        return False

    async def _sync_learned_jargon(self, term: str, meaning: str) -> None:
        clean_term = str(term).strip()
        if not clean_term:
            return
        await self.jargon_mgr.add(clean_term, str(meaning or "").strip())

    async def _run_background_observers(self, *tasks: Awaitable[Any], stage: str) -> None:
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                self._logger.warning("Background observer failed: stage=%s err=%s", stage, item)

    async def _on_memory_summary(self, session_id: str, summary: str, updated_at: datetime) -> None:
        self._history.set_summary(session_id, summary, updated_at=updated_at)
        self._logger.info(
            "Memory summary updated: session=%s updated_at=%s",
            session_id,
            updated_at.isoformat(),
        )

    async def get_admin_status(self) -> dict[str, Any]:
        status = await self.status_engine.get_snapshot()
        short_term = await self.memory_mgr.get_short_term_snapshot(max_sessions=8, turn_limit=4)
        history_summary = self._history.short_term_summary(max_sessions=8, max_messages=3)
        return {
            "energy": round(status.energy, 2),
            "energy_tier": status.energy_tier,
            "fatigue_mode": status.fatigue_mode,
            "forced_rest": status.forced_rest,
            "last_active_at": status.last_active_at.isoformat(),
            "runtime_started_at": self._started_at_utc.isoformat(),
            "uptime_seconds": self.uptime_seconds(),
            "models": {
                "chat": self.llm.get_loaded_models(),
                "auxiliary": self._evolution_llm.get_loaded_models(),
                "embedding": self.cfg.embedding.model,
                "vision": self.cfg.vision_llm.model,
            },
            "short_term_memory": {
                "memory_manager": short_term,
                "history": history_summary,
            },
        }

    async def reset_runtime_state(
        self,
        *,
        fill_energy: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        snapshot = await self.status_engine.reset(fill_energy=fill_energy)
        if fill_energy:
            current = self.personality.get_current_mood()
            self.personality.set_mood(
                valence=current.valence,
                energy=snapshot.energy / 100.0,
                sociability=current.sociability,
            )

        cleared = False
        if session_id and session_id.strip():
            cleared = self._history.clear_session(session_id.strip())
            cleared = await self.memory_mgr.clear_session_memory(session_id.strip()) or cleared

        return {
            "status": {
                "energy": round(snapshot.energy, 2),
                "energy_tier": snapshot.energy_tier,
                "fatigue_mode": snapshot.fatigue_mode,
                "forced_rest": snapshot.forced_rest,
            },
            "cleared_session": session_id.strip() if session_id and cleared else "",
            "cleared": cleared,
        }

    async def _sync_personality_with_status_engine(self) -> StatusSnapshot:
        snapshot = await self.status_engine.get_snapshot()
        current = self.personality.get_current_mood()
        self.personality.set_mood(
            valence=current.valence,
            energy=snapshot.energy / 100.0,
            sociability=current.sociability,
        )
        return snapshot

    def update_system_prompt(self, prompt: str) -> None:
        self.cfg.persona.system_prompt = prompt
        self.personality.cfg.system_prompt = prompt

    def uptime_seconds(self) -> int:
        return max(0, int((datetime.now(timezone.utc) - self._started_at_utc).total_seconds()))

    @staticmethod
    def _build_status_prompt(status: StatusSnapshot) -> str:
        tier = status.energy_tier
        if tier == "充沛":
            policy = "维持常规清冷人设，按正常节奏短答。"
        elif tier == "一般":
            policy = "回复更短，体现轻微疲惫，避免展开。"
        else:
            policy = "极度冷淡，拒绝长谈，必要时可不回复。"
        return (
            "## 状态分档\n"
            f"- 当前状态：[精力:{tier}]\n"
            f"- 回复策略：{policy}\n"
            "- 不要输出任何数值化状态（例如 0-100 分）。"
        )

    async def emit_debug_message(self, group_id: int, content: str, user_id: int = 10000) -> None:
        packet: dict[str, Any] = {
            "post_type": "message",
            "message_type": "group",
            "group_id": group_id,
            "user_id": user_id,
            "self_id": self.cfg.persona.qq,
            "text": content,
            "raw": {"sender": {"nickname": "debug-user"}},
        }
        await self.handle_message(packet)

    async def _reply(self, message: dict[str, Any], reply: str) -> None:
        parts = self._split_reply_parts(reply)
        if not parts:
            return

        message_type = str(message.get("message_type", "")).strip() or "private"
        if message_type == "group":
            group_id = self._to_int(message.get("group_id"))
            if group_id is None:
                self._logger.warning("Skip group reply: missing group_id")
                return
            for idx, part in enumerate(parts):
                await self.bot_client.send_group_msg(group_id=group_id, message=part)
                self._logger.info(
                    "Queue.Reply: target=group group_id=%s part=%s/%s",
                    group_id,
                    idx + 1,
                    len(parts),
                )
                if idx + 1 < len(parts):
                    await asyncio.sleep(self._reply_gap_seconds(part))
            return

        user_id = self._to_int(message.get("user_id"))
        if user_id is None:
            self._logger.warning("Skip private reply: missing user_id")
            return
        for idx, part in enumerate(parts):
            await self.bot_client.send_private_msg(user_id=user_id, message=part)
            self._logger.info(
                "Queue.Reply: target=private user_id=%s part=%s/%s",
                user_id,
                idx + 1,
                len(parts),
            )
            if idx + 1 < len(parts):
                await asyncio.sleep(self._reply_gap_seconds(part))

    @classmethod
    def _split_reply_parts(cls, reply: str) -> list[str]:
        text = reply.strip()
        if not text:
            return []

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        by_lines = [item.strip() for item in cls._REPLY_LINE_SPLIT_PATTERN.split(normalized) if item.strip()]
        if len(by_lines) > 1:
            return cls._merge_short_parts(by_lines)

        by_sentences = [item.strip() for item in cls._REPLY_SENTENCE_PATTERN.findall(normalized) if item.strip()]
        if len(by_sentences) > 1:
            return cls._merge_short_parts(by_sentences)

        return [text]

    @classmethod
    def _merge_short_parts(cls, parts: list[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""
        for part in parts:
            item = part.strip()
            if not item:
                continue
            if not buffer:
                buffer = item
                continue
            if len(buffer) < 8:
                buffer = cls._concat_reply_text(buffer, item)
                continue
            merged.append(buffer)
            buffer = item
        if buffer:
            merged.append(buffer)
        return merged

    @staticmethod
    def _concat_reply_text(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
            return f"{left} {right}"
        return f"{left}{right}"

    def _reply_gap_seconds(self, part: str) -> float:
        # 小间隔让多条消息看起来更像自然打字，而不是一次性刷屏。
        base = 0.18
        length_factor = min(0.6, max(0.0, len(part)) * 0.012)
        jitter = self._rng.uniform(0.04, 0.18)
        return min(1.2, base + length_factor + jitter)

    def _strip_self_prefix(self, text: str) -> str:
        clean = str(text or "").strip()
        if not clean:
            return ""

        prefix = ""
        tail = clean
        while tail and tail[0] in self._LEADING_QUOTES:
            prefix += tail[0]
            tail = tail[1:].lstrip()

        names = self._self_names_for_prefix()
        lowered = tail.lower()
        for name in names:
            if not lowered.startswith(name.lower()):
                continue
            remain = tail[len(name):].lstrip()
            if not remain.startswith(("：", ":")):
                continue
            stripped = remain[1:].lstrip()
            if not stripped:
                return ""
            return f"{prefix}{stripped}"

        stripped_log = self._strip_log_style_self_prefix(tail, names)
        if stripped_log is not None:
            if not stripped_log:
                return ""
            return f"{prefix}{stripped_log}"
        return clean

    @classmethod
    def _strip_log_style_self_prefix(cls, text: str, names: list[str]) -> str | None:
        match = cls._LOG_STYLE_SELF_PREFIX_PATTERN.match(text)
        if match is None:
            return None

        speaker = str(match.group("speaker") or "").strip()
        if not speaker:
            return None

        lowered_speaker = speaker.lower()
        if not any(lowered_speaker == name.lower() for name in names):
            return None

        body = str(match.group("body") or "").strip()
        return body

    def _self_names_for_prefix(self) -> list[str]:
        candidates = [
            str(self.cfg.persona.name or "").strip(),
            *[str(item or "").strip() for item in getattr(self.cfg.persona, "alias_names", [])],
        ]
        unique: list[str] = []
        seen: set[str] = set()
        for item in sorted(candidates, key=len, reverse=True):
            if not item:
                continue
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(item)
        return unique

    def _build_debounce_key(self, message: dict[str, Any]) -> str:
        message_type = str(message.get("message_type", "")).strip() or "private"
        if message_type == "group":
            group_id = self._to_int(message.get("group_id"))
            if group_id is not None:
                return f"group:{group_id}"

        user_id = self._to_int(message.get("user_id"))
        if user_id is not None:
            return f"user:{user_id}"

        message_id = self._to_int(message.get("message_id"))
        if message_id is not None:
            return f"message:{message_id}"
        return "unknown"

    def _is_packet_mentioned(self, message: dict[str, Any]) -> bool:
        text = str(message.get("text", "")).strip()
        if text and self.personality.is_mentioned(text):
            return True

        return self._is_packet_explicit_at(message)

    def _is_packet_explicit_at(self, message: dict[str, Any]) -> bool:
        target_ids: set[int] = set()
        self_id = self._to_int(message.get("self_id"))
        if self_id is not None:
            target_ids.add(self_id)
        persona_qq = self._to_int(self.cfg.persona.qq)
        if persona_qq is not None:
            target_ids.add(persona_qq)
        if not target_ids:
            return False

        raw = message.get("raw")
        if isinstance(raw, dict):
            if bool(raw.get("at_me")):
                return True
            if self._has_at_segment(raw.get("message"), target_ids):
                return True
            if self._has_cq_at(raw.get("raw_message"), target_ids):
                return True

        if self._has_at_segment(message.get("message"), target_ids):
            return True
        if self._has_cq_at(message.get("raw_message"), target_ids):
            return True
        return False

    def _build_session_id(self, message: dict[str, Any]) -> str:
        message_type = str(message.get("message_type", "")).strip() or "private"
        if message_type == "group":
            group_id = self._to_int(message.get("group_id"))
            if group_id is not None:
                return f"group:{group_id}"

        user_id = self._to_int(message.get("user_id"))
        if user_id is not None:
            return f"private:{user_id}"
        return "unknown"

    async def _try_handle_admin_command(
        self,
        *,
        message: dict[str, Any],
        text: str,
        user_id: int | None,
        speaker: str,
        group_id: int | None,
        is_admin_sender: bool,
    ) -> bool:
        admin_cfg = self.cfg.admin_commands
        if not bool(admin_cfg.enabled):
            return False

        prefix = str(admin_cfg.prefix or "").strip()
        if not prefix:
            return False

        clean_text = str(text or "").strip()
        if not clean_text.startswith(prefix):
            return False

        command_text = clean_text[len(prefix):].strip()
        if not is_admin_sender:
            self._logger.warning(
                "AdminCommand denied: user_id=%s speaker=%s text=%s",
                user_id,
                speaker,
                clean_text,
            )
            await self._reply(message, "你没有权限使用管理指令。")
            return True

        if not command_text:
            await self._reply(message, self._admin_command_help(prefix))
            return True
        if self._is_admin_help_subcommand(command_text):
            await self._reply(message, self._admin_custom_trigger_list(prefix))
            return True

        resolved = self._resolve_admin_command(command_text)
        if resolved is None:
            await self._reply(message, f"未知管理指令：{command_text}\n{self._admin_command_help(prefix)}")
            return True

        action, args, trigger = resolved
        self._logger.info(
            "AdminCommand accepted: user_id=%s group_id=%s action=%s trigger=%s args=%s",
            user_id,
            group_id,
            action,
            trigger,
            args,
        )

        if action == "toggle_group_chat":
            result = await self._admin_toggle_group_chat(args=args, fallback_group_id=group_id)
            await self._reply(message, result)
            return True

        if action == "join_group_chat":
            result = await self._admin_join_group_chat(args=args, fallback_group_id=group_id)
            await self._reply(message, result)
            return True

        if action == "shutdown":
            result = await self._admin_shutdown()
            await self._reply(message, result)
            return True

        await self._reply(message, f"未支持的管理动作：{action}")
        return True

    def _resolve_admin_command(self, command_text: str) -> tuple[str, str, str] | None:
        clean_text = str(command_text or "").strip()
        if not clean_text:
            return None

        candidates: list[tuple[str, str]] = []
        for item in self.cfg.admin_commands.commands:
            action = str(item.action or "").strip()
            if action not in self._SUPPORTED_ADMIN_ACTIONS:
                continue
            for raw_trigger in list(item.triggers or []):
                trigger = str(raw_trigger or "").strip()
                if not trigger:
                    continue
                candidates.append((trigger, action))

        candidates.sort(key=lambda row: len(row[0]), reverse=True)
        for trigger, action in candidates:
            if not clean_text.startswith(trigger):
                continue
            args = clean_text[len(trigger):].strip()
            return action, args, trigger
        return None

    def _admin_command_help(self, prefix: str) -> str:
        rows: list[str] = []
        for item in self.cfg.admin_commands.commands:
            action = str(item.action or "").strip()
            if action not in self._SUPPORTED_ADMIN_ACTIONS:
                continue
            triggers = [str(trigger).strip() for trigger in list(item.triggers or []) if str(trigger).strip()]
            if not triggers:
                continue
            joined = " / ".join(triggers)
            rows.append(f"- {joined}")

        if not rows:
            return f"没有可用管理指令，当前前缀：{prefix}"
        body = "\n".join(rows)
        return (
            f"管理指令前缀：{prefix}\n"
            "可用指令：\n"
            f"{body}\n"
            f"查看触发词：{prefix} 帮助 或 {prefix} list\n"
            "群号参数可写在指令后，例如：\n"
            f"{prefix} 加入群聊 123456789"
        )

    def _is_admin_help_subcommand(self, command_text: str) -> bool:
        normalized = str(command_text or "").strip().lower()
        return normalized in self._ADMIN_HELP_SUBCOMMANDS

    def _admin_custom_trigger_list(self, prefix: str) -> str:
        rows: list[str] = []
        for item in self.cfg.admin_commands.commands:
            action = str(item.action or "").strip()
            if action not in self._SUPPORTED_ADMIN_ACTIONS:
                continue
            triggers = self._normalize_admin_triggers(item.triggers)
            if not triggers:
                continue
            label = self._ADMIN_ACTION_LABELS.get(action, action)
            rows.append(f"- {label}: {' / '.join(triggers)}")

        if not rows:
            return f"管理指令前缀：{prefix}\n当前没有生效的自定义触发词。"

        body = "\n".join(rows)
        return (
            f"管理指令前缀：{prefix}\n"
            "当前生效的自定义触发词：\n"
            f"{body}"
        )

    @staticmethod
    def _normalize_admin_triggers(raw_triggers: list[str] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in list(raw_triggers or []):
            trigger = str(raw or "").strip()
            if not trigger:
                continue
            lowered = trigger.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(trigger)
        return out

    async def _admin_toggle_group_chat(self, *, args: str, fallback_group_id: int | None) -> str:
        group_id = self._resolve_target_group_id(args=args, fallback_group_id=fallback_group_id)
        if group_id is None:
            prefix = str(self.cfg.admin_commands.prefix or "").strip() or "<prefix>"
            return f"缺少群号。示例：{prefix} 开关群聊 123456789"

        group = self.cfg.get_group(group_id)
        created = False
        if group is None:
            group = GroupConfig(group_id=group_id, enabled=True, extra_prompt="")
            self.cfg.groups.append(group)
            created = True
        else:
            group.enabled = not bool(group.enabled)

        persist_error = await self._try_persist_group_config(
            group_id=group_id,
            enabled=bool(group.enabled),
            extra_prompt=group.extra_prompt,
        )
        if created:
            base = f"群 {group_id} 未在配置中，已自动加入并开启聊天。"
            if persist_error:
                return f"{base}\n注意：写入配置失败：{persist_error}"
            return base
        state = "开启" if group.enabled else "关闭"
        if persist_error:
            return f"群 {group_id} 聊天已{state}。\n注意：写入配置失败：{persist_error}"
        return f"群 {group_id} 聊天已{state}。"

    async def _admin_join_group_chat(self, *, args: str, fallback_group_id: int | None) -> str:
        group_id = self._resolve_target_group_id(args=args, fallback_group_id=fallback_group_id)
        if group_id is None:
            prefix = str(self.cfg.admin_commands.prefix or "").strip() or "<prefix>"
            return f"缺少群号。示例：{prefix} 加入群聊 123456789"

        group = self.cfg.get_group(group_id)
        if group is None:
            group = GroupConfig(group_id=group_id, enabled=True, extra_prompt="")
            self.cfg.groups.append(group)
            created = True
        else:
            group.enabled = True
            created = False

        persist_error = await self._try_persist_group_config(
            group_id=group_id,
            enabled=True,
            extra_prompt=group.extra_prompt,
        )
        if created:
            base = f"已加入群 {group_id} 的聊天列表，并开启回复。"
            if persist_error:
                return f"{base}\n注意：写入配置失败：{persist_error}"
            return base
        if persist_error:
            return f"群 {group_id} 已开启聊天。\n注意：写入配置失败：{persist_error}"
        return f"群 {group_id} 已开启聊天。"

    async def _admin_shutdown(self) -> str:
        if self._shutdown_handler is None:
            return "关闭失败：shutdown handler 不可用。"

        async with self._shutdown_lock:
            if self._shutdown_requested:
                return "关闭流程已在进行中。"
            self._shutdown_requested = True

        asyncio.create_task(self._run_shutdown_handler(), name="chat-admin-shutdown")
        return "收到，正在关闭程序。"

    async def _run_shutdown_handler(self) -> None:
        await asyncio.sleep(0.2)
        try:
            handler = self._shutdown_handler
            if handler is None:
                return
            await handler()
        except Exception:
            self._logger.exception("Admin shutdown handler failed")
            async with self._shutdown_lock:
                self._shutdown_requested = False

    def _resolve_target_group_id(self, *, args: str, fallback_group_id: int | None) -> int | None:
        if fallback_group_id is not None and not args.strip():
            return fallback_group_id

        match = re.search(r"\d+", str(args or ""))
        if match is None:
            return None
        try:
            value = int(match.group(0))
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    async def _persist_group_config(
        self,
        *,
        group_id: int,
        enabled: bool,
        extra_prompt: str,
    ) -> None:
        if self._config_path is None:
            return

        async with self._config_lock:
            config_data: dict[str, Any] = {}
            if self._config_path.exists():
                raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    config_data = raw

            self._upsert_group_config_data(
                config_data=config_data,
                group_id=group_id,
                enabled=enabled,
                extra_prompt=extra_prompt,
            )

            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
            temp_file.write_text(
                yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            temp_file.replace(self._config_path)

    async def _try_persist_group_config(
        self,
        *,
        group_id: int,
        enabled: bool,
        extra_prompt: str,
    ) -> str:
        try:
            await self._persist_group_config(
                group_id=group_id,
                enabled=enabled,
                extra_prompt=extra_prompt,
            )
            return ""
        except Exception as exc:
            self._logger.warning(
                "Persist group config failed: group_id=%s enabled=%s err=%s",
                group_id,
                enabled,
                exc,
            )
            return str(exc)

    @staticmethod
    def _upsert_group_config_data(
        *,
        config_data: dict[str, Any],
        group_id: int,
        enabled: bool,
        extra_prompt: str,
    ) -> None:
        groups = config_data.setdefault("groups", [])
        if not isinstance(groups, list):
            groups = []
            config_data["groups"] = groups

        for row in groups:
            if not isinstance(row, dict):
                continue
            raw_group_id = row.get("group_id")
            try:
                if int(raw_group_id) != int(group_id):
                    continue
            except (TypeError, ValueError):
                continue
            row["enabled"] = bool(enabled)
            row["extra_prompt"] = str(extra_prompt or "")
            return

        groups.append(
            {
                "group_id": int(group_id),
                "enabled": bool(enabled),
                "extra_prompt": str(extra_prompt or ""),
            },
        )

    async def _resolve_chat_speaker(
        self,
        *,
        message: dict[str, Any],
        user_id: int | None,
        group_id: int | None,
        identity: SenderIdentity,
    ) -> str:
        del message
        current_group_card = str(identity.group_card or "").strip()
        current_nickname = str(identity.nickname or "").strip()
        if group_id is not None and group_id > 0 and current_group_card:
            return current_group_card

        profile = await self.user_profiler.get_user_profile(user_id)
        display_name = self.user_profiler.display_name_for_group(
            profile,
            group_id=group_id,
            current_group_card=current_group_card,
            current_nickname=current_nickname,
        )
        if display_name:
            return display_name

        if current_group_card:
            return current_group_card
        if current_nickname:
            return current_nickname
        return str(user_id) if user_id is not None else "unknown"

    @staticmethod
    def _extract_sender_identity(message: dict[str, Any]) -> SenderIdentity:
        raw = message.get("raw", {})
        if not isinstance(raw, dict):
            return SenderIdentity(
                nickname=str(message.get("nickname", "")).strip(),
                group_card="",
            )
        sender = raw.get("sender", {})
        if not isinstance(sender, dict):
            return SenderIdentity(
                nickname=str(message.get("nickname", "")).strip(),
                group_card="",
            )
        return SenderIdentity(
            nickname=str(sender.get("nickname", "")).strip(),
            group_card=str(sender.get("card", "")).strip(),
        )

    def _extract_speaker(self, message: dict[str, Any]) -> str:
        identity = self._extract_sender_identity(message)
        if identity.group_card:
            return identity.group_card
        if identity.nickname:
            return identity.nickname
        user_id = self._to_int(message.get("user_id"))
        return str(user_id) if user_id is not None else "unknown"

    def _is_master(self, user_id: int | None) -> bool:
        master_id = self.cfg.persona.master_id
        return bool(master_id and user_id is not None and user_id == master_id)

    def _is_admin_sender(self, *, user_id: int | None, speaker: str) -> bool:
        if self._is_master(user_id) or self._is_master_name(speaker):
            return True

        admin_ids = {
            item
            for item in (
                self._to_int(value)
                for value in list(getattr(self.cfg.admin_commands, "admin_user_ids", []) or [])
            )
            if item is not None and item > 0
        }
        if user_id is not None and user_id in admin_ids:
            return True

        speaker_clean = str(speaker or "").strip()
        if not speaker_clean:
            return False
        admin_names = {
            str(item).strip().lower()
            for item in list(getattr(self.cfg.admin_commands, "admin_names", []) or [])
            if str(item).strip()
        }
        if not admin_names:
            return False
        return speaker_clean.lower() in admin_names

    def _is_master_name(self, speaker: str) -> bool:
        master_name = str(self.cfg.persona.master_name or "").strip()
        if not master_name:
            return False
        return master_name == str(speaker or "").strip()

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_sticker_request(cls, text: str) -> bool:
        clean_text = str(text or "").strip()
        if not clean_text:
            return False
        return bool(cls._STICKER_INTENT_PATTERN.search(clean_text))

    @classmethod
    def _is_low_signal_passive_group_message(cls, text: str) -> bool:
        clean_text = str(text or "").strip()
        if not clean_text:
            return True
        if len(clean_text) > 24:
            return False
        if cls._PASSIVE_GROUP_REPLY_CUE_PATTERN.search(clean_text):
            return False

        compact = re.sub(r"[\s，,。.!！?？；;:：~～…·`'\"“”‘’\-_/\(\)（）\[\]{}<>]+", "", clean_text).lower()
        if not compact:
            return True
        if len(compact) <= 1:
            return True
        return bool(cls._LOW_SIGNAL_GROUP_TEXT_PATTERN.fullmatch(compact))

    @staticmethod
    def _is_tarot_trigger(text: str) -> bool:
        return str(text or "").strip().lower() == "tarot"

    @staticmethod
    def _is_tarot3_trigger(text: str) -> bool:
        return str(text or "").strip().lower() == "tarot3"

    async def _try_handle_tarot_command(self, *, message: dict[str, Any], text: str) -> bool:
        is_tarot_single = self._is_tarot_trigger(text)
        is_tarot_three = self._is_tarot3_trigger(text)
        if not is_tarot_single and not is_tarot_three:
            return False

        mention_prefix = self._build_tarot_mention_prefix(message)
        cooldown_key = self._build_tarot_cooldown_key(message)
        now = datetime.now(timezone.utc)
        remaining_sec = self._tarot_cooldown_remaining(cooldown_key=cooldown_key, now=now)
        if remaining_sec > 0:
            await self._reply_single(
                message,
                f"{mention_prefix}抽牌冷却中，请在 {remaining_sec} 秒后再试。",
            )
            return True

        try:
            if is_tarot_three:
                draws = self._tarot_knowledge.draw_many(3, self._rng)
            else:
                draws = [self._tarot_knowledge.draw(self._rng)]
        except RuntimeError as exc:
            await self._reply_single(message, f"{mention_prefix}塔罗知识库异常：{exc}")
            self._logger.error("Tarot.Draw failed: mode=%s err=%s", "tarot3" if is_tarot_three else "tarot", exc)
            return True

        if is_tarot_three:
            parts = [f"{mention_prefix}抽出了三张牌："]
            for idx, draw in enumerate(draws, start=1):
                parts.append(f"{idx}. {draw.card.display_name_cn}（{draw.orientation_label}）：{draw.meaning}")
            content = "\n".join(parts)
            await self._reply_single(message, content)
            if cooldown_key:
                self._tarot_last_draw_at[cooldown_key] = now
            self._logger.info(
                "Tarot.Draw3: cards=%s user_id=%s group_id=%s",
                ", ".join(f"{draw.card.display_name_cn}/{draw.orientation_label}" for draw in draws),
                message.get("user_id"),
                message.get("group_id"),
            )
        else:
            draw = draws[0]
            summary = f"抽出了一张 {draw.card.display_name_cn}，{draw.orientation_label}，解释是“{draw.meaning}”。"
            image_path = self._tarot_knowledge.resolve_draw_image_path(draw)
            if image_path is not None:
                image_cq = self._build_local_image_cq(image_path)
                content = f"{mention_prefix}{image_cq}\n{summary}"
            else:
                content = f"{mention_prefix}{summary}"
            await self._reply_single(message, content)
            if cooldown_key:
                self._tarot_last_draw_at[cooldown_key] = now
            self._logger.info(
                "Tarot.Draw: card=%s orientation=%s image=%s user_id=%s group_id=%s",
                draw.card.display_name_cn,
                draw.orientation_label,
                image_path if image_path is not None else "",
                message.get("user_id"),
                message.get("group_id"),
            )
        return True

    def _build_tarot_cooldown_key(self, message: dict[str, Any]) -> str:
        user_id = self._to_int(message.get("user_id"))
        if user_id is not None and user_id > 0:
            return f"user:{user_id}"
        speaker = self._extract_speaker(message).strip().lower()
        if speaker and speaker != "unknown":
            return f"speaker:{speaker}"
        return ""

    def _tarot_cooldown_remaining(self, *, cooldown_key: str, now: datetime) -> int:
        if not cooldown_key:
            return 0
        previous = self._tarot_last_draw_at.get(cooldown_key)
        if previous is None:
            return 0
        elapsed = (now - previous).total_seconds()
        if elapsed >= float(self._TAROT_DRAW_COOLDOWN_SEC):
            return 0
        remaining = int(self._TAROT_DRAW_COOLDOWN_SEC - elapsed + 0.999)
        if remaining <= 0:
            return 1
        return remaining

    def _build_tarot_mention_prefix(self, message: dict[str, Any]) -> str:
        message_type = str(message.get("message_type", "")).strip() or "private"
        if message_type != "group":
            return ""

        user_id = self._to_int(message.get("user_id"))
        if user_id is not None and user_id > 0:
            return f"[CQ:at,qq={user_id}] "

        speaker = self._extract_speaker(message)
        if speaker and speaker != "unknown":
            return f"@{speaker} "
        return ""

    @staticmethod
    def _build_local_image_cq(path: Path) -> str:
        return f"[CQ:image,file=file:///{path.as_posix()}]"

    async def _reply_single(self, message: dict[str, Any], content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return

        message_type = str(message.get("message_type", "")).strip() or "private"
        if message_type == "group":
            group_id = self._to_int(message.get("group_id"))
            if group_id is None:
                self._logger.warning("Skip group single reply: missing group_id")
                return
            await self.bot_client.send_group_msg(group_id=group_id, message=text)
            self._logger.info("Queue.ReplySingle: target=group group_id=%s", group_id)
            return

        user_id = self._to_int(message.get("user_id"))
        if user_id is None:
            self._logger.warning("Skip private single reply: missing user_id")
            return
        await self.bot_client.send_private_msg(user_id=user_id, message=text)
        self._logger.info("Queue.ReplySingle: target=private user_id=%s", user_id)

    def _allow_auto_sticker_send(self, *, ctx: ThinkContext) -> bool:
        now = datetime.now(timezone.utc)
        is_direct = ctx.message_type != "group"
        cooldown = self._AUTO_STICKER_DIRECT_COOLDOWN_SEC if is_direct else self._AUTO_STICKER_GROUP_COOLDOWN_SEC
        previous = self._sticker_last_sent_at.get(ctx.session_id)
        if previous is not None:
            elapsed = (now - previous).total_seconds()
            if elapsed < float(cooldown):
                self._logger.info(
                    "Sticker.SkipAutoSend: session=%s reason=cooldown elapsed=%.1fs cooldown=%ss direct=%s",
                    ctx.session_id,
                    elapsed,
                    cooldown,
                    is_direct,
                )
                return False

        probability = self._clamp_probability(self._AUTO_STICKER_SEND_PROBABILITY)
        if probability <= 0.0:
            self._logger.info(
                "Sticker.SkipAutoSend: session=%s reason=probability_zero direct=%s",
                ctx.session_id,
                is_direct,
            )
            return False

        roll = self._rng.random()
        if roll > probability:
            self._logger.info(
                "Sticker.SkipAutoSend: session=%s reason=probability roll=%.4f threshold=%.4f direct=%s",
                ctx.session_id,
                roll,
                probability,
                is_direct,
            )
            return False
        return True

    def _mark_sticker_reply(self, *, session_id: str) -> None:
        self._sticker_last_sent_at[session_id] = datetime.now(timezone.utc)

    def _active_reply_probability(self, status_energy: float) -> float:
        base = self._clamp_probability(self.cfg.agent.active_reply_probability)
        if status_energy >= 70.0:
            tier_scale = 1.0
        elif status_energy >= 30.0:
            tier_scale = 0.55
        else:
            tier_scale = 0.15
        return self._clamp_probability(base * tier_scale)

    def _llm_route_probability(
        self,
        *,
        message_type: str,
        mentioned_in_window: bool,
        status_energy: float,
        is_master: bool,
        is_admin_sender: bool,
        sticker_intent: bool = False,
        source_text: str = "",
    ) -> float:
        if is_master or is_admin_sender:
            return 1.0

        if message_type != "group":
            return 1.0
        if mentioned_in_window or sticker_intent:
            return 1.0
        if self._is_low_signal_passive_group_message(source_text):
            return 0.0

        base = self._active_reply_probability(status_energy)
        return base

    def _allow_low_energy_reply(
        self,
        *,
        session_id: str,
        message_type: str,
        mentioned_in_window: bool,
        status_energy: float,
        forced_rest: bool,
    ) -> bool:
        if forced_rest:
            self._logger.info(
                (
                    "Queue.SkipReply: session=%s reason=forced_rest "
                    "status_energy=%.1f direct=%s"
                ),
                session_id,
                status_energy,
                message_type != "group" or mentioned_in_window,
            )
            return False

        now = datetime.now(timezone.utc)
        is_direct = message_type != "group" or mentioned_in_window
        cooldown = self._LOW_ENERGY_DIRECT_COOLDOWN_SEC if is_direct else self._LOW_ENERGY_GROUP_COOLDOWN_SEC

        previous = self._low_energy_last_reply_at.get(session_id)
        if previous is not None:
            elapsed = (now - previous).total_seconds()
            if elapsed < float(cooldown):
                self._logger.info(
                    (
                        "Queue.SkipReply: session=%s reason=low_energy_cooldown "
                        "status_energy=%.1f elapsed=%.1fs cooldown=%ss direct=%s"
                    ),
                    session_id,
                    status_energy,
                    elapsed,
                    cooldown,
                    is_direct,
                )
                return False

        probability = self._low_energy_reply_probability(
            status_energy=status_energy,
            is_direct=is_direct,
        )
        roll = self._rng.random()
        if roll >= probability:
            self._logger.info(
                (
                    "Queue.SkipReply: session=%s reason=low_energy_probability "
                    "status_energy=%.1f roll=%.4f threshold=%.4f direct=%s"
                ),
                session_id,
                status_energy,
                roll,
                probability,
                is_direct,
            )
            return False
        return True

    def _low_energy_reply_probability(self, *, status_energy: float, is_direct: bool) -> float:
        threshold = max(1.0, float(self._LOW_ENERGY_THRESHOLD))
        energy_ratio = self._clamp_probability(float(status_energy) / threshold)
        if is_direct:
            base = self._LOW_ENERGY_DIRECT_BASE_REPLY_PROBABILITY
            span = self._LOW_ENERGY_DIRECT_REPLY_ENERGY_SPAN
        else:
            base = self._LOW_ENERGY_GROUP_BASE_REPLY_PROBABILITY
            span = self._LOW_ENERGY_GROUP_REPLY_ENERGY_SPAN
        scale = self._clamp_probability(self.cfg.agent.active_reply_probability)
        return self._clamp_probability((base + span * energy_ratio) * scale)

    def _pick_low_energy_reply(self, *, session_id: str) -> str:
        options = list(self._LOW_ENERGY_REPLIES)
        last_reply = self._low_energy_last_reply_text.get(session_id)
        if last_reply and len(options) > 1:
            options = [item for item in options if item != last_reply] or options
        return str(self._rng.choice(options))

    def _mark_low_energy_reply(self, *, session_id: str, reply: str) -> None:
        self._low_energy_last_reply_at[session_id] = datetime.now(timezone.utc)
        self._low_energy_last_reply_text[session_id] = str(reply or "").strip()

    @staticmethod
    def _set_response_token_cap(extra_fields: dict[str, Any], cap: int) -> None:
        token_cap = max(1, int(cap))
        key = "max_completion_tokens" if "max_completion_tokens" in extra_fields else "max_tokens"
        existing = ZhiyueAgent._parse_positive_int(extra_fields.get(key))
        extra_fields[key] = min(existing, token_cap) if existing is not None else token_cap

    @staticmethod
    def _parse_positive_int(value: Any) -> int | None:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric > 0 else None

    @staticmethod
    def _clamp_probability(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _has_at_segment(segments: Any, target_ids: set[int]) -> bool:
        if not isinstance(segments, list):
            return False

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("type", "")).strip().lower() != "at":
                continue
            data = segment.get("data")
            if not isinstance(data, dict):
                continue

            for key in ("qq", "uid", "user_id"):
                target = ZhiyueAgent._to_int(data.get(key))
                if target is not None and target in target_ids:
                    return True
        return False

    @classmethod
    def _has_cq_at(cls, raw_message: Any, target_ids: set[int]) -> bool:
        if not isinstance(raw_message, str) or not raw_message.strip():
            return False

        for match in cls._CQ_AT_PATTERN.findall(raw_message):
            target = cls._to_int(match)
            if target is not None and target in target_ids:
                return True
        return False


AsyncReactAgent = ZhiyueAgent
