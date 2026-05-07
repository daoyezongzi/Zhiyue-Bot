from __future__ import annotations

import asyncio
import inspect
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, get_args, get_origin

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
from internal.topic import TopicManager
from plugins.context import ToolContext
from plugins.registry import PluginRegistry


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
    mentioned_in_window: bool
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


@dataclass(slots=True, frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class LLMThinkResult:
    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    force_silence: bool = False


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
    _STICKER_MARKER_LOOSE_PATTERN = re.compile(
        r"\[\[\s*(?:sticker|表情包)\s*[:：]\s*(?P<query>[^\]\r\n，。,;；!！?？]{1,120})\s*(?:\]\]?)?",
        re.IGNORECASE,
    )
    _STICKER_FUNC_CALL_PATTERN = re.compile(
        r"sendSticker\s*[\(（]\s*(?P<query>[^)\）\r\n]{1,120})\s*[\)）]",
        re.IGNORECASE,
    )
    _STICKER_FUNC_CALL_LOOSE_PATTERN = re.compile(
        r"sendSticker\s*[\(（]\s*(?P<query>[^)\）\r\n，。,;；!！?？]{1,120})\s*[\)）]?",
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
            r"(?:沉默观察|保持沉默|保持安静即可|保持安静|安静即可|先观察|继续观察|无必要回应|无需回应|无需回复|不需要回应|不用回应|无需额外回应|不需要额外回应(?:了)?|暂不回应|不作回应|没有新内容需要回应|先潜水|先潜水了|先潜水啦)"
            r"(?:\s*[，,、；;]\s*"
            r"(?:沉默观察|保持沉默|保持安静即可|保持安静|安静即可|先观察|继续观察|无必要回应|无需回应|无需回复|不需要回应|不用回应|无需额外回应|不需要额外回应(?:了)?|暂不回应|不作回应|没有新内容需要回应|先潜水|先潜水了|先潜水啦))*"
            r"\s*[）)]?\s*$"
        ),
        re.IGNORECASE,
    )
    _STAY_QUIET_PLACEHOLDER_PATTERN = re.compile(
        (
            r"^\s*(?:"
            r"(?:(?:请|就|那就|直接)?\s*(?:调用|call|use|invoke|执行|选择)?\s*stay[_\s-]?quiet(?:\s*[\(（][^)\r\n]{0,120}[\)）])?(?:\s*(?:工具|结束推理|结束|即可|就行|就好|吧))?)|"
            r"stayQuiet|stay_quiet|保持沉默|保持安静即可|保持安静|安静即可|不回复|无需回复|无必要回应|不需要回应|不用回应|无需额外回应|不需要额外回应(?:了)?|"
            r"\{.*?(?:stayQuiet|stay_quiet).*?\}"
            r")\s*$"
        ),
        re.IGNORECASE | re.DOTALL,
    )
    _ELLIPSIS_ONLY_PATTERN = re.compile(r"^\s*(?:\.{2,}|…{1,}|⋯{1,}|(?:\.\s*){2,})\s*$")
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
        "沉默",
        "保持沉默",
        "保持安静",
        "保持安静即可",
        "安静即可",
        "沉默观察",
        "不回复",
        "无需回复",
        "无必要回应",
        "不需要回应",
        "不用回应",
        "无需额外回应",
        "不需要额外回应",
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
    _TERMINAL_PERIOD_PATTERN = re.compile(r"。+\s*$")
    _JOKING_REPLY_CUE_PATTERN = re.compile(
        r"(?:哈哈|哈{2,}|嘿嘿|笑死|绷不住|乐子|整活|开玩笑|逗你|狗头|doge|233|xD|XD|qwq|捏|欸嘿)",
        re.IGNORECASE,
    )
    _LOG_STYLE_SELF_PREFIX_PATTERN = re.compile(
        r"^\s*(?:>\s*)?(?:(?:\[[^\]\r\n]+\]\s*)+(?:\(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\)\s*)?|\(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\)\s*)(?P<speaker>[^:\r\n]{1,32})\s*[:：]\s*(?P<body>.*)$",
        re.DOTALL,
    )
    _LEADING_QUOTES = ("\"", "'", "“", "‘", "「", "『")
    _MOOD_FAST_REPLY_HIGH_ENERGY_TOKEN_CAP = 192
    _MOOD_FAST_REPLY_MID_ENERGY_TOKEN_CAP = 144
    _MOOD_FAST_REPLY_LOW_ENERGY_TOKEN_CAP = 112
    _AUTO_STICKER_SEND_PROBABILITY = 0.22
    _AUTO_STICKER_GROUP_COOLDOWN_SEC = 300
    _AUTO_STICKER_DIRECT_COOLDOWN_SEC = 180
    _SILENCE_STICKER_SEND_PROBABILITY = 0.12
    _SILENCE_STICKER_GROUP_COOLDOWN_SEC = 420
    _TAROT_DRAW_COOLDOWN_SEC = 60
    _SUPPORTED_ADMIN_ACTIONS = {"toggle_group_chat", "join_group_chat", "shutdown"}
    _ADMIN_HELP_SUBCOMMANDS = {"帮助", "列表", "list", "help", "帮助/list", "help/list"}
    _ADMIN_ACTION_LABELS = {
        "toggle_group_chat": "开关群聊",
        "join_group_chat": "加入群聊",
        "shutdown": "关闭程序",
    }
    _SILENCE_FORCE_REPLY_SKIP_THRESHOLD = 3
    _SILENCE_FORCE_REPLY_IDLE_SEC = 180
    _NON_IMMEDIATE_REPLY_WINDOW_SEC = 180
    _TOOL_MAX_STEP = 6

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
            reply_cost_per_turn=self.cfg.personality.energy_reply_cost_per_turn,
            fatigue_silence_threshold=self.cfg.personality.energy_fatigue_silence_threshold,
            rest_lock_threshold=self.cfg.personality.energy_rest_lock_threshold,
            rest_unlock_threshold=self.cfg.personality.energy_rest_unlock_threshold,
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
        self.topic_mgr = TopicManager(cfg=self.cfg, llm=self._evolution_llm)
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
        # Keep message aggregation short to reduce response latency.
        debounce_ms = self._parse_positive_int(getattr(self.cfg.agent, "think_debounce_ms", None))
        if debounce_ms is None:
            debounce_ms = 260
        self._debounce_window_sec = max(0.08, min(float(debounce_ms) / 1000.0, 0.35))
        self._consecutive_skip_count: dict[str, int] = {}
        self._last_reply_at: dict[str, datetime] = {}
        self._sticker_last_sent_at: dict[str, datetime] = {}
        self._tarot_last_draw_at: dict[str, datetime] = {}
        self._rng = random.Random()
        self._plugin_registry = PluginRegistry()
        self._plugin_registry.register_defaults()
        self._llm_tool_schemas = self._build_llm_tool_schemas()
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
        await self.memory_mgr.start()
        await self.topic_mgr.start()
        await self.user_profiler.start()
        await self.jargon_engine.start()
        await self._reload_jargon_matcher_from_store()
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
        await self.topic_mgr.close()
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
            self._logger.info(
                (
                    "Skip group message: group not enabled group_id=%s "
                    "mentioned=%s explicit_at=%s is_master=%s is_admin=%s"
                ),
                group_id,
                mentioned_in_window,
                explicit_at_in_window,
                is_master,
                is_admin_sender,
            )
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
        await self.memory_mgr.ingest_message_memory(
            group_id=memory_group_id,
            user_id=user_id,
            content=text,
            source_ref=f"message:{self._to_int(message.get('message_id')) or 0}:user",
            source_kind="message",
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
        topic_assignment = await self.topic_mgr.ingest_user_message(
            group_id=memory_group_id,
            message_id=self._to_int(message.get("message_id")) or 0,
            user_id=user_id or 0,
            speaker=speaker,
            content=text,
            created_at=now_utc,
        )
        topic_interest_score = self._apply_topic_interest_mood(
            session_id=session_id,
            source_text=text,
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
                "status_energy=%.1f status_tier=%s fatigue=%s rest_locked=%s topic_interest=%.2f"
            ),
            session_id,
            message_type,
            user_id,
            message.get("group_id"),
            is_master,
            status_after_user.energy,
            status_after_user.energy_tier,
            status_after_user.fatigue_mode,
            status_after_user.rest_locked,
            topic_interest_score,
        )

        low_energy_mode = bool(status_after_user.fatigue_mode)
        rest_locked = bool(status_after_user.rest_locked)
        final_reply = ""
        forced_rest = rest_locked

        if rest_locked:
            self._logger.info(
                "Queue.SkipReply: session=%s reason=rest_locked status_energy=%.1f",
                session_id,
                status_after_user.energy,
            )
            self._track_reply_skip(session_id=session_id, reason="rest_locked")
            return

        if low_energy_mode:
            self._logger.info(
                "Queue.SkipReply: session=%s reason=low_energy_silent status_energy=%.1f",
                session_id,
                status_after_user.energy,
            )
            self._track_reply_skip(session_id=session_id, reason="low_energy_silent")
            return

        if not final_reply:
            force_active_reply = self._should_force_active_reply(
                session_id=session_id,
                message_type=message_type,
                mentioned_in_window=mentioned_in_window,
                source_text=text,
                is_master=is_master,
                is_admin_sender=is_admin_sender,
            )
            llm_route_probability = 1.0
            if not force_active_reply:
                llm_route_probability = self._llm_route_probability(
                    message_type=message_type,
                    mentioned_in_window=mentioned_in_window,
                    status_energy=status_after_user.energy,
                    is_master=is_master,
                    is_admin_sender=is_admin_sender,
                    sticker_intent=sticker_intent,
                    source_text=text,
                    topic_interest_score=topic_interest_score,
                )
            else:
                self._logger.info(
                    "Queue.ForceReply: session=%s reason=silence_guard skip_count=%s",
                    session_id,
                    self._consecutive_skip_count.get(session_id, 0),
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
                    self._track_reply_skip(session_id=session_id, reason="llm_route_probability")
                    return

            retrieval = await self.memory_mgr.retrieve_for_prompt(
                text=text,
                session_id=session_id,
                group_id=group_id,
                top_k=self.cfg.memory.rag_top_k,
            )
            topic_context = await self.topic_mgr.build_prompt_context(
                group_id=memory_group_id,
                session_id=session_id,
                query_text=text,
                current_topic_id=self._to_int(topic_assignment.get("topic_id")),
            )
            history_background = list(retrieval.history_background)
            related_knowledge = list(retrieval.related_knowledge)
            current_topic_block = str(topic_context.get("current_topic", "") or "").strip()
            if current_topic_block:
                history_background.insert(0, f"当前话题上下文:\n{current_topic_block}")
            for archived_topic_block in list(topic_context.get("archived_topics", []) or []):
                clean_archived_topic_block = str(archived_topic_block or "").strip()
                if clean_archived_topic_block:
                    history_background.append(f"相关历史话题:\n{clean_archived_topic_block}")

            social_background = await self.user_profiler.build_social_background(user_id, speaker)
            ctx = await self._build_context(
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
            ctx = await self._before_llm_think(ctx)
            reply = await self._llm_think(
                ctx,
                max_tokens_override=self._mood_reply_token_cap(ctx),
            )
            reply = await self._after_llm_think(ctx, reply)
            if not reply and force_active_reply:
                reply = await self._recover_required_reply(ctx, reason="silence_guard")
            if not reply:
                if await self._maybe_send_silence_sticker(ctx=ctx, reason="empty_after_llm_think"):
                    self._mark_reply_sent(session_id=session_id)
                    return
                self._logger.info("Queue.SkipReply: session=%s reason=empty_after_llm_think", session_id)
                self._track_reply_skip(session_id=session_id, reason="empty_after_llm_think")
                return

            final_reply = await self._response_post_process(reply, ctx)
            if not final_reply:
                if await self._maybe_send_silence_sticker(ctx=ctx, reason="empty_after_post_process"):
                    self._mark_reply_sent(session_id=session_id)
                    return
                self._logger.info("Queue.SkipReply: session=%s reason=empty_after_post_process", session_id)
                self._track_reply_skip(session_id=session_id, reason="empty_after_post_process")
                return

            final_reply, forced_rest = await self.status_engine.apply_reply_policy(final_reply)
            if not final_reply:
                self._logger.info("Queue.SkipReply: session=%s reason=reply_policy_empty", session_id)
                self._track_reply_skip(session_id=session_id, reason="reply_policy_empty")
                return

        try:
            await self._reply(message, final_reply)
        except Exception:
            self._track_reply_skip(session_id=session_id, reason="send_failed")
            raise
        self._mark_reply_sent(session_id=session_id)

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
        await self.memory_mgr.ingest_message_memory(
            group_id=memory_group_id,
            user_id=self.cfg.persona.qq,
            content=final_reply,
            source_ref=f"message:{self._to_int(message.get('message_id')) or 0}:assistant",
            source_kind="message",
        )
        await self.topic_mgr.ingest_assistant_reply(
            group_id=memory_group_id,
            topic_id=self._to_int(topic_assignment.get("topic_id")),
            speaker=self.cfg.persona.name,
            content=final_reply,
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
            mentioned_in_window=mentioned_in_window,
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
    ) -> LLMThinkResult:
        extra_fields: dict[str, Any] = dict(self.cfg.llm.extra_fields)
        if self.cfg.llm.max_response_tokens > 0:
            if "max_tokens" not in extra_fields and "max_completion_tokens" not in extra_fields:
                extra_fields["max_tokens"] = self.cfg.llm.max_response_tokens

        if max_tokens_override is not None and max_tokens_override > 0:
            self._set_response_token_cap(extra_fields, int(max_tokens_override))
        tools_enabled = bool(self._llm_tool_schemas)
        if tools_enabled:
            extra_fields.setdefault("tools", self._llm_tool_schemas)
            extra_fields.setdefault("tool_choice", "auto")

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

        conversation = [dict(item) for item in llm_messages]
        tool_ctx = self._build_tool_context(ctx)
        last_text_reply = ""
        force_silence = False

        for step in range(1, self._TOOL_MAX_STEP + 1):
            data = await self.llm.request_chat_completion(conversation, extra_fields)
            if not data and tools_enabled:
                fallback_fields = dict(extra_fields)
                fallback_fields.pop("tools", None)
                fallback_fields.pop("tool_choice", None)
                data = await self.llm.request_chat_completion(conversation, fallback_fields)
                if data:
                    tools_enabled = False
                    extra_fields = fallback_fields
                    self._logger.info(
                        "LLM.Tool fallback: session=%s reason=provider_rejected_tools",
                        ctx.session_id,
                    )
            think_result = self._parse_llm_think_response(data)
            last_text_reply = think_result.content or last_text_reply
            self._logger.info(
                "LLM.Think step=%s session=%s tone=%s has_reply=%s tool_calls=%s preview=%s",
                step,
                ctx.session_id,
                ctx.style.tone,
                bool(think_result.content),
                len(think_result.tool_calls),
                think_result.content[:120],
            )

            if not think_result.tool_calls:
                return LLMThinkResult(content=think_result.content, tool_calls=[], force_silence=force_silence)

            conversation.append(
                self._build_assistant_tool_call_message(think_result.content, think_result.tool_calls),
            )

            handled_any = False
            for tool_call in think_result.tool_calls:
                tool_name = str(tool_call.name or "").strip()
                if not tool_name:
                    continue
                handled_any = True
                tool_success = False
                tool_error = ""
                plugin = self._plugin_registry.get(tool_name)
                if plugin is None:
                    tool_payload: Any = {"ok": False, "error": "tool_not_found", "name": tool_name}
                    tool_error = "tool_not_found"
                    self._logger.info(
                        "LLM.Tool ignored: session=%s tool=%s reason=not_registered",
                        ctx.session_id,
                        tool_name,
                    )
                else:
                    arguments = self._filter_tool_arguments(plugin, self._parse_tool_arguments(tool_call.arguments))
                    try:
                        tool_payload = await plugin.run(tool_ctx, **arguments)
                        tool_success = True
                        self._logger.info(
                            "LLM.Tool executed: session=%s tool=%s result=%s",
                            ctx.session_id,
                            tool_name,
                            str(tool_payload)[:160],
                        )
                    except Exception as exc:
                        tool_payload = {"ok": False, "error": str(exc), "name": tool_name}
                        tool_error = str(exc)
                        self._logger.warning(
                            "LLM.Tool failed: session=%s tool=%s err=%s",
                            ctx.session_id,
                            tool_name,
                            exc,
                        )

                try:
                    await self.memory_mgr.record_tool_call(
                        session_id=ctx.session_id,
                        message_type=ctx.message_type,
                        group_id=ctx.group_id if ctx.group_id is not None else 0,
                        user_id=ctx.user_id if ctx.user_id is not None else 0,
                        speaker=ctx.speaker,
                        step=step,
                        tool_call_id=str(tool_call.id or ""),
                        tool_name=tool_name,
                        arguments=self._parse_tool_arguments(tool_call.arguments),
                        success=tool_success,
                        result=tool_payload,
                        error=tool_error,
                    )
                except Exception as exc:
                    self._logger.warning(
                        "LLM.Tool log persist failed: session=%s tool=%s err=%s",
                        ctx.session_id,
                        tool_name,
                        exc,
                    )

                conversation.append(
                    self._build_tool_result_message(
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        payload=tool_payload,
                    ),
                )

                if tool_name == "stayQuiet":
                    force_silence = True
                    break

            if force_silence:
                return LLMThinkResult(content="", tool_calls=[], force_silence=True)
            if not handled_any:
                break

        self._logger.info(
            "LLM.Think loop reached max steps: session=%s max_step=%s",
            ctx.session_id,
            self._TOOL_MAX_STEP,
        )
        return LLMThinkResult(content=last_text_reply, tool_calls=[], force_silence=force_silence)

    async def _after_llm_think(self, ctx: ThinkContext, think_result: LLMThinkResult) -> str:
        must_reply = self._must_reply_in_context(ctx)
        if think_result.force_silence:
            if must_reply:
                fallback = await self._recover_required_reply(ctx, reason="stayQuiet_tool")
                if fallback:
                    return fallback
            self._logger.info("LLM.Reply suppressed: session=%s reason=stayQuiet_tool", ctx.session_id)
            return ""

        tool_result = await self._apply_tool_results(ctx, think_result)
        if bool(tool_result.get("force_silence", False)):
            if must_reply:
                fallback = await self._recover_required_reply(ctx, reason="silence_state_flag")
                if fallback:
                    return fallback
            self._logger.info("LLM.Reply suppressed: session=%s reason=silence_state_flag", ctx.session_id)
            return ""
        clean_reply = str(tool_result.get("clean_reply", think_result.content) or "").strip()
        if self._should_force_silence(clean_reply):
            if must_reply:
                fallback = await self._recover_required_reply(ctx, reason="silence_placeholder")
                if fallback:
                    return fallback
            self._logger.info("LLM.Reply suppressed: session=%s reason=silence_placeholder", ctx.session_id)
            return ""
        if not clean_reply and must_reply:
            fallback = await self._recover_required_reply(ctx, reason="empty_reply")
            if fallback:
                return fallback
        return clean_reply

    def _must_reply_in_context(self, ctx: ThinkContext) -> bool:
        if ctx.message_type != "group":
            return True
        if ctx.mentioned_in_window:
            return True
        clean_source = str(ctx.source_text or "").strip()
        if not clean_source:
            return False
        if self._PASSIVE_GROUP_REPLY_CUE_PATTERN.search(clean_source):
            return True
        compact = re.sub(r"\s+", "", clean_source)
        if len(compact) >= 20 and not self._is_low_signal_passive_group_message(clean_source):
            return True
        return False

    async def _recover_required_reply(self, ctx: ThinkContext, *, reason: str) -> str:
        extra_fields: dict[str, Any] = dict(self.cfg.llm.extra_fields)
        if self.cfg.llm.max_response_tokens > 0:
            if "max_tokens" not in extra_fields and "max_completion_tokens" not in extra_fields:
                extra_fields["max_tokens"] = self.cfg.llm.max_response_tokens
        self._set_response_token_cap(extra_fields, 96)

        prompt = (
            "你上一轮没有给出可直接发送的回复。"
            "现在必须回复且只输出正文：1-2句，简短自然。"
            "禁止输出 stayQuiet、沉默说明、工具调用、JSON、代码块。"
        )
        messages = [dict(item) for item in ctx.llm_messages]
        messages.append({"role": "user", "content": prompt})
        data = await self.llm.request_chat_completion(messages, extra_fields)
        recovered = self._parse_llm_think_response(data)
        clean = self._normalize_reply_spaces(str(recovered.content or "")).strip()
        if not clean:
            self._logger.info(
                "LLM.Reply recover failed: session=%s reason=%s detail=empty",
                ctx.session_id,
                reason,
            )
            return ""
        if self._should_force_silence(clean):
            self._logger.info(
                "LLM.Reply recover failed: session=%s reason=%s detail=silence_like",
                ctx.session_id,
                reason,
            )
            return ""
        self._logger.info(
            "LLM.Reply recovered: session=%s reason=%s preview=%s",
            ctx.session_id,
            reason,
            clean[:120],
        )
        return clean

    async def _plan_tool_calls(self, ctx: ThinkContext) -> list[str]:
        del ctx
        return []

    async def _classify_style_context(self, ctx: ThinkContext) -> StyleClassification:
        energy_ratio = self._clamp_probability(
            (ctx.status.energy / 100.0) if ctx.status is not None else ctx.mood.energy,
        )
        mood_value = max(-1.0, min(1.0, float(ctx.mood.valence)))
        return self.jargon_mgr.classify_style(
            mood_value,
            energy_ratio,
            speaker_is_master=ctx.is_master,
        )

    async def _apply_tool_results(self, ctx: ThinkContext, think_result: LLMThinkResult) -> dict[str, Any]:
        raw_reply = str(think_result.content or "")
        clean_reply = raw_reply

        explicit_sticker_request = self._is_sticker_request(ctx.source_text)
        sticker_queries = self._extract_sticker_queries(raw_reply)

        if explicit_sticker_request and not sticker_queries:
            fallback_query = self._extract_sticker_fallback_query(raw_reply, ctx.source_text)
            if fallback_query:
                sticker_queries.append(fallback_query)

        sticker_sent = False
        used_query = ""
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

        clean_reply = self._strip_sticker_control_leaks(clean_reply)
        if sticker_sent:
            clean_reply = self._FAKE_STICKER_TEXT_PATTERN.sub("", clean_reply)
            clean_reply = self._normalize_reply_spaces(clean_reply)
            if not clean_reply and not explicit_sticker_request:
                clean_reply = "嗯。"
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

    def _build_llm_tool_schemas(self) -> list[dict[str, Any]]:
        preferred = [
            "searchStickers",
            "sendSticker",
            "queryMemory",
            "saveMemory",
            "searchJargon",
            "searchStyleCards",
            "getMemberInfo",
            "getRecentMessages",
            "getGroupMemberDetail",
            "stayQuiet",
        ]
        out: list[dict[str, Any]] = []
        for name in preferred:
            plugin = self._plugin_registry.get(name)
            if plugin is None:
                continue
            schema = self._plugin_to_tool_schema(plugin)
            if schema:
                out.append(schema)
        return out

    def _plugin_to_tool_schema(self, plugin: Any) -> dict[str, Any] | None:
        plugin_name = str(getattr(plugin, "name", "")).strip()
        if not plugin_name:
            return None
        description = str(getattr(plugin, "description", "")).strip() or plugin_name

        handler = getattr(plugin, "handler", None)
        if not callable(handler):
            return {
                "type": "function",
                "function": {
                    "name": plugin_name,
                    "description": description,
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
                },
            }

        signature = inspect.signature(handler)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in signature.parameters.items():
            if name in {"self", "ctx"}:
                continue
            schema = self._annotation_to_json_schema(param.annotation)
            if param.default is inspect._empty:
                required.append(name)
            properties[name] = schema

        params_schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
        if required:
            params_schema["required"] = required
        return {
            "type": "function",
            "function": {
                "name": plugin_name,
                "description": description,
                "parameters": params_schema,
            },
        }

    @staticmethod
    def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
        if annotation in {inspect._empty, Any}:
            return {"type": "string"}
        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}

        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in {list, tuple, set}:
            item_annotation = args[0] if args else Any
            return {"type": "array", "items": ZhiyueAgent._annotation_to_json_schema(item_annotation)}
        if origin in {dict}:
            return {"type": "object"}
        if args:
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return ZhiyueAgent._annotation_to_json_schema(non_none[0])
        return {"type": "string"}

    def _filter_tool_arguments(self, plugin: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        if not arguments:
            return {}
        handler = getattr(plugin, "handler", None)
        if not callable(handler):
            return dict(arguments)
        signature = inspect.signature(handler)
        allowed: set[str] = set()
        for name in signature.parameters:
            if name in {"self", "ctx"}:
                continue
            allowed.add(name)
        return {key: value for key, value in arguments.items() if key in allowed}

    def _build_tool_context(self, ctx: ThinkContext) -> ToolContext:
        return ToolContext(
            group_id=ctx.group_id if ctx.group_id is not None else 0,
            memory_mgr=self.memory_mgr,
            bot=self.bot_client,
            agent=self,
            speak_callback=self._make_tool_speak_callback(ctx),
        )

    def _make_tool_speak_callback(self, ctx: ThinkContext) -> Callable[[int, str, int | None, list[int] | None], Awaitable[int]]:
        async def _callback(
            group_id: int,
            content: str,
            reply_to: int | None = None,
            mentions: list[int] | None = None,
        ) -> int:
            if ctx.message_type == "group" and ctx.group_id is not None and ctx.group_id > 0:
                return await self.bot_client.send_group_message(
                    group_id=ctx.group_id,
                    content=str(content or ""),
                    reply_to=reply_to,
                    mentions=mentions,
                )
            if ctx.user_id is not None and ctx.user_id > 0:
                return await self.bot_client.send_private_message(
                    user_id=ctx.user_id,
                    content=str(content or ""),
                )
            if group_id > 0:
                return await self.bot_client.send_group_message(
                    group_id=group_id,
                    content=str(content or ""),
                    reply_to=reply_to,
                    mentions=mentions,
                )
            raise ValueError("speak callback has no valid target")

        return _callback

    @staticmethod
    def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
        raw = str(raw_arguments or "").strip()
        if not raw:
            return {}
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(loaded, dict):
            return loaded
        return {}

    @staticmethod
    def _build_assistant_tool_call_message(content: str, tool_calls: list[LLMToolCall]) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": str(content or ""),
        }
        normalized_calls: list[dict[str, Any]] = []
        for idx, call in enumerate(tool_calls):
            normalized_calls.append(
                {
                    "id": str(call.id or f"tool_call_{idx}"),
                    "type": "function",
                    "function": {
                        "name": str(call.name or ""),
                        "arguments": str(call.arguments or "{}"),
                    },
                },
            )
        message["tool_calls"] = normalized_calls
        return message

    @staticmethod
    def _build_tool_result_message(*, tool_call_id: str, tool_name: str, payload: Any) -> dict[str, Any]:
        content: str
        if isinstance(payload, str):
            content = payload
        else:
            try:
                content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
            except TypeError:
                content = str(payload)
        return {
            "role": "tool",
            "tool_call_id": str(tool_call_id or ""),
            "name": str(tool_name or ""),
            "content": content[:4000],
        }

    @staticmethod
    def _parse_llm_message_content(message: Any) -> str:
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type", "")).strip() != "text":
                    continue
                text_value = part.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
            return "".join(text_parts).strip()
        return ""

    @classmethod
    def _parse_llm_think_response(cls, data: dict[str, Any]) -> LLMThinkResult:
        if not isinstance(data, dict):
            return LLMThinkResult(content="", tool_calls=[])
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return LLMThinkResult(content="", tool_calls=[])
        first = choices[0]
        if not isinstance(first, dict):
            return LLMThinkResult(content="", tool_calls=[])
        message = first.get("message")
        content = cls._parse_llm_message_content(message)

        tool_calls: list[LLMToolCall] = []
        if isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for idx, row in enumerate(raw_tool_calls):
                    if not isinstance(row, dict):
                        continue
                    function = row.get("function")
                    if not isinstance(function, dict):
                        continue
                    tool_name = str(function.get("name", "")).strip()
                    if not tool_name:
                        continue
                    arguments = str(function.get("arguments", "") or "")
                    call_id = str(row.get("id", "") or f"tool_call_{idx}").strip()
                    tool_calls.append(LLMToolCall(id=call_id, name=tool_name, arguments=arguments))
            elif isinstance(message.get("function_call"), dict):
                legacy = message.get("function_call") or {}
                tool_name = str(legacy.get("name", "")).strip()
                if tool_name:
                    tool_calls.append(
                        LLMToolCall(
                            id="function_call_0",
                            name=tool_name,
                            arguments=str(legacy.get("arguments", "") or ""),
                        ),
                    )
        return LLMThinkResult(content=content, tool_calls=tool_calls)

    def _extract_sticker_queries(self, reply: str) -> list[str]:
        clean_reply = str(reply or "")
        if not clean_reply:
            return []

        out: list[str] = []
        seen: set[str] = set()

        def _append(raw_query: str) -> None:
            query = str(raw_query or "").strip()
            query = re.sub(r"^[\s\"'`“”‘’\[\]\(\)\{\}<>]+", "", query)
            query = re.sub(r"[\s\"'`“”‘’\[\]\(\)\{\}<>,，。.!！?？;；:：]+$", "", query)
            if not query:
                return
            lowered = query.lower()
            if lowered in seen:
                return
            seen.add(lowered)
            out.append(query)

        for pattern in (
            self._STICKER_MARKER_PATTERN,
            self._STICKER_MARKER_LOOSE_PATTERN,
            self._STICKER_FUNC_CALL_PATTERN,
            self._STICKER_FUNC_CALL_LOOSE_PATTERN,
        ):
            for match in pattern.finditer(clean_reply):
                _append(match.group("query"))
        return out

    @classmethod
    def _strip_sticker_control_leaks(cls, text: str) -> str:
        clean = str(text or "")
        for pattern in (
            cls._STICKER_MARKER_PATTERN,
            cls._STICKER_MARKER_LOOSE_PATTERN,
            cls._STICKER_FUNC_CALL_PATTERN,
            cls._STICKER_FUNC_CALL_LOOSE_PATTERN,
        ):
            clean = pattern.sub("", clean)
        return clean

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

            file_path = str(self.sticker_collector.resolve_local_file_path(file_name))
            as_sticker = self.sticker_collector.is_sticker_item(item)
            if group_id is not None and hasattr(self.bot_client, "send_group_image"):
                await self.bot_client.send_group_image(group_id=group_id, file_path=file_path, as_sticker=as_sticker)
            elif user_id is not None and hasattr(self.bot_client, "send_private_image"):
                await self.bot_client.send_private_image(user_id=user_id, file_path=file_path, as_sticker=as_sticker)
            else:
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
                    file_path = str(self.sticker_collector.resolve_local_file_path(file_name))
                    as_sticker = self.sticker_collector.is_sticker_item(random_item)
                    if group_id is not None and hasattr(self.bot_client, "send_group_image"):
                        await self.bot_client.send_group_image(group_id=group_id, file_path=file_path, as_sticker=as_sticker)
                    elif user_id is not None and hasattr(self.bot_client, "send_private_image"):
                        await self.bot_client.send_private_image(user_id=user_id, file_path=file_path, as_sticker=as_sticker)
                    else:
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
            mood=max(-1.0, min(1.0, float(ctx.mood.valence))),
            energy=energy_ratio,
            speaker_is_master=ctx.is_master,
        )
        processed = await self.jargon_engine.apply_to_reply(styled)
        clean = self._strip_self_prefix(processed)
        clean = self._strip_sticker_control_leaks(clean)
        clean = self._normalize_reply_spaces(clean)
        clean = self._polish_reply_punctuation(clean, tone_key=ctx.style.tone_key)
        if self._is_silence_placeholder_reply(clean):
            self._logger.info("LLM.Reply suppressed: session=%s reason=silence_placeholder_post", ctx.session_id)
            return ""
        return clean

    @classmethod
    def _polish_reply_punctuation(cls, text: str, *, tone_key: str) -> str:
        clean = str(text or "").strip()
        if not clean:
            return ""

        lines = clean.split("\n")
        out: list[str] = []
        for line in lines:
            current = line.rstrip()
            if not current:
                out.append(current)
                continue
            if not cls._TERMINAL_PERIOD_PATTERN.search(current):
                out.append(current)
                continue

            if cls._looks_like_joking_reply(current, tone_key=tone_key):
                current = cls._TERMINAL_PERIOD_PATTERN.sub("）", current)
                out.append(current)
                continue

            if cls._should_weaken_terminal_period(current, tone_key=tone_key):
                current = cls._TERMINAL_PERIOD_PATTERN.sub("", current)
            out.append(current)
        return "\n".join(out).strip()

    @classmethod
    def _looks_like_joking_reply(cls, text: str, *, tone_key: str) -> bool:
        if cls._JOKING_REPLY_CUE_PATTERN.search(str(text or "")):
            return True
        if tone_key not in {"light", "exaggerate"}:
            return False
        lowered = str(text or "").lower()
        return any(token in lowered for token in ("不是吧", "你小子", "离谱", "好好好", "行行行"))

    @staticmethod
    def _should_weaken_terminal_period(text: str, *, tone_key: str) -> bool:
        if tone_key in {"light", "direct", "exaggerate"}:
            return True
        compact = re.sub(r"\s+", "", str(text or ""))
        return len(compact) <= 20

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

    @classmethod
    def _should_force_silence(cls, text: str) -> bool:
        clean = str(text or "").strip()
        if not clean:
            return False

        if cls._is_silence_placeholder_reply(clean):
            return True
        return bool(cls._ELLIPSIS_ONLY_PATTERN.fullmatch(clean))

    async def _sync_learned_jargon(self, term: str, meaning: str) -> None:
        clean_term = str(term).strip()
        if not clean_term:
            return
        await self.jargon_mgr.add(clean_term, str(meaning or "").strip())

    async def _reload_jargon_matcher_from_store(self) -> None:
        rows = await self.jargon_lexicon_store.get_entries()
        mapping: dict[str, str] = {}
        for row in rows:
            term = str(getattr(row, "jargon", "") or "").strip()
            if not term:
                continue
            meaning = str(getattr(row, "meaning", "") or "").strip() or str(getattr(row, "standard", "") or "").strip()
            mapping[term] = meaning
        await self.jargon_mgr.reload(mapping)

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
        long_term = await self.memory_mgr.get_runtime_snapshot()
        tool_call_stats = await self.memory_mgr.get_tool_call_stats()
        history_summary = self._history.short_term_summary(max_sessions=8, max_messages=3)
        topic_snapshot = await self.topic_mgr.get_runtime_snapshot()
        return {
            "energy": round(status.energy, 2),
            "energy_tier": status.energy_tier,
            "fatigue_mode": status.fatigue_mode,
            "forced_rest": status.forced_rest,
            "rest_locked": status.rest_locked,
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
            "long_term_memory": long_term,
            "tool_calls": tool_call_stats,
            "topic_system": topic_snapshot,
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
                "rest_locked": snapshot.rest_locked,
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
        if status.rest_locked:
            policy = "休息锁定中，本轮不回复。"
        elif tier == "充沛":
            policy = "维持常规清冷人设，按正常节奏短答。"
        elif tier == "一般":
            policy = "保持简洁回复，不展开闲聊。"
        else:
            policy = "低精力时保持沉默。"
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
        safe_reply = self._normalize_reply_spaces(self._strip_sticker_control_leaks(reply))
        if safe_reply != str(reply or "").strip():
            self._logger.info(
                "SendChain.Sanitize: removed_sticker_control_text target=%s message_id=%s",
                str(message.get("message_type", "")).strip() or "private",
                message.get("message_id"),
            )

        parts = self._split_reply_parts(safe_reply)
        if not parts:
            self._logger.info("SendChain.Skip: reason=empty_reply_parts")
            return

        message_type = str(message.get("message_type", "")).strip() or "private"
        self._logger.info(
            "SendChain.Prepare: target=%s message_id=%s parts=%s",
            message_type,
            message.get("message_id"),
            len(parts),
        )
        if message_type == "group":
            group_id = self._to_int(message.get("group_id"))
            if group_id is None:
                self._logger.warning("Skip group reply: missing group_id")
                return
            reply_to = self._resolve_reply_to_message_id(message)
            for idx, part in enumerate(parts):
                try:
                    part_reply_to = reply_to if idx == 0 else None
                    echo = await self.bot_client.send_group_message(
                        group_id=group_id,
                        content=part,
                        reply_to=part_reply_to,
                    )
                except Exception as exc:
                    self._logger.warning(
                        "SendChain.Failed: target=group group_id=%s part=%s/%s reply_to=%s err=%s",
                        group_id,
                        idx + 1,
                        len(parts),
                        part_reply_to,
                        exc,
                    )
                    raise
                self._logger.info(
                    "SendChain.Sent: target=group group_id=%s part=%s/%s reply_to=%s echo=%s preview=%s",
                    group_id,
                    idx + 1,
                    len(parts),
                    part_reply_to,
                    echo,
                    part[:80],
                )
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
            try:
                echo = await self.bot_client.send_private_msg(user_id=user_id, message=part)
            except Exception as exc:
                self._logger.warning(
                    "SendChain.Failed: target=private user_id=%s part=%s/%s err=%s",
                    user_id,
                    idx + 1,
                    len(parts),
                    exc,
                )
                raise
            self._logger.info(
                "SendChain.Sent: target=private user_id=%s part=%s/%s echo=%s preview=%s",
                user_id,
                idx + 1,
                len(parts),
                echo,
                part[:80],
            )
            self._logger.info(
                "Queue.Reply: target=private user_id=%s part=%s/%s",
                user_id,
                idx + 1,
                len(parts),
            )
            if idx + 1 < len(parts):
                await asyncio.sleep(self._reply_gap_seconds(part))

    def _resolve_reply_to_message_id(self, message: dict[str, Any]) -> int | None:
        if str(message.get("message_type", "")).strip() != "group":
            return None

        message_id = self._to_int(message.get("message_id"))
        if message_id is None or message_id <= 0:
            return None

        message_time = self._extract_message_time_utc(message)
        if message_time is None:
            return None

        age_seconds = (datetime.now(timezone.utc) - message_time).total_seconds()
        if age_seconds < float(self._NON_IMMEDIATE_REPLY_WINDOW_SEC):
            return None

        self._logger.info(
            "SendChain.ReplyMode: target=group message_id=%s age_seconds=%.1f threshold=%s mode=reply",
            message_id,
            age_seconds,
            self._NON_IMMEDIATE_REPLY_WINDOW_SEC,
        )
        return message_id

    @staticmethod
    def _extract_message_time_utc(message: dict[str, Any]) -> datetime | None:
        raw_time = message.get("time")
        if raw_time is None:
            return None
        try:
            timestamp = float(raw_time)
        except (TypeError, ValueError):
            return None
        if timestamp <= 0:
            return None
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

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
        base = 0.12
        length_factor = min(0.35, max(0.0, len(part)) * 0.008)
        jitter = self._rng.uniform(0.02, 0.10)
        return min(0.72, base + length_factor + jitter)

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

    async def _maybe_send_silence_sticker(self, *, ctx: ThinkContext, reason: str) -> bool:
        if not self._allow_silence_sticker_send(ctx=ctx):
            return False

        sent = await self._send_sticker_from_library(ctx, "随机")
        if not sent:
            self._logger.info(
                "Sticker.Silence skipped: session=%s group_id=%s reason=%s detail=send_failed",
                ctx.session_id,
                ctx.group_id,
                reason,
            )
            return False

        self._mark_sticker_reply(session_id=ctx.session_id)
        self._logger.info(
            "Sticker.Silence sent: session=%s group_id=%s reason=%s",
            ctx.session_id,
            ctx.group_id,
            reason,
        )
        return True

    def _allow_silence_sticker_send(self, *, ctx: ThinkContext) -> bool:
        if ctx.message_type != "group":
            return False
        if ctx.group_id is None or ctx.group_id <= 0:
            return False
        if ctx.mentioned_in_window:
            return False
        if not bool(getattr(self.cfg.sticker, "enabled", True)):
            return False

        now = datetime.now(timezone.utc)
        previous = self._sticker_last_sent_at.get(ctx.session_id)
        if previous is not None:
            elapsed = (now - previous).total_seconds()
            if elapsed < float(self._SILENCE_STICKER_GROUP_COOLDOWN_SEC):
                self._logger.info(
                    "Sticker.Silence skipped: session=%s reason=cooldown elapsed=%.1fs cooldown=%ss",
                    ctx.session_id,
                    elapsed,
                    self._SILENCE_STICKER_GROUP_COOLDOWN_SEC,
                )
                return False

        probability = self._clamp_probability(self._SILENCE_STICKER_SEND_PROBABILITY)
        if probability <= 0.0:
            return False

        roll = self._rng.random()
        if roll > probability:
            self._logger.info(
                "Sticker.Silence skipped: session=%s reason=probability roll=%.4f threshold=%.4f",
                ctx.session_id,
                roll,
                probability,
            )
            return False
        return True

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

    def _mood_reply_token_cap(self, ctx: ThinkContext) -> int:
        configured_cap = self._parse_positive_int(getattr(self.cfg.llm, "max_response_tokens", None))
        energy = float(ctx.status.energy) if ctx.status is not None else 60.0
        if energy >= 70.0:
            target_cap = self._MOOD_FAST_REPLY_HIGH_ENERGY_TOKEN_CAP
        elif energy >= 40.0:
            target_cap = self._MOOD_FAST_REPLY_MID_ENERGY_TOKEN_CAP
        else:
            target_cap = self._MOOD_FAST_REPLY_LOW_ENERGY_TOKEN_CAP
        if configured_cap is not None:
            return max(1, min(configured_cap, target_cap))
        return target_cap

    def _active_reply_probability(self, status_energy: float) -> float:
        base = self._clamp_probability(self.cfg.agent.active_reply_probability)
        if status_energy >= 70.0:
            tier_scale = 1.15
        elif status_energy >= 30.0:
            tier_scale = 0.92
        else:
            tier_scale = 0.55
        probability = self._clamp_probability(base * tier_scale)
        if status_energy >= 70.0:
            probability = max(probability, 0.50)
        elif status_energy >= 30.0:
            probability = max(probability, 0.35)
        else:
            probability = max(probability, 0.18)
        return self._clamp_probability(probability)

    def _apply_topic_interest_mood(self, *, session_id: str, source_text: str) -> float:
        if not bool(getattr(self.cfg.personality, "topic_interest_enabled", True)):
            return 0.0

        clean_text = str(source_text or "").strip()
        if not clean_text:
            return 0.0
        if self._is_low_signal_passive_group_message(clean_text):
            return 0.0

        interest_score = self._clamp_probability(self.personality.topic_interest_score(clean_text))
        if interest_score <= 0:
            return 0.0

        valence_boost = max(0.0, float(getattr(self.cfg.personality, "topic_interest_mood_boost", 0.08)))
        sociability_boost = max(
            0.0,
            float(getattr(self.cfg.personality, "topic_interest_sociability_boost", 0.06)),
        )
        valence_delta = valence_boost * interest_score
        sociability_delta = sociability_boost * interest_score
        if valence_delta <= 0 and sociability_delta <= 0:
            return interest_score

        current = self.personality.get_current_mood()
        updated = self.personality.set_mood(
            valence=current.valence + valence_delta,
            energy=current.energy,
            sociability=current.sociability + sociability_delta,
        )
        self._logger.info(
            (
                "Mood.TopicInterest: session=%s score=%.2f valence_delta=%.3f "
                "sociability_delta=%.3f valence=%.2f sociability=%.2f"
            ),
            session_id,
            interest_score,
            valence_delta,
            sociability_delta,
            updated.valence,
            updated.sociability,
        )
        return interest_score

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
        topic_interest_score: float = 0.0,
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
        clean_text = str(source_text or "").strip()
        if self._PASSIVE_GROUP_REPLY_CUE_PATTERN.search(clean_text):
            base = max(base, 0.72)
        compact_len = len(re.sub(r"\s+", "", clean_text))
        if compact_len >= 18:
            base = max(base, 0.42)
        interest_score = self._clamp_probability(topic_interest_score)
        if interest_score > 0:
            boost = max(
                0.0,
                float(getattr(self.cfg.personality, "topic_interest_reply_probability_boost", 0.18)),
            )
            base = base + (boost * interest_score)
        return self._clamp_probability(base)

    def _track_reply_skip(self, *, session_id: str, reason: str) -> None:
        skip_count = int(self._consecutive_skip_count.get(session_id, 0)) + 1
        self._consecutive_skip_count[session_id] = skip_count
        self._logger.info(
            "Queue.Silence: session=%s reason=%s skip_count=%s",
            session_id,
            reason,
            skip_count,
        )

    def _mark_reply_sent(self, *, session_id: str) -> None:
        previous_skip_count = int(self._consecutive_skip_count.pop(session_id, 0))
        self._last_reply_at[session_id] = datetime.now(timezone.utc)
        if previous_skip_count > 0:
            self._logger.info(
                "Queue.SilenceRecovered: session=%s skipped_before_reply=%s",
                session_id,
                previous_skip_count,
            )

    def _should_force_active_reply(
        self,
        *,
        session_id: str,
        message_type: str,
        mentioned_in_window: bool,
        source_text: str,
        is_master: bool,
        is_admin_sender: bool,
    ) -> bool:
        if is_master or is_admin_sender:
            return False
        if message_type != "group" or mentioned_in_window:
            return False

        clean_text = str(source_text or "").strip()
        if self._is_low_signal_passive_group_message(clean_text):
            return False

        skip_count = int(self._consecutive_skip_count.get(session_id, 0))
        if skip_count >= int(self._SILENCE_FORCE_REPLY_SKIP_THRESHOLD):
            return True

        last_reply_at = self._last_reply_at.get(session_id)
        if last_reply_at is None:
            return False
        idle_seconds = (datetime.now(timezone.utc) - last_reply_at).total_seconds()
        if idle_seconds < float(self._SILENCE_FORCE_REPLY_IDLE_SEC):
            return False

        if self._PASSIVE_GROUP_REPLY_CUE_PATTERN.search(clean_text):
            return True
        compact = re.sub(r"\s+", "", clean_text)
        return len(compact) >= 14

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
