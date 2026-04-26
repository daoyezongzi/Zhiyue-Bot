from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable

from adapters.llm.chat import ChatLLMAdapter
from adapters.onebot.client import OneBotClient
from core.memory import ConversationMessage, MessageHistory
from internal.config.schema import Config
from internal.jargon import JargonManager, StyleClassification
from internal.jargon.jargon_engine import JargonEvolutionEngine, JargonLexiconStore
from internal.learning.user_profiling import UserProfileStore, UserProfilingEngine
from internal.logger import get_logger
from internal.memory import MemoryManager, MessageLog
from internal.persona import MoodInfo, PersonalityManager, PromptContext, StatusEngine, StatusSnapshot


@dataclass(slots=True)
class ThinkContext:
    session_id: str
    message_type: str
    user_id: int | None
    group_id: int | None
    speaker: str
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
    timer_task: asyncio.Task[None]


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

        current_text = str(message.get("text", "")).strip()
        jargon_matches = await self._jargon_mgr.match(current_text)

        context_messages = self._history.get_structured_messages(session_id)
        chat_context = "\n".join(item["content"] for item in context_messages)
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
                group_extra = group_cfg.extra_prompt.strip()

        summary_hint = self._history.summary_prompt(session_id)
        if summary_hint:
            prompt_ctx.related_memories.append(summary_hint)
        elif self._history.should_refresh_summary(session_id):
            prompt_ctx.related_memories.append("历史上下文窗口已裁剪，暂无可用总结。")
        if history_background:
            prompt_ctx.related_memories.extend(history_background[:5])

        recent_people = self._build_recent_people(session_id)
        dynamic_hobbies = self._build_hobbies_hint(jargon_matches)
        dynamic_styles = [f"当前建议语气：{style.tone}（意图：{style.intent}）"]

        system_prompt = self._personality.get_system_prompt(
            hobbies=dynamic_hobbies,
            styles=dynamic_styles,
            is_master=is_master,
        )
        retrieval_block = self._build_retrieval_block(history_background, related_knowledge)
        if retrieval_block:
            system_prompt = f"{system_prompt}\n\n{retrieval_block}"
        if social_background.strip():
            system_prompt = f"{system_prompt}\n\n## 社交背景\n{social_background.strip()}"
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
        for row in reversed(rows):
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
    _REPLY_LINE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)+")
    _REPLY_SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|$)")
    _LEADING_QUOTES = ("\"", "'", "“", "‘", "「", "『")

    def __init__(self, bot_client: OneBotClient, cfg: Config, llm: ChatLLMAdapter) -> None:
        self.bot_client = bot_client
        self.cfg = cfg
        self.llm = llm

        self._logger = get_logger("ZhiyueAgent")
        self._started_at_utc = datetime.now(timezone.utc)
        self._history = MessageHistory(self.cfg.agent.context_window_size)

        self.personality = PersonalityManager(self.cfg.persona, self.cfg.personality)
        self.status_engine = StatusEngine(
            initial_energy=float(self.cfg.personality.energy) * 100.0,
            initial_mood=50.0 + (float(self.cfg.personality.mood) * 50.0),
            heartbeat_interval_sec=600,
            idle_threshold_sec=180,
            recovery_step=8.0,
            mood_recovery_step=2.0,
        )
        self.jargon_mgr = JargonManager(self.cfg.jargon)
        self.memory_mgr = MemoryManager(cfg=self.cfg, llm=self.llm, on_summary=self._on_memory_summary)
        self._evolution_llm = ChatLLMAdapter(self.cfg.auxiliary_model, self.cfg.llm)
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
        self._debounce_window_sec = 0.5
        self._rng = random.Random()
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
        self._inbound_worker = asyncio.create_task(self._inbound_loop(), name="zhiyue-inbound-worker")
        self._dispatch_worker = asyncio.create_task(self._dispatch_loop(), name="zhiyue-dispatch-worker")
        self._started = True
        self._logger.info("Agent started at %s", self._started_at_utc.isoformat())

    async def stop(self) -> None:
        self._started = False
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

    async def _debounce(self, packet: dict[str, Any]) -> None:
        key = self._build_debounce_key(packet)
        mentioned = self._is_packet_mentioned(packet)
        async with self._debounce_lock:
            previous = self._debounce_entries.get(key)
            if previous is not None:
                previous.timer_task.cancel()
                generation = previous.generation + 1
                merged_count = previous.merged_count + 1
                mentioned = mentioned or previous.mentioned
                timer_task = asyncio.create_task(
                    self._flush_debounce_after(key, generation),
                    name=f"debounce-{key}",
                )
                self._debounce_entries[key] = DebounceEntry(
                    packet=packet,
                    generation=generation,
                    merged_count=merged_count,
                    mentioned=mentioned,
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
        async with self._debounce_lock:
            entry = self._debounce_entries.get(key)
            if entry is None or entry.generation != generation:
                return
            packet = dict(entry.packet)
            merged_count = entry.merged_count
            mentioned = entry.mentioned
            del self._debounce_entries[key]

        assert packet is not None
        packet["_debounced_mention"] = mentioned
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
            if not self.cfg.is_group_enabled(group_id):
                self._logger.info("Skip group message: group not enabled group_id=%s", group_id)
                return

        mentioned_in_window = bool(message.get("_debounced_mention"))
        text = str(message.get("text", "")).strip()
        if not text:
            if message_type == "group" and mentioned_in_window:
                text = "[用户仅@了你]"
            else:
                return

        user_id = self._to_int(message.get("user_id"))
        self_id = self._to_int(message.get("self_id"))
        if user_id is not None and self_id is not None and user_id == self_id:
            return

        now_utc = datetime.now(timezone.utc)
        session_id = self._build_session_id(message)
        speaker = self._extract_speaker(message)
        is_master = self._is_master(user_id)
        merged_count = int(message.get("_debounced_count", 1) or 1)
        memory_group_id = group_id if group_id is not None else 0

        self._logger.info(
            "Queue.Dispatch: session=%s message_id=%s merged=%s mentioned=%s",
            session_id,
            message.get("message_id"),
            merged_count,
            mentioned_in_window,
        )

        status_after_user = await self.status_engine.apply_user_message(text)
        retrieval = await self.memory_mgr.retrieve_for_prompt(
            text=text,
            session_id=session_id,
            group_id=group_id,
            top_k=self.cfg.memory.rag_top_k,
        )

        mood = self.personality.update_state(now=now_utc, speaker_is_master=is_master)
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
                "valence=%.2f energy=%.2f sociability=%.2f status_energy=%.1f status_mood=%.1f fatigue=%s"
            ),
            session_id,
            message_type,
            user_id,
            message.get("group_id"),
            is_master,
            mood.valence,
            mood.energy,
            mood.sociability,
            status_after_user.energy,
            status_after_user.mood,
            status_after_user.fatigue_mode,
        )

        if message_type == "group" and not mentioned_in_window:
            active_reply_probability = self._clamp_probability(self.cfg.agent.active_reply_probability)
            if active_reply_probability < 1.0:
                roll = self._rng.random()
                if roll >= active_reply_probability:
                    self._logger.info(
                        "Queue.SkipReply: session=%s reason=active_probability roll=%.4f threshold=%.4f",
                        session_id,
                        roll,
                        active_reply_probability,
                    )
                    return

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

        now_after_reply = datetime.now(timezone.utc)
        status_after_reply = await self.status_engine.consume_reply(final_reply)
        self.personality.update_state(
            now=now_after_reply,
            speaker_is_master=False,
            is_bot_reply=True,
            apply_interaction=False,
        )
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
            "ReplyStatus: session=%s forced_rest=%s status_energy=%.1f status_mood=%.1f fatigue=%s",
            session_id,
            forced_rest,
            status_after_reply.energy,
            status_after_reply.mood,
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
        status = await self.status_engine.get_snapshot()
        status_block = self._build_status_prompt(status)
        built.llm_messages[0]["content"] = f"{built.llm_messages[0]['content']}\n\n{status_block}"
        if status.fatigue_mode:
            built.llm_messages[1]["content"] = (
                f"{built.llm_messages[1]['content']}\n"
                "注意：你当前精力不足，回复应该更短、更冷淡，避免连续输出长句。"
            )
        if status.forced_rest:
            built.llm_messages[1]["content"] = (
                f"{built.llm_messages[1]['content']}\n"
                "注意：你已经进入强制休眠状态，尽量用一句话结束对话。"
            )

        return ThinkContext(
            session_id=session_id,
            message_type=str(message.get("message_type", "")).strip() or "private",
            user_id=user_id,
            group_id=self._to_int(message.get("group_id")),
            speaker=speaker,
            is_master=is_master,
            mood=built.mood,
            style=built.style,
            prompt_context=built.prompt_context,
            llm_messages=built.llm_messages,
            status=status,
            social_background=social_background,
        )

    async def _before_llm_think(self, ctx: ThinkContext) -> ThinkContext:
        # 濡澘瀚弳鈧柨娑欒壘椤曨喗顬?ReAct 婵炵繝鑳堕埢鍏肩▔椤撶姵鐣遍柍銉︾矊娴兼劙宕楅悿顖ｆ綈闁?+ 濡炲瀛╅悧鎼佸礆閸℃瑨顫﹂柍銉︾箞濡礁鈻撻悙鍏夊亾?
        ctx.planned_tools = await self._plan_tool_calls(ctx)
        ctx.style = await self._classify_style_context(ctx)
        return ctx

    async def _llm_think(self, ctx: ThinkContext) -> str:
        extra_fields: dict[str, Any] = dict(self.cfg.llm.extra_fields)
        if self.cfg.llm.max_response_tokens > 0:
            if "max_tokens" not in extra_fields and "max_completion_tokens" not in extra_fields:
                extra_fields["max_tokens"] = self.cfg.llm.max_response_tokens

        reply = await self.llm.generate_from_messages(ctx.llm_messages, extra_fields)
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
        # 濡澘瀚弳鈧柨娑欒壘椤曨喗顬?ReAct 婵炵繝鑳堕埢鍏肩▔椤撶姵鐣遍柍銉︾矊娴兼劙宕楅柨瀣挃閻炴稑鑻ú鏍礃?闁告艾楠搁ˇ鈺呮偠閸″繆鍋撳┑瀣枆婵炲牏鍋ｉ埀?
        _ = await self._apply_tool_results(ctx, reply)
        return reply

    async def _plan_tool_calls(self, ctx: ThinkContext) -> list[str]:
        del ctx
        return []

    async def _classify_style_context(self, ctx: ThinkContext) -> StyleClassification:
        return self.jargon_mgr.classify_style(
            ctx.mood.valence,
            ctx.mood.energy,
            speaker_is_master=ctx.is_master,
        )

    async def _apply_tool_results(self, ctx: ThinkContext, reply: str) -> dict[str, Any]:
        del ctx
        del reply
        return {}

    async def _response_post_process(self, reply: str, ctx: ThinkContext) -> str:
        styled = self.jargon_mgr.apply_post_process(
            reply,
            mood=ctx.mood.valence,
            energy=ctx.mood.energy,
            speaker_is_master=ctx.is_master,
        )
        processed = await self.jargon_engine.apply_to_reply(styled)
        return self._strip_self_prefix(processed)

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
            "mood": round(status.mood, 2),
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
        reset_mood: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        snapshot = await self.status_engine.reset(fill_energy=fill_energy, reset_mood=reset_mood)
        if fill_energy or reset_mood:
            self.personality.set_mood(
                valence=(snapshot.mood - 50.0) / 50.0,
                energy=snapshot.energy / 100.0,
                sociability=0.5,
            )

        cleared = False
        if session_id and session_id.strip():
            cleared = self._history.clear_session(session_id.strip())
            cleared = await self.memory_mgr.clear_session_memory(session_id.strip()) or cleared

        return {
            "status": {
                "energy": round(snapshot.energy, 2),
                "mood": round(snapshot.mood, 2),
                "fatigue_mode": snapshot.fatigue_mode,
                "forced_rest": snapshot.forced_rest,
            },
            "cleared_session": session_id.strip() if session_id and cleared else "",
            "cleared": cleared,
        }

    def update_system_prompt(self, prompt: str) -> None:
        self.cfg.persona.system_prompt = prompt
        self.personality.cfg.system_prompt = prompt

    def uptime_seconds(self) -> int:
        return max(0, int((datetime.now(timezone.utc) - self._started_at_utc).total_seconds()))

    @staticmethod
    def _build_status_prompt(status: StatusSnapshot) -> str:
        mode = "疲劳模式" if status.fatigue_mode else "正常模式"
        return (
            "## 生理状态\n"
            f"- 精力: {status.energy:.1f}/100\n"
            f"- 心情: {status.mood:.1f}/100\n"
            f"- 当前模式: {mode}\n"
            "- 当精力低于 10 时请主动缩短回复，并保持冷淡语气。"
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
        return clean

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

    def _extract_speaker(self, message: dict[str, Any]) -> str:
        raw = message.get("raw", {})
        if isinstance(raw, dict):
            sender = raw.get("sender", {})
            if isinstance(sender, dict):
                nickname = str(sender.get("nickname", "")).strip()
                if nickname:
                    return nickname
                card = str(sender.get("card", "")).strip()
                if card:
                    return card
        user_id = self._to_int(message.get("user_id"))
        return str(user_id) if user_id is not None else "unknown"

    def _is_master(self, user_id: int | None) -> bool:
        master_id = self.cfg.persona.master_id
        return bool(master_id and user_id is not None and user_id == master_id)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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
