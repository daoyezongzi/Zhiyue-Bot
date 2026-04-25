from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from adapters.llm.chat import ChatLLMAdapter
from adapters.onebot.client import OneBotClient
from core.memory import ConversationMessage, MessageHistory
from internal.config.schema import Config
from internal.jargon import JargonManager, StyleClassification
from internal.logger import get_logger
from internal.persona import MoodInfo, PersonalityManager, PromptContext


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
    style_source: str = "heuristic"
    planned_tools: list[str] | None = None


@dataclass(slots=True)
class ContextBuildResult:
    mood: MoodInfo
    style: StyleClassification
    prompt_context: PromptContext
    llm_messages: list[dict[str, str]]


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

        recent_people = self._build_recent_people(session_id)
        dynamic_hobbies = self._build_hobbies_hint(jargon_matches)
        dynamic_styles = [f"当前建议语气：{style.tone}（意图：{style.intent}）"]

        system_prompt = self._personality.get_system_prompt(
            hobbies=dynamic_hobbies,
            styles=dynamic_styles,
        )
        think_prompt = self._personality.get_think_prompt(
            prompt_ctx,
            chat_context,
            group_extra,
            recent_people,
        )

        if self._personality.is_mentioned(current_text):
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
    def __init__(self, bot_client: OneBotClient, cfg: Config, llm: ChatLLMAdapter) -> None:
        self.bot_client = bot_client
        self.cfg = cfg
        self.llm = llm

        self._logger = get_logger("ZhiyueAgent")
        self._started_at_utc = datetime.now(timezone.utc)
        self._history = MessageHistory(self.cfg.agent.context_window_size)

        self.personality = PersonalityManager(self.cfg.persona, self.cfg.personality)
        self.jargon_mgr = JargonManager(self.cfg.jargon)
        self.context_builder = ContextBuilder(
            cfg=self.cfg,
            history=self._history,
            personality=self.personality,
            jargon_mgr=self.jargon_mgr,
        )

    async def start(self) -> None:
        self._started_at_utc = datetime.now(timezone.utc)
        await self.jargon_mgr.reload()
        self._logger.info("Agent started at %s", self._started_at_utc.isoformat())

    async def stop(self) -> None:
        self._logger.info("Agent stopped")

    async def handle_message(self, message: dict[str, Any]) -> None:
        if str(message.get("post_type", "")).strip() != "message":
            return

        message_type = str(message.get("message_type", "")).strip() or "private"
        if message_type == "group":
            group_id = self._to_int(message.get("group_id"))
            if group_id is None or not self.cfg.is_group_enabled(group_id):
                return

        text = str(message.get("text", "")).strip()
        if not text:
            return

        user_id = self._to_int(message.get("user_id"))
        self_id = self._to_int(message.get("self_id"))
        if user_id is not None and self_id is not None and user_id == self_id:
            return

        now_utc = datetime.now(timezone.utc)
        session_id = self._build_session_id(message)
        speaker = self._extract_speaker(message)
        is_master = self._is_master(user_id)

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

        self._logger.info(
            "HandleMessage: session=%s type=%s user_id=%s group_id=%s master=%s valence=%.2f energy=%.2f sociability=%.2f",
            session_id,
            message_type,
            user_id,
            message.get("group_id"),
            is_master,
            mood.valence,
            mood.energy,
            mood.sociability,
        )

        ctx = await self._build_context(
            session_id=session_id,
            message=message,
            speaker=speaker,
            user_id=user_id,
            is_master=is_master,
        )
        ctx = await self._before_llm_think(ctx)
        reply = await self._llm_think(ctx)
        reply = await self._after_llm_think(ctx, reply)
        if not reply:
            return

        final_reply = self._response_post_process(reply, ctx)
        if not final_reply:
            return

        await self._reply(message, final_reply)

        now_after_reply = datetime.now(timezone.utc)
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

    async def _build_context(
        self,
        *,
        session_id: str,
        message: dict[str, Any],
        speaker: str,
        user_id: int | None,
        is_master: bool,
    ) -> ThinkContext:
        built = await self.context_builder.build(
            session_id=session_id,
            message=message,
            speaker=speaker,
            user_id=user_id,
            is_master=is_master,
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
        )

    async def _before_llm_think(self, ctx: ThinkContext) -> ThinkContext:
        # 预留：对齐 ReAct 流程中的“工具规划 + 风格分类”阶段。
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
        # 预留：对齐 ReAct 流程中的“工具执行回写/后处理”阶段。
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

    def _response_post_process(self, reply: str, ctx: ThinkContext) -> str:
        return self.jargon_mgr.apply_post_process(
            reply,
            mood=ctx.mood.valence,
            energy=ctx.mood.energy,
            speaker_is_master=ctx.is_master,
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
        message_type = str(message.get("message_type", "")).strip() or "private"
        if message_type == "group":
            group_id = self._to_int(message.get("group_id"))
            if group_id is None:
                self._logger.warning("Skip group reply: missing group_id")
                return
            await self.bot_client.send_group_msg(group_id=group_id, message=reply)
            self._logger.info("Reply sent to group %s", group_id)
            return

        user_id = self._to_int(message.get("user_id"))
        if user_id is None:
            self._logger.warning("Skip private reply: missing user_id")
            return
        await self.bot_client.send_private_msg(user_id=user_id, message=reply)
        self._logger.info("Reply sent to user %s", user_id)

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


AsyncReactAgent = ZhiyueAgent
