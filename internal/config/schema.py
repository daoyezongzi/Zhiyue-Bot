from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    debug: bool = False
    log_level: str = "INFO"


class PersonaConfig(BaseModel):
    name: str = "Zhiyue"
    qq: int = 0
    master_name: str = ""
    master_id: int = 0
    alias_names: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)
    speaking_style: str = "清冷、文艺、克制"
    styles: List[str] = Field(default_factory=list)
    personality: str = ""
    system_prompt: str = ""
    admin_system_prompt: str = ""


class PersonalityConfig(BaseModel):
    enabled: bool = True
    mood: float = 0.0
    energy: float = 0.6
    sociability: float = 0.5
    neutral_energy: float = 0.55
    mood_decay: float = 0.06
    energy_recovery: float = 0.05
    interaction_window_sec: int = 120
    burst_mood_boost: float = 0.06
    burst_energy_boost: float = 0.08
    master_mood_boost: float = 0.15
    master_energy_boost: float = 0.08
    other_mood_boost: float = 0.03
    other_energy_delta: float = -0.01
    reply_energy_cost: float = 0.06
    energy_timezone_offset_hours: int = 8
    energy_active_start_hour: int = 8
    energy_active_end_hour: int = 21
    energy_active_recovery_multiplier: float = 0.9
    energy_active_reply_cost_multiplier: float = 0.9
    energy_rest_recovery_multiplier: float = 1.12
    energy_rest_reply_cost_multiplier: float = 1.12


class JargonReplaceRuleConfig(BaseModel):
    pattern: str
    replacement: str


class JargonConfig(BaseModel):
    enabled: bool = True
    conversion_rate: float = 0.0
    lexicon_store_path: str = "data/jargon_lexicon.json"
    learn_trigger_count: int = 20
    learn_context_limit: int = 40
    low_mood_threshold: float = -0.35
    high_mood_threshold: float = 0.45
    keyword_aliases: Dict[str, str] = Field(default_factory=dict)
    tone_particles: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "direct": [],
            "light": [],
            "exaggerate": [],
            "restrained": [],
        }
    )
    style_rules: Dict[str, List[JargonReplaceRuleConfig]] = Field(default_factory=dict)


class OneBotConfig(BaseModel):
    ws_mode: str = "reverse"
    ws_url: str = "ws://127.0.0.1:18001/ws"
    access_token: str = ""
    reconnect_interval: int = 5


class GroupConfig(BaseModel):
    group_id: int
    enabled: bool = True
    group_name: str = ""
    remark: str = ""
    extra_prompt: str = ""


class AdminCommandItemConfig(BaseModel):
    action: str = ""
    triggers: List[str] = Field(default_factory=list)


class AdminCommandConfig(BaseModel):
    enabled: bool = True
    prefix: str = "##zy"
    admin_user_ids: List[int] = Field(default_factory=list)
    admin_names: List[str] = Field(default_factory=list)
    commands: List[AdminCommandItemConfig] = Field(
        default_factory=lambda: [
            AdminCommandItemConfig(
                action="toggle_group_chat",
                triggers=["开关群聊", "开关某群聊天", "开关某群的聊天"],
            ),
            AdminCommandItemConfig(
                action="join_group_chat",
                triggers=["加入群聊", "加入某群聊天", "加入某群的聊天"],
            ),
            AdminCommandItemConfig(
                action="shutdown",
                triggers=["关闭程序", "关闭机器人", "停止程序"],
            ),
        ]
    )


class AgentConfig(BaseModel):
    observe_window: int = 30
    think_interval: int = 8
    think_debounce_ms: int = 800
    message_buffer_size: int = 20
    context_window_size: int = 20
    max_step: int = 12
    max_coroutine: int = 4
    enable_active_retrieval: bool = True
    active_reply_probability: float = 0.35
    prompt_cache_heartbeat_enabled: bool = False
    prompt_cache_heartbeat_interval_sec: int = 600


class LearningConfig(BaseModel):
    enabled: bool = True
    interval_minutes: int = 10
    review_interval_minutes: int = 30
    profile_store_path: str = "data/user_profiles.json"
    profile_trigger_count: int = 20
    profile_context_limit: int = 40
    profile_max_tags: int = 12


class LLMConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    max_response_tokens: int = 256
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"


class VisionConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"


class MemoryConfig(BaseModel):
    mysql_dsn: str = ""
    milvus_address: str = "127.0.0.1:19530"
    vector_dim: int = 1536
    chroma_path: str = "./data/chroma"
    short_term_threshold: int = 20
    short_term_keep_last: int = 3
    topic_shift_similarity_threshold: float = 0.35
    topic_shift_min_messages: int = 8
    rag_top_k: int = 5


class StickerConfig(BaseModel):
    enabled: bool = True
    collection_rate: float = 0.1
    storage_mode: str = "local"
    local_dir: str = "data/stickers"
    filter_keywords: List[str] = Field(
        default_factory=lambda: ["浮夸", "低俗", "吵闹", "恶臭", "擦边", "鬼畜"],
    )
    user_weights: Dict[str, float] = Field(default_factory=dict)
    enable_persona_filter: bool = True
    llm_filter_enabled: bool = True
    llm_filter_probability: float = 0.2
    llm_filter_mood_threshold: float = 50.0
    llm_filter_cache_ttl_seconds: int = 86400
    llm_filter_max_tokens: int = 10
    cloud_actions: List[str] = Field(
        default_factory=lambda: [
            "set_msg_favorite",
            "nc_set_msg_favorite",
            "mark_msg_as_favorite",
        ],
    )
    download_timeout_sec: float = 12.0


class UISettingsConfig(BaseModel):
    background_url: str = ""


class WebConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 18002
    access_token: str = ""
    admin_key: str = ""
    ui_settings: UISettingsConfig = Field(default_factory=UISettingsConfig)

    def resolved_access_token(self) -> str:
        primary = str(self.access_token or "").strip()
        if primary:
            return primary
        return str(self.admin_key or "").strip()


class PathsConfig(BaseModel):
    napcat_path: str = ""
    napcat_args: List[str] = Field(default_factory=list)
    knowledge_dir: str = "data/knowledge"
    knowledge_exclude_dirs: List[str] = Field(default_factory=list)


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    jargon: JargonConfig = Field(default_factory=JargonConfig)
    onebot: OneBotConfig = Field(default_factory=OneBotConfig)
    groups: List[GroupConfig] = Field(default_factory=list)
    admin_commands: AdminCommandConfig = Field(default_factory=AdminCommandConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    auxiliary_model: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vision_llm: VisionConfig = Field(default_factory=VisionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sticker: StickerConfig = Field(default_factory=StickerConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def get_group(self, group_id: int) -> Optional[GroupConfig]:
        for item in self.groups:
            if item.group_id == group_id:
                return item
        return None

    def is_group_enabled(self, group_id: int) -> bool:
        item = self.get_group(group_id)
        return bool(item and item.enabled)


DEFAULT_CONFIG_PATH = Path("config/config.yaml")
