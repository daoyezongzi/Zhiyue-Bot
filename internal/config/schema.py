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
    ws_url: str = "ws://127.0.0.1:3001"
    access_token: str = ""
    reconnect_interval: int = 5


class GroupConfig(BaseModel):
    group_id: int
    enabled: bool = True
    extra_prompt: str = ""


class AgentConfig(BaseModel):
    observe_window: int = 30
    think_interval: int = 8
    think_debounce_ms: int = 800
    message_buffer_size: int = 20
    context_window_size: int = 20
    max_step: int = 12
    max_coroutine: int = 4
    enable_active_retrieval: bool = True


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


class UISettingsConfig(BaseModel):
    background_url: str = ""


class WebConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    access_token: str = ""
    admin_key: str = ""
    ui_settings: UISettingsConfig = Field(default_factory=UISettingsConfig)

    def resolved_access_token(self) -> str:
        primary = str(self.access_token or "").strip()
        if primary:
            return primary
        return str(self.admin_key or "").strip()


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    jargon: JargonConfig = Field(default_factory=JargonConfig)
    onebot: OneBotConfig = Field(default_factory=OneBotConfig)
    groups: List[GroupConfig] = Field(default_factory=list)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    auxiliary_model: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vision_llm: VisionConfig = Field(default_factory=VisionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    def get_group(self, group_id: int) -> Optional[GroupConfig]:
        for item in self.groups:
            if item.group_id == group_id:
                return item
        return None

    def is_group_enabled(self, group_id: int) -> bool:
        item = self.get_group(group_id)
        return bool(item and item.enabled)


DEFAULT_CONFIG_PATH = Path("config/config.yaml")
