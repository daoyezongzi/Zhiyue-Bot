import yaml
from pydantic import BaseModel, Field
from typing import List, Optional
import os

class LLMConfig(BaseModel):
    # 如果环境变量里有 API_KEY，就用环境变量的，否则读 YAML
    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY"))

# --- 子配置模型 ---

class AppConfig(BaseModel):
    log_level: str = "INFO"
    debug: bool = False

class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str

class PersonaConfig(BaseModel):
    name: str
    qq: int
    core_prompt: str

class GroupConfig(BaseModel):
    group_id: int
    enabled: bool = True
    extra_prompt: Optional[str] = ""

# --- 主配置模型 ---

class Config(BaseModel):
    app: AppConfig
    llm: LLMConfig
    persona: PersonaConfig
    groups: List[GroupConfig]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return cls(**data)

# 全局单例占位
_config = None

def load_config(path: str = "config/config.yaml") -> Config:
    global _config
    _config = Config.from_yaml(path)
    return _config

def get_config() -> Config:
    if _config is None:
        raise RuntimeError("配置尚未加载，请先调用 load_config()")
    return _config