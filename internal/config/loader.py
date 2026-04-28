from __future__ import annotations

import os
import shlex
from pathlib import Path

import yaml
from dotenv import load_dotenv

from internal.config.schema import Config, DEFAULT_CONFIG_PATH


_config: Config | None = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
ENV_EXAMPLE_TEMPLATE = """# Copy this file to .env and fill in required values.
# Primary chat model (default: DeepSeek)
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
LLM_API_KEY=
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# Auxiliary chat model (optional, can be another provider/model)
AUX_LLM_PROVIDER=
AUX_LLM_API_KEY=
AUX_LLM_BASE_URL=
AUX_LLM_MODEL=

# Embedding model (optional)
EMBEDDING_PROVIDER=
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=

# Memory/RAG
MEMORY_CHROMA_PATH=./data/chroma
MEMORY_SHORT_TERM_THRESHOLD=20
MEMORY_SHORT_TERM_KEEP_LAST=3
MEMORY_TOPIC_SHIFT_THRESHOLD=0.35
MEMORY_TOPIC_SHIFT_MIN_MESSAGES=8
MEMORY_RAG_TOP_K=5

# Vision model (optional)
VISION_LLM_PROVIDER=
VISION_LLM_API_KEY=
VISION_LLM_BASE_URL=
VISION_LLM_MODEL=

# Legacy/other overrides
ZHIYUE_ONEBOT_TOKEN=
BOT_QQ=
ONEBOT_WS_MODE=reverse
ONEBOT_WS_URL=ws://127.0.0.1:18001/ws
WEB_ENABLED=true
WEB_HOST=127.0.0.1
WEB_PORT=18002
WEB_ACCESS_TOKEN=
WEB_BACKGROUND_URL=

# Optional managed OneBot process (NapCat)
NAPCAT_PATH=
NAPCAT_ARGS=
KNOWLEDGE_DIR=data/knowledge

# Persona privacy overrides
PERSONA_MASTER_NAME=
PERSONA_MASTER_ID=
"""


def _load_project_env() -> None:
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)


def _ensure_env_example() -> None:
    if ENV_EXAMPLE_PATH.exists():
        return
    ENV_EXAMPLE_PATH.write_text(ENV_EXAMPLE_TEMPLATE, encoding="utf-8")


def _read_first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _normalize_env_key(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in value.strip().upper())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _parse_env_bool(raw: str) -> bool | None:
    normalized = str(raw or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _read_provider_api_key(provider: str | None) -> str | None:
    if not provider:
        return None
    env_prefix = _normalize_env_key(provider)
    if not env_prefix:
        return None
    return _read_first_env(f"{env_prefix}_API_KEY", f"ZHIYUE_{env_prefix}_API_KEY")


def _apply_model_env_overrides(
    target: object,
    *,
    provider_env_names: tuple[str, ...],
    api_key_env_names: tuple[str, ...],
    base_url_env_names: tuple[str, ...],
    model_env_names: tuple[str, ...],
    fallback_api_key: str | None = None,
) -> None:
    provider = _read_first_env(*provider_env_names)
    provider_api_key = _read_provider_api_key(provider)

    explicit_api_key = _read_first_env(*api_key_env_names)
    api_key = explicit_api_key or provider_api_key
    if api_key:
        setattr(target, "api_key", api_key)
    elif fallback_api_key and not getattr(target, "api_key", ""):
        setattr(target, "api_key", fallback_api_key)

    base_url = _read_first_env(*base_url_env_names)
    if base_url:
        setattr(target, "base_url", base_url)

    model = _read_first_env(*model_env_names)
    if model:
        setattr(target, "model", model)


def _apply_env_overrides(cfg: Config) -> None:
    _apply_model_env_overrides(
        cfg.llm,
        provider_env_names=("LLM_PROVIDER", "ZHIYUE_LLM_PROVIDER"),
        api_key_env_names=("LLM_API_KEY", "ZHIYUE_LLM_API_KEY"),
        base_url_env_names=("LLM_BASE_URL", "ZHIYUE_LLM_BASE_URL"),
        model_env_names=("LLM_MODEL", "ZHIYUE_LLM_MODEL"),
    )

    _apply_model_env_overrides(
        cfg.auxiliary_model,
        provider_env_names=("AUX_LLM_PROVIDER", "ZHIYUE_AUX_PROVIDER"),
        api_key_env_names=("AUX_LLM_API_KEY", "ZHIYUE_AUX_API_KEY"),
        base_url_env_names=("AUX_LLM_BASE_URL", "ZHIYUE_AUX_BASE_URL"),
        model_env_names=("AUX_LLM_MODEL", "ZHIYUE_AUX_MODEL"),
        fallback_api_key=cfg.llm.api_key,
    )

    _apply_model_env_overrides(
        cfg.embedding,
        provider_env_names=("EMBEDDING_PROVIDER", "ZHIYUE_EMBEDDING_PROVIDER"),
        api_key_env_names=("EMBEDDING_API_KEY", "ZHIYUE_EMBEDDING_API_KEY"),
        base_url_env_names=("EMBEDDING_BASE_URL", "ZHIYUE_EMBEDDING_BASE_URL"),
        model_env_names=("EMBEDDING_MODEL", "ZHIYUE_EMBEDDING_MODEL"),
    )

    _apply_model_env_overrides(
        cfg.vision_llm,
        provider_env_names=("VISION_LLM_PROVIDER", "ZHIYUE_VISION_PROVIDER"),
        api_key_env_names=("VISION_LLM_API_KEY", "ZHIYUE_VISION_API_KEY"),
        base_url_env_names=("VISION_LLM_BASE_URL", "ZHIYUE_VISION_BASE_URL"),
        model_env_names=("VISION_LLM_MODEL", "ZHIYUE_VISION_MODEL"),
    )

    onebot_token = _read_first_env("ZHIYUE_ONEBOT_TOKEN")
    if onebot_token:
        cfg.onebot.access_token = onebot_token

    onebot_mode = _read_first_env("ONEBOT_WS_MODE", "ZHIYUE_ONEBOT_WS_MODE")
    if onebot_mode:
        cfg.onebot.ws_mode = onebot_mode.strip()

    onebot_ws_url = _read_first_env("ONEBOT_WS_URL", "ZHIYUE_ONEBOT_WS_URL")
    if onebot_ws_url:
        cfg.onebot.ws_url = onebot_ws_url.strip()

    web_host = _read_first_env("WEB_HOST", "ZHIYUE_WEB_HOST")
    if web_host:
        cfg.web.host = web_host

    web_enabled = _read_first_env("WEB_ENABLED", "ZHIYUE_WEB_ENABLED")
    if web_enabled is not None:
        parsed_enabled = _parse_env_bool(web_enabled)
        if parsed_enabled is not None:
            cfg.web.enabled = parsed_enabled

    web_port = _read_first_env("WEB_PORT", "ZHIYUE_WEB_PORT")
    if web_port:
        try:
            cfg.web.port = max(1, int(web_port))
        except ValueError:
            pass

    web_access_token = _read_first_env(
        "WEB_ACCESS_TOKEN",
        "ZHIYUE_WEB_ACCESS_TOKEN",
        "WEB_ADMIN_KEY",
        "ZHIYUE_WEB_ADMIN_KEY",
    )
    if web_access_token:
        cfg.web.access_token = web_access_token

    chroma_path = _read_first_env("MEMORY_CHROMA_PATH", "ZHIYUE_MEMORY_CHROMA_PATH")
    if chroma_path:
        cfg.memory.chroma_path = chroma_path

    short_term_threshold = _read_first_env("MEMORY_SHORT_TERM_THRESHOLD", "ZHIYUE_MEMORY_SHORT_TERM_THRESHOLD")
    if short_term_threshold:
        try:
            cfg.memory.short_term_threshold = max(1, int(short_term_threshold))
        except ValueError:
            pass

    short_term_keep_last = _read_first_env("MEMORY_SHORT_TERM_KEEP_LAST", "ZHIYUE_MEMORY_SHORT_TERM_KEEP_LAST")
    if short_term_keep_last:
        try:
            cfg.memory.short_term_keep_last = max(1, int(short_term_keep_last))
        except ValueError:
            pass

    topic_shift_threshold = _read_first_env("MEMORY_TOPIC_SHIFT_THRESHOLD", "ZHIYUE_MEMORY_TOPIC_SHIFT_THRESHOLD")
    if topic_shift_threshold:
        try:
            cfg.memory.topic_shift_similarity_threshold = float(topic_shift_threshold)
        except ValueError:
            pass

    topic_shift_min_messages = _read_first_env(
        "MEMORY_TOPIC_SHIFT_MIN_MESSAGES",
        "ZHIYUE_MEMORY_TOPIC_SHIFT_MIN_MESSAGES",
    )
    if topic_shift_min_messages:
        try:
            cfg.memory.topic_shift_min_messages = max(1, int(topic_shift_min_messages))
        except ValueError:
            pass

    rag_top_k = _read_first_env("MEMORY_RAG_TOP_K", "ZHIYUE_MEMORY_RAG_TOP_K")
    if rag_top_k:
        try:
            cfg.memory.rag_top_k = max(1, int(rag_top_k))
        except ValueError:
            pass

    background_url = _read_first_env("WEB_BACKGROUND_URL", "ZHIYUE_WEB_BACKGROUND_URL")
    if background_url:
        cfg.web.ui_settings.background_url = background_url

    napcat_path = _read_first_env("NAPCAT_PATH", "ZHIYUE_NAPCAT_PATH")
    if napcat_path:
        cfg.paths.napcat_path = napcat_path

    napcat_args = _read_first_env("NAPCAT_ARGS", "ZHIYUE_NAPCAT_ARGS")
    if napcat_args:
        try:
            cfg.paths.napcat_args = [part for part in shlex.split(napcat_args, posix=False) if part.strip()]
        except ValueError:
            cfg.paths.napcat_args = [part for part in napcat_args.split(" ") if part.strip()]

    knowledge_dir = _read_first_env("KNOWLEDGE_DIR", "ZHIYUE_KNOWLEDGE_DIR")
    if knowledge_dir:
        cfg.paths.knowledge_dir = knowledge_dir

    master_name = _read_first_env("PERSONA_MASTER_NAME", "MASTER_NAME", "ZHIYUE_MASTER_NAME")
    if master_name is not None:
        cfg.persona.master_name = master_name

    master_id = _read_first_env("PERSONA_MASTER_ID", "MASTER_ID", "ZHIYUE_MASTER_ID")
    if master_id:
        try:
            cfg.persona.master_id = int(master_id)
        except ValueError:
            pass


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    global _config
    _load_project_env()
    _ensure_env_example()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _config = Config(**data)
    _apply_env_overrides(_config)
    return _config


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("config is not loaded, call load_config first")
    return _config
