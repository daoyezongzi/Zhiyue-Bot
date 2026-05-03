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
MEMORY_STORE_PATH=./data/memory/memory_items.json
MEMORY_TOOL_CALL_STORE_PATH=./data/memory/tool_calls.json
MEMORY_TOOL_CALL_MAX_ENTRIES=5000
MEMORY_AUTO_INGEST_ENABLED=true
MEMORY_CONVERGENCE_INTERVAL_MINUTES=15
MEMORY_CANDIDATE_GRACE_HOURS=72
MEMORY_CANDIDATE_PROMOTE_EVIDENCE=2
MEMORY_SHORT_TERM_THRESHOLD=20
MEMORY_SHORT_TERM_KEEP_LAST=3
MEMORY_TOPIC_SHIFT_THRESHOLD=0.35
MEMORY_TOPIC_SHIFT_MIN_MESSAGES=8
MEMORY_RAG_TOP_K=5
MEMORY_TOPIC_ENABLED=true
MEMORY_TOPIC_STORE_PATH=./data/topics/topic_threads.json
MEMORY_TOPIC_MAX_ACTIVE_PER_GROUP=5
MEMORY_TOPIC_SUMMARY_TRIGGER_MESSAGES=10
MEMORY_TOPIC_ARCHIVE_INACTIVE_MINUTES=180
MEMORY_TOPIC_REUSE_THRESHOLD=0.42
MEMORY_TOPIC_RECALL_TOP_K=3
MEMORY_TOPIC_MESSAGE_TAIL_SIZE=80

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
KNOWLEDGE_EXCLUDE_DIRS=workspace

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


def _parse_env_list(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    normalized = text.replace(";", ",").replace("\n", ",")
    parts = [item.strip() for item in normalized.split(",")]
    return [item for item in parts if item]


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

    memory_store_path = _read_first_env("MEMORY_STORE_PATH", "ZHIYUE_MEMORY_STORE_PATH")
    if memory_store_path:
        cfg.memory.memory_store_path = memory_store_path

    tool_call_store_path = _read_first_env("MEMORY_TOOL_CALL_STORE_PATH", "ZHIYUE_MEMORY_TOOL_CALL_STORE_PATH")
    if tool_call_store_path:
        cfg.memory.tool_call_store_path = tool_call_store_path

    tool_call_max_entries = _read_first_env("MEMORY_TOOL_CALL_MAX_ENTRIES", "ZHIYUE_MEMORY_TOOL_CALL_MAX_ENTRIES")
    if tool_call_max_entries:
        try:
            cfg.memory.tool_call_max_entries = max(100, int(tool_call_max_entries))
        except ValueError:
            pass

    memory_auto_ingest = _read_first_env("MEMORY_AUTO_INGEST_ENABLED", "ZHIYUE_MEMORY_AUTO_INGEST_ENABLED")
    if memory_auto_ingest is not None:
        parsed_auto_ingest = _parse_env_bool(memory_auto_ingest)
        if parsed_auto_ingest is not None:
            cfg.memory.memory_auto_ingest_enabled = parsed_auto_ingest

    convergence_interval = _read_first_env(
        "MEMORY_CONVERGENCE_INTERVAL_MINUTES",
        "ZHIYUE_MEMORY_CONVERGENCE_INTERVAL_MINUTES",
    )
    if convergence_interval:
        try:
            cfg.memory.memory_convergence_interval_minutes = max(1, int(convergence_interval))
        except ValueError:
            pass

    candidate_grace_hours = _read_first_env(
        "MEMORY_CANDIDATE_GRACE_HOURS",
        "ZHIYUE_MEMORY_CANDIDATE_GRACE_HOURS",
    )
    if candidate_grace_hours:
        try:
            cfg.memory.memory_candidate_grace_hours = max(1, int(candidate_grace_hours))
        except ValueError:
            pass

    candidate_promote_evidence = _read_first_env(
        "MEMORY_CANDIDATE_PROMOTE_EVIDENCE",
        "ZHIYUE_MEMORY_CANDIDATE_PROMOTE_EVIDENCE",
    )
    if candidate_promote_evidence:
        try:
            cfg.memory.memory_candidate_promote_evidence = max(1, int(candidate_promote_evidence))
        except ValueError:
            pass

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

    topic_enabled = _read_first_env("MEMORY_TOPIC_ENABLED", "ZHIYUE_MEMORY_TOPIC_ENABLED")
    if topic_enabled is not None:
        parsed_topic_enabled = _parse_env_bool(topic_enabled)
        if parsed_topic_enabled is not None:
            cfg.memory.topic_enabled = parsed_topic_enabled

    topic_store_path = _read_first_env("MEMORY_TOPIC_STORE_PATH", "ZHIYUE_MEMORY_TOPIC_STORE_PATH")
    if topic_store_path:
        cfg.memory.topic_store_path = topic_store_path

    topic_max_active = _read_first_env(
        "MEMORY_TOPIC_MAX_ACTIVE_PER_GROUP",
        "ZHIYUE_MEMORY_TOPIC_MAX_ACTIVE_PER_GROUP",
    )
    if topic_max_active:
        try:
            cfg.memory.topic_max_active_per_group = max(1, int(topic_max_active))
        except ValueError:
            pass

    topic_summary_trigger = _read_first_env(
        "MEMORY_TOPIC_SUMMARY_TRIGGER_MESSAGES",
        "ZHIYUE_MEMORY_TOPIC_SUMMARY_TRIGGER_MESSAGES",
    )
    if topic_summary_trigger:
        try:
            cfg.memory.topic_summary_trigger_messages = max(1, int(topic_summary_trigger))
        except ValueError:
            pass

    topic_archive_inactive = _read_first_env(
        "MEMORY_TOPIC_ARCHIVE_INACTIVE_MINUTES",
        "ZHIYUE_MEMORY_TOPIC_ARCHIVE_INACTIVE_MINUTES",
    )
    if topic_archive_inactive:
        try:
            cfg.memory.topic_archive_inactive_minutes = max(1, int(topic_archive_inactive))
        except ValueError:
            pass

    topic_reuse_threshold = _read_first_env(
        "MEMORY_TOPIC_REUSE_THRESHOLD",
        "ZHIYUE_MEMORY_TOPIC_REUSE_THRESHOLD",
    )
    if topic_reuse_threshold:
        try:
            cfg.memory.topic_reuse_threshold = float(topic_reuse_threshold)
        except ValueError:
            pass

    topic_recall_top_k = _read_first_env(
        "MEMORY_TOPIC_RECALL_TOP_K",
        "ZHIYUE_MEMORY_TOPIC_RECALL_TOP_K",
    )
    if topic_recall_top_k:
        try:
            cfg.memory.topic_recall_top_k = max(1, int(topic_recall_top_k))
        except ValueError:
            pass

    topic_tail_size = _read_first_env(
        "MEMORY_TOPIC_MESSAGE_TAIL_SIZE",
        "ZHIYUE_MEMORY_TOPIC_MESSAGE_TAIL_SIZE",
    )
    if topic_tail_size:
        try:
            cfg.memory.topic_message_tail_size = max(20, int(topic_tail_size))
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

    knowledge_exclude_dirs = _read_first_env(
        "KNOWLEDGE_EXCLUDE_DIRS",
        "ZHIYUE_KNOWLEDGE_EXCLUDE_DIRS",
    )
    if knowledge_exclude_dirs:
        cfg.paths.knowledge_exclude_dirs = _parse_env_list(knowledge_exclude_dirs)

    master_name = _read_first_env("PERSONA_MASTER_NAME", "MASTER_NAME", "ZHIYUE_MASTER_NAME")
    if master_name is not None:
        cfg.persona.master_name = master_name

    master_id = _read_first_env("PERSONA_MASTER_ID", "MASTER_ID", "ZHIYUE_MASTER_ID")
    if master_id:
        try:
            cfg.persona.master_id = int(master_id)
        except ValueError:
            pass


def _resolve_persona_prompt_path(config_path: Path) -> Path:
    candidates = [
        config_path.parent / "persona.prompt",
        PROJECT_ROOT / "config" / "persona.prompt",
    ]
    for item in candidates:
        if item.exists():
            return item
    # Keep the first candidate for clearer error path.
    return candidates[0]


def _load_persona_prompt(config_path: Path) -> str:
    prompt_path = _resolve_persona_prompt_path(config_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"persona prompt file not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError(f"persona prompt body is empty: {prompt_path}")
    return normalized


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
    _config.persona.system_prompt = _load_persona_prompt(file_path)
    _apply_env_overrides(_config)
    return _config


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("config is not loaded, call load_config first")
    return _config
