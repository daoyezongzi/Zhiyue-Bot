from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from internal.config.schema import Config, DEFAULT_CONFIG_PATH


_config: Config | None = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
ENV_EXAMPLE_TEMPLATE = """# Copy this file to .env and fill in required values.
LLM_API_KEY=your_llm_api_key_here

# Optional overrides
ZHIYUE_AUX_API_KEY=
ZHIYUE_ONEBOT_TOKEN=
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


def _apply_env_overrides(cfg: Config) -> None:
    llm_api_key = _read_first_env("LLM_API_KEY", "ZHIYUE_LLM_API_KEY")
    if llm_api_key:
        cfg.llm.api_key = llm_api_key

    aux_api_key = _read_first_env("ZHIYUE_AUX_API_KEY")
    if aux_api_key:
        cfg.auxiliary_model.api_key = aux_api_key
    elif not cfg.auxiliary_model.api_key:
        cfg.auxiliary_model.api_key = cfg.llm.api_key

    onebot_token = _read_first_env("ZHIYUE_ONEBOT_TOKEN")
    if onebot_token:
        cfg.onebot.access_token = onebot_token


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
