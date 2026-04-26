from adapters.llm.chat import ChatLLMAdapter


def get_llm_client(cfg, fallback_cfg=None):
    return ChatLLMAdapter(cfg, fallback_cfg)


__all__ = ["ChatLLMAdapter", "get_llm_client"]
