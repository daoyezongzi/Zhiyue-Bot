from adapters.llm.chat import ChatLLMAdapter


def get_llm_client(cfg):
    return ChatLLMAdapter(cfg)


__all__ = ["ChatLLMAdapter", "get_llm_client"]
