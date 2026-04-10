import logging
import threading
from openai import OpenAI
from typing import Optional

class LLMClientManager:
    """
    LLM 客户端管理器（单例模式）
    对应 client.go 中的 defaultClientOnce 逻辑
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMClientManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cfg=None):
        if self._initialized:
            return
        
        self.cfg = cfg
        self.logger = logging.getLogger("LLMClient")
        
        # 初始化主模型 (对应 NewClient)
        self._main_client = self._create_client(cfg)
        
        # 初始化辅助模型 (对应 NewAuxClient)
        # 如果配置里没有专门的辅助模型，则复用主模型
        self._aux_client = self._main_client 
        
        self._initialized = True
        self.logger.info(f"LLM 客户端已初始化，模型: {cfg.model}")

    def _create_client(self, cfg):
        try:
            return OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=60.0
            )
        except Exception as e:
            self.logger.error(f"创建 LLM 客户端失败: {e}")
            return None

    def ask(self, system_prompt: str, user_prompt: str, is_aux: bool = False) -> str:
        """
        统一的请求入口
        :param is_aux: 是否使用辅助模型执行任务
        """
        client = self._aux_client if is_aux else self._main_client
        if not client:
            return "（大脑连接中断...）"

        try:
            response = client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM 请求异常: {e}")
            return "（纸月似乎失神了...）"

# 为了方便调用，模仿 Go 的 NewClient()
def get_llm_client(cfg=None):
    return LLMClientManager(cfg)