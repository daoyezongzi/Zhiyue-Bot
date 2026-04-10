import signal
import sys
import time
import logging
from internal.config.config import load_config
from internal.logger.logger import init_logger
from internal.memory.manager import MemoryManager
from internal.agent.agent import ZhiyueAgent
from internal.llm.client import LLMClient

def main():
    # 1. 加载配置 (对应 main.go 第 51 行)
    config_path = "config/config.yaml"
    cfg = load_config(config_path)
    if not cfg:
        print(f"无法加载配置文件: {config_path}")
        sys.exit(1)

    # 2. 初始化日志 (对应 main.go 第 57 行)
    init_logger(cfg.app.log_level, cfg.app.debug)
    logger = logging.getLogger("Main")
    logger.info(f"配置已从 {config_path} 加载")

    # 3. 初始化模型客户端 (对应 main.go 第 60-64 行)
    # 这里的 LLMClient 将负责 Embedding 和 Chat 任务
    llm_client = LLMClient(cfg.llm)

    # 4. 初始化记忆系统 (对应 main.go 第 70-74 行)
    # 将模型客户端传入，以便记忆系统进行向量化检索
    memory_mgr = MemoryManager(llm_client)
    logger.info("记忆系统已初始化")

    # 5. 创建并启动 Agent (对应 main.go 第 77-81 行)
    # 纸月的大脑，持有记忆引用
    zhiyue = ZhiyueAgent(memory_mgr, cfg)
    zhiyue.start()

    # 6. 设置退出信号处理 (对应 main.go 第 101-102 行)
    def handle_exit(sig, frame):
        logger.info("正在关闭纸月...")
        zhiyue.stop()
        memory_mgr.close()
        logger.info("再见！")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    logger.info("纸月已上线，按 Ctrl+C 退出")
    
    # 保持主线程运行 (对应 main.go 第 105 行的 <-quit)
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()