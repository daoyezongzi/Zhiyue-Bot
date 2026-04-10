# test_brain.py
from internal.config.config import load_config
from internal.llm.client import get_llm_client

def test():
    # 1. 加载你之前写好的配置
    cfg = load_config("config/config.yaml")
    
    # 2. 获取客户端
    brain = get_llm_client(cfg.llm)
    
    # 3. 测试对话
    print("正在呼唤纸月...")
    reply = brain.ask("你是一个性格清冷的少女，名叫纸月。", "现在几点了？")
    print(f"纸月回复：{reply}")

if __name__ == "__main__":
    test()