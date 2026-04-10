import logging
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque

class ZhiyueAgent:
    def __init__(self, memory_mgr, cfg):
        self.cfg = cfg
        self.memory = memory_mgr
        self.logger = logging.getLogger("ZhiyueAgent")
        
        # 1. 结构化缓冲区：对应 Go 的 RingBuffer
        # 存储格式改为 Dict[int, deque[dict]] 以保留时间戳元数据
        self.buffers: Dict[int, deque] = {}
        self.buffer_size = cfg.app.debug and 15 or 10
        
        # 2. 思考状态管理：对应 Go 的 pendingThinks 和 processing
        self.pending_thinks = {}  # Dict[group_id, threading.Timer]
        self.last_processed_time = {} # Dict[group_id, float]
        self._lock = threading.Lock()
        
        # 模拟工具名，用于屏蔽注入
        self.tools = [{"name": "speak"}, {"name": "query_memory"}, {"name": "save_memory"}]

    def _parse_message_content(self, msg: dict) -> str:
        """
        解析消息并生成 Mumubot 标准行
        格式: [15:04:05] #ID 昵称(账号): [回复] 内容
        """
        ts = datetime.fromtimestamp(msg.get("time", time.time())).strftime("%H:%M:%S")
        content = msg.get("content", "").strip()
        
        # 处理回复
        reply_text = ""
        if msg.get("reply"):
            r = msg["reply"]
            reply_text = f"[回复 #{r.get('id')} {r.get('nickname')}:\"{r.get('content', '')[:50]}\"] "

        # 处理多模态占位符
        media = []
        for f in msg.get("faces", []): media.append(f"[表情:{f.get('name', '未知')}]")
        for i in msg.get("images", []): 
            tag = "表情包" if i.get("is_sticker") else "图片"
            media.append(f"[{tag}:{i.get('summary', '')}]")
        
        full_content = f"{reply_text}{content} {' '.join(media)}".strip()
        user_id = "你" if msg.get("user_id") == self.cfg.persona.qq else f"{msg.get('user_id')}"
        
        line = f"[{ts}] #{msg.get('id')} {msg.get('nickname')}({user_id}): {full_content}"
        
        # 动态屏蔽工具名防止注入
        for t in self.tools:
            if t["name"] in line:
                line = line.replace(t["name"], "“危险指令，已屏蔽”")
        return line

    def on_message(self, raw_msg: dict):
        """
        消息处理主入口：对应 Go 的 onMessage
        """
        group_id = raw_msg.get("group_id")
        if not group_id or not self.cfg.groups: # 简单校验群是否启用
            return

        # 1. 解析内容
        parsed_line = self._parse_message_content(raw_msg)
        
        # 2. 压入结构化缓冲区
        with self._lock:
            if group_id not in self.buffers:
                self.buffers[group_id] = deque(maxlen=self.buffer_size)
            self.buffers[group_id].append({
                "text": parsed_line,
                "time": raw_msg.get("time", time.time())
            })

        # 3. 异步持久化记忆 (骨架：调用 memory_mgr)
        # self.memory.add_message(raw_msg, parsed_line)

        # 4. 触发去抖动思考
        is_mentioned = raw_msg.get("is_mentioned", False)
        self._schedule_think(group_id, is_mentioned)

    def _schedule_think(self, group_id: int, is_mentioned: bool):
        """
        去抖动逻辑：对应 Go 的 scheduleThink
        """
        with self._lock:
            # 如果已有定时器，取消它（重新计时）
            if group_id in self.pending_thinks:
                self.pending_thinks[group_id].cancel()

            # 如果没被提及且不是特定的逻辑触发，可以不思考（此处简化，按原版逻辑提及必触发）
            # 原版逻辑见 react_agent.go 第 920 行
            
            # 设置延迟触发（模仿 ThinkDebounceMS）
            debounce_sec = 0.8 # 对应 800ms
            timer = threading.Timer(debounce_sec, self._execute_think, args=(group_id, is_mentioned))
            self.pending_thinks[group_id] = timer
            timer.start()

    def _execute_think(self, group_id: int, is_mentioned: bool):
        """
        真正开始思考：对应 Go 的 think 函数
        """
        self.logger.info(f"Group {group_id} 开始思考 (Mentioned: {is_mentioned})")
        
        # 获取上下文
        last_time = self.last_processed_time.get(group_id, 0)
        chat_context = self._build_chat_context(group_id, last_time)
        
        if not chat_context:
            return

        # 更新最后处理时间
        self.last_processed_time[group_id] = time.time()
        
        # TODO: 调用 LLMClient 进行生成
        # print(f"发送给 LLM 的上下文:\n{chat_context}")

    def _build_chat_context(self, group_id: int, last_time: float) -> str:
        """
        构建上下文并添加 (OLD) 标记：对应 Go 的 buildChatContext
        """
        if group_id not in self.buffers:
            return ""

        lines = []
        for msg in self.buffers[group_id]:
            line = msg["text"]
            # 如果时间早于上次处理时间，加上 (OLD)
            if msg["time"] <= last_time:
                line = f"(OLD) {line}"
            lines.append(line)
        
        return "\n".join(lines)