"""
消息采集器 —— 学习模式的核心

将群聊消息按时间间隔构建成 Q&A 链：

  消息流:  msgA  --- 3s --- msgB  --- 2s --- msgC  --- 15s --- msgD
           │                  │                 │                  │
           └── 问题 ──────────┘                 │                  │
           └── msgA ← msgB 答案 ────────────────┘                  │
           └── msgB ← msgC 答案 ───────────────────────────────────┘
                                                    新窗口
           └── msgD = 新问题（间隔 > interval）────────────────────┘

规则：
  1. 时间窗口内第一条消息 = 问题
  2. 窗口内后续消息 = 上一个问题的答案，同时自身也作为问题记录
  3. 超过 interval 秒 → 开启新窗口
"""

import time
from typing import Optional


class Collector:
    """消息采集器：按群维护时间窗口，在内存中搭建 Q&A 链后写入词库。"""

    def __init__(self, interval: float):
        """
        Args:
            interval: 时间窗口阈值（秒），超过此间隔的消息视为新窗口
        """
        self.interval = interval
        # 每个群的窗口状态
        self._windows: dict[str, _Window] = {}

    @property
    def group_count(self) -> int:
        return len(self._windows)

    def feed(
        self,
        group_id: str,
        message_text: str,
        message_raw: str,
        timestamp: float | None = None,
    ) -> Optional["QAPair"]:
        """
        喂入一条消息，返回一个待写入的 Q&A 对（或 None 表示仅更新窗口无产出）。

        Args:
            group_id: 群/会话 ID
            message_text: 消息纯文本（用于 embedding 和显示）
            message_raw: 消息完整 JSON（消息链）
            timestamp: Unix 时间戳，默认当前时间

        Returns:
            QAPair 或 None
        """
        if timestamp is None:
            timestamp = time.time()

        window = self._windows.get(group_id)
        if window is None or (timestamp - window.last_time) > self.interval:
            # ── 开启新窗口 ──
            window = _Window(
                question_text=message_text,
                question_raw=message_raw,
                last_time=timestamp,
            )
            self._windows[group_id] = window
            return None  # 首个消息仅作问题，暂无答案

        # ── 窗口内：生成 Q&A 对 ──
        qa = QAPair(
            group_id=group_id,
            question_text=window.question_text,
            question_raw=window.question_raw,
            answer_text=message_text,
            answer_raw=message_raw,
        )

        # 当前消息成为新的「上一个问题」
        window.question_text = message_text
        window.question_raw = message_raw
        window.last_time = timestamp

        return qa

    def set_interval(self, seconds: float):
        self.interval = seconds

    def clear_group(self, group_id: str):
        self._windows.pop(group_id, None)


class _Window:
    """单个群的时间窗口状态。"""

    __slots__ = ("question_text", "question_raw", "last_time")

    def __init__(self, question_text: str, question_raw: str, last_time: float):
        self.question_text = question_text
        self.question_raw = question_raw
        self.last_time = last_time


class QAPair:
    """一个待写入词库的 Q&A 对。"""

    __slots__ = (
        "group_id",
        "question_text",
        "question_raw",
        "answer_text",
        "answer_raw",
    )

    def __init__(
        self,
        group_id: str,
        question_text: str,
        question_raw: str,
        answer_text: str,
        answer_raw: str,
    ):
        self.group_id = group_id
        self.question_text = question_text
        self.question_raw = question_raw
        self.answer_text = answer_text
        self.answer_raw = answer_raw

    def __repr__(self):
        return (
            f"QAPair(group={self.group_id}, "
            f"Q='{self.question_text[:30]}...', "
            f"A='{self.answer_text[:30]}...')"
        )
