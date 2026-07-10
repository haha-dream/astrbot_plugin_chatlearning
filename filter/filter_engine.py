"""
过滤器

控制消息是否被记录 / 是否触发回复。
"""

import re


class FilterEngine:
    """三级过滤器：敏感词（不记录） / 黑名单用户（不记录） / 回复屏蔽词（不回复）"""

    _RE_AT_QQ = re.compile(r"@(\S+?)\(\d+\)")
    _RE_CQ_AT = re.compile(r"\[CQ:at,\s*qq=\d+\]")

    def __init__(
        self,
        sensitive_words: list[str] | None = None,
        blocked_users: list[str] | None = None,
        reply_filter_words: list[str] | None = None,
    ):
        self.sensitive_words = set(sensitive_words or [])
        self.blocked_users = {str(u) for u in (blocked_users or [])}
        self.reply_filter_words = set(reply_filter_words or [])

    # ── 文本清洗 ──

    @classmethod
    def clean_at_mentions(cls, text: str) -> str:
        """剔除 @用户(QQ号) 中的括号及QQ号，以及 [CQ:at,qq=...] 段。"""
        text = cls._RE_AT_QQ.sub(r"@\1", text)
        text = cls._RE_CQ_AT.sub("", text)
        return text

    # ── 学习侧过滤 ──

    def should_record(self, text: str, user_id: str) -> tuple[bool, str]:
        """判断消息是否应该被记录。

        Returns:
            (是否记录, 拒绝原因 or "")
        """
        if str(user_id) in self.blocked_users:
            return False, "用户在黑名单中"

        if self._contains_any(text, self.sensitive_words):
            return False, "包含敏感词"

        return True, ""

    # ── 回复侧过滤 ──

    def should_reply(self, text: str) -> bool:
        """判断匹配到的消息是否应该被回复。"""
        return not self._contains_any(text, self.reply_filter_words)

    # ── 工具 ──

    @staticmethod
    def _contains_any(text: str, words: set[str]) -> bool:
        if not words:
            return False
        for w in words:
            if w and w in text:
                return True
        return False
