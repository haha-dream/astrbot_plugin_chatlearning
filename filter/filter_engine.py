"""
过滤器

控制消息是否被记录 / 是否触发回复。
"""

import re

from astrbot.api.message_components import At, Face, Image, Plain, Reply


class FilterEngine:
    """三级过滤器：敏感词（不记录） / 黑名单用户（不记录） / 回复屏蔽词（不回复）"""

    _ALLOWED_MESSAGE_COMPONENTS = (Plain, Face, Image, Reply)
    _RE_AT_QQ = re.compile(r"@(\S+?)\(\d+\)")
    _RE_CQ_AT = re.compile(r"\[CQ:at,\s*qq=\d+\]")
    _OFFICIAL_BOT_MARKERS = (
        "mqqapi://",
        "q.qlogo.cn/qqapp/",
        "qqbot.ugcimg.cn",
    )
    _RE_OFFICIAL_BOT_LINK = re.compile(
        r"\[\]\([^)]*(?:%7b[^)]*version|\{[^)]*version)[^)]*\)",
        re.IGNORECASE,
    )

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

    # ── 消息级过滤 ──

    @classmethod
    def has_at_mention(cls, components: list | None, text: str = "") -> bool:
        """判断消息是否 @ 了用户（含 @全体成员）。

        Args:
            components: AstrBot 消息链组件列表。
            text: 消息纯文本，用于没有 At 组件的平台或历史消息兜底。

        Returns:
            消息包含 @ 用户时为 True。
        """
        if any(isinstance(comp, At) for comp in components or []):
            return True
        return bool(cls._RE_AT_QQ.search(text) or cls._RE_CQ_AT.search(text))

    @staticmethod
    def is_bot_author(raw_message: object) -> bool:
        """判断消息发送者是否为平台标记的机器人账号。

        QQ 官方机器人 API 会在 author.bot 中标记机器人账号；QQ 个人号适配器
        没有该字段，因此调用方只应对 QQ 官方平台使用此判断。

        Args:
            raw_message: 平台原始消息对象或事件字典。

        Returns:
            发送者被标记为机器人时为 True。
        """
        author = getattr(raw_message, "author", None)
        if author is None and isinstance(raw_message, dict):
            author = raw_message.get("author")
        if isinstance(author, dict):
            return bool(author.get("bot"))
        return bool(getattr(author, "bot", False))

    @classmethod
    def is_official_bot_message(cls, text: str) -> bool:
        """判断消息内容是否为 QQ 官方机器人发送的富文本消息。

        QQ 个人号协议无法从消息段判断发送者是否为官方机器人，但官方机器人
        的 Markdown 消息会带有固定的内联命令、应用头像或版本标记。

        Args:
            text: 消息纯文本。

        Returns:
            命中官方机器人消息特征时为 True。
        """
        lowered = text.lower()
        if any(marker in lowered for marker in cls._OFFICIAL_BOT_MARKERS):
            return True
        return bool(cls._RE_OFFICIAL_BOT_LINK.search(text))

    @classmethod
    def has_unsupported_component(cls, components: list | None) -> bool:
        """判断消息是否包含文本、表情或图片以外的组件。

        语音、视频、文件、转发、JSON 卡片等消息即使带有文字说明，也不应
        参与学习或回复；纯文本、QQ 表情（Face）、图片/图片表情（Image）和
        引用回复（Reply，正文仍由 Plain 提供）可以通过。

        Args:
            components: AstrBot 消息链组件列表。

        Returns:
            存在不允许的组件时为 True。
        """
        return any(
            not isinstance(comp, cls._ALLOWED_MESSAGE_COMPONENTS)
            for comp in components or []
        )

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
