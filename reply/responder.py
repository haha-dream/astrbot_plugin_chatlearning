"""
回复器 —— 向量匹配 + LLM 语义检查 + 跨群搜索 + 加权随机 + CD/概率控制
"""

import random
from collections.abc import Callable
from typing import Optional

from astrbot.api import logger


class Responder:
    """从词库中匹配消息并生成回复。"""

    def __init__(
        self,
        reply_chance: float = 0.5,
        threshold: float = 0.75,
        reply_cd: float = 30.0,
        llm_check_fn: Callable | None = None,
        llm_gray_threshold: float = 0.5,
    ):
        """
        Args:
            reply_chance: 回复触发概率 [0.0, 1.0]
            threshold: 向量相似度阈值 [0.0, 1.0]
            reply_cd: 回复冷却时间（秒）
            llm_check_fn: async (query_text, candidate_text) -> bool
            llm_gray_threshold: 触发 LLM 判断的灰色区间下限
        """
        self.reply_chance = reply_chance
        self.threshold = threshold
        self.reply_cd = reply_cd
        self.llm_check_fn = llm_check_fn
        self.llm_gray_threshold = llm_gray_threshold

    # ── 主入口 ──

    async def try_reply(
        self,
        search_fn,  # async (group_id, query_vec) -> list[dict]
        group_id: str,
        query_vec: list[float],
        query_text: str = "",
        last_reply_time: float = 0,
        cross_search_fn=None,  # async (exclude_group_id, query_vec) -> list[dict]
    ) -> Optional["ReplyResult"]:
        """完整的匹配 → 回复流程，含 LLM 检查和跨群回退。"""
        import time

        now = time.time()
        if now - last_reply_time < self.reply_cd:
            return None

        if random.random() > self.reply_chance:
            return None

        # ── 阶段 1：本群搜索 ──
        candidates = await search_fn(group_id, query_vec)
        result = await self._filter_and_pick(candidates, query_text, strict=True)

        # ── 阶段 2：跨群回退 ──
        if result is None and cross_search_fn:
            candidates = await cross_search_fn(group_id, query_vec)
            result = await self._filter_and_pick(candidates, query_text, strict=False)

        if result is None:
            return None

        # 附加跨群标识
        if candidates and candidates[0].get("group_id") != group_id:
            result.cross_group = True
        return result

    async def _filter_and_pick(
        self,
        candidates: list[dict],
        query_text: str,
        strict: bool,
    ) -> Optional["ReplyResult"]:
        """候选集 → 阈值过滤 → (可选LLM) → 加权选取。"""
        if not candidates:
            return None

        threshold = self.threshold if strict else max(self.threshold - 0.05, 0.6)

        # 阈值过滤
        filtered = [c for c in candidates if c.get("score", 0) >= threshold]

        # 灰色区间：LLM 二次确认
        if self.llm_check_fn and query_text:
            gray = [c for c in filtered if c["score"] < self.threshold]
            if gray:
                confirmed = await self._llm_confirm(query_text, gray)
                if confirmed:
                    # LLM 确认的合并回 filtered
                    confirmed_texts = {c["question_text"] for c in confirmed}
                    filtered = [
                        c
                        for c in filtered
                        if c["score"] >= self.threshold
                        or c["question_text"] in confirmed_texts
                    ]
                else:
                    # 移除被 LLM 否决的低分候选
                    confirmed_texts = {c["question_text"] for c in confirmed}
                    filtered = [
                        c
                        for c in filtered
                        if c["score"] >= self.threshold
                        or c["question_text"] not in confirmed_texts
                    ]
            # 只保留通过阈值的 + LLM 确认的
            filtered = [
                c
                for c in filtered
                if c["score"] >= self.threshold or c.get("_llm_confirmed", False)
            ]

        if not filtered:
            return None

        selected_question = self._weighted_pick(filtered, weight_key="freq")
        answers = selected_question.get("answers", [])
        if not answers:
            return None

        selected_answer = self._weighted_pick(answers, weight_key="same")

        return ReplyResult(
            question_text=selected_question["question_text"],
            answer_text=selected_answer.get("answertext", ""),
            answer_raw=selected_answer.get("answer_raw", ""),
            score=selected_question["score"],
        )

    async def _llm_confirm(
        self,
        query_text: str,
        candidates: list[dict],
    ) -> list[dict]:
        """用 LLM 批量确认候选问题是否与查询语义相似。"""
        if not self.llm_check_fn:
            return []

        confirmed = []
        for c in candidates:
            if c["score"] < self.llm_gray_threshold:
                continue  # 低于灰色下限的直接排除
            try:
                ok = await self.llm_check_fn(query_text, c["question_text"])
                if ok:
                    c["_llm_confirmed"] = True
                    confirmed.append(c)
            except Exception as e:
                logger.warning(f"[Responder] LLM 确认失败: {e}")
        return confirmed

    # ── 工具 ──

    @staticmethod
    def _weighted_pick(items: list[dict], weight_key: str) -> dict:
        """按指定字段加权随机选取一项。"""
        if len(items) == 1:
            return items[0]
        weights = [item.get(weight_key, 1) for item in items]
        total = sum(weights)
        if total == 0:
            return random.choice(items)
        return random.choices(items, weights=weights, k=1)[0]


class ReplyResult:
    """一次回复匹配的结果。"""

    __slots__ = ("question_text", "answer_text", "answer_raw", "score", "cross_group")

    def __init__(
        self, question_text: str, answer_text: str, answer_raw: str, score: float
    ):
        self.question_text = question_text
        self.answer_text = answer_text
        self.answer_raw = answer_raw
        self.score = score
        self.cross_group = False

    def __repr__(self):
        return (
            f"ReplyResult(Q='{self.question_text[:20]}...', "
            f"A='{self.answer_text[:20]}...', score={self.score:.3f})"
        )
