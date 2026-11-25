"""
回复器 —— 向量匹配 + LLM 语义检查 + 跨群搜索 + 加权随机 + CD/概率控制
"""

import random
import time
from collections.abc import Callable
from typing import Optional

from astrbot.api import logger


class Responder:
    """从词库中匹配消息并生成回复。"""

    _MAX_LLM_GRAY_CANDIDATES = 5

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

    def should_attempt(
        self, last_reply_time: float = 0, now: float | None = None
    ) -> bool:
        """检查当前消息是否通过冷却时间与回复概率门槛。"""
        if now is None:
            now = time.time()
        if now - last_reply_time < self.reply_cd:
            return False
        if random.random() > self.reply_chance:
            return False
        return True

    # ── 主入口 ──

    async def try_reply(
        self,
        search_fn,  # async (group_id, query_vec) -> list[dict]
        group_id: str,
        query_vec: list[float],
        query_text: str = "",
        last_reply_time: float = 0,
        cross_search_fn=None,  # async (exclude_group_id, query_vec) -> list[dict]
        *,
        gate: bool = True,
    ) -> Optional["ReplyResult"]:
        """完整的匹配 → 回复流程，含 LLM 检查和跨群回退。"""
        if gate and not self.should_attempt(last_reply_time):
            return None

        # ── 阶段 1：本群搜索 ──
        candidates = await search_fn(group_id, query_vec)
        result = await self._filter_and_pick(candidates, query_text, strict=True)
        if result is not None:
            result.cross_group = False
            return result

        # ── 阶段 2：跨群回退 ──
        if cross_search_fn:
            candidates = await cross_search_fn(group_id, query_vec)
            result = await self._filter_and_pick(candidates, query_text, strict=False)
            if result is not None:
                result.cross_group = True
                return result

        return None

    async def _filter_and_pick(
        self,
        candidates: list[dict],
        query_text: str,
        strict: bool,
    ) -> Optional["ReplyResult"]:
        """候选集 → 有效答案过滤 → 阈值过滤 → (可选LLM) → 加权选取。"""
        if not candidates:
            return None

        valid = [c for c in candidates if c.get("answers")]
        if not valid:
            return None

        main_threshold = self.threshold if strict else max(self.threshold - 0.05, 0.6)
        direct = [c for c in valid if c.get("score", 0) >= main_threshold]

        # 灰色区间：LLM 二次确认
        confirmed: list[dict] = []
        if self.llm_check_fn and query_text:
            gray_floor = min(self.llm_gray_threshold, main_threshold)
            gray = [
                c
                for c in valid
                if gray_floor <= c.get("score", 0) < main_threshold
            ]
            if gray:
                gray.sort(key=lambda c: c.get("score", 0), reverse=True)
                confirmed = await self._llm_confirm(
                    query_text, gray[: self._MAX_LLM_GRAY_CANDIDATES]
                )

        pool = direct + confirmed
        if not pool:
            return None

        selected_question = self._weighted_pick(pool, weight_key="freq")
        answers = selected_question.get("answers", [])
        if not answers:
            return None

        selected_answer = self._weighted_pick(answers, weight_key="same")

        return ReplyResult(
            question_text=selected_question["question_text"],
            answer_text=selected_answer.get("answertext", ""),
            answer_raw=selected_answer.get("answer_raw", ""),
            score=selected_question["score"],
            question_id=int(selected_question.get("id", 0) or 0),
            group_id=str(selected_question.get("group_id", "") or ""),
        )

    async def _llm_confirm(
        self,
        query_text: str,
        candidates: list[dict],
    ) -> list[dict]:
        """用 LLM 确认灰色区间候选问题是否与查询语义相似。"""
        if not self.llm_check_fn:
            return []

        confirmed = []
        for c in candidates:
            if c.get("score", 0) < self.llm_gray_threshold:
                continue  # 低于灰色下限的直接排除
            try:
                ok = await self.llm_check_fn(query_text, c["question_text"])
                if ok:
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
        weights = [max(0, item.get(weight_key, 1)) for item in items]
        if sum(weights) == 0:
            return random.choice(items)
        return random.choices(items, weights=weights, k=1)[0]


class ReplyResult:
    """一次回复匹配的结果。"""

    __slots__ = (
        "question_text",
        "answer_text",
        "answer_raw",
        "score",
        "cross_group",
        "question_id",
        "group_id",
    )

    def __init__(
        self,
        question_text: str,
        answer_text: str,
        answer_raw: str,
        score: float,
        question_id: int = 0,
        group_id: str = "",
    ):
        self.question_text = question_text
        self.answer_text = answer_text
        self.answer_raw = answer_raw
        self.score = score
        self.cross_group = False
        self.question_id = question_id
        self.group_id = group_id

    def __repr__(self):
        return (
            f"ReplyResult(Q='{self.question_text[:20]}...', "
            f"A='{self.answer_text[:20]}...', score={self.score:.3f})"
        )
