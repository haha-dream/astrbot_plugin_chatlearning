"""Responder 单元测试 — 向量匹配、灰色区间 LLM 确认、空答案过滤与跨群标识。"""

import time

import pytest
from reply.responder import Responder


def _cand(
    q: str,
    score: float,
    answers: list[dict] | None = None,
    freq: int = 1,
    cid: int = 1,
    group_id: str | None = None,
) -> dict:
    item = {
        "id": cid,
        "question_text": q,
        "question_raw": q,
        "answers": [{"answertext": "ok", "answer_raw": "ok", "same": 1}]
        if answers is None
        else answers,
        "freq": freq,
        "score": score,
    }
    if group_id is not None:
        item["group_id"] = group_id
    return item


class TestResponder:
    @pytest.mark.asyncio
    async def test_same_group_reply_has_cross_group_false_even_without_group_id(self):
        r = Responder(reply_chance=1.0, threshold=0.75, reply_cd=0)

        async def search_fn(_gid, _vec):
            return [_cand("你好", 0.90, cid=42)]

        res = await r.try_reply(search_fn, "g1", [0.1], query_text="你好")
        assert res is not None
        assert res.cross_group is False
        assert res.question_id == 42

    @pytest.mark.asyncio
    async def test_ignores_zero_answer_candidates(self):
        r = Responder(reply_chance=1.0, threshold=0.75, reply_cd=0)

        async def search_fn(_gid, _vec):
            return [
                _cand("orphan", 0.99, answers=[], freq=999, cid=1),
                _cand("valid", 0.80, answers=[{"answertext": "答", "answer_raw": "答", "same": 1}], cid=2),
            ]

        res = await r.try_reply(search_fn, "g1", [0.1], query_text="test")
        assert res is not None
        assert res.question_text == "valid"
        assert res.answer_text == "答"

    @pytest.mark.asyncio
    async def test_gray_zone_llm_runs_for_same_group_and_caps_calls(self):
        checked: list[str] = []

        async def llm_check(_q, cand):
            checked.append(cand)
            return cand == "g3"

        r = Responder(
            reply_chance=1.0,
            threshold=0.75,
            reply_cd=0,
            llm_check_fn=llm_check,
            llm_gray_threshold=0.5,
        )

        # 6 gray-zone candidates + 1 below floor; cap is 5 highest scores
        async def search_fn(_gid, _vec):
            return [
                _cand("below", 0.45, cid=0),
                _cand("g1", 0.52, cid=1),
                _cand("g2", 0.55, cid=2),
                _cand("g3", 0.68, cid=3),
                _cand("g4", 0.62, cid=4),
                _cand("g5", 0.60, cid=5),
                _cand("g6", 0.58, cid=6),
            ]

        res = await r.try_reply(search_fn, "g1", [0.1], query_text="query")
        assert res is not None
        assert res.question_text == "g3"
        assert res.question_id == 3
        assert len(checked) == 5
        assert "g1" not in checked
        assert "below" not in checked

    @pytest.mark.asyncio
    async def test_cross_group_fallback_sets_cross_group_true(self):
        r = Responder(reply_chance=1.0, threshold=0.75, reply_cd=0)

        async def search_fn(_gid, _vec):
            return []

        async def cross_fn(_gid, _vec):
            return [_cand("cross_q", 0.72, cid=99, group_id="g2")]

        res = await r.try_reply(
            search_fn, "g1", [0.1], query_text="hi", cross_search_fn=cross_fn
        )
        assert res is not None
        assert res.cross_group is True
        assert res.question_id == 99
        assert res.group_id == "g2"

    def test_should_attempt_respects_cd_and_chance(self):
        now = time.time()
        r = Responder(reply_chance=1.0, reply_cd=30)
        assert not r.should_attempt(last_reply_time=now - 5, now=now)
        assert r.should_attempt(last_reply_time=now - 31, now=now)

        r_zero = Responder(reply_chance=0.0, reply_cd=0)
        assert not r_zero.should_attempt(last_reply_time=0, now=now)
