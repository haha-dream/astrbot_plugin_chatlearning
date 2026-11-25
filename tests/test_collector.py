"""Collector 单元测试 — 时间窗口 Q&A 链构建与过期清理。"""

from learning.collector import Collector


class TestCollector:
    def test_first_message_opens_window(self):
        c = Collector(interval=10)
        assert c.feed("g1", "你好", "你好_raw", timestamp=100.0) is None
        assert c.group_count == 1

    def test_chain_within_interval(self):
        c = Collector(interval=10)
        assert c.feed("g1", "msgA", "msgA_raw", timestamp=100.0) is None
        qa1 = c.feed("g1", "msgB", "msgB_raw", timestamp=103.0)
        assert qa1 is not None
        assert qa1.question_text == "msgA"
        assert qa1.answer_text == "msgB"

        qa2 = c.feed("g1", "msgC", "msgC_raw", timestamp=108.0)
        assert qa2 is not None
        assert qa2.question_text == "msgB"
        assert qa2.answer_text == "msgC"

    def test_expired_interval_starts_new_window(self):
        c = Collector(interval=10)
        c.feed("g1", "msgA", "msgA_raw", timestamp=100.0)
        qa = c.feed("g1", "msgB", "msgB_raw", timestamp=111.0)
        assert qa is None
        qa2 = c.feed("g1", "msgC", "msgC_raw", timestamp=115.0)
        assert qa2 is not None
        assert qa2.question_text == "msgB"
        assert qa2.answer_text == "msgC"

    def test_prune_expired_and_clear(self):
        c = Collector(interval=10)
        c.feed("g1", "a", "a", timestamp=100.0)
        c.feed("g2", "b", "b", timestamp=115.0)
        removed = c.prune_expired(now=118.0)
        assert removed == 1
        assert c.group_count == 1
        c.clear_group("g2")
        assert c.group_count == 0
