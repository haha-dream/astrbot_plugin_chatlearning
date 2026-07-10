"""FilterEngine.clean_at_mentions 单元测试 — @用户(QQ号) 清洗逻辑。"""

import pytest
from filter.filter_engine import FilterEngine


class TestCleanAtMentions:
    """测试 clean_at_mentions 对 @提及的各种模式处理。"""

    # ── 正常场景 ──

    def test_strip_qq_from_at(self):
        assert FilterEngine.clean_at_mentions("@小明(123456)") == "@小明"

    def test_pure_numeric_name(self):
        """纯数字昵称不应被误删。"""
        assert FilterEngine.clean_at_mentions("@123(456789)") == "@123"

    def test_alphanumeric_name(self):
        assert FilterEngine.clean_at_mentions("@abc123(111)") == "@abc123"

    def test_name_with_parens(self):
        """昵称内含括号时，仅剥离末尾 QQ 号括号。"""
        assert FilterEngine.clean_at_mentions("@小明(xx)(123)") == "@小明(xx)"

    def test_name_contains_digits_no_qq_parens(self):
        assert FilterEngine.clean_at_mentions("@user123") == "@user123"

    # ── 边界场景 ──

    def test_qq_in_middle_not_at_end(self):
        assert FilterEngine.clean_at_mentions("@小明(123)abc") == "@小明abc"

    def test_no_at_sign(self):
        """无 @ 的消息不应被修改。"""
        assert FilterEngine.clean_at_mentions("你好世界") == "你好世界"

    def test_only_at_no_qq(self):
        assert FilterEngine.clean_at_mentions("@用户名") == "@用户名"

    def test_empty_string(self):
        assert FilterEngine.clean_at_mentions("") == ""

    def test_at_with_whitespace(self):
        assert FilterEngine.clean_at_mentions("@小明 (123)") == "@小明 (123)"

    # ── 多条 @ 场景 ──

    def test_multiple_at_mentions(self):
        text = "@张三(111) @李四(222) 在吗"
        expected = "@张三 @李四 在吗"
        assert FilterEngine.clean_at_mentions(text) == expected

    # ── [CQ:at] 格式 ──

    def test_strip_cq_at(self):
        assert FilterEngine.clean_at_mentions("[CQ:at, qq=123456]") == ""

    def test_mixed_at_and_cq(self):
        text = "@小明(123) [CQ:at, qq=456789] 你好"
        assert FilterEngine.clean_at_mentions(text) == "@小明  你好"

    # ── 非 at 场景（不应误伤） ──

    def test_url_like_not_matched(self):
        """非 @ 场景不被误伤 —— 没有 @ 前缀不匹配。"""
        assert FilterEngine.clean_at_mentions("访问 https://site(8080) 端口") == "访问 https://site(8080) 端口"

    def test_email_like(self):
        assert FilterEngine.clean_at_mentions("联系 user@host.com") == "联系 user@host.com"

    def test_function_call_like(self):
        """非 @ 前缀的括号不匹配，保持原样。"""
        assert FilterEngine.clean_at_mentions("调用 foo(1) bar(2)") == "调用 foo(1) bar(2)"

    # ── 空格变体 ──

    def test_at_with_space_before_qq(self):
        assert FilterEngine.clean_at_mentions("@小明 (123)") == "@小明 (123)"

    def test_cq_at_with_varied_spacing(self):
        assert FilterEngine.clean_at_mentions("[CQ:at,qq=123]") == ""
