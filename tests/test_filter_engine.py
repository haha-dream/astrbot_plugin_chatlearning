"""FilterEngine 单元测试 — @用户清洗、@消息过滤与机器人发送者识别。"""

from types import SimpleNamespace

from filter.filter_engine import FilterEngine

from astrbot.api.message_components import At, AtAll, Plain


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
        assert (
            FilterEngine.clean_at_mentions("访问 https://site(8080) 端口")
            == "访问 https://site(8080) 端口"
        )

    def test_email_like(self):
        assert (
            FilterEngine.clean_at_mentions("联系 user@host.com") == "联系 user@host.com"
        )

    def test_function_call_like(self):
        """非 @ 前缀的括号不匹配，保持原样。"""
        assert (
            FilterEngine.clean_at_mentions("调用 foo(1) bar(2)") == "调用 foo(1) bar(2)"
        )

    # ── 空格变体 ──

    def test_at_with_space_before_qq(self):
        assert FilterEngine.clean_at_mentions("@小明 (123)") == "@小明 (123)"

    def test_cq_at_with_varied_spacing(self):
        assert FilterEngine.clean_at_mentions("[CQ:at,qq=123]") == ""


class TestHasAtMention:
    """测试消息级 @ 用户过滤。"""

    def test_detects_at_component(self):
        components = [Plain(text="你好 "), At(qq="123456", name="小明")]
        assert FilterEngine.has_at_mention(components, "你好 @小明(123456)")

    def test_detects_at_all_component(self):
        assert FilterEngine.has_at_mention([AtAll()], "")

    def test_plain_at_text_is_not_a_mention(self):
        assert not FilterEngine.has_at_mention([Plain(text="@用户名")], "@用户名")

    def test_detects_at_qq_text_fallback(self):
        assert FilterEngine.has_at_mention([], "@小明(123456)")

    def test_detects_cq_at_text_fallback(self):
        assert FilterEngine.has_at_mention([], "[CQ:at, qq=123456]")

    def test_email_is_not_a_mention(self):
        assert not FilterEngine.has_at_mention([], "联系 user@host.com")

    def test_empty_message(self):
        assert not FilterEngine.has_at_mention([], "")


class TestIsBotAuthor:
    """测试 QQ 官方机器人发送者识别。"""

    def test_object_author_marked_bot(self):
        raw = SimpleNamespace(author=SimpleNamespace(bot=True))
        assert FilterEngine.is_bot_author(raw)

    def test_dict_author_marked_bot(self):
        assert FilterEngine.is_bot_author({"author": {"bot": True}})

    def test_human_author(self):
        raw = SimpleNamespace(author=SimpleNamespace(bot=False))
        assert not FilterEngine.is_bot_author(raw)

    def test_missing_author(self):
        assert not FilterEngine.is_bot_author(SimpleNamespace())
        assert not FilterEngine.is_bot_author({})
        assert not FilterEngine.is_bot_author(None)


class TestIsOfficialBotMessage:
    """测试通过消息内容识别 QQ 官方机器人富文本消息。"""

    def test_encoded_version_marker(self):
        assert FilterEngine.is_official_bot_message("[](%7B%22version%22%3A2%7D)")

    def test_decoded_version_marker(self):
        assert FilterEngine.is_official_bot_message('[]({"version": 2})')

    def test_inline_command_link(self):
        text = "> 发送 [/趴场次](mqqapi://aio/inlinecmd?command=x) 可查看"
        assert FilterEngine.is_official_bot_message(text)

    def test_official_app_avatar(self):
        text = (
            "![头像 #28px #28px]"
            "(https://q.qlogo.cn/qqapp/1903662881/41FF18E7CDEEA0E7760CFDCBDB37890C/100)"
        )
        assert FilterEngine.is_official_bot_message(text)

    def test_official_media_cdn(self):
        text = "![视奸 #960px #1593px](https://qqbot.ugcimg.cn/1903662881/image.png)"
        assert FilterEngine.is_official_bot_message(text)

    def test_full_official_bot_example(self):
        text = (
            "[](%7B%22version%22%3A2%7D)\n"
            "![头像 #28px #28px]"
            "(https://q.qlogo.cn/qqapp/1903662881/41FF18E7CDEEA0E7760CFDCBDB37890C/100)"
            " **趴视奸**\n"
            "> 我会一直看着你👁，直到永远......\n"
            "> 发送 [/趴场次](mqqapi://aio/inlinecmd?command=x) 可自动视奸游戏好友\n"
            "> [新增视奸](mqqapi://aio/inlinecmd?command=y)"
        )
        assert FilterEngine.is_official_bot_message(text)

    def test_normal_chat_is_not_filtered(self):
        assert not FilterEngine.is_official_bot_message("今天吃什么")

    def test_normal_markdown_is_not_filtered(self):
        assert not FilterEngine.is_official_bot_message(
            "**你好** [链接](https://example.com)"
        )
