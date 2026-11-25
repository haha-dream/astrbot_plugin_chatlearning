"""ChatLearningPlugin 集成单元测试 — Embedding Provider、学习去重、!d 快速删除与过滤同步。"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.api.message_components import At, Plain, Reply
from filter.filter_engine import FilterEngine
from learning.collector import Collector
from reply.responder import Responder
from storage.wordstock import WordStock

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_chatlearning_main_test"


@pytest.fixture(scope="module")
def plugin_module():
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(PLUGIN_ROOT)]
    sys.modules[PACKAGE] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.main", PLUGIN_ROOT / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    for name in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
        sys.modules.pop(name, None)


def _make_config(overrides: dict | None = None):
    data = {
        "learning": True,
        "reply": True,
        "interval": 10,
        "learning_groups": [],
        "reply_groups": [],
        "reply_chance": 100,
        "reply_cd": 0,
        "cos_match_threshold": 0.75,
        "bot_names": [],
        "embedding_provider_id": "emb-1",
        "enable_local_embedding": False,
        "local_embedding_model": "BAAI/bge-small-zh-v1.5",
        "hf_endpoint": "https://huggingface.co",
        "llm_semantic_check": False,
        "llm_provider_id": "",
        "llm_semantic_threshold": 0.5,
        "cross_group_search": False,
        "auto_cleanup_days": 30,
        "version_cleanup_interval_hours": 6,
        "version_retention_hours": 2,
        "max_answers_per_question": 20,
        "auto_schedule_enable": False,
        "sensitive_words": [],
        "blocked_users": [],
    }
    if overrides:
        data.update(overrides)

    class FakeConfig(dict):
        def __init__(self, initial):
            super().__init__(initial)
            self.saved = 0

        async def save_config_async(self):
            self.saved += 1

        def save_config(self):
            self.saved += 1

    return FakeConfig(data)


def _make_event(
    text: str,
    *,
    group_id: str | None = "1001",
    sender_id: str = "u1",
    self_id: str = "bot1",
    messages: list | None = None,
    is_at_or_wake_command: bool = False,
):
    stopped = {"flag": False}
    event = SimpleNamespace(
        message_str=text,
        message_obj=SimpleNamespace(timestamp=1_700_000_000.0, raw_message=None),
        is_at_or_wake_command=is_at_or_wake_command,
        get_group_id=lambda: group_id,
        get_sender_id=lambda: sender_id,
        get_sender_name=lambda: "测试用户",
        get_self_id=lambda: self_id,
        get_platform_name=lambda: "aiocqhttp",
        get_messages=lambda: messages if messages is not None else [Plain(text=text)],
        plain_result=lambda msg: SimpleNamespace(text=msg),
        stop_event=lambda: stopped.__setitem__("flag", True),
        is_stopped=lambda: stopped["flag"],
    )
    return event


class FakeEmbeddingProvider:
    def __init__(self, pid: str, vec: list[float]):
        self._pid = pid
        self._vec = vec
        self.calls: list[str] = []

    def meta(self):
        return SimpleNamespace(id=self._pid)

    async def get_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vec)


class TestOnlineEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_finds_provider_via_meta_id(self, plugin_module):
        prov = FakeEmbeddingProvider("silicon-emb", [0.1, 0.2, 0.3, 0.4])
        ctx = SimpleNamespace(
            get_provider_by_id=lambda pid: None,
            get_all_embedding_providers=lambda: [prov],
        )
        plugin = plugin_module.ChatLearningPlugin(
            ctx, _make_config({"embedding_provider_id": "silicon-emb"})
        )
        plugin._embedding_ready = True

        found = plugin._find_embedding_provider("silicon-emb")
        assert found is prov
        vec = await plugin._get_embedding("你好")
        assert vec == [0.1, 0.2, 0.3, 0.4]
        assert prov.calls == ["你好"]


class TestLearningAndReply:
    @pytest.mark.asyncio
    async def test_learning_skips_orphan_and_handles_same_q_a(self, plugin_module):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = WordStock(str(Path(tmpdir) / "lancedb"))
            await ws.initialize()
            try:
                prov = FakeEmbeddingProvider("emb-1", [0.25, 0.25, 0.25, 0.25])
                ctx = SimpleNamespace(
                    get_provider_by_id=lambda pid: prov if pid == "emb-1" else None,
                    get_all_embedding_providers=lambda: [prov],
                )
                plugin = plugin_module.ChatLearningPlugin(ctx, _make_config())
                plugin.wordstock = ws
                plugin.collector = Collector(interval=10)
                plugin.filter_engine = FilterEngine()
                plugin._embedding_ready = True

                # 1) First message only opens window -> 0 DB writes
                await plugin._handle_learning("g1", "u1", "草", 100.0)
                assert await ws.count("g1") == 0

                # 2) Identical second message within window -> 1 record, 1 embed, freq=1
                await plugin._handle_learning("g1", "u2", "草", 102.0)
                assert await ws.count("g1") == 1
                rec = await ws.get_by_text("g1", "草")
                assert rec is not None
                assert rec["freq"] == 1
                assert len(rec["answers"]) == 1
                assert rec["answers"][0]["answertext"] == "草"
                assert prov.calls == ["草"]

                # 3) Third distinct message -> updates "草" (freq=2), does NOT insert "确实" as 0-answer orphan
                await plugin._handle_learning("g1", "u3", "确实", 105.0)
                assert await ws.count("g1") == 1
                rec = await ws.get_by_text("g1", "草")
                assert rec["freq"] == 2
                assert await ws.get_by_text("g1", "确实") is None
            finally:
                await ws.close()

    @pytest.mark.asyncio
    async def test_reply_skips_embedding_during_cooldown(self, plugin_module):
        prov = FakeEmbeddingProvider("emb-1", [0.1, 0.2, 0.3, 0.4])
        ctx = SimpleNamespace(
            get_provider_by_id=lambda pid: prov,
            get_all_embedding_providers=lambda: [prov],
        )
        plugin = plugin_module.ChatLearningPlugin(ctx, _make_config({"reply_cd": 30}))
        plugin.wordstock = MagicMock()
        plugin.responder = Responder(reply_chance=1.0, threshold=0.75, reply_cd=30)
        plugin._embedding_ready = True
        plugin._reply_cd["g1"] = 9e18  # active cooldown

        event = _make_event("你好", group_id="g1")
        outs = [r async for r in plugin._handle_reply(event, "g1", "你好")]
        assert outs == []
        assert prov.calls == []


class TestQuickDeleteAndCommands:
    @pytest.mark.asyncio
    async def test_quick_delete_matches_replied_bot_message_even_when_woken(
        self, plugin_module
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = WordStock(str(Path(tmpdir) / "lancedb"))
            await ws.initialize()
            try:
                id1 = await ws.add_question(
                    "g1", "q1", "q1", [0.1, 0.2, 0.3, 0.4], "回复一", "回复一"
                )
                id2 = await ws.add_question(
                    "g1", "q2", "q2", [0.4, 0.3, 0.2, 0.1], "回复二", "回复二"
                )
                ctx = SimpleNamespace(
                    get_provider_by_id=lambda _pid: None,
                    get_all_embedding_providers=lambda: [],
                )
                plugin = plugin_module.ChatLearningPlugin(ctx, _make_config())
                plugin.wordstock = ws
                plugin.collector = Collector(interval=10)
                plugin.filter_engine = FilterEngine()
                plugin._recent_replies["g1"] = [
                    ("回复一", id1, 100.0),
                    ("回复二", id2, 101.0),
                ]

                # User replies to the OLDER bot message ("回复一"), with At + is_at_or_wake_command=True
                reply_seg = Reply(id="m1", sender_id="bot1", message_str="回复一")
                event = _make_event(
                    "!d",
                    group_id="g1",
                    sender_id="normal_member",
                    self_id="bot1",
                    messages=[reply_seg, At(qq="bot1", name="Bot"), Plain(text=" !d")],
                    is_at_or_wake_command=True,
                )

                results = [r async for r in plugin.on_group_message(event)]
                assert len(results) == 1
                assert f"#{id1}" in results[0].text
                assert event.is_stopped()
                assert await ws.get_by_id(id1) is None
                assert await ws.get_by_id(id2) is not None

                # Replying !d to an unknown/unmatched bot message must NOT delete id2
                miss_reply = Reply(id="m_old", sender_id="bot1", message_str="不在记录里的消息")
                miss_event = _make_event(
                    "!d",
                    group_id="g1",
                    sender_id="normal_member",
                    self_id="bot1",
                    messages=[miss_reply, Plain(text="!d")],
                    is_at_or_wake_command=True,
                )
                miss_results = [r async for r in plugin.on_group_message(miss_event)]
                assert len(miss_results) == 1
                assert "未找到对应" in miss_results[0].text
                assert await ws.get_by_id(id2) is not None
            finally:
                await ws.close()

    @pytest.mark.asyncio
    async def test_filter_commands_update_runtime_engine(self, plugin_module):
        ctx = SimpleNamespace(
            get_provider_by_id=lambda _pid: None,
            get_all_embedding_providers=lambda: [],
        )
        plugin = plugin_module.ChatLearningPlugin(ctx, _make_config())
        plugin.filter_engine = FilterEngine()

        event = _make_event("/filter sensitive add 违禁词")
        _ = [r async for r in plugin.cmd_filter_sensitive_add(event, "违禁词")]
        ok, _ = plugin.filter_engine.should_record("包含违禁词", "u1")
        assert not ok

        _ = [r async for r in plugin.cmd_filter_blocked_add(event, "999")]
        ok, _ = plugin.filter_engine.should_record("正常消息", "999")
        assert not ok

        _ = [r async for r in plugin.cmd_filter_sensitive_remove(event, "违禁词")]
        _ = [r async for r in plugin.cmd_filter_blocked_remove(event, "999")]
        assert plugin.filter_engine.should_record("包含违禁词", "999") == (True, "")

    @pytest.mark.asyncio
    async def test_private_chat_commands_do_not_stringify_none(self, plugin_module):
        ctx = SimpleNamespace(
            get_provider_by_id=lambda _pid: None,
            get_all_embedding_providers=lambda: [],
        )
        plugin = plugin_module.ChatLearningPlugin(ctx, _make_config())
        plugin.wordstock = AsyncMock()
        plugin.wordstock.count = AsyncMock(return_value=0)

        event = _make_event("status", group_id=None)
        results = [r async for r in plugin.cmd_wordstock_stats(event)]
        assert "全部" in results[0].text
        plugin.wordstock.count.assert_awaited_once_with(None)

        results_add = [r async for r in plugin.cmd_learn_group_add(event, "")]
        assert "请在群聊中使用" in results_add[0].text
