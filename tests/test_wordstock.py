"""
WordStock 单元测试 — 覆盖缓存、merge_insert、答案合并逻辑。
使用 LanceDB memory:// 模式，无需磁盘 IO。
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from storage.wordstock import WordStock

# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def ws():
    """创建临时目录 LanceDB 的 WordStock 实例并初始化。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stock = WordStock(tmpdir)
        await stock.initialize()
        yield stock
        await stock.close()


@pytest_asyncio.fixture
async def ws_with_data(ws):
    """预置一条问题 + 答案的 WordStock。"""
    vec = [0.1] * 384  # bge-small-zh-v1.5 维数
    rid = await ws.add_question(
        "g1", "你好", "你好_raw", vec,
        answer_text="你好呀", answer_raw="你好呀_raw",
    )
    assert rid > 0
    return ws, rid, vec


# ═══════════════════════════════════════════════════════════════
# 缓存单元测试（纯逻辑，无需 DB）
# ═══════════════════════════════════════════════════════════════

class TestCachePure:
    """测试 _text_cache_key / _cache_put / _cache_invalidate 纯逻辑。"""

    def test_cache_key_generation(self):
        key = WordStock._text_cache_key("g123", "你好")
        assert "g123" in key
        assert "你好" in key
        assert "\x00" in key

    def test_cache_key_no_collision(self):
        k1 = WordStock._text_cache_key("g1", "ab\x00c")  # text 内含 null
        k2 = WordStock._text_cache_key("g1\x00a", "bc")
        assert k1 != k2  # 复合键防止碰撞

    def test_cache_put_and_eviction(self, ws):
        ws._text_cache_max = 3
        for i in range(5):
            ws._cache_put(f"key_{i}", {"id": i})
        assert len(ws._text_cache) == 3
        # LRU: key_0 和 key_1 应被驱逐
        assert "key_0" not in ws._text_cache
        assert "key_1" not in ws._text_cache
        assert "key_2" in ws._text_cache
        assert "key_4" in ws._text_cache

    def test_cache_put_updates_order(self, ws):
        ws._text_cache_max = 3
        ws._cache_put("a", {"id": 1})
        ws._cache_put("b", {"id": 2})
        ws._cache_put("c", {"id": 3})
        ws._cache_put("a", {"id": 1})  # 重新访问 a，移到末尾
        ws._cache_put("d", {"id": 4})  # 应驱逐 b（最久未用）
        assert "a" in ws._text_cache
        assert "b" not in ws._text_cache
        assert "c" in ws._text_cache
        assert "d" in ws._text_cache

    def test_cache_invalidate(self, ws):
        ws._cache_put("key_x", {"id": 99})
        ws._cache_invalidate("g", "text")  # 不匹配的不影响
        assert "key_x" in ws._text_cache
        ws._cache_invalidate("g", "text")  # 不存在的静默忽略

    def test_cache_invalidate_real_key(self, ws):
        key = WordStock._text_cache_key("g1", "hello")
        ws._cache_put(key, {"id": 42})
        ws._cache_invalidate("g1", "hello")
        assert key not in ws._text_cache


# ═══════════════════════════════════════════════════════════════
# 缓存 + DB 集成测试
# ═══════════════════════════════════════════════════════════════

class TestCacheWithDB:

    @pytest.mark.asyncio
    async def test_get_by_text_cache_hit(self, ws_with_data):
        ws, _, _ = ws_with_data
        # 第一次 → DB 查询并缓存
        rec1 = await ws.get_by_text("g1", "你好")
        assert rec1 is not None
        assert rec1["question_text"] == "你好"

        # 第二次 → 缓存命中，不应再查 DB（通过 mock 验证）
        with patch.object(ws._table, "query", wraps=ws._table.query) as spy:
            rec2 = await ws.get_by_text("g1", "你好")
            spy.assert_not_called()  # 缓存命中，跳过 DB
        assert rec2["id"] == rec1["id"]

    @pytest.mark.asyncio
    async def test_get_by_text_cache_miss_queries_db(self, ws):
        # 空库，缓存无命中
        rec = await ws.get_by_text("g1", "不存在")
        assert rec is None

    @pytest.mark.asyncio
    async def test_add_question_populates_cache(self, ws):
        rid = await ws.add_question(
            "g1", "测试", "测试_raw", [0.1] * 384
        )
        # 缓存命中
        key = WordStock._text_cache_key("g1", "测试")
        assert key in ws._text_cache
        assert ws._text_cache[key]["id"] == rid

    @pytest.mark.asyncio
    async def test_delete_invalidates_cache(self, ws_with_data):
        ws, rid, _ = ws_with_data
        key = WordStock._text_cache_key("g1", "你好")
        assert key in ws._text_cache

        await ws.delete(rid)
        assert key not in ws._text_cache
        # DB 中也应消失
        assert await ws.get_by_id(rid) is None

    @pytest.mark.asyncio
    async def test_add_answer_invalidates_cache(self, ws_with_data):
        ws, rid, _ = ws_with_data
        key = WordStock._text_cache_key("g1", "你好")
        assert key in ws._text_cache

        await ws.add_answer(rid, "新答案", "新答案_raw")
        # merge_insert → 缓存被清
        assert key not in ws._text_cache

        # 再次查询重新缓存
        rec = await ws.get_by_text("g1", "你好")
        assert rec is not None


# ═══════════════════════════════════════════════════════════════
# add_answer 答案合并逻辑
# ═══════════════════════════════════════════════════════════════

class TestAddAnswerMerge:

    @pytest.mark.asyncio
    async def test_new_answer_appended(self, ws_with_data):
        ws, rid, _ = ws_with_data
        ok = await ws.add_answer(rid, "新回复", "新回复_raw")
        assert ok

        rec = await ws.get_by_id(rid)
        answers = rec["answers"]
        assert len(answers) == 2
        texts = [str(a["answertext"]) for a in answers]
        assert "你好呀" in texts
        assert "新回复" in texts

    @pytest.mark.asyncio
    async def test_existing_answer_increments_same(self, ws_with_data):
        ws, rid, _ = ws_with_data
        # 重复添加同一答案
        await ws.add_answer(rid, "你好呀", "你好呀_raw")

        rec = await ws.get_by_id(rid)
        answers = rec["answers"]
        assert len(answers) == 1  # 不新增，合并
        assert int(answers[0]["same"]) == 2  # same 从 1 → 2

    @pytest.mark.asyncio
    async def test_add_answer_increments_freq(self, ws_with_data):
        ws, rid, _ = ws_with_data
        before = await ws.get_by_id(rid)
        await ws.add_answer(rid, "回答2", "回答2_raw")
        after = await ws.get_by_id(rid)
        assert after["freq"] == before["freq"] + 1

    @pytest.mark.asyncio
    async def test_max_answers_enforcement(self, ws_with_data):
        ws, rid, _ = ws_with_data
        ws.max_answers = 3

        for i in range(5):
            await ws.add_answer(rid, f"答案{i}", f"答案{i}_raw")

        rec = await ws.get_by_id(rid)
        assert len(rec["answers"]) == 3  # 最多保留 3 条
        # 应保留最新的（按 added_at）
        texts = [str(a["answertext"]) for a in rec["answers"]]
        for i in range(2):  # 答案0, 答案1 应被裁剪
            assert f"答案{i}" not in texts
        for i in range(2, 5):  # 答案2, 答案3, 答案4 应保留
            assert f"答案{i}" in texts

    @pytest.mark.asyncio
    async def test_add_answer_nonexistent_id(self, ws):
        ok = await ws.add_answer(99999, "答", "答_raw")
        assert not ok


# ═══════════════════════════════════════════════════════════════
# merge_insert 行为验证（mock）
# ═══════════════════════════════════════════════════════════════

class TestMergeInsert:

    @pytest.mark.asyncio
    async def test_merge_insert_called_not_delete_add(self, ws_with_data):
        """验证 add_answer 调用的是 merge_insert 而非 delete + add。"""
        ws, rid, _ = ws_with_data

        # Mock merge_insert builder 链
        mock_execute = AsyncMock(return_value=MagicMock())
        mock_builder = MagicMock()
        mock_builder.when_matched_update_all.return_value = mock_builder
        mock_builder.when_not_matched_insert_all.return_value = mock_builder
        mock_builder.execute = mock_execute

        orig_add = AsyncMock()
        orig_delete = AsyncMock()

        with (
            patch.object(ws._table, "merge_insert", return_value=mock_builder) as mi_spy,
            patch.object(ws._table, "add", orig_add),
            patch.object(ws._table, "delete", orig_delete),
        ):
            await ws.add_answer(rid, "merge测试", "merge_raw")

            # merge_insert 被调用
            mi_spy.assert_called_once_with("id")
            # when_matched_update_all / when_not_matched_insert_all 被链式调用
            mock_builder.when_matched_update_all.assert_called_once()
            mock_builder.when_not_matched_insert_all.assert_called_once()
            # execute 被调用
            mock_execute.assert_called_once()

            # add / delete 不应被调用
            orig_add.assert_not_called()
            orig_delete.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# 向量搜索与精确查询
# ═══════════════════════════════════════════════════════════════

class TestSearch:

    @pytest.mark.asyncio
    async def test_search_similar_empty_db(self, ws):
        results = await ws.search_similar("g1", [0.1] * 384)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_finds_match(self, ws_with_data):
        ws, _, vec = ws_with_data
        results = await ws.search_similar("g1", vec, threshold=0.0)
        assert len(results) >= 1
        assert results[0]["question_text"] == "你好"
        assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_search_similar_threshold_filters(self, ws_with_data):
        ws, _, vec = ws_with_data
        # 用相反的向量 → 余弦距离应很大，高阈值应过滤掉
        opposite = [-v for v in vec]
        results = await ws.search_similar("g1", opposite, threshold=0.9999)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_wrong_group(self, ws_with_data):
        ws, _, vec = ws_with_data
        results = await ws.search_similar("other_group", vec, threshold=0.0)
        assert results == []

    @pytest.mark.asyncio
    async def test_get_by_text_exact(self, ws_with_data):
        ws, _, _ = ws_with_data
        rec = await ws.get_by_text("g1", "你好")
        assert rec["question_text"] == "你好"
        assert rec["group_id"] == "g1"

    @pytest.mark.asyncio
    async def test_get_by_text_no_match(self, ws):
        assert await ws.get_by_text("g1", "不存在") is None

    @pytest.mark.asyncio
    async def test_get_by_text_different_group(self, ws_with_data):
        ws, _, _ = ws_with_data
        assert await ws.get_by_text("other_group", "你好") is None


# ═══════════════════════════════════════════════════════════════
# 跨群搜索
# ═══════════════════════════════════════════════════════════════

class TestCrossGroupSearch:

    @pytest.mark.asyncio
    async def test_excludes_current_group(self, ws):
        # 写入两个群的数据
        await ws.add_question("g1", "你好", "g1_raw", [0.1] * 384)
        await ws.add_question("g2", "你好", "g2_raw", [0.1] * 384)

        results = await ws.search_cross_group("g1", [0.1] * 384, threshold=0.0)
        # 不应包含 g1 的结果
        for r in results:
            assert r["group_id"] != "g1"
        # 应包含 g2 的结果
        assert any(r["group_id"] == "g2" for r in results)


# ═══════════════════════════════════════════════════════════════
# 统计与管理
# ═══════════════════════════════════════════════════════════════

class TestAdmin:

    @pytest.mark.asyncio
    async def test_count(self, ws_with_data):
        ws, _, _ = ws_with_data
        assert await ws.count("g1") == 1
        assert await ws.count("g2") == 0
        assert await ws.count() == 1

    @pytest.mark.asyncio
    async def test_touch_increments_freq(self, ws_with_data):
        ws, rid, _ = ws_with_data
        before = await ws.get_by_id(rid)
        await ws.touch(rid)
        after = await ws.get_by_id(rid)
        assert after["freq"] == before["freq"] + 1

    @pytest.mark.asyncio
    async def test_export_data(self, ws_with_data):
        ws, _, _ = ws_with_data
        data = await ws.export_data("g1")
        assert len(data) == 1
        assert "vec" not in data[0]  # 不含向量
        assert data[0]["question_text"] == "你好"

    @pytest.mark.asyncio
    async def test_cleanup_low_freq(self, ws):
        await ws.add_question("g1", "rare", "r", [0.1] * 384)
        await ws.add_question("g1", "freq", "f", [0.2] * 384)
        # 手动调高 freq 避免被清理
        rec = await ws.get_by_text("g1", "freq")
        await ws.touch(rec["id"])
        await ws.touch(rec["id"])

        # days=-1 → cutoff 在未来，所有记录通过时间过滤，仅按 min_freq 筛选
        cleaned = await ws.cleanup_low_freq(days=-1, min_freq=2)
        # rare (freq=1) 应被清理，freq (freq=3) 应保留
        assert cleaned >= 1
        assert await ws.get_by_text("g1", "rare") is None
        assert await ws.get_by_text("g1", "freq") is not None
