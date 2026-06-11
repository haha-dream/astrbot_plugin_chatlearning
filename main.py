"""
astrbot_plugin_chatlearning — 主插件

从群聊中自动学习对话模式，用向量嵌入做语义匹配回复。
"""

import asyncio
import os
import random
import time
from collections import OrderedDict

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .filter.filter_engine import FilterEngine
from .learning.collector import Collector
from .reply.responder import Responder
from .storage.wordstock import WordStock

LANCE_DB_DIR = "lancedb"


@register(
    "astrbot_plugin_chatlearning",
    "satori",
    "群聊对话学习与智能回复——基于向量嵌入的语义对话引擎",
    "v0.2.0",
)
class ChatLearningPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self._cfg = config
        self.wordstock: WordStock | None = None
        self.collector: Collector | None = None
        self.responder: Responder | None = None
        self.filter_engine: FilterEngine | None = None
        self._reply_cd: dict[str, float] = {}
        self._embedding_ready: bool = False
        self._embedding_warned: bool = False
        self._local_model: object | None = None
        self._model_lock = asyncio.Lock()
        self._embed_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._embed_cache_max = 3000
        self._recent_replies: dict[str, list[tuple]] = {}
        self._cleanup_task: asyncio.Task | None = None

    # ═══ 配置快捷读取 ═══════════════════════════════════════

    def _C(self, key, default=None):
        """读配置项。"""
        return self._cfg.get(key, default)

    async def _save(self):
        """持久化配置。"""
        self._cfg.save_config()

    def _is_group_learning(self, gid: str) -> bool:
        groups = self._C("learning_groups", []) or []
        return not groups or gid in {str(g) for g in groups}

    def _is_group_reply(self, gid: str) -> bool:
        groups = self._C("reply_groups", []) or []
        return not groups or gid in {str(g) for g in groups}

    # ═══ 生命周期 ═════════════════════════════════════════════

    async def initialize(self):
        logger.debug("[ChatLearning] 初始化中...")

        from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

        data_root = os.path.join(get_astrbot_plugin_data_path(), self.name)
        os.makedirs(data_root, exist_ok=True)

        self.wordstock = WordStock(os.path.join(data_root, LANCE_DB_DIR))
        await self.wordstock.initialize()

        self.collector = Collector(interval=self._C("interval", 10))

        # LLM 语义检查
        llm_check_fn = None
        self._llm_check_provider_id = ""
        if self._C("llm_semantic_check"):
            llm_pid = (self._C("llm_provider_id", "") or "").strip()
            if llm_pid:
                llm_p = self.context.get_provider_by_id(llm_pid)
                if llm_p:
                    self._llm_check_provider_id = llm_pid
                    llm_check_fn = self._llm_semantic_check
                    logger.debug(f"[ChatLearning] LLM 语义检查: {llm_pid}")
                else:
                    logger.warning(f"[ChatLearning] llm_provider_id '{llm_pid}' 未找到")
            else:
                logger.warning(
                    "[ChatLearning] llm_semantic_check=true 但 llm_provider_id 为空"
                )

        self.responder = Responder(
            reply_chance=self._C("reply_chance", 50) / 100.0,
            threshold=self._C("cos_match_threshold", 0.75),
            reply_cd=self._C("reply_cd", 30),
            llm_check_fn=llm_check_fn,
            llm_gray_threshold=self._C("llm_semantic_threshold", 0.5),
        )

        self.filter_engine = FilterEngine(
            sensitive_words=self._C("sensitive_words"),
            blocked_users=self._C("blocked_users"),
        )

        self.wordstock.max_answers = self._C("max_answers_per_question", 20)

        # Embedding
        if self._C("enable_local_embedding"):
            self._embedding_ready = True
            logger.info(
                f"[ChatLearning] 本地向量: {self._C('local_embedding_model', 'BAAI/bge-small-zh-v1.5')}"
            )
        else:
            target_id = (self._C("embedding_provider_id", "") or "").strip()
            if not target_id:
                logger.warning(
                    "[ChatLearning] ⚠ 未配置 embedding_provider_id 且未启用本地向量"
                )
                self._embedding_ready = False
            else:
                providers = self.context.get_all_embedding_providers()
                found = any(
                    (getattr(p, "provider_id", "") or getattr(p, "name", ""))
                    == target_id
                    for p in providers
                )
                if found:
                    self._embedding_ready = True
                    logger.debug(f"[ChatLearning] Embedding Provider: {target_id}")
                else:
                    self._embedding_ready = False
                    logger.warning(
                        f"[ChatLearning] Embedding Provider '{target_id}' 未找到"
                    )

        logger.info("[ChatLearning] ✅ 初始化完成")

        # Pages API
        PN = "astrbot_plugin_chatlearning"
        self.context.register_web_api(
            f"/{PN}/stats", self._api_stats, ["GET"], "词库统计"
        )
        self.context.register_web_api(
            f"/{PN}/search", self._api_search, ["GET"], "搜索词库"
        )
        self.context.register_web_api(
            f"/{PN}/entry", self._api_entry_detail, ["GET"], "词条详情"
        )
        self.context.register_web_api(
            f"/{PN}/delete", self._api_delete_entry, ["POST"], "删除词条"
        )

        self._cleanup_task = asyncio.create_task(self._periodic_maintenance())

    async def terminate(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        if self.wordstock:
            await self.wordstock.close()

    # ═══ 群消息 ═══════════════════════════════════════════════

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        if not self.wordstock or not self.filter_engine or not self.collector:
            return

        group_id = event.get_group_id()
        if not group_id:
            return
        group_id = str(group_id)

        if str(event.get_sender_id()) == str(event.get_self_id()):
            return

        user_id = str(event.get_sender_id())
        timestamp = event.message_obj.timestamp if event.message_obj else time.time()
        plain_text = event.message_str.strip()
        if not plain_text:
            return
        if self._is_noise(plain_text):
            return

        # 指令/@唤醒消息跳过学习/回复，留给 AstrBot LLM
        if event.is_at_or_wake_command:
            return

        # 快速删除
        handled, qd_reply = await self._handle_quick_delete(event, group_id, plain_text)
        if handled:
            if qd_reply:
                yield qd_reply
            return

        # 学习
        if self._C("learning") and self._is_group_learning(group_id):
            await self._handle_learning(group_id, user_id, plain_text, timestamp)

        # 回复
        if self._C("reply") and self._is_group_reply(group_id):
            async for result in self._handle_reply(event, group_id, plain_text):
                if result is not None:
                    yield result

    # ═══ 学习 ═════════════════════════════════════════════════

    async def _handle_learning(self, group_id, user_id, text, timestamp):
        ok, _ = self.filter_engine.should_record(text, user_id)
        if not ok:
            return

        qa = self.collector.feed(group_id, text, text, timestamp)
        if qa is None:
            return

        # 先查 DB（快），确定哪些需要 embedding，再批量生成
        ans_exist = await self.wordstock.get_by_text(qa.group_id, qa.answer_text)
        q_exist = await self.wordstock.get_by_text(qa.group_id, qa.question_text)

        need_embed = []
        if not ans_exist:
            need_embed.append(qa.answer_text)
        if not q_exist:
            need_embed.append(qa.question_text)

        vecs = {}
        if need_embed:
            batch = await self._get_embeddings_batch(need_embed)
            for t, v in zip(need_embed, batch):
                if v is not None:
                    vecs[t] = v

        # 1) 答案侧
        if ans_exist:
            await self.wordstock.touch(ans_exist["id"])
        elif qa.answer_text in vecs:
            await self.wordstock.add_question(
                qa.group_id, qa.answer_text, qa.answer_raw, vecs[qa.answer_text]
            )
        else:
            return

        # 2) 问题侧
        if q_exist:
            await self.wordstock.add_answer(
                q_exist["id"], qa.answer_text, qa.answer_raw
            )
        elif qa.question_text in vecs:
            await self.wordstock.add_question(
                qa.group_id,
                qa.question_text,
                qa.question_raw,
                vecs[qa.question_text],
                answer_text=qa.answer_text,
                answer_raw=qa.answer_raw,
            )

        self._write_count = getattr(self, "_write_count", 0) + 1
        if self._write_count % 100 == 0:
            await self.wordstock.ensure_index()
        if self._write_count % 10 == 0:
            logger.info(f"[ChatLearning] 已学习 {self._write_count} 条")

    # ═══ 回复 ═════════════════════════════════════════════════

    async def _handle_reply(self, event, group_id, text):
        vec = await self._get_embedding(text)
        if vec is None:
            return

        threshold = self._C("cos_match_threshold", 0.75)

        async def _search(gid, qvec):
            return await self.wordstock.search_similar(
                gid, qvec, threshold=threshold * 0.8, top_k=20
            )

        cross_fn = (
            self.wordstock.search_cross_group if self._C("cross_group_search") else None
        )

        last_reply = self._reply_cd.get(group_id, 0)
        result = await self.responder.try_reply(
            _search,
            group_id,
            vec,
            query_text=text,
            last_reply_time=last_reply,
            cross_search_fn=cross_fn,
        )
        if result is None:
            event.stop_event()
            return

        self._reply_cd[group_id] = time.time()

        q_exist = await self.wordstock.get_by_text(group_id, result.question_text)
        q_id = q_exist["id"] if q_exist else 0
        recent = self._recent_replies.get(group_id, [])
        recent.append((result.answer_text, q_id, time.time()))
        if len(recent) > 30:
            recent = recent[-30:]
        self._recent_replies[group_id] = recent

        fusion_mode = self._C("persona_fusion_mode", "off")
        if fusion_mode in ("blend", "full"):
            fused = await self._persona_fusion(
                event, text, result.answer_text, fusion_mode
            )
            if fused:
                await asyncio.sleep(random.uniform(0.5, 2.0))
                self._reply_count = getattr(self, "_reply_count", 0) + 1
                logger.info(
                    f"[ChatLearning] 回复 #{self._reply_count}: "
                    f"Q={text[:20]} → A={fused[:20]}"
                )
                event.stop_event()
                yield event.plain_result(fused)
            return

        await asyncio.sleep(random.uniform(0.5, 2.0))
        text_out = self._apply_placeholders(result.answer_text, event)
        self._reply_count = getattr(self, "_reply_count", 0) + 1
        logger.info(
            f"[ChatLearning] 回复 #{self._reply_count}: "
            f"Q={text[:20]} → A={result.answer_text[:20]}"
        )
        event.stop_event()
        yield event.plain_result(text_out)

    async def _persona_fusion(self, event, query, matched_answer, mode):
        """用配置的融合 LLM 润色或生成回复（带会话上下文）。"""
        pid = (self._C("persona_fusion_provider", "") or "").strip()
        if not pid:
            return None
        prov = self.context.get_provider_by_id(pid)
        if not prov:
            return None

        persona_prompt = await self._get_fusion_persona_prompt()
        prompt = self._build_fusion_prompt(query, matched_answer, mode)

        # 获取会话历史
        import json as _json

        cm = self.context.conversation_manager
        cid = await cm.get_curr_conversation_id(event.unified_msg_origin)
        contexts = []
        if cid:
            conv = await cm.get_conversation(event.unified_msg_origin, cid)
            if conv and conv.history:
                try:
                    contexts = _json.loads(conv.history)
                except Exception:
                    pass

        try:
            resp = await prov.text_chat(
                prompt=prompt,
                system_prompt=persona_prompt or None,
                contexts=contexts if contexts else None,
            )
            return (resp.completion_text or "").strip()
        except Exception as e:
            logger.warning(f"[ChatLearning] 融合生成失败: {e}")
            return None

    async def _get_fusion_persona_prompt(self):
        pid = (self._C("persona_fusion_persona", "") or "").strip()
        if not pid:
            return ""
        try:
            p = await self.context.persona_manager.get_persona(pid)
            return p.system_prompt if p else ""
        except Exception:
            return ""

    @staticmethod
    def _build_fusion_prompt(query, answer, mode):
        if mode == "blend":
            return f"用户说：{query}\n词库匹配的参考回复：{answer}\n请用你自己的风格改写这条回复，保持原意但让表达更自然。只输出改写后的文本。"
        return f"用户说：{query}\n参考上下文（来自群聊词库）：{answer}\n请用你自己的风格回复这条消息。只输出回复文本。"

    # ═══ 快速删除 ═════════════════════════════════════════════

    async def _handle_quick_delete(self, event, group_id, text):
        t = text.strip().lower()
        if t not in ("!d", "!delete", "！d", "！delete"):
            return False, None
        for seg in event.get_messages():
            if hasattr(seg, "type") and str(seg.type) == "Reply":
                break
        else:
            return False, None

        recent = self._recent_replies.get(group_id, [])
        if not recent:
            return True, event.plain_result("没有可删除的最近回复")

        _, ws_id, _ = recent.pop()
        self._recent_replies[group_id] = recent
        if ws_id > 0:
            await self.wordstock.delete(ws_id)
            return True, event.plain_result(f"🗑 已从词库删除 #{ws_id}")
        return True, event.plain_result("🗑 已移除最近回复")

    # ═══ 后台维护 ═════════════════════════════════════════════

    async def _periodic_maintenance(self):
        last_cleanup = time.time()
        last_states = {"learn": None, "reply": None}
        while True:
            await asyncio.sleep(60)
            try:
                now = time.time()
                if self._C("auto_schedule_enable"):
                    await self._check_schedule(last_states)
                if now - last_cleanup > 6 * 3600:
                    days = int(self._C("auto_cleanup_days", 30))
                    if days > 0:
                        cleaned = await self.wordstock.cleanup_low_freq(
                            days=days, min_freq=2
                        )
                        if cleaned:
                            logger.debug(f"[ChatLearning] 自动清理: {cleaned} 条")
                    await self.wordstock.build_index()
                    freed = await self.wordstock.cleanup_versions()
                    if freed:
                        logger.info(
                            f"[ChatLearning] LanceDB 清理: {freed / 1024 / 1024:.0f} MB"
                        )
                    last_cleanup = now
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[ChatLearning] 后台维护失败: {e}")

    async def _check_schedule(self, last_states):
        now = time.localtime()
        current = f"{now.tm_hour:02d}:{now.tm_min:02d}"
        for mode_key, cfg_key, state_key in [
            ("学习", "auto_schedule_learn_on", "learn"),
            ("学习", "auto_schedule_learn_off", "learn"),
            ("回复", "auto_schedule_reply_on", "reply"),
            ("回复", "auto_schedule_reply_off", "reply"),
        ]:
            expected = (self._C(cfg_key, "") or "").strip()
            if expected != current or last_states.get(state_key) == current:
                continue
            last_states[state_key] = current
            enable = "on" in cfg_key
            if state_key == "learn":
                self._cfg["learning"] = enable
                await self._save()
            else:
                self._cfg["reply"] = enable
                await self._save()
            logger.info(
                f"[ChatLearning] 定时: {mode_key}{'开启' if enable else '关闭'}"
            )

    # ═══ 占位符 ═══════════════════════════════════════════════

    def _apply_placeholders(self, text, event):
        names = self._C("bot_names", []) or []
        if names:
            text = text.replace("{me}", random.choice(names))
        sender = event.get_sender_name()
        if sender:
            text = text.replace("{name}", sender)
        return text.replace("{segment}", "\n")

    _NOISE = frozenset(
        {
            "",
            "。",
            "？",
            "！",
            "…",
            "...",
            "……",
            "。。",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
            "+",
            "-",
            "=",
            ".",
            ",",
            "!",
            "?",
            "~",
            "～",
            "[图片]",
            "[表情]",
            "[视频]",
            "[语音]",
            "[文件]",
        }
    )

    @classmethod
    def _is_noise(cls, text):
        t = text.strip()
        if len(t) <= 3 and t in cls._NOISE:
            return True
        if t.startswith(("http://", "https://")):
            return True
        if len(t) <= 2 and all(
            c in "0123456789.,;:!?。，、；：！？…~～-+=/\\()[]{}（）【】" for c in t
        ):
            return True
        return False

    # ═══ LLM 语义检查 ════════════════════════════════════════

    async def _llm_semantic_check(self, query, candidate):
        if not self._llm_check_provider_id:
            return False
        prov = self.context.get_provider_by_id(self._llm_check_provider_id)
        if not prov:
            return False
        try:
            resp = await prov.text_chat(
                prompt=f"判断以下两句话是否表达了相同的意思，仅回答 YES 或 NO。\n句子A: {query}\n句子B: {candidate}"
            )
            return (resp.completion_text or "").strip().upper().startswith("YES")
        except Exception:
            return False

    # ═══ Embedding ════════════════════════════════════════════

    async def _get_embedding(self, text):
        if not text.strip():
            return None
        if text in self._embed_cache:
            self._embed_cache.move_to_end(text)
            return self._embed_cache[text]
        if not self._embedding_ready:
            if not self._embedding_warned:
                self._embedding_warned = True
                logger.debug("[ChatLearning] embedding 不可用")
            return None

        result = None
        if self._C("enable_local_embedding"):
            result = await self._local_embed(text)
        else:
            target_id = (self._C("embedding_provider_id", "") or "").strip()
            for p in self.context.get_all_embedding_providers():
                if (
                    getattr(p, "provider_id", "") or getattr(p, "name", "")
                ) == target_id:
                    try:
                        result = await p.get_embedding(text)
                    except Exception:
                        pass
                    break

        if result is not None:
            self._embed_cache[text] = result
            while len(self._embed_cache) > self._embed_cache_max:
                self._embed_cache.popitem(last=False)
        return result

    async def _local_embed(self, text):
        async with self._model_lock:
            if self._local_model is None:
                try:
                    from astrbot.core.utils.astrbot_path import (
                        get_astrbot_plugin_data_path,
                    )

                    model_name = self._C(
                        "local_embedding_model", "BAAI/bge-small-zh-v1.5"
                    )
                    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                    cache_dir = os.path.join(
                        get_astrbot_plugin_data_path(), self.name, "hf_cache"
                    )
                    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)

                    from sentence_transformers import SentenceTransformer

                    logger.debug(f"[ChatLearning] 加载本地模型: {model_name}")
                    try:
                        self._local_model = await asyncio.to_thread(
                            SentenceTransformer, model_name, local_files_only=True
                        )
                    except Exception:
                        logger.debug("[ChatLearning] 缓存未命中，从远程下载...")
                        self._local_model = await asyncio.to_thread(
                            SentenceTransformer, model_name
                        )
                    logger.debug("[ChatLearning] 本地模型加载完成")
                except ImportError:
                    logger.error("[ChatLearning] sentence-transformers 未安装")
                    self._embedding_ready = False
                    return None
                except Exception as e:
                    logger.error(f"[ChatLearning] 本地模型加载失败: {e}")
                    self._embedding_ready = False
                    return None
        try:
            vec = await asyncio.to_thread(
                self._local_model.encode, text, normalize_embeddings=True
            )
            return vec.tolist()
        except Exception as e:
            logger.error(f"[ChatLearning] 本地 embedding 失败: {e}")
            return None

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float] | None]:
        """批量生成向量，比逐个调用效率高得多。"""
        cleaned = [t for t in texts if t.strip()]
        if not cleaned:
            return [None] * len(texts)

        result = [None] * len(texts)
        clean_indices = [i for i, t in enumerate(texts) if t.strip()]

        if self._C("enable_local_embedding"):
            async with self._model_lock:
                if self._local_model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        from astrbot.core.utils.astrbot_path import (
                            get_astrbot_plugin_data_path,
                        )

                        model_name = self._C(
                            "local_embedding_model", "BAAI/bge-small-zh-v1.5"
                        )
                        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                        cache_dir = os.path.join(
                            get_astrbot_plugin_data_path(), self.name, "hf_cache"
                        )
                        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
                        logger.debug(f"[ChatLearning] 加载本地模型: {model_name}")
                        try:
                            self._local_model = await asyncio.to_thread(
                                SentenceTransformer, model_name, local_files_only=True
                            )
                        except Exception:
                            self._local_model = await asyncio.to_thread(
                                SentenceTransformer, model_name
                            )
                        logger.debug("[ChatLearning] 本地模型加载完成")
                    except ImportError:
                        logger.error("[ChatLearning] sentence-transformers 未安装")
                        self._embedding_ready = False
                        return result
                    except Exception as e:
                        logger.error(f"[ChatLearning] 本地模型加载失败: {e}")
                        self._embedding_ready = False
                        return result
                vecs = await asyncio.to_thread(
                    self._local_model.encode,
                    cleaned,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for idx, vec in zip(clean_indices, vecs.tolist()):
                    result[idx] = vec
                    self._embed_cache[texts[idx]] = vec
                    while len(self._embed_cache) > self._embed_cache_max:
                        self._embed_cache.popitem(last=False)
        else:
            for idx in clean_indices:
                result[idx] = await self._get_embedding(texts[idx])

        return result

    # ═══ 指令: /learn ═════════════════════════════════════════

    @filter.command_group("learn")
    def _grp_learn(self):
        pass

    @_grp_learn.command("on")
    async def cmd_learn_on(self, event):
        self._cfg["learning"] = True
        await self._save()
        yield event.plain_result("✅ 学习模式已开启")

    @_grp_learn.command("off")
    async def cmd_learn_off(self, event):
        self._cfg["learning"] = False
        await self._save()
        yield event.plain_result("⏸ 学习模式已关闭")

    @_grp_learn.command("status")
    async def cmd_learn_status(self, event):
        gid = str(event.get_group_id() or "?")
        count = await self.wordstock.count(gid)
        yield event.plain_result(
            f"📊 学习状态\n模式: {'开启' if self._C('learning') else '关闭'}\n本群词条: {count}\n间隔: {self._C('interval', 10)}s"
        )

    @_grp_learn.group("group")
    def _grp_learn_group(self):
        pass

    @_grp_learn_group.command("add")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_learn_group_add(self, event, group_id=""):
        gid = group_id or str(event.get_group_id())
        lst = list(self._C("learning_groups", []) or [])
        if gid not in lst:
            lst.append(gid)
            self._cfg["learning_groups"] = lst
            await self._save()
        yield event.plain_result(f"✅ 已添加学习群: {gid}")

    @_grp_learn_group.command("remove")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_learn_group_remove(self, event, group_id):
        lst = [g for g in (self._C("learning_groups", []) or []) if g != group_id]
        self._cfg["learning_groups"] = lst
        await self._save()
        yield event.plain_result(f"❌ 已移除学习群: {group_id}")

    @_grp_learn_group.command("list")
    async def cmd_learn_group_list(self, event):
        groups = self._C("learning_groups", []) or []
        yield event.plain_result(
            "📋 学习群:\n" + "\n".join(sorted(groups)) if groups else "📋 学习群: 全部"
        )

    # ═══ 指令: /reply ════════════════════════════════════════

    @filter.command_group("reply")
    def _grp_reply(self):
        pass

    @_grp_reply.command("on")
    async def cmd_reply_on(self, event):
        self._cfg["reply"] = True
        await self._save()
        yield event.plain_result("✅ 回复模式已开启")

    @_grp_reply.command("off")
    async def cmd_reply_off(self, event):
        self._cfg["reply"] = False
        await self._save()
        yield event.plain_result("⏸ 回复模式已关闭")

    @_grp_reply.command("status")
    async def cmd_reply_status(self, event):
        gid = str(event.get_group_id() or "?")
        count = await self.wordstock.count(gid)
        yield event.plain_result(
            f"💬 回复状态\n模式: {'开启' if self._C('reply') else '关闭'}\n概率: {self._C('reply_chance', 50)}%\nCD: {self._C('reply_cd', 30)}s\n阈值: {self._C('cos_match_threshold', 0.75)}\n词条: {count}"
        )

    @_grp_reply.group("group")
    def _grp_reply_group(self):
        pass

    @_grp_reply_group.command("add")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_reply_group_add(self, event, group_id=""):
        gid = group_id or str(event.get_group_id())
        lst = list(self._C("reply_groups", []) or [])
        if gid not in lst:
            lst.append(gid)
            self._cfg["reply_groups"] = lst
            await self._save()
        yield event.plain_result(f"✅ 已添加回复群: {gid}")

    @_grp_reply_group.command("remove")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_reply_group_remove(self, event, group_id):
        lst = [g for g in (self._C("reply_groups", []) or []) if g != group_id]
        self._cfg["reply_groups"] = lst
        await self._save()
        yield event.plain_result(f"❌ 已移除回复群: {group_id}")

    @_grp_reply_group.command("list")
    async def cmd_reply_group_list(self, event):
        groups = self._C("reply_groups", []) or []
        yield event.plain_result(
            "📋 回复群:\n" + "\n".join(sorted(groups)) if groups else "📋 回复群: 全部"
        )

    # ═══ 指令: /wordstock ═════════════════════════════════════

    @filter.command_group("wordstock")
    def _grp_wordstock(self):
        pass

    @_grp_wordstock.command("stats")
    async def cmd_wordstock_stats(self, event):
        gid = str(event.get_group_id())
        total = await self.wordstock.count(gid)
        yield event.plain_result(f"📚 词库统计 (群 {gid})\n词条总数: {total}")

    @_grp_wordstock.command("search")
    async def cmd_wordstock_search(self, event, keyword=""):
        if not keyword:
            yield event.plain_result("用法: /wordstock search <关键词>")
            return
        gid = str(event.get_group_id())
        rec = await self.wordstock.get_by_text(gid, keyword)
        if rec:
            yield event.plain_result(
                f"🔍 找到: '{keyword}'\n出现 {rec['freq']} 次 | {len(rec.get('answers', []))} 条答案"
            )
        else:
            vec = await self._get_embedding(keyword)
            if vec:
                results = await self.wordstock.search_similar(
                    gid, vec, threshold=0.5, top_k=5
                )
                if results:
                    yield event.plain_result(
                        "🔍 近似匹配:\n"
                        + "\n".join(
                            f"{r['question_text'][:40]} ({r['score']:.2f})"
                            for r in results
                        )
                    )
                    return
            yield event.plain_result("🔍 未找到匹配")

    @_grp_wordstock.command("delete")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_wordstock_delete(self, event, entry_id):
        try:
            await self.wordstock.delete(int(entry_id))
            yield event.plain_result(f"🗑 已删除词条 #{entry_id}")
        except ValueError:
            yield event.plain_result("❌ 无效的 ID")

    @_grp_wordstock.command("panel")
    async def cmd_wordstock_panel(self, event):
        import datetime

        gid = str(event.get_group_id())
        stats = await self.wordstock.get_stats(gid if gid else None)
        tmpl_path = os.path.join(os.path.dirname(__file__), "templates", "stats.html")
        try:
            with open(tmpl_path, encoding="utf-8") as f:
                tmpl = f.read()
            url = await self.html_render(
                tmpl,
                {
                    "stats": stats,
                    "group_label": gid if gid else "全局",
                    "now": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            )
            yield event.image_result(url)
        except Exception as e:
            logger.error(f"[ChatLearning] T2I 失败: {e}")
            yield event.plain_result(
                f"📊 词条: {stats['total']} | 答案: {stats['total_answers']} | 热度: {stats['avg_freq']}"
            )

    @_grp_wordstock.command("export")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_wordstock_export(self, event):
        import json

        from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

        gid = str(event.get_group_id())
        data = await self.wordstock.export_data(gid if gid else None)
        if not data:
            yield event.plain_result("📭 词库为空")
            return
        export_path = os.path.join(
            get_astrbot_plugin_data_path(),
            self.name,
            f"export_{gid}_{int(time.time())}.json",
        )
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        yield event.plain_result(f"📤 已导出 {len(data)} 条到\n{export_path}")

    @_grp_wordstock.command("cleanup")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_wordstock_cleanup(self, event):
        cleaned = await self.wordstock.cleanup_low_freq()
        yield event.plain_result(f"🧹 已清理 {cleaned} 条低频词条")

    @_grp_wordstock.command("rebuild_index")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_wordstock_rebuild(self, event):
        await self.wordstock.build_index()
        yield event.plain_result("🔧 索引重建完成")

    @_grp_wordstock.command("cleanup_versions")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_wordstock_cleanup_versions(self, event):
        freed = await self.wordstock.cleanup_versions()
        yield event.plain_result(f"🧹 已清理 {freed / 1024 / 1024:.0f} MB 历史版本")

    # ═══ 指令: /filter ═══════════════════════════════════════

    @filter.command_group("filter")
    @filter.permission_type(filter.PermissionType.ADMIN)
    def _grp_filter(self):
        pass

    @_grp_filter.group("sensitive")
    def _grp_filter_sensitive(self):
        pass

    @_grp_filter_sensitive.command("add")
    async def cmd_filter_sensitive_add(self, event, word):
        lst = list(self._C("sensitive_words", []) or [])
        if word not in lst:
            lst.append(word)
            self._cfg["sensitive_words"] = lst
            await self._save()
        yield event.plain_result(f"🚫 已添加敏感词: {word}")

    @_grp_filter_sensitive.command("remove")
    async def cmd_filter_sensitive_remove(self, event, word):
        lst = [w for w in (self._C("sensitive_words", []) or []) if w != word]
        self._cfg["sensitive_words"] = lst
        await self._save()
        yield event.plain_result(f"✅ 已移除敏感词: {word}")

    @_grp_filter_sensitive.command("list")
    async def cmd_filter_sensitive_list(self, event):
        words = self._C("sensitive_words", []) or []
        yield event.plain_result(
            "🚫 敏感词:\n" + "\n".join(words) if words else "🚫 敏感词: 无"
        )

    @_grp_filter.group("blocked")
    def _grp_filter_blocked(self):
        pass

    @_grp_filter_blocked.command("add")
    async def cmd_filter_blocked_add(self, event, user_id):
        lst = list(self._C("blocked_users", []) or [])
        uid = str(user_id)
        if uid not in lst:
            lst.append(uid)
            self._cfg["blocked_users"] = lst
            await self._save()
        yield event.plain_result(f"🚫 已添加黑名单: {user_id}")

    @_grp_filter_blocked.command("remove")
    async def cmd_filter_blocked_remove(self, event, user_id):
        lst = [u for u in (self._C("blocked_users", []) or []) if u != str(user_id)]
        self._cfg["blocked_users"] = lst
        await self._save()
        yield event.plain_result(f"✅ 已移除黑名单: {user_id}")

    @_grp_filter_blocked.command("list")
    async def cmd_filter_blocked_list(self, event):
        users = self._C("blocked_users", []) or []
        yield event.plain_result(
            "🚫 黑名单:\n" + "\n".join(users) if users else "🚫 黑名单: 无"
        )

    # ═══ Pages API ═══════════════════════════════════════════

    async def _api_stats(self):
        from quart import jsonify, request

        gid = request.args.get("group_id", "")
        return jsonify(await self.wordstock.get_stats(gid if gid else None))

    async def _api_search(self):
        from quart import jsonify, request

        q = (request.args.get("q", "") or "").strip()
        if not q:
            return jsonify([])
        results = []
        # 全表扫描 + Python 过滤（LanceDB 不支持 SQL LIKE）
        raw = await self.wordstock._table.query().to_arrow()
        for row in raw.to_pylist():
            if q in row.get("question_text", ""):
                results.append(
                    {
                        "id": row["id"],
                        "question_text": row["question_text"],
                        "freq": row["freq"],
                        "answer_count": len(row.get("answers", [])),
                    }
                )
                if len(results) >= 30:
                    break
        return jsonify(results)

    async def _api_entry_detail(self):
        from quart import jsonify, request

        eid = request.args.get("id", "")
        if not eid:
            return jsonify({"error": "missing id"}), 400
        rec = await self.wordstock.get_by_id(int(eid))
        if not rec:
            return jsonify({"error": "not found"}), 404
        ans = [
            {
                "answertext": a.get("answertext", ""),
                "same": a.get("same", 1),
                "added_at": a.get("added_at", 0),
            }
            for a in rec.get("answers", [])
        ]
        return jsonify(
            {
                "id": rec["id"],
                "question_text": rec["question_text"],
                "freq": rec["freq"],
                "created_at": rec["created_at"],
                "answers": ans,
            }
        )

    async def _api_delete_entry(self):
        from quart import jsonify, request

        data = await request.get_json()
        eid = (data or {}).get("id")
        if not eid:
            return jsonify({"error": "missing id"}), 400
        await self.wordstock.delete(int(eid))
        return jsonify({"ok": True})
