"""
astrbot_plugin_chatlearning — 主插件

从群聊中自动学习对话模式，用向量嵌入做语义匹配回复。
"""

import asyncio
import json
import os
import random
import time
from collections import OrderedDict

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Reply
from astrbot.api.star import Context, Star

from .filter.filter_engine import FilterEngine
from .learning.collector import Collector
from .reply.responder import Responder
from .storage.wordstock import WordStock

LANCE_DB_DIR = "lancedb"


class ChatLearningPlugin(Star):
    """群聊对话学习与智能回复——基于向量嵌入的语义对话引擎。"""

    name = "astrbot_plugin_chatlearning"

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
        self._recent_replies: dict[str, list[tuple[str, int, float]]] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._data_root: str | None = None
        self._maintenance_state_path: str | None = None
        self._llm_check_provider_id: str = ""
        self._write_count: int = 0
        self._reply_count: int = 0
        self._group_locks: dict[str, asyncio.Lock] = {}

    # ═══ 配置快捷读取 ═══════════════════════════════════════

    def _C(self, key, default=None):
        """读配置项。"""
        return self._cfg.get(key, default)

    async def _save(self):
        """持久化配置。"""
        save_async = getattr(self._cfg, "save_config_async", None)
        if callable(save_async):
            await save_async()
        else:
            self._cfg.save_config()

    def _is_group_learning(self, gid: str) -> bool:
        groups = self._C("learning_groups", []) or []
        return not groups or gid in {str(g) for g in groups}

    def _is_group_reply(self, gid: str) -> bool:
        groups = self._C("reply_groups", []) or []
        return not groups or gid in {str(g) for g in groups}

    @staticmethod
    def _resolve_group_id(event: AstrMessageEvent, group_id: str = "") -> str | None:
        gid = str(group_id).strip() if group_id else str(event.get_group_id() or "").strip()
        return gid or None

    # ═══ 生命周期 ═════════════════════════════════════════════

    async def initialize(self):
        logger.debug("[ChatLearning] 初始化中...")

        from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

        data_root = os.path.join(get_astrbot_plugin_data_path(), self.name)
        os.makedirs(data_root, exist_ok=True)
        self._data_root = data_root
        self._maintenance_state_path = os.path.join(data_root, "maintenance_state.json")

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
            elif self._find_embedding_provider(target_id) is not None:
                self._embedding_ready = True
                logger.debug(f"[ChatLearning] Embedding Provider: {target_id}")
            else:
                self._embedding_ready = False
                logger.warning(
                    f"[ChatLearning] Embedding Provider '{target_id}' 未找到"
                )

        logger.info("[ChatLearning] ✅ 初始化完成")

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

        # QQ 官方机器人账号发送的消息不参与学习/回复
        platform_name = event.get_platform_name()
        if platform_name.startswith("qq_official") and self.filter_engine.is_bot_author(
            getattr(event.message_obj, "raw_message", None)
        ):
            logger.debug("[ChatLearning] 跳过 QQ 官方机器人消息")
            return

        # 官方机器人的 Markdown 消息带有固定标记，个人号协议也能据此识别
        if self.filter_engine.is_official_bot_message(event.message_str):
            logger.debug("[ChatLearning] 跳过 QQ 官方机器人 Markdown 消息")
            return

        # 快速删除需放在媒体/@/wake 检查之前：回复 bot 时可能附带 At 且 WakingStage 会标记 is_at_or_wake_command
        handled, qd_reply = await self._handle_quick_delete(event, group_id)
        if handled:
            if qd_reply:
                yield qd_reply
            return

        # 视频、语音等非纯文本消息不参与学习/回复；表情和图片除外
        if self.filter_engine.has_unsupported_component(event.get_messages()):
            logger.debug("[ChatLearning] 跳过非纯文本消息")
            return

        # 含 @用户 的消息不参与学习/回复
        if self.filter_engine.has_at_mention(event.get_messages(), event.message_str):
            logger.debug("[ChatLearning] 跳过包含 @ 用户的消息")
            return

        user_id = str(event.get_sender_id())
        raw_ts = getattr(event.message_obj, "timestamp", None) if event.message_obj else None
        timestamp = float(raw_ts) if raw_ts else time.time()
        plain_text = self.filter_engine.clean_at_mentions(event.message_str.strip())
        if not plain_text:
            return
        if self._is_noise(plain_text):
            return

        # 指令/@唤醒消息跳过学习/回复，留给 AstrBot LLM
        if event.is_at_or_wake_command:
            return

        # 学习失败不应影响后续回复
        if self._C("learning") and self._is_group_learning(group_id):
            try:
                await self._handle_learning(group_id, user_id, plain_text, timestamp)
            except Exception as e:
                logger.warning(f"[ChatLearning] 学习失败: {e}")

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

        lock = self._group_locks.setdefault(group_id, asyncio.Lock())
        async with lock:
            qa = self.collector.feed(group_id, text, text, timestamp)
            if qa is None:
                return

            try:
                q_exist = await self.wordstock.get_by_text(qa.group_id, qa.question_text)
            except Exception as e:
                logger.warning(f"[ChatLearning] 查询词库失败: {e}")
                return

            if q_exist:
                await self.wordstock.add_answer(
                    q_exist["id"], qa.answer_text, qa.answer_raw
                )
            else:
                vec = await self._get_embedding(qa.question_text)
                if vec is None:
                    return
                rid = await self.wordstock.add_question(
                    qa.group_id,
                    qa.question_text,
                    qa.question_raw,
                    vec,
                    answer_text=qa.answer_text,
                    answer_raw=qa.answer_raw,
                )
                if not rid:
                    return

            self._write_count += 1
            if self._write_count % 100 == 0:
                await self.wordstock.ensure_index()
            if self._write_count % 10 == 0:
                logger.info(f"[ChatLearning] 已学习 {self._write_count} 条")

    # ═══ 回复 ═════════════════════════════════════════════════

    async def _handle_reply(self, event, group_id, text):
        if not self.responder:
            return

        last_reply = self._reply_cd.get(group_id, 0)
        if not self.responder.should_attempt(last_reply):
            return

        vec = await self._get_embedding(text)
        if vec is None:
            return

        threshold = self._C("cos_match_threshold", 0.75)
        if self._C("llm_semantic_check") and self._llm_check_provider_id:
            search_threshold = min(threshold * 0.8, self._C("llm_semantic_threshold", 0.5))
        else:
            search_threshold = threshold

        async def _search(gid, qvec):
            return await self.wordstock.search_similar(
                gid, qvec, threshold=search_threshold, top_k=20
            )

        cross_fn = None
        if self._C("cross_group_search"):
            cross_threshold = min(max(threshold - 0.05, 0.6), search_threshold)

            async def _cross_search(exclude_gid, qvec):
                return await self.wordstock.search_cross_group(
                    exclude_gid, qvec, threshold=cross_threshold, top_k=20
                )

            cross_fn = _cross_search

        result = await self.responder.try_reply(
            _search,
            group_id,
            vec,
            query_text=text,
            last_reply_time=last_reply,
            cross_search_fn=cross_fn,
            gate=False,
        )
        if result is None:
            return

        text_out = self._apply_placeholders(result.answer_text, event)
        q_id = result.question_id
        if not q_id and not result.cross_group:
            q_exist = await self.wordstock.get_by_text(group_id, result.question_text)
            q_id = q_exist["id"] if q_exist else 0

        recent = self._recent_replies.get(group_id, [])
        recent.append((text_out, q_id, time.time()))
        if len(recent) > 30:
            recent = recent[-30:]
        self._recent_replies[group_id] = recent

        self._reply_cd[group_id] = time.time()
        await asyncio.sleep(random.uniform(0.5, 2.0))
        self._reply_count += 1
        logger.info(
            f"[ChatLearning] 回复 #{self._reply_count}: "
            f"Q={text[:20]} → A={text_out[:20]}"
        )
        yield event.plain_result(text_out)

    # ═══ 快速删除 ═════════════════════════════════════════════

    @staticmethod
    def _extract_reply_segment(components: list | None):
        for seg in components or []:
            if isinstance(seg, Reply) or str(getattr(seg, "type", "")) == "Reply":
                return seg
        return None

    def _extract_command_text(self, event: AstrMessageEvent) -> str:
        plain_parts = [
            getattr(seg, "text", "")
            for seg in (event.get_messages() or [])
            if isinstance(seg, Plain)
        ]
        if plain_parts:
            return "".join(plain_parts).strip()
        return self.filter_engine.clean_at_mentions(event.message_str.strip()).strip()

    @staticmethod
    def _extract_reply_text(reply_seg) -> str:
        text = (
            getattr(reply_seg, "message_str", None)
            or getattr(reply_seg, "text", None)
            or ""
        )
        if not text and getattr(reply_seg, "chain", None):
            text = "".join(
                getattr(seg, "text", "")
                for seg in reply_seg.chain
                if isinstance(seg, Plain) or str(getattr(seg, "type", "")) == "Plain"
            )
        return str(text).strip()

    async def _handle_quick_delete(self, event: AstrMessageEvent, group_id: str):
        reply_seg = self._extract_reply_segment(event.get_messages())
        if reply_seg is None:
            return False, None

        t = self._extract_command_text(event).lower()
        if t not in ("!d", "!delete", "！d", "！delete"):
            return False, None

        reply_sender = getattr(reply_seg, "sender_id", None) or getattr(
            reply_seg, "qq", None
        )
        self_id = event.get_self_id()
        if reply_sender and self_id and str(reply_sender) != str(self_id):
            return False, None

        event.stop_event()

        recent = self._recent_replies.get(group_id, [])
        if not recent:
            return True, event.plain_result("没有可删除的最近回复")

        reply_text = self._extract_reply_text(reply_seg)
        match_idx = None
        if reply_text:
            for idx in range(len(recent) - 1, -1, -1):
                if recent[idx][0].strip() == reply_text:
                    match_idx = idx
                    break
            if match_idx is None:
                return True, event.plain_result("未找到对应的最近回复记录")
        else:
            match_idx = len(recent) - 1

        _, ws_id, _ = recent.pop(match_idx)
        self._recent_replies[group_id] = recent
        if ws_id > 0:
            deleted = await self.wordstock.delete(ws_id)
            if deleted:
                return True, event.plain_result(f"🗑 已从词库删除 #{ws_id}")
        return True, event.plain_result("🗑 已移除最近回复")

    # ═══ 后台维护 ═════════════════════════════════════════════

    def _save_maintenance_state(
        self, last_cleanup: float, last_version_cleanup: float
    ) -> None:
        """原子写入维护时间戳，避免重启后重复或永久跳过清理。"""
        if not self._maintenance_state_path:
            return
        tmp_path = self._maintenance_state_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_cleanup": last_cleanup,
                        "last_version_cleanup": last_version_cleanup,
                    },
                    f,
                )
            os.replace(tmp_path, self._maintenance_state_path)
        except Exception as e:
            logger.warning(f"[ChatLearning] 保存维护状态失败: {e}")

    async def _periodic_maintenance(self):
        now = time.time()
        last_cleanup = now
        last_version_cleanup = 0.0
        if self._maintenance_state_path:
            try:
                with open(self._maintenance_state_path, encoding="utf-8") as f:
                    state = json.load(f)
                last_cleanup = float(state.get("last_cleanup", now))
                last_version_cleanup = float(state.get("last_version_cleanup", 0.0))
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"[ChatLearning] 读取维护状态失败: {e}")

        version_retry_at = 0.0
        last_states = {"learn": None, "reply": None}
        while True:
            await asyncio.sleep(60)
            try:
                now = time.time()
                if self._C("auto_schedule_enable"):
                    await self._check_schedule(last_states)
                if now - last_cleanup > 24 * 3600:
                    days = int(self._C("auto_cleanup_days", 30))
                    if days > 0:
                        cleaned = await self.wordstock.cleanup_low_freq(
                            days=days, min_freq=2
                        )
                        if cleaned:
                            logger.debug(f"[ChatLearning] 自动清理: {cleaned} 条")
                    await self.wordstock.build_index()
                    last_cleanup = now
                    self._save_maintenance_state(last_cleanup, last_version_cleanup)
                version_interval = int(self._C("version_cleanup_interval_hours", 6))
                if (
                    version_interval > 0
                    and now - last_version_cleanup > version_interval * 3600
                    and now >= version_retry_at
                ):
                    retention = int(self._C("version_retention_hours", 2))
                    freed = await self.wordstock.cleanup_versions(
                        retention_hours=retention
                    )
                    if freed is not None:
                        if freed:
                            logger.info(
                                f"[ChatLearning] LanceDB 版本清理: {freed / 1024 / 1024:.0f} MB"
                            )
                        last_version_cleanup = now
                        version_retry_at = 0.0
                        self._save_maintenance_state(last_cleanup, last_version_cleanup)
                    else:
                        # Back off briefly instead of retrying a broken table every minute.
                        version_retry_at = now + 300
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

    @staticmethod
    def _provider_matches_id(provider, target_id: str) -> bool:
        """兼容 AstrBot Provider.meta().id 以及测试桩属性。"""
        meta_fn = getattr(provider, "meta", None)
        if callable(meta_fn):
            try:
                if getattr(meta_fn(), "id", "") == target_id:
                    return True
            except Exception:
                pass
        cfg = getattr(provider, "provider_config", None)
        if isinstance(cfg, dict) and cfg.get("id") == target_id:
            return True
        return (
            getattr(provider, "provider_id", "") or getattr(provider, "name", "")
        ) == target_id

    def _find_embedding_provider(self, target_id: str):
        if not target_id:
            return None
        prov = self.context.get_provider_by_id(target_id)
        if prov is not None and callable(getattr(prov, "get_embedding", None)):
            return prov
        for p in self.context.get_all_embedding_providers() or []:
            if self._provider_matches_id(p, target_id):
                return p
        return None

    def _cache_embed_put(self, text: str, vec: list[float]) -> None:
        self._embed_cache[text] = vec
        self._embed_cache.move_to_end(text)
        while len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.popitem(last=False)

    async def _ensure_local_model(self):
        if self._local_model is not None:
            return self._local_model
        async with self._model_lock:
            if self._local_model is not None:
                return self._local_model
            try:
                from astrbot.core.utils.astrbot_path import (
                    get_astrbot_plugin_data_path,
                )
                from sentence_transformers import SentenceTransformer

                model_name = self._C(
                    "local_embedding_model", "BAAI/bge-small-zh-v1.5"
                )
                os.environ["HF_ENDPOINT"] = self._C(
                    "hf_endpoint", "https://huggingface.co"
                )
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
        return self._local_model

    async def _get_embedding(self, text: str) -> list[float] | None:
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
            prov = self._find_embedding_provider(target_id)
            if prov is not None:
                try:
                    result = await prov.get_embedding(text)
                except Exception as e:
                    logger.warning(f"[ChatLearning] 在线 embedding 失败: {e}")

        if result is not None:
            self._cache_embed_put(text, result)
        return result

    async def _local_embed(self, text: str) -> list[float] | None:
        model = await self._ensure_local_model()
        if model is None:
            return None
        try:
            vec = await asyncio.to_thread(
                model.encode,
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vec.tolist()
        except Exception as e:
            logger.error(f"[ChatLearning] 本地 embedding 失败: {e}")
            return None

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float] | None]:
        """批量生成向量，优先命中缓存并利用批量接口。"""
        result: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_map: dict[str, list[int]] = {}

        for idx, text in enumerate(texts):
            if not text.strip():
                continue
            if text in self._embed_cache:
                self._embed_cache.move_to_end(text)
                result[idx] = self._embed_cache[text]
                continue
            if text not in missing_map:
                missing_map[text] = []
                missing_texts.append(text)
            missing_map[text].append(idx)

        if not missing_texts or not self._embedding_ready:
            return result

        vecs: list[list[float] | None] = []
        if self._C("enable_local_embedding"):
            model = await self._ensure_local_model()
            if model is None:
                return result
            try:
                encoded = await asyncio.to_thread(
                    model.encode,
                    missing_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                vecs = encoded.tolist()
            except Exception as e:
                logger.error(f"[ChatLearning] 批量本地 embedding 失败: {e}")
                return result
        else:
            target_id = (self._C("embedding_provider_id", "") or "").strip()
            prov = self._find_embedding_provider(target_id)
            if prov is None:
                return result
            batch_fn = getattr(prov, "get_embeddings", None)
            if callable(batch_fn):
                try:
                    vecs = await batch_fn(missing_texts)
                except Exception as e:
                    logger.warning(f"[ChatLearning] 在线批量 embedding 失败，回退单条: {e}")
                    vecs = []
            if len(vecs) != len(missing_texts):
                vecs = []
                for t in missing_texts:
                    try:
                        vecs.append(await prov.get_embedding(t))
                    except Exception as e:
                        logger.warning(f"[ChatLearning] 在线 embedding 失败: {e}")
                        vecs.append(None)

        for t, vec in zip(missing_texts, vecs):
            if vec is None:
                continue
            self._cache_embed_put(t, vec)
            for idx in missing_map[t]:
                result[idx] = vec

        return result

    # ═══ 指令: /learn ═════════════════════════════════════════

    @filter.command_group("learn")
    def _grp_learn(self):
        pass

    @_grp_learn.command("on")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_learn_on(self, event):
        self._cfg["learning"] = True
        await self._save()
        yield event.plain_result("✅ 学习模式已开启")

    @_grp_learn.command("off")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_learn_off(self, event):
        self._cfg["learning"] = False
        await self._save()
        yield event.plain_result("⏸ 学习模式已关闭")

    @_grp_learn.command("status")
    async def cmd_learn_status(self, event):
        gid = self._resolve_group_id(event)
        count = await self.wordstock.count(gid)
        yield event.plain_result(
            f"📊 学习状态\n模式: {'开启' if self._C('learning') else '关闭'}\n{'本群' if gid else '全部'}词条: {count}\n间隔: {self._C('interval', 10)}s"
        )

    @_grp_learn.group("group")
    def _grp_learn_group(self):
        pass

    @_grp_learn_group.command("add")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_learn_group_add(self, event, group_id=""):
        gid = self._resolve_group_id(event, group_id)
        if not gid:
            yield event.plain_result("❌ 请在群聊中使用或指定群号")
            return
        lst = [str(g) for g in (self._C("learning_groups", []) or [])]
        if gid not in lst:
            lst.append(gid)
            self._cfg["learning_groups"] = lst
            await self._save()
        yield event.plain_result(f"✅ 已添加学习群: {gid}")

    @_grp_learn_group.command("remove")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_learn_group_remove(self, event, group_id):
        target = str(group_id)
        lst = [str(g) for g in (self._C("learning_groups", []) or []) if str(g) != target]
        self._cfg["learning_groups"] = lst
        await self._save()
        yield event.plain_result(f"❌ 已移除学习群: {target}")

    @_grp_learn_group.command("list")
    async def cmd_learn_group_list(self, event):
        groups = [str(g) for g in (self._C("learning_groups", []) or [])]
        yield event.plain_result(
            "📋 学习群:\n" + "\n".join(sorted(groups)) if groups else "📋 学习群: 全部"
        )

    # ═══ 指令: /reply ════════════════════════════════════════

    @filter.command_group("reply")
    def _grp_reply(self):
        pass

    @_grp_reply.command("on")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_reply_on(self, event):
        self._cfg["reply"] = True
        await self._save()
        yield event.plain_result("✅ 回复模式已开启")

    @_grp_reply.command("off")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_reply_off(self, event):
        self._cfg["reply"] = False
        await self._save()
        yield event.plain_result("⏸ 回复模式已关闭")

    @_grp_reply.command("status")
    async def cmd_reply_status(self, event):
        gid = self._resolve_group_id(event)
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
        gid = self._resolve_group_id(event, group_id)
        if not gid:
            yield event.plain_result("❌ 请在群聊中使用或指定群号")
            return
        lst = [str(g) for g in (self._C("reply_groups", []) or [])]
        if gid not in lst:
            lst.append(gid)
            self._cfg["reply_groups"] = lst
            await self._save()
        yield event.plain_result(f"✅ 已添加回复群: {gid}")

    @_grp_reply_group.command("remove")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_reply_group_remove(self, event, group_id):
        target = str(group_id)
        lst = [str(g) for g in (self._C("reply_groups", []) or []) if str(g) != target]
        self._cfg["reply_groups"] = lst
        await self._save()
        yield event.plain_result(f"❌ 已移除回复群: {target}")

    @_grp_reply_group.command("list")
    async def cmd_reply_group_list(self, event):
        groups = [str(g) for g in (self._C("reply_groups", []) or [])]
        yield event.plain_result(
            "📋 回复群:\n" + "\n".join(sorted(groups)) if groups else "📋 回复群: 全部"
        )

    # ═══ 指令: /wordstock ═════════════════════════════════════

    @filter.command_group("wordstock")
    def _grp_wordstock(self):
        pass

    @_grp_wordstock.command("stats")
    async def cmd_wordstock_stats(self, event):
        gid = self._resolve_group_id(event)
        total = await self.wordstock.count(gid)
        scope = f"群 {gid}" if gid else "全部"
        yield event.plain_result(f"📚 词库统计 ({scope})\n词条总数: {total}")

    @_grp_wordstock.command("search")
    async def cmd_wordstock_search(self, event, keyword=""):
        if not keyword:
            yield event.plain_result("用法: /wordstock search <关键词>")
            return
        gid = self._resolve_group_id(event)
        if not gid:
            yield event.plain_result("❌ 请在群聊中使用该指令")
            return
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
            deleted = await self.wordstock.delete(int(entry_id))
            if deleted:
                yield event.plain_result(f"🗑 已删除词条 #{entry_id}")
            else:
                yield event.plain_result(f"未找到词条 #{entry_id}")
        except ValueError:
            yield event.plain_result("❌ 无效的 ID")

    @_grp_wordstock.command("export")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_wordstock_export(self, event):
        from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

        gid = self._resolve_group_id(event)
        data = await self.wordstock.export_data(gid)
        if not data:
            yield event.plain_result("📭 词库为空")
            return
        export_path = os.path.join(
            get_astrbot_plugin_data_path(),
            self.name,
            f"export_{gid or 'all'}_{int(time.time())}.json",
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
        if freed is None:
            yield event.plain_result("❌ 版本清理失败，详见日志")
            return
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
        if self.filter_engine:
            self.filter_engine.sensitive_words = set(lst)
        yield event.plain_result(f"🚫 已添加敏感词: {word}")

    @_grp_filter_sensitive.command("remove")
    async def cmd_filter_sensitive_remove(self, event, word):
        lst = [w for w in (self._C("sensitive_words", []) or []) if w != word]
        self._cfg["sensitive_words"] = lst
        await self._save()
        if self.filter_engine:
            self.filter_engine.sensitive_words = set(lst)
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
        lst = [str(u) for u in (self._C("blocked_users", []) or [])]
        uid = str(user_id)
        if uid not in lst:
            lst.append(uid)
            self._cfg["blocked_users"] = lst
            await self._save()
        if self.filter_engine:
            self.filter_engine.blocked_users = set(lst)
        yield event.plain_result(f"🚫 已添加黑名单: {user_id}")

    @_grp_filter_blocked.command("remove")
    async def cmd_filter_blocked_remove(self, event, user_id):
        uid = str(user_id)
        lst = [str(u) for u in (self._C("blocked_users", []) or []) if str(u) != uid]
        self._cfg["blocked_users"] = lst
        await self._save()
        if self.filter_engine:
            self.filter_engine.blocked_users = set(lst)
        yield event.plain_result(f"✅ 已移除黑名单: {user_id}")

    @_grp_filter_blocked.command("list")
    async def cmd_filter_blocked_list(self, event):
        users = [str(u) for u in (self._C("blocked_users", []) or [])]
        yield event.plain_result(
            "🚫 黑名单:\n" + "\n".join(users) if users else "🚫 黑名单: 无"
        )
