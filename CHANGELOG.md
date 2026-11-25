# Changelog

All notable changes to `astrbot_plugin_chatlearning` will be documented in this file.

## [0.4.4] - 2026-11-25

### Fixed
- **在线 Embedding Provider 查找失败**：通过 `context.get_provider_by_id()` 与 `provider.meta().id`（兼容 `provider_config["id"]`）解析提供商，修复原先读取不存在的 `.provider_id` / `.name` 属性导致在线向量模式不可用的问题。
- **`!d` 快速删除不可达与误删问题**：将 `_handle_quick_delete` 提前至媒体/`@`/唤醒检查之前，使其在引用回复 bot 消息时可正常触发并调用 `event.stop_event()`；支持按所回复的 bot 消息文本精确匹配最近回复记录（无需管理员权限），未命中时不误删其他记录。
- **LLM 灰色区间二次判断与跨群标识**：修复 `Responder._filter_and_pick` 在同群检索时预先过滤掉灰色区间候选导致 `llm_semantic_check` 永不执行的问题（并限制单次最多校验前 5 条高分候选）；修复 `ReplyResult.cross_group` 在同群回复时被误标为 `True` 的问题；在选取问题前过滤无答案记录。
- **学习链路数据一致性**：停止向词库写入末尾无答案的孤立记录，修复 `freq` 重复累加、`Q == A` 重复建行及同群并发写入竞态。
- **空库与命令边界异常**：修复空表时 `/learn status`、`/reply status`、`/wordstock stats`、`/wordstock export` 及 `delete()` / `touch()` 崩溃的问题；为 `/learn on|off` 与 `/reply on|off` 补充管理员权限校验；修复私聊触发群指令时群号变为 `"None"` 的问题。
- **动态过滤生效延迟**：`/filter sensitive` 与 `/filter blocked` 增删后立即同步到运行时 `FilterEngine`，无需重载插件。
- **LanceDB 存储与迁移脚本**：补全 SQL 字符串单引号转义、单调递增记录 ID、空向量拦截、`touch()` 缓存失效，以及 `migrate.py` 固定维度向量 schema 与 `--batch-size` 参数支持。

### Changed
- 移除已废弃的 `@register` 装饰器，统一由 `metadata.yaml` 提供插件元数据。
- 使用 `save_config_async()` 异步持久化配置；同步更新 `README.md`、`AGENTS.md` 与 `_conf_schema.json`。
- 补充 `Collector`、`Responder`、`main.py` 与 `WordStock` 边界用例单元测试，并新增 `tests/conftest.py` 支持直接运行测试。

### Performance
- 回复前先检查冷却时间（`reply_cd`）与触发概率（`reply_chance`），未通过时不调用向量模型。
- `WordStock.count()` 改用 `count_rows()`，`cleanup_low_freq()` 改用列裁剪与单条谓词 `delete()`，减少内存占用与 LanceDB 历史版本膨胀；`ensure_index()` 启动时检查 `list_indices()` 并保证 IVF-PQ 子向量数整除向量维度。
- 统一本地向量模型加载并仅在初始化阶段加锁；`Collector` 增加过期窗口清理防止长期运行状态累积。

### Removed
- 移除未完整接入或已废弃的 `persona_fusion`（人格融合）配置与逻辑、`FilterEngine.reply_filter_words` / `should_reply()`、`WordStock.import_data()` 以及 `README.md` 中的无效配置项说明。

## [0.4.3] - 2026-10-26

### Added
- 跳过包含视频、语音、文件、转发、JSON 卡片等非纯文本媒体的消息（保留纯文本、QQ 表情、图片与引用回复）。
- 跳过 QQ 官方机器人消息（`author.bot` 标记与官方 Markdown 特征）及包含 `@用户` / `@全体成员` 的消息。

## [0.4.1] - 2026-10-26

### Fixed
- 加固 LanceDB 并发写入与历史版本清理：引入文件写锁（`.lance_writer.lock`）、`merge_insert` 原子更新、维护状态持久化与失败退避重试。
