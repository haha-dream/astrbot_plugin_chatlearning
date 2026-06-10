# ChatLearning — AstrBot 群聊对话学习插件

从群聊中自动学习对话模式，用向量嵌入做语义匹配，智能回复。

> 灵感来源于 [ChatLearning](https://github.com/Nana-Miko/ChatLearning)（Mirai 词库插件），基于 AstrBot 框架重写，使用 LanceDB 向量检索替代传统分词+FTS。

## 核心机制

```
群聊消息 ──时间间隔──→ 构建 Q&A 词库 ──Embedding──→ LanceDB
                                           ↓
新消息 ──Embedding──→ 向量搜索最相似问题 ──→ 加权随机选答案 ──→ 发送
```

1. **学习**：监听群消息，按时间间隔自动构建 Q&A 链写入词库
2. **回复**：新消息 → 向量语义搜索 → 高相似度匹配 → 加权随机回复

## 环境要求

- AstrBot ≥ 4.0.0
- Python ≥ 3.10
- 若不启用本地模型，则**必须**在 WebUI 配置一个 Embedding Provider

## 快速开始

### 1. 安装

```bash
# 进入 AstrBot 插件目录
cd AstrBot/data/plugins
git clone https://github.com/user/astrbot_plugin_chatlearning
# 在 AstrBot WebUI → 插件管理 → 重载插件
```

### 2. 配置

> ⭐ **推荐启用本地向量模型**（内置 sentence-transformers + bge-small-zh-v1.5，约 96MB），零费用零延迟。
>
> 1. 插件配置中勾选「启用本地向量模型」
> 2. 重载插件（首次加载模型约 2 秒）
>
> 若不使用本地模型，则需要在「向量模型」中配置在线 Embedding Provider。

在 AstrBot WebUI → 插件管理 → ChatLearning → 管理，配置：

| 必填项                  | 说明                                                     |
| ----------------------- | -------------------------------------------------------- |
| `embedding_provider_id` | 向量模型提供商 ID（与 WebUI 提供商管理中显示的名称一致） |

| 常用项                | 默认值 | 说明                                               |
| --------------------- | ------ | -------------------------------------------------- |
| `interval`            | 10     | 词库链间隔（秒），两条消息间隔超过此时间视为新对话 |
| `reply_chance`        | 50     | 回复概率（%）                                      |
| `reply_cd`            | 30     | 回复冷却时间（秒）                                 |
| `cos_match_threshold` | 0.75   | 向量相似度阈值                                     |

### 3. 使用

在群内发送指令（需要有权限）：

```bash
/learn on                    # 开启学习模式
/learn group add             # 添加当前群到学习白名单
/reply on                    # 开启回复模式
/reply group add             # 添加当前群到回复白名单
```

## 指令参考

### 学习控制

| 指令                         | 说明                               |
| ---------------------------- | ---------------------------------- |
| `/learn on`                  | 开启学习模式                       |
| `/learn off`                 | 关闭学习模式                       |
| `/learn status`              | 查看学习状态                       |
| `/learn group add [群号]`    | 添加学习群（管理员，不填则当前群） |
| `/learn group remove <群号>` | 移除学习群（管理员）               |
| `/learn group list`          | 列出学习群                         |

### 回复控制

| 指令                         | 说明                 |
| ---------------------------- | -------------------- |
| `/reply on`                  | 开启回复模式         |
| `/reply off`                 | 关闭回复模式         |
| `/reply status`              | 查看回复状态         |
| `/reply group add [群号]`    | 添加回复群（管理员） |
| `/reply group remove <群号>` | 移除回复群（管理员） |
| `/reply group list`          | 列出回复群           |

### 词库管理

| 指令                         | 说明                       |
| ---------------------------- | -------------------------- |
| `/wordstock stats`           | 查看本群词库统计           |
| `/wordstock panel`           | T2I 统计面板（渲染为图片） |
| `/wordstock search <关键词>` | 搜索词库（精确+近似）      |
| `/wordstock delete <id>`     | 删除词条（管理员）         |
| `/wordstock export`          | 导出词库为 JSON（管理员）  |
| `/wordstock cleanup`         | 清理低频词条（管理员）     |
| `/wordstock rebuild_index`   | 重建向量索引（管理员）     |

### 过滤管理（管理员）

| 指令                                       | 说明       |
| ------------------------------------------ | ---------- |
| `/filter sensitive add/remove/list <词>`   | 敏感词管理 |
| `/filter blocked add/remove/list <用户ID>` | 黑名单管理 |

### 其他

| 操作               | 说明             |
| ------------------ | ---------------- |
| 回复 bot 消息 `!d` | 快速删除对应词条 |

## 配置项完整列表

在 WebUI 插件配置面板中可调整：

| 配置项                     | 类型   | 默认  | 说明                                |
| -------------------------- | ------ | ----- | ----------------------------------- |
| `learning`                 | bool   | false | 学习模式开关                        |
| `reply`                    | bool   | false | 回复模式开关                        |
| `interval`                 | int    | 10    | 词库链间隔（秒）                    |
| `learning_groups`          | list   | []    | 学习群白名单（空=全部）             |
| `reply_groups`             | list   | []    | 回复群白名单（空=全部）             |
| `reply_chance`             | int    | 50    | 回复概率（%）                       |
| `reply_cd`                 | int    | 30    | 回复冷却（秒）                      |
| `cos_match_threshold`      | float  | 0.75  | 相似度阈值                          |
| `at_reply_only`            | bool   | false | 仅@回复                             |
| `bot_names`                | list   | []    | Bot 昵称（`{me}` 占位符）           |
| `sensitive_words`          | list   | []    | 敏感词                              |
| `blocked_users`            | list   | []    | 黑名单                              |
| `embedding_provider_id`    | string | ""    | **必填** 向量模型 ID                |
| `llm_semantic_check`       | bool   | false | LLM 语义二次判断                    |
| `llm_provider_id`          | string | ""    | 语义判断用 LLM ID（建议低成本模型） |
| `llm_semantic_threshold`   | float  | 0.5   | LLM 灰色区间下限                    |
| `cross_group_search`       | bool   | false | 跨群词库搜索                        |
| `auto_cleanup_days`        | int    | 30    | 自动清理天数（0=禁用）              |
| `max_answers_per_question` | int    | 20    | 每个问题最大答案数                  |
| `persona_fusion_mode`      | string | off   | 人格融合：off/blend/full            |
| `persona_fusion_provider`  | string | ""    | 融合用 LLM（下拉选择）              |
| `persona_fusion_persona`   | string | ""    | 融合用人格（下拉选择）              |
| `auto_schedule_enable`     | bool   | false | 定时调度开关                        |
| `auto_schedule_learn_on`   | string | ""    | 自动开启学习（HH:MM）               |
| `auto_schedule_learn_off`  | string | ""    | 自动关闭学习（HH:MM）               |
| `auto_schedule_reply_on`   | string | ""    | 自动开启回复（HH:MM）               |
| `auto_schedule_reply_off`  | string | ""    | 自动关闭回复（HH:MM）               |

## 技术架构

```
astrbot_plugin_chatlearning/
├── main.py              # Star 插件入口 + 事件 + 26 指令
├── learning/
│   └── collector.py     # 时间窗口 Q&A 链采集
├── storage/
│   └── wordstock.py     # LanceDB Arrow list<struct> 词库
├── reply/
│   └── responder.py     # 向量匹配 + LLM 语义检查 + 跨群
├── filter/
│   └── filter_engine.py # 敏感词/黑名单过滤
├── config/
│   └── config_mgr.py    # AstrBotConfig 访问层
├── _conf_schema.json    # WebUI 配置
└── requirements.txt     # lancedb / pyarrow / numpy
```

- **存储**：LanceDB（Arrow 原生 `list<struct>` 嵌套类型，零 JSON 开销）
- **匹配**：Embedding Provider → 向量余弦相似度，LLM 可选二次确认
- **索引**：IVF-PQ 自动构建（5000+ 条时每 100 次写入触发一次检查）
- **优化**：Embedding LRU 缓存（3000 条，命中率 30-50%）+ 短消息噪音过滤
