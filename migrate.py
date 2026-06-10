#!/usr/bin/env python3
"""ChatLearning 旧版 SQLite → 新版 LanceDB 数据迁移

用法:
  python migrate.py \
    --old-wordstock /path/to/old/WordStock \
    --new-lancedb /path/to/new/plugin_data/astrbot_plugin_chatlearning/lancedb \
    --embedding-provider siliconflow \
    [--hf-mirror https://hf-mirror.com]
"""

import argparse
import ast
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import lancedb
import pyarrow as pa

# ── 旧版 SQLite schema ─────────────────────────────────────

OLD_TABLE = "wordstock"
OLD_COLS = "id, question, freq, time, answer_json"


def iter_old_wordstock(db_path: str):
    """遍历旧版 SQLite 词库文件，yield (id, question_text, question_raw, freq, answers)。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT {OLD_COLS} FROM {OLD_TABLE}").fetchall()
    conn.close()

    for row in rows:
        # 从旧版消息链 JSON 提取纯文本
        try:
            chain = ast.literal_eval(row["question"])
            texts = []
            for seg in chain:
                if isinstance(seg, dict) and seg.get("type") in ("Plain", "text"):
                    texts.append(seg.get("text", ""))
            question_text = " ".join(texts).strip()
            if not question_text:
                continue
        except Exception:
            question_text = str(row["question"]).strip()
            if not question_text:
                continue

        question_raw = str(row["question"])
        freq = int(row["freq"] or 0) or 1

        # 解析答案列表
        answers = []
        try:
            raw_answers = json.loads(row["answer_json"] or "[]")
        except Exception:
            raw_answers = []

        for a in raw_answers:
            answer_raw_text = a.get("answertext", "")
            # 答案文本也可能是消息链 JSON
            try:
                ans_chain = ast.literal_eval(answer_raw_text)
                ans_texts = []
                for seg in ans_chain:
                    if isinstance(seg, dict) and seg.get("type") in ("Plain", "text"):
                        ans_texts.append(seg.get("text", ""))
                ans_plain = " ".join(ans_texts).strip()
            except Exception:
                ans_plain = str(answer_raw_text).strip()

            if ans_plain:
                answers.append({
                    "answertext": ans_plain,
                    "answer_raw": str(answer_raw_text),
                    "added_at": time.time(),
                    "same": max(1, int(a.get("same", 1))),
                })

        yield row["id"], question_text, question_raw, freq, answers


# ── 新版 LanceDB schema ─────────────────────────────────────

_answer_struct = pa.struct([
    pa.field("answertext", pa.string()),
    pa.field("answer_raw", pa.string()),
    pa.field("added_at", pa.float64()),
    pa.field("same", pa.int32()),
])

_new_schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("group_id", pa.string()),
    pa.field("question_text", pa.string()),
    pa.field("question_raw", pa.string()),
    pa.field("vec", pa.list_(pa.float32(), -1)),
    pa.field("answers", pa.list_(_answer_struct)),
    pa.field("freq", pa.int32()),
    pa.field("created_at", pa.float64()),
    pa.field("updated_at", pa.float64()),
])


# ── 纯文本提取（从旧版消息链） ─────────────────────────────

def extract_plain_from_chain(raw) -> str:
    """从旧版消息链 JSON 字符串提取纯文本。"""
    try:
        chain = ast.literal_eval(str(raw))
        parts = []
        for seg in chain:
            if isinstance(seg, dict) and seg.get("type") in ("Plain", "text"):
                parts.append(seg.get("text", ""))
        return " ".join(parts).strip()
    except Exception:
        return str(raw).strip()


# ── 主流程 ─────────────────────────────────────────────────

async def migrate(args):
    old_dir = Path(args.old_wordstock)
    new_dir = Path(args.new_lancedb)
    os.makedirs(new_dir, exist_ok=True)

    # 初始化 embedding
    if args.local_model:
        os.environ.setdefault("HF_ENDPOINT", args.hf_mirror or "https://hf-mirror.com")
        from sentence_transformers import SentenceTransformer

        print(f"加载本地模型: {args.local_model} ...")
        cache_dir = os.path.join(str(new_dir.parent), "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        model = await asyncio.to_thread(
            SentenceTransformer, args.local_model, cache_folder=cache_dir
        )
        print("模型加载完成")

        async def embed(text: str) -> list[float]:
            vec = await asyncio.to_thread(model.encode, text, normalize_embeddings=True)
            return vec.tolist()

    else:
        if not args.embedding_provider:
            print("错误: 未指定 --local-model 时需要 --embedding-provider")
            sys.exit(1)
        # 这里需要用户通过环境变量或其他方式提供 API，简化处理
        print(f"在线 embedding 需要在脚本中自行接入 API，当前仅支持 --local-model")
        print("示例: --local-model BAAI/bge-small-zh-v1.5")
        sys.exit(1)

    # 打开/创建 LanceDB
    db = await lancedb.connect_async(str(new_dir))
    existing = await db.table_names()
    table_name = "qa_pairs"
    if table_name in existing:
        table = await db.open_table(table_name)
        print(f"已打开现有 LanceDB 表: {new_dir}")
    else:
        table = await db.create_table(table_name, schema=_new_schema, mode="create")
        print(f"已创建 LanceDB 表: {new_dir}")

    # 扫描旧数据库文件
    db_files = sorted(old_dir.glob("*.db"))
    if not db_files:
        print(f"未在 {old_dir} 找到 .db 文件")
        return

    total_questions = 0
    total_answers = 0
    total_skipped = 0

    for db_file in db_files:
        group_id = db_file.stem
        print(f"\n处理群 {group_id} ({db_file}) ...")

        batch = []
        batch_size = 50

        for old_id, qtext, qraw, freq, answers in iter_old_wordstock(str(db_file)):
            vec = await embed(qtext)
            if not vec:
                total_skipped += 1
                continue

            row = {
                "id": int(time.time() * 1_000_000) + total_questions,
                "group_id": group_id,
                "question_text": qtext,
                "question_raw": qraw,
                "vec": vec,
                "answers": answers,
                "freq": freq,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            batch.append(row)
            total_questions += 1
            total_answers += len(answers)

            if len(batch) >= batch_size:
                await table.add(batch)
                print(f"  已写入 {total_questions} 条...", end="\r")
                batch.clear()

        if batch:
            await table.add(batch)

        print(f"  群 {group_id}: 迁移完成")

    print(f"\n✅ 迁移完成: {total_questions} 个问题, {total_answers} 个答案")
    if total_skipped:
        print(f"⚠ 跳过 {total_skipped} 条（embedding 失败）")

    await db.close()


def main():
    parser = argparse.ArgumentParser(description="ChatLearning 旧版 SQLite → 新版 LanceDB 迁移")
    parser.add_argument("--old-wordstock", required=True, help="旧版 WordStock 目录路径")
    parser.add_argument("--new-lancedb", required=True, help="新版 LanceDB 数据目录路径")
    parser.add_argument(
        "--local-model",
        default="BAAI/bge-small-zh-v1.5",
        help="本地 sentence-transformers 模型名（默认 BAAI/bge-small-zh-v1.5）",
    )
    parser.add_argument("--hf-mirror", default="https://hf-mirror.com", help="HuggingFace 镜像")
    parser.add_argument("--batch-size", type=int, default=50, help="批量写入大小")
    parser.add_argument(
        "--embedding-provider",
        help="在线 embedding provider（暂不支持，请用 --local-model）",
    )
    args = parser.parse_args()

    asyncio.run(migrate(args))


if __name__ == "__main__":
    main()
