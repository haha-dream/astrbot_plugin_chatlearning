"""
LanceDB 词库封装层

Schema:

  id: int64              — 自增主键
  group_id: string       — 群/会话标识
  question_text: string  — 问题纯文本（embedding 输入）
  question_raw: string   — 问题完整消息链 JSON（回复时用于消息重建）
  vec: list<float32>     — 问题的 embedding 向量
  answers: list<struct<  — 该问题的所有答案（LanceDB 原生嵌套类型）
    answertext: string   — 答案纯文本
    answer_raw: string   — 答案完整消息链 JSON
    added_at: float64    — 添加时间戳
    same: int32          — 相同答案计数（权重）
  >>
  freq: int32            — 问题被提问的总次数
  created_at: float64    — 首次创建时间
  updated_at: float64    — 最后更新时间
"""

import os
import time

import lancedb
import numpy as np
import pyarrow as pa

from astrbot.api import logger

# ── Arrow Schema ──────────────────────────────────────────────

_answer_struct = pa.struct(
    [
        pa.field("answertext", pa.string()),
        pa.field("answer_raw", pa.string()),
        pa.field("added_at", pa.float64()),
        pa.field("same", pa.int32()),
    ]
)

_wordstock_schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("group_id", pa.string()),
        pa.field("question_text", pa.string()),
        pa.field("question_raw", pa.string()),
        pa.field("vec", pa.list_(pa.float32(), -1)),  # -1 = variable dim
        pa.field("answers", pa.list_(_answer_struct)),
        pa.field("freq", pa.int32()),
        pa.field("created_at", pa.float64()),
        pa.field("updated_at", pa.float64()),
    ]
)

TABLE_NAME = "qa_pairs"


class WordStock:
    """LanceDB 词库：管理 Q&A 对的存储、查询、更新、删除。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._vec_dim: int | None = None

    # ── 生命周期 ────────────────────────────────────────────

    async def initialize(self):
        """打开或创建 LanceDB 数据库与表。"""
        os.makedirs(self.db_path, exist_ok=True)
        self._db = await lancedb.connect_async(self.db_path)

        existing = await self._db.table_names()
        if TABLE_NAME in existing:
            self._table = await self._db.open_table(TABLE_NAME)
            logger.info(f"LanceDB 词库已打开: {self.db_path}")
        else:
            # 空表：先创建，后续第一次写入时确定 vec 维度
            self._table = await self._db.create_table(
                TABLE_NAME,
                schema=_wordstock_schema,
                mode="create",
            )
            logger.info(f"LanceDB 词库已创建: {self.db_path}")

        # 探测已有向量维度
        records = await self._table.query().limit(1).to_arrow()
        if records.num_rows > 0:
            vec_col = records.column("vec")
            if vec_col[0].as_py():
                self._vec_dim = len(vec_col[0].as_py())

    async def close(self):
        """LanceDB 无需显式关闭，这里做清理标记。"""
        self._table = None
        self._db = None

    # ── 写入 ────────────────────────────────────────────────

    async def add_question(
        self,
        group_id: str,
        question_text: str,
        question_raw: str,
        vector: list[float],
        answer_text: str = "",
        answer_raw: str = "",
    ) -> int:
        """新增一条问题记录（可选附带首个答案）。返回写入后的 id。"""
        now = time.time()
        if self._vec_dim is None:
            self._vec_dim = len(vector)

        answers = []
        if answer_text or answer_raw:
            answers = [
                {
                    "answertext": answer_text or answer_raw,
                    "answer_raw": answer_raw or answer_text,
                    "added_at": now,
                    "same": 1,
                }
            ]

        row = [
            {
                "id": int(now * 1_000_000),  # 微秒级时间戳作 ID
                "group_id": group_id,
                "question_text": question_text,
                "question_raw": question_raw,
                "vec": vector,
                "answers": answers,
                "freq": 1,
                "created_at": now,
                "updated_at": now,
            }
        ]

        await self._table.add(row)
        logger.debug(
            f"[WordStock] 新增问题: group={group_id} text={question_text[:50]}..."
        )
        return row[0]["id"]

    async def add_answer(
        self,
        record_id: int,
        answer_text: str,
        answer_raw: str,
    ) -> bool:
        """向已有问题追加（或合并）答案。如答案已存在则增加 same 计数。"""
        # 读取当前记录
        result = await self._table.query().where(f"id = {record_id}").to_arrow()
        if result.num_rows == 0:
            logger.warning(f"[WordStock] add_answer: id={record_id} 不存在")
            return False

        row = result.to_pylist()[0]
        answers: list = row["answers"] or []
        now = time.time()

        # 检查是否已有相同答案（比较纯文本）
        merged = False
        for ans in answers:
            if ans["answertext"] == answer_text:
                ans["same"] = ans.get("same", 1) + 1
                ans["added_at"] = now  # 更新最近出现时间
                merged = True
                break

        if not merged:
            answers.append(
                {
                    "answertext": answer_text,
                    "answer_raw": answer_raw,
                    "added_at": now,
                    "same": 1,
                }
            )

        # 超出上限时裁剪最早的答案
        max_ans = getattr(self, "max_answers", 0) or 20
        if len(answers) > max_ans:
            answers.sort(key=lambda a: a.get("added_at", 0))
            answers = answers[-max_ans:]

        # LanceDB update
        await self._table.update(
            where=f"id = {record_id}",
            values={
                "answers": answers,
                "freq": row["freq"] + 1,
                "updated_at": now,
            },
        )
        logger.debug(f"[WordStock] 追加答案: id={record_id} merged={merged}")
        return True

    # ── 查询 ────────────────────────────────────────────────

    async def search_similar(
        self,
        group_id: str,
        query_vec: list[float],
        threshold: float = 0.75,
        top_k: int = 20,
    ) -> list[dict]:
        """向量相似搜索，返回 top_k 条满足阈值的记录。"""
        if self._table is None:
            return []

        try:
            results = (
                await self._table.query()
                .nearest_to(query_vec)
                .where(f"group_id = '{group_id}'")
                .distance_type("cosine")
                .limit(top_k)
                .to_arrow()
            )
        except Exception as e:
            logger.error(f"[WordStock] 搜索失败: {e}")
            return []

        records = []
        for row in results.to_pylist():
            score = row.get("_distance", 0)
            if 1.0 - score >= threshold:  # cosine distance → similarity
                records.append(
                    {
                        "id": row["id"],
                        "question_text": row["question_text"],
                        "question_raw": row["question_raw"],
                        "answers": row["answers"] or [],
                        "freq": row["freq"],
                        "score": 1.0 - score,
                    }
                )

        logger.debug(f"[WordStock] 搜索: group={group_id} found={len(records)}")
        return records

    async def get_by_id(self, record_id: int) -> dict | None:
        """按 ID 获取单条记录。"""
        result = await self._table.query().where(f"id = {record_id}").to_arrow()
        if result.num_rows == 0:
            return None
        return result.to_pylist()[0]

    async def get_by_text(self, group_id: str, question_text: str) -> dict | None:
        """按 group_id + 精确文本 获取单条记录（用于判断问题是否已存在）。"""
        escaped = question_text.replace("'", "''")
        result = (
            await self._table.query()
            .where(f"group_id = '{group_id}' AND question_text = '{escaped}'")
            .to_arrow()
        )
        if result.num_rows == 0:
            return None
        return result.to_pylist()[0]

    # ── 管理 ────────────────────────────────────────────────

    async def touch(self, record_id: int) -> bool:
        """增加问题的出现频率（问题复现时调用）。"""
        rec = await self.get_by_id(record_id)
        if rec is None:
            return False
        await self._table.update(
            where=f"id = {record_id}",
            values={"freq": rec["freq"] + 1, "updated_at": time.time()},
        )
        return True

    async def delete(self, record_id: int) -> bool:
        """按 ID 删除单条记录。"""
        await self._table.delete(f"id = {record_id}")
        return True

    async def count(self, group_id: str | None = None) -> int:
        """统计记录数。"""
        if group_id:
            result = (
                await self._table.query().where(f"group_id = '{group_id}'").to_arrow()
            )
        else:
            result = await self._table.query().to_arrow()
        return result.num_rows

    async def get_all_group_ids(self) -> list[str]:
        """获取所有不重复的 group_id。"""
        result = await self._table.query().to_arrow()
        if result.num_rows == 0:
            return []
        group_ids: set[str] = set()
        for gid in result.column("group_id").to_pylist():
            if gid:
                group_ids.add(gid)
        return sorted(group_ids)

    async def search_cross_group(
        self,
        exclude_group_id: str,
        query_vec: list[float],
        threshold: float = 0.75,
        top_k: int = 20,
    ) -> list[dict]:
        """跨群向量搜索：排除当前群，搜索其他所有群的词库。"""
        if self._table is None:
            return []

        try:
            # LanceDB 不支持 != 操作符直接做 nearest search，
            # 所以我们：1) 全局 nearest search 取足够多候选 → 2) Python 侧过滤
            results = (
                await self._table.query()
                .nearest_to(query_vec)
                .distance_type("cosine")
                .limit(top_k * 5)  # 多取一些以补偿过滤损失
                .to_arrow()
            )
        except Exception as e:
            logger.error(f"[WordStock] 跨群搜索失败: {e}")
            return []

        records = []
        seen_group_ids = set()
        for row in results.to_pylist():
            if row["group_id"] == exclude_group_id:
                continue
            score = row.get("_distance", 0)
            sim = 1.0 - score
            if sim < threshold:
                continue
            records.append(
                {
                    "id": row["id"],
                    "question_text": row["question_text"],
                    "question_raw": row["question_raw"],
                    "answers": row["answers"] or [],
                    "freq": row["freq"],
                    "score": sim,
                    "group_id": row["group_id"],
                }
            )
            seen_group_ids.add(row["group_id"])
            if len(records) >= top_k:
                break

        logger.debug(
            f"[WordStock] 跨群搜索: exclude={exclude_group_id} "
            f"from {len(seen_group_ids)} groups, found={len(records)}"
        )
        return records

    async def ensure_index(self):
        """确保索引存在：数据量达标且未构建索引时自动构建 IVF-PQ。"""
        if self._table is None or self._vec_dim is None:
            return

        if getattr(self, "_index_built", False):
            return

        row_count = await self.count()
        if row_count < 5000:
            return

        try:
            nlist = min(int(np.sqrt(row_count)), 256)
            n_sub = min(self._vec_dim // 8, 96)
            logger.info(
                f"[WordStock] 自动构建 IVF-PQ 索引: "
                f"rows={row_count} nlist={nlist} n_sub_vectors={n_sub}"
            )
            await self._table.create_index(
                metric="cosine",
                num_partitions=nlist,
                num_sub_vectors=n_sub,
            )
            self._index_built = True
            logger.info("[WordStock] IVF-PQ 索引构建完成")
        except Exception as e:
            logger.warning(f"[WordStock] 索引构建失败（非致命）: {e}")

    async def build_index(self):
        """强制重建 IVF-PQ 索引。"""
        if self._table is None or self._vec_dim is None:
            return

        row_count = await self.count()
        if row_count < 5000:
            logger.info("[WordStock] 数据量 < 5000，跳过索引构建（暴力搜索已够快）")
            return

        nlist = min(int(np.sqrt(row_count)), 256)
        n_sub = min(self._vec_dim // 8, 96)
        logger.info(f"[WordStock] 重建 IVF-PQ 索引: nlist={nlist} n_sub={n_sub}")
        await self._table.create_index(
            metric="cosine",
            num_partitions=nlist,
            num_sub_vectors=n_sub,
            replace=True,
        )
        self._index_built = True
        logger.info("[WordStock] 索引重建完成")

    async def cleanup_low_freq(self, days: float = 30, min_freq: int = 2) -> int:
        """清理低频旧词条：updated_at 距今超过 days 且 freq < min_freq。返回清理数。"""
        cutoff = time.time() - days * 86400
        result = await self._table.query().to_arrow()
        to_delete = []
        for row in result.to_pylist():
            if row["freq"] < min_freq and row["updated_at"] < cutoff:
                to_delete.append(row["id"])
        if to_delete:
            # LanceDB delete 逐条执行
            for rid in to_delete:
                await self._table.delete(f"id = {rid}")
        logger.info(f"[WordStock] 清理低频词条: {len(to_delete)} 条")
        return len(to_delete)

    async def get_stats(self, group_id: str | None = None) -> dict:
        """获取词库统计信息。"""
        if group_id:
            result = (
                await self._table.query().where(f"group_id = '{group_id}'").to_arrow()
            )
        else:
            result = await self._table.query().to_arrow()

        rows = result.to_pylist()
        if not rows:
            return {"total": 0, "total_answers": 0, "top_questions": [], "groups": []}

        total_answers = sum(len(r.get("answers", [])) for r in rows)
        total_freq = sum(r.get("freq", 0) for r in rows)
        avg_score = total_freq / len(rows) if rows else 0

        # Top 10 热门问题
        sorted_rows = sorted(rows, key=lambda r: r.get("freq", 0), reverse=True)
        top = [
            {
                "text": r["question_text"][:50],
                "freq": r["freq"],
                "answers": len(r.get("answers", [])),
            }
            for r in sorted_rows[:10]
        ]

        # 群分布
        group_counts: dict[str, int] = {}
        for r in rows:
            gid = r.get("group_id", "?")
            group_counts[gid] = group_counts.get(gid, 0) + 1
        groups = [{"id": k, "count": v} for k, v in group_counts.items()]

        return {
            "total": len(rows),
            "total_answers": total_answers,
            "avg_freq": round(avg_score, 1),
            "top_questions": top,
            "groups": groups,
        }

    async def export_data(self, group_id: str | None = None) -> list[dict]:
        """导出词库数据（不含向量，导入时需重新生成 embedding）。"""
        if group_id:
            result = (
                await self._table.query().where(f"group_id = '{group_id}'").to_arrow()
            )
        else:
            result = await self._table.query().to_arrow()
        return [
            {
                "group_id": r["group_id"],
                "question_text": r["question_text"],
                "question_raw": r["question_raw"],
                "answers": r.get("answers", []),
                "freq": r["freq"],
            }
            for r in result.to_pylist()
        ]

    async def import_data(
        self,
        data: list[dict],
        embedding_fn,  # async (text) -> list[float]
    ) -> tuple[int, int]:
        """导入词库数据，重新生成 embedding。返回 (新增数, 跳过数)。"""
        added = 0
        skipped = 0
        for item in data:
            gid = item["group_id"]
            qtext = item["question_text"]
            # 避免重复
            exist = await self.get_by_text(gid, qtext)
            if exist:
                skipped += 1
                # 合并答案
                for a in item.get("answers", []):
                    try:
                        await self.add_answer(
                            exist["id"],
                            a.get("answertext", ""),
                            a.get("answer_raw", a.get("answertext", "")),
                        )
                    except Exception:
                        pass
                continue
            vec = await embedding_fn(qtext)
            if vec is None:
                skipped += 1
                continue
            # 逐个添加答案（避免 add_question 只带一个答案的限制）
            first_answer = ""
            first_raw = ""
            if item.get("answers"):
                first_answer = item["answers"][0].get("answertext", "")
                first_raw = item["answers"][0].get("answer_raw", first_answer)

            rid = await self.add_question(
                group_id=gid,
                question_text=qtext,
                question_raw=item.get("question_raw", qtext),
                vector=vec,
                answer_text=first_answer,
                answer_raw=first_raw,
            )
            # 追加剩余答案
            for a in item.get("answers", [])[1:]:
                await self.add_answer(
                    rid,
                    a.get("answertext", ""),
                    a.get("answer_raw", a.get("answertext", "")),
                )
            added += 1

        logger.info(f"[WordStock] 导入完成: added={added} skipped={skipped}")
        return added, skipped
