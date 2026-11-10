#!/usr/bin/env python3
"""Memory-v2 搜索空间的 chunk_size 变体（运行时改写，不改共享模块）。

背景
====
`rq1_memory_v2_jit_random.SearchConfig` 原本是 5 维：

    text_top_k(4) × image_top_k(6) × text_weight(3) × rrf_k(4) × backend(3) = 864

其中 text_weight 与 rrf_k 是在 chunk=512 下实测 η²≈0.003/0.008 才被判为摆设的；
而 chunk_size 从未作为维度出现过，且它同时改变"同一页的上榜机会"（顺序统计量
效应）与"页级多样性"（top-k 可能全来自同一页）——是最可能造出非单调响应的旋钮。

本模块把搜索空间换成 4 维：

    text_top_k(4) × image_top_k(6) × chunk_size(4) × embedding_backend(3) = 288

⚠️ text_weight / rrf_k 被固定为 pharma(512) 上的中位取值（FIXED_TEXT_WEIGHT /
FIXED_RRF_K）。这是一个**必须写进 manifest 的假设**：两个旋钮是在 chunk=512 下
被判定为摆设的，而 chunk 变细会改变 RRF 的排名分布，固定其值不保证无害。

实现方式
========
`search` / `experiment` / `paired` 在 `rq1_memory_v2_jit_random`、
`rq1_fixed_eval_paired5`、`jev_initial_batch_repeat6` 里全部指向**同一个模块
对象**，且这些模块都在调用时做 `search.X` 的属性查找。因此 `install()` 只替换
该模块的属性即可全局生效：`mgrs/rag_pipeline/*`、`environment/runner.py`
以及 scienceqa 那条 864 配置的线都不受影响，随时可回退。

刻意**不**改写 `search.RetrievalCache`：全仓有 14 处 `search.RetrievalCache(...)`
构造点，它们用的是 3 参数签名；chunk 版检索缓存由本模块的
`ChunkRetrievalCache` 单独提供，只有新入口使用。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mgrs.controllers.llm_controller import LLMController
from mgrs.pipeline import rq1_memory_v2_jit_random as search


LOGGER = logging.getLogger("ao_explore.rq1_memory_v2_chunk_dim")

# 升版即同时升级 retrieval_cache.json 的 schema，旧缓存会被自动忽略（不会静默复用
# chunk=512 的文本排名）。
SPACE_VERSION = "memory_v2_absolute_scope_rrf_embedding_chunk_v1"

TEXT_VALUES = (5, 10, 20, 40)
IMAGE_VALUES = (0, 1, 3, 5, 10, 20)
CHUNK_VALUES = (64, 128, 256, 512)
EMBEDDING_VALUES = ("jina_nano", "jina_small", "qwen3_vl_2b")

PARAM_NAMES = ("text_top_k", "image_top_k", "chunk_size", "embedding_backend")
# 每个 ordinal 维度的分母 = len(values) - 1，使该维度距离归一化到 [0, 1]。
ORDINAL_DENOMINATORS = (len(TEXT_VALUES) - 1, len(IMAGE_VALUES) - 1, len(CHUNK_VALUES) - 1)
DISTANCE_WEIGHT = 1.0 / len(PARAM_NAMES)

# 固定旋钮：pharma(chunk=512) 上的中位取值。见模块文档的假设说明。
FIXED_TEXT_WEIGHT = 1.2
FIXED_RRF_K = 60

# chunk=512 的索引沿用既有命名，**没有** _c512 后缀。
LEGACY_CHUNK = 512

# 已经建好索引的 chunk 取值全集；set_chunk_values 只能在其中取子集。
ALL_BUILT_CHUNKS = (64, 128, 256, 512)
# 另外三维的取值全集：set_*_values 只能在其中取子集（索引未覆盖的取值会 404）。
ALL_TEXT_VALUES = (5, 10, 20, 40)
ALL_IMAGE_VALUES = (0, 1, 3, 5, 10, 20)
ALL_EMBEDDING_VALUES = ("jina_nano", "jina_small", "qwen3_vl_2b")


def _refresh_denominators() -> None:
    """按当前取值重算 ordinal 分母；单取值收窄时兜底 1，避免除零。"""
    global ORDINAL_DENOMINATORS
    ORDINAL_DENOMINATORS = (
        max(len(TEXT_VALUES) - 1, 1),
        max(len(IMAGE_VALUES) - 1, 1),
        max(len(CHUNK_VALUES) - 1, 1),
    )


def _narrow(name: str, values: Iterable[Any], allowed: tuple[Any, ...], cast: Any):
    resolved = tuple(cast(v) for v in values)
    if not resolved:
        raise ValueError(f"{name} 取值不能为空")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} 取值有重复: {resolved}")
    unknown = [v for v in resolved if v not in allowed]
    if unknown:
        raise ValueError(f"以下 {name} 取值没有对应索引: {unknown}（可选 {list(allowed)}）")
    return resolved


def set_chunk_values(values: Iterable[int]) -> tuple[int, ...]:
    """收窄 chunk 维度（用于小规模探针），必须在 install() 之前调用。

    只允许在 ALL_BUILT_CHUNKS 内取子集：超出范围会查到不存在的索引。
    """
    global CHUNK_VALUES
    CHUNK_VALUES = _narrow("chunk", values, ALL_BUILT_CHUNKS, int)
    _refresh_denominators()
    return CHUNK_VALUES


def set_text_values(values: Iterable[int]) -> tuple[int, ...]:
    """收窄 text_top_k 维度，必须在 install() 之前调用。"""
    global TEXT_VALUES
    TEXT_VALUES = _narrow("text_top_k", values, ALL_TEXT_VALUES, int)
    _refresh_denominators()
    return TEXT_VALUES


def set_image_values(values: Iterable[int]) -> tuple[int, ...]:
    """收窄 image_top_k 维度，必须在 install() 之前调用。"""
    global IMAGE_VALUES
    IMAGE_VALUES = _narrow("image_top_k", values, ALL_IMAGE_VALUES, int)
    _refresh_denominators()
    return IMAGE_VALUES


def set_embedding_values(values: Iterable[str]) -> tuple[str, ...]:
    """收窄检索后端维度，必须在 install() 之前调用。"""
    global EMBEDDING_VALUES
    EMBEDDING_VALUES = _narrow("embedding_backend", values, ALL_EMBEDDING_VALUES, str)
    _refresh_denominators()
    return EMBEDDING_VALUES


@dataclass(frozen=True)
class SearchConfig:
    config_id: str
    text_top_k: int
    image_top_k: int
    chunk_size: int
    embedding_backend: str
    param_idx: tuple[int, int, int, int]

    @property
    def params(self) -> dict[str, Any]:
        return {
            "text_top_k": self.text_top_k,
            "image_top_k": self.image_top_k,
            "chunk_size": self.chunk_size,
            "embedding_backend": self.embedding_backend,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "params": self.params,
            "param_idx": list(self.param_idx),
        }


def all_configs() -> list[SearchConfig]:
    rows: list[SearchConfig] = []
    for ti, ii, ci, ei in product(
        range(len(TEXT_VALUES)),
        range(len(IMAGE_VALUES)),
        range(len(CHUNK_VALUES)),
        range(len(EMBEDDING_VALUES)),
    ):
        text_k = TEXT_VALUES[ti]
        image_k = IMAGE_VALUES[ii]
        chunk = CHUNK_VALUES[ci]
        backend = EMBEDDING_VALUES[ei]
        rows.append(
            SearchConfig(
                config_id=f"t{text_k}_i{image_k}_c{chunk}_e{backend}",
                text_top_k=text_k,
                image_top_k=image_k,
                chunk_size=chunk,
                embedding_backend=backend,
                param_idx=(ti, ii, ci, ei),
            )
        )
    return rows


def record_distance(config_idx: Iterable[int], record_idx: Iterable[int]) -> float:
    candidate = tuple(int(v) for v in config_idx)
    history = tuple(int(v) for v in record_idx)
    ordinal = sum(
        DISTANCE_WEIGHT * abs(candidate[dim] - history[dim]) / ORDINAL_DENOMINATORS[dim]
        for dim in range(3)
    )
    embedding = DISTANCE_WEIGHT * (0.0 if candidate[3] == history[3] else 1.0)
    return float(ordinal + embedding)


def config_distance(left: SearchConfig, right: SearchConfig) -> float:
    return record_distance(left.param_idx, right.param_idx)


def shared_space_filling_batch(
    candidates: list[SearchConfig], count: int = 5
) -> list[SearchConfig]:
    """与共享模块同构的确定性贪心 maximin 设计（只依赖 mock 后的距离函数）。"""
    if count > len(candidates):
        raise ValueError("space-filling batch exceeds candidate count")
    ordered = sorted(candidates, key=lambda item: item.config_id)
    selected = [ordered[0]]
    available = {item.config_id: item for item in ordered[1:]}
    while len(selected) < count:
        # 字典序 config id 让距离完全相同的并列可复现。
        chosen = max(
            available.values(),
            key=lambda cfg: (min(config_distance(cfg, old) for old in selected), cfg.config_id),
        )
        selected.append(chosen)
        del available[chosen.config_id]
    return selected


def text_index_name(prefix: str, backend: str, chunk: int) -> str:
    """文本索引名。chunk=512 沿用既有索引（无 _c512 后缀）。"""
    chunk = int(chunk)
    suffix = "" if chunk == LEGACY_CHUNK else f"_c{chunk}"
    return f"{prefix}{suffix}_text_embedding_v2_{backend}"


def image_index_name(prefix: str, backend: str) -> str:
    """图像索引名与切分无关，因此不含 chunk。"""
    return f"{prefix}_image_embedding_v2_{backend}"


def apply_backends(prefix: str) -> dict[str, str]:
    """改写 search.BACKENDS 的索引名，返回 (backend, chunk) → 文本索引名 的映射。

    文本索引随 chunk_size 变化，所以 `BACKENDS[backend]["text_index"]` 只能放一个
    占位值；这里放 chunk=512（既有索引）的名字，真正的解析一律走
    `text_index_name()`。
    """
    mapping: dict[str, str] = {}
    for backend, info in search.BACKENDS.items():
        info["text_index"] = text_index_name(prefix, backend, LEGACY_CHUNK)
        info["image_index"] = image_index_name(prefix, backend)
        for chunk in CHUNK_VALUES:
            mapping[f"{backend}|c{chunk}"] = text_index_name(prefix, backend, chunk)
    return mapping


def expected_index_names(prefix: str) -> dict[str, list[str]]:
    """本空间会访问到的全部索引名，供校验与 manifest 归档。"""
    text = sorted(
        text_index_name(prefix, backend, chunk)
        for chunk in CHUNK_VALUES
        for backend in EMBEDDING_VALUES
    )
    image = sorted(image_index_name(prefix, backend) for backend in EMBEDDING_VALUES)
    return {"text": text, "image": image}


def validate_indices(es: Any, prefix: str) -> dict[str, Any]:
    """校验本空间需要的全部索引：存在、非空、dims 与 BACKENDS 声明一致。

    刻意不校验精确文档数：文本索引按各 backend 的分词器切 chunk，文档数本就不同。
    """
    names = expected_index_names(prefix)
    checked: dict[str, Any] = {}
    for name in names["text"]:
        backend = name.rsplit("_text_embedding_v2_", 1)[1]
        checked[name] = _check_one(es, name, backend, "text_index")
    for name in names["image"]:
        backend = name.rsplit("_image_embedding_v2_", 1)[1]
        checked[name] = _check_one(es, name, backend, "image_index")
    return checked


def _check_one(es: Any, name: str, backend: str, kind: str) -> dict[str, Any]:
    if not es.indices.exists(index=name):
        raise RuntimeError(
            f"缺少索引 {kind}[{backend}]: {name}（确认已建库且 --index-prefix 正确）"
        )
    count = int(es.count(index=name)["count"])
    if count <= 0:
        raise RuntimeError(f"索引为空 {kind}[{backend}]: {name}")
    dims = int(
        es.indices.get_mapping(index=name)[name]["mappings"]["properties"]["embedding"]["dims"]
    )
    declared = int(search.BACKENDS[backend]["dims"])
    if dims != declared:
        raise RuntimeError(f"dims 不一致 {name}: {dims} != {declared}")
    return {"backend": backend, "kind": kind, "count": count, "dims": dims}


class ChunkRetrievalCache:
    """每 (backend, chunk) 的原始 ES 结果，对控制器与 top-k 截断不变。

    与共享模块 `RetrievalCache` 的唯一结构差异：文本两路按 chunk 分开缓存，图像
    一路仍与 chunk 无关（因此跨 chunk 复用，不重复打图像索引）。

        entries["{backend}:c{chunk}:{query_key}"] = {kind: text,  vector, lexical}
        entries["{backend}:image:{query_key}"]     = {kind: image, image}
    """

    def __init__(
        self,
        path: Path,
        es: Any,
        embedding_cache: Any,
        prefix: str,
        *,
        text_raw_top_k: int | None = None,
        image_raw_top_k: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.es = es
        self.embedding_cache = embedding_cache
        self.prefix = str(prefix)
        self.text_raw_top_k = int(
            search.TEXT_RAW_TOP_K if text_raw_top_k is None else text_raw_top_k
        )
        self.image_raw_top_k = int(
            search.IMAGE_RAW_TOP_K if image_raw_top_k is None else image_raw_top_k
        )
        self.rows: dict[str, dict[str, Any]] = {}
        # 多线程（并行配置）共享同一个 rows：写入要上锁，flush 要快照，
        # 否则 atomic_json 的 json.dump 会遍历到正在变形的 dict。
        self._lock = threading.Lock()
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if loaded.get("schema") == SPACE_VERSION:
                    self.rows = dict(loaded.get("entries", {}))
                    LOGGER.info(
                        "reuse retrieval cache entries=%s path=%s", len(self.rows), self.path
                    )
                else:
                    LOGGER.warning(
                        "检索缓存 schema 不匹配（%s != %s），忽略旧缓存并重建 path=%s",
                        loaded.get("schema"),
                        SPACE_VERSION,
                        self.path,
                    )
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("ignoring unreadable retrieval cache path=%s", self.path)

    # --- 键 -----------------------------------------------------------------
    def _text_key(self, backend: str, chunk: int, item: dict[str, Any]) -> str:
        return f"{backend}:c{int(chunk)}:{self.embedding_cache.query_key(item)}"

    def _image_key(self, backend: str, item: dict[str, Any]) -> str:
        return f"{backend}:image:{self.embedding_cache.query_key(item)}"

    @staticmethod
    def _text_rows(response: dict[str, Any], route: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source") or {}
            rows.append(
                {
                    "id": str(hit.get("_id") or ""),
                    "text": str(source.get("content") or ""),
                    "metadata": dict(source.get("metadata") or {}),
                    "score": float(hit.get("_score") or 0.0),
                    "route": route,
                }
            )
        return rows

    # --- 取数 ---------------------------------------------------------------
    def _fetch_text(self, backend: str, chunk: int, item: dict[str, Any]) -> dict[str, Any]:
        index = text_index_name(self.prefix, backend, chunk)
        vector = self.embedding_cache.vector(backend, item)
        vector_response = self.es.search(
            index=index,
            knn={
                "field": "embedding",
                "query_vector": vector.tolist(),
                "k": self.text_raw_top_k,
                "num_candidates": max(self.text_raw_top_k, 100),
            },
            size=self.text_raw_top_k,
            source=["content", "metadata"],
        )
        lexical_response = self.es.search(
            index=index,
            size=self.text_raw_top_k,
            query={
                "multi_match": {
                    "query": str(item.get("query", "")),
                    "fields": ["content^2", "metadata.source_id"],
                    "type": "best_fields",
                }
            },
            source=["content", "metadata"],
        )
        return {
            "kind": "text",
            "chunk_size": int(chunk),
            "index": index,
            "vector": self._text_rows(vector_response, "vector"),
            "lexical": self._text_rows(lexical_response, "lexical"),
        }

    def _fetch_image(self, backend: str, item: dict[str, Any]) -> dict[str, Any]:
        index = image_index_name(self.prefix, backend)
        vector = self.embedding_cache.vector(backend, item)
        response = self.es.search(
            index=index,
            knn={
                "field": "embedding",
                "query_vector": vector.tolist(),
                "k": self.image_raw_top_k,
                "num_candidates": max(self.image_raw_top_k, 100),
            },
            size=self.image_raw_top_k,
            source=["source_id", "image_path"],
        )
        rows = [
            {
                "rank": rank,
                "id": str((hit.get("_source") or {}).get("source_id") or hit.get("_id") or ""),
                "image_path": str((hit.get("_source") or {}).get("image_path") or ""),
                "score": float(hit.get("_score") or 0.0),
            }
            for rank, hit in enumerate(response["hits"]["hits"], start=1)
        ]
        return {"kind": "image", "index": index, "image": rows}

    def raw(self, backend: str, chunk: int, item: dict[str, Any]) -> dict[str, Any]:
        text_key = self._text_key(backend, chunk, item)
        text_row = self.rows.get(text_key)
        if text_row is None:
            # ES 查询在锁外做（保持并行），只有落库这一步上锁；
            # 并发同键重复查询是无害的，setdefault 保证只留一份。
            text_row = self.rows.setdefault(text_key, self._fetch_text(backend, chunk, item))
        image_key = self._image_key(backend, item)
        if image_key not in self.rows:
            image_row = self.rows.setdefault(image_key, self._fetch_image(backend, item))
        else:
            image_row = self.rows[image_key]
        return {
            "vector": text_row["vector"],
            "lexical": text_row["lexical"],
            "image": image_row["image"],
        }

    def evidence(
        self, config: SearchConfig, item: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """chunk 级 RRF 合并后截前 text_top_k 条；固定 text_weight / rrf_k。"""
        raw = self.raw(config.embedding_backend, config.chunk_size, item)
        merged: dict[str, dict[str, Any]] = {}
        for route, rows, route_weight in (
            ("vector", raw["vector"], 1.0),
            ("lexical", raw["lexical"], FIXED_TEXT_WEIGHT),
        ):
            for rank, row in enumerate(rows, start=1):
                target = merged.setdefault(row["id"], {**row, "rrf_score": 0.0, "routes": []})
                target["rrf_score"] += route_weight / (FIXED_RRF_K + rank)
                target["routes"].append(
                    {"route": route, "rank": rank, "score": float(row["score"])}
                )
        ranked = sorted(
            merged.values(), key=lambda row: (-float(row["rrf_score"]), str(row["id"]))
        )
        text_rows = [
            {**row, "rank": rank, "score": round(float(row["rrf_score"]), 8)}
            for rank, row in enumerate(ranked[: config.text_top_k], start=1)
        ]
        image_rows = [] if config.image_top_k == 0 else raw["image"][: config.image_top_k]
        return text_rows, image_rows

    def flush(self) -> None:
        # 必须先快照：atomic_json 内的 json.dump 是在 Python 层遍历字典的，
        # 并发配置的线程若此刻插入新键，会抛
        # `RuntimeError: dictionary changed size during iteration`。
        with self._lock:
            entries = dict(self.rows)
        search.atomic_json(
            self.path,
            {
                "schema": SPACE_VERSION,
                "prefix": self.prefix,
                "raw_text_top_k": self.text_raw_top_k,
                "raw_image_top_k": self.image_raw_top_k,
                "text_key_template": "{backend}:c{chunk}:{query_key}",
                "image_key_template": "{backend}:image:{query_key}",
                "fixed_params": {"text_weight": FIXED_TEXT_WEIGHT, "rrf_k": FIXED_RRF_K},
                "entries": entries,
            },
        )


NEW_SYSTEM_PROMPT = """你是药品多模态 RAG 检索配置搜索的决策器。
目标是在有限评估预算内尽快找到 retrieval support 高 的配置。
你只能依据给出的本组已评估配置与分数决策；没有其他组的结果。
每次选 5 个未评估的完整配置，适度探索不同参数组合。
只输出 JSON：{"configs": [{"text_top_k": 5, "image_top_k": 0,
"chunk_size": 128, "embedding_backend": "jina_small"}],
"reason": "简短理由"}。configs 必须恰好有 5 项，不要输出额外文字。"""


class ChunkLLMController(LLMController):
    """把决策 JSON 的参数域换成 chunk 版本（4 维、288 空间）。"""

    name = "llm"
    PARAM_FIELDS = PARAM_NAMES

    def _allowed_values(self) -> dict[str, list[Any]]:
        return {
            "text_top_k": list(TEXT_VALUES),
            "image_top_k": list(IMAGE_VALUES),
            "chunk_size": list(CHUNK_VALUES),
            "embedding_backend": list(EMBEDDING_VALUES),
        }


def install(*, six_module: Any = None) -> dict[str, Any]:
    """把 chunk 版本装进 `search`（以及可选的 `six`）模块，进程内全局生效。

    幂等：重复调用结果一致。刻意不改写 `search.RetrievalCache`（见模块文档）。
    """
    search.SEARCH_SPACE_VERSION = SPACE_VERSION
    search.PARAM_NAMES = PARAM_NAMES
    search.ORDINAL_DENOMINATORS = ORDINAL_DENOMINATORS
    search.DISTANCE_WEIGHT = DISTANCE_WEIGHT
    search.TEXT_VALUES = TEXT_VALUES
    search.IMAGE_VALUES = IMAGE_VALUES
    search.EMBEDDING_VALUES = EMBEDDING_VALUES
    search.CHUNK_VALUES = CHUNK_VALUES
    search.FIXED_TEXT_WEIGHT = FIXED_TEXT_WEIGHT
    search.FIXED_RRF_K = FIXED_RRF_K
    search.SearchConfig = SearchConfig
    search.all_configs = all_configs
    search.record_distance = record_distance
    search.config_distance = config_distance
    search.shared_space_filling_batch = shared_space_filling_batch
    search.text_index_name = text_index_name
    search.image_index_name = image_index_name
    search.ChunkRetrievalCache = ChunkRetrievalCache
    if six_module is not None:
        # six 在 import 时用 `from ... import LLMController, SYSTEM_PROMPT` 绑定了
        # 名字，所以补丁必须打在 six 上，打 llm_controller 上无效。
        six_module.LLMController = ChunkLLMController
        six_module.SYSTEM_PROMPT = NEW_SYSTEM_PROMPT
    return describe()


def describe() -> dict[str, Any]:
    """本空间的静态描述，用于 manifest 归档与自检。"""
    candidates = all_configs()
    return {
        "space_version": SPACE_VERSION,
        "param_names": list(PARAM_NAMES),
        "domains": {
            "text_top_k": list(TEXT_VALUES),
            "image_top_k": list(IMAGE_VALUES),
            "chunk_size": list(CHUNK_VALUES),
            "embedding_backend": list(EMBEDDING_VALUES),
        },
        "ordinal_denominators": list(ORDINAL_DENOMINATORS),
        "distance_weight": DISTANCE_WEIGHT,
        "fixed_params": {"text_weight": FIXED_TEXT_WEIGHT, "rrf_k": FIXED_RRF_K},
        "candidate_count": len(candidates),
        "config_id_shape": "t{text_top_k}_i{image_top_k}_c{chunk_size}_e{backend}",
    }


def verify_space() -> dict[str, Any]:
    """自检：计数、config_id 唯一、param_idx 与取值域一致、距离函数对称。"""
    candidates = all_configs()
    expected = (
        len(TEXT_VALUES)
        * len(IMAGE_VALUES)
        * len(CHUNK_VALUES)
        * len(EMBEDDING_VALUES)
    )
    if len(candidates) != expected:
        raise AssertionError(f"空间大小异常: {len(candidates)} != {expected}")
    ids = [cfg.config_id for cfg in candidates]
    if len(set(ids)) != len(ids):
        raise AssertionError("config_id 存在重复（会导致冻结评分缓存串味）")
    for cfg in candidates:
        if cfg.text_top_k != TEXT_VALUES[cfg.param_idx[0]]:
            raise AssertionError(f"param_idx 与取值域不一致: {cfg.config_id}")
        if cfg.image_top_k != IMAGE_VALUES[cfg.param_idx[1]]:
            raise AssertionError(f"param_idx 与取值域不一致: {cfg.config_id}")
        if cfg.chunk_size != CHUNK_VALUES[cfg.param_idx[2]]:
            raise AssertionError(f"param_idx 与取值域不一致: {cfg.config_id}")
        if cfg.embedding_backend != EMBEDDING_VALUES[cfg.param_idx[3]]:
            raise AssertionError(f"param_idx 与取值域不一致: {cfg.config_id}")
    for left, right in zip(candidates[:40], reversed(candidates[-40:])):
        if abs(record_distance(left.param_idx, right.param_idx)
               - record_distance(right.param_idx, left.param_idx)) > 1e-12:
            raise AssertionError("record_distance 不对称")
    by_chunk: dict[int, int] = {}
    for cfg in candidates:
        by_chunk[cfg.chunk_size] = by_chunk.get(cfg.chunk_size, 0) + 1
    per_chunk = len(TEXT_VALUES) * len(IMAGE_VALUES) * len(EMBEDDING_VALUES)
    for chunk in CHUNK_VALUES:
        if by_chunk.get(chunk) != per_chunk:
            raise AssertionError(f"chunk={chunk} 的配置数异常: {by_chunk.get(chunk)}")
    return {
        **describe(),
        "per_chunk_configs": per_chunk,
        "numpy_backend": np.__version__,
        "text_index_examples": {
            f"c{chunk}": text_index_name("PREFIX", "jina_small", chunk)
            for chunk in CHUNK_VALUES
        },
        "image_index_example": image_index_name("PREFIX", "jina_small"),
    }