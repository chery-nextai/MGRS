#!/usr/bin/env python3
"""Independent RQ1 experiment: Memory-v2 JIT, seeded Random, optional LLM.

This module deliberately does not import the old controllers, ExperimentRunner,
or global memory/cache.  It implements the absolute-score Memory-v2
specification in one self-contained entry point.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import time
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from elasticsearch import Elasticsearch

from mgrs import config as settings
from mgrs.controllers.llm_controller import LLMController
from mgrs.environment.data import load_dataset
from mgrs.rag_pipeline.llm import MultimodalLLM, RETRIEVAL_JUDGE_PROMPT_VERSION, RetrievalJudgeResult


LOGGER = logging.getLogger("ao_explore.rq1_memory_v2")

DOMAIN_ID = "scienceqa"
SEARCH_SPACE_VERSION = "memory_v2_absolute_scope_rrf_embedding_v1"
JUDGE_CACHE_SCHEMA = "retrieval_support_evidence_fingerprint_v1"
METRIC_SCHEMA = f"retrieval_support_ground_truth_mm_{RETRIEVAL_JUDGE_PROMPT_VERSION}"

TEXT_VALUES = (5, 10, 20, 40)
IMAGE_VALUES = (0, 1, 3, 5, 10, 20)
WEIGHT_VALUES = (0.8, 1.2, 1.8)
RRF_VALUES = (20, 40, 60, 100)
EMBEDDING_VALUES = ("jina_nano", "jina_small", "qwen3_vl_2b")
PARAM_NAMES = (
    "text_top_k",
    "image_top_k",
    "text_weight",
    "rrf_k",
    "embedding_backend",
)
ORDINAL_DENOMINATORS = (3, 5, 2, 3)
DISTANCE_WEIGHT = 1.0 / len(PARAM_NAMES)
K_NEIGHBORS = 5
KNN_TAU = 0.25
EXPLORE_LAMBDA = 0.1
RULE_PROMOTION_SUPPORT = 3
CONFIDENCE_EPSILON = 0.5
TEXT_RAW_TOP_K = 20
IMAGE_RAW_TOP_K = 20

# 每个后端的编码器权重目录（本地离线加载）。默认在仓库根的 models/ 下按名字查找，
# 可用环境变量覆盖，避免把机器相关的绝对路径写进代码：
#   MGRS_EMBED_MODEL_ROOT            权重公共根目录，默认 <repo>/models
#   MGRS_EMBED_MODEL_DIR_<BACKEND>   单个后端的权重目录（后端名大写）
#   例：MGRS_EMBED_MODEL_DIR_QWEN3_VL_2B=/models/Qwen3-VL-Embedding-2B
# 注：AO 格式数据集跑之前会用 apply_backends() 按数据集覆盖 text_index/image_index，
# 这里的索引名只是旧 ScienceQA 语料的默认值。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EMBED_MODEL_ROOT = Path(
    os.environ.get("MGRS_EMBED_MODEL_ROOT", "").strip() or (_REPO_ROOT / "models")
)


def _embed_model_dir(backend: str, default_name: str) -> str:
    override = os.environ.get(f"MGRS_EMBED_MODEL_DIR_{backend.upper()}", "").strip()
    return override or str(_EMBED_MODEL_ROOT / default_name)


BACKENDS: dict[str, dict[str, Any]] = {
    "jina_nano": {
        "dims": 768,
        "text_index": "aoexplore_scienceqa_train3944_text_embedding_v2_jina_nano",
        "image_index": "aoexplore_scienceqa_train3944_image_embedding_v2_jina_nano",
        "model_path": _embed_model_dir(
            "jina_nano", "jinaai--jina-embeddings-v5-omni-nano"
        ),
    },
    "jina_small": {
        "dims": 1024,
        "text_index": "aoexplore_scienceqa_train3944_text_embedding_v2_jina_small",
        "image_index": "aoexplore_scienceqa_train3944_image_embedding_v2_jina_small",
        "model_path": _embed_model_dir(
            "jina_small", "jinaai--jina-embeddings-v5-omni-small"
        ),
    },
    "qwen3_vl_2b": {
        "dims": 2048,
        "text_index": "aoexplore_scienceqa_train3944_text_embedding_v2_qwen3_vl_2b",
        "image_index": "aoexplore_scienceqa_train3944_image_embedding_v2_qwen3_vl_2b",
        "model_path": _embed_model_dir("qwen3_vl_2b", "Qwen3-VL-Embedding-2B"),
    },
}


@dataclass(frozen=True)
class SearchConfig:
    config_id: str
    text_top_k: int
    image_top_k: int
    text_weight: float
    rrf_k: int
    embedding_backend: str
    param_idx: tuple[int, int, int, int, int]

    @property
    def params(self) -> dict[str, Any]:
        return {
            "text_top_k": self.text_top_k,
            "image_top_k": self.image_top_k,
            "text_weight": self.text_weight,
            "rrf_k": self.rrf_k,
            "embedding_backend": self.embedding_backend,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "params": self.params,
            "param_idx": list(self.param_idx),
        }


def _weight_label(value: float) -> str:
    return str(value).replace(".", "p")


def all_configs() -> list[SearchConfig]:
    rows: list[SearchConfig] = []
    for ti, ii, wi, ri, ei in product(
        range(len(TEXT_VALUES)),
        range(len(IMAGE_VALUES)),
        range(len(WEIGHT_VALUES)),
        range(len(RRF_VALUES)),
        range(len(EMBEDDING_VALUES)),
    ):
        text_k = TEXT_VALUES[ti]
        image_k = IMAGE_VALUES[ii]
        weight = WEIGHT_VALUES[wi]
        rrf_k = RRF_VALUES[ri]
        backend = EMBEDDING_VALUES[ei]
        rows.append(
            SearchConfig(
                config_id=(
                    f"t{text_k}_i{image_k}_w{_weight_label(weight)}_r{rrf_k}_e{backend}"
                ),
                text_top_k=text_k,
                image_top_k=image_k,
                text_weight=weight,
                rrf_k=rrf_k,
                embedding_backend=backend,
                param_idx=(ti, ii, wi, ri, ei),
            )
        )
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_distance(config_idx: Iterable[int], record_idx: Iterable[int]) -> float:
    candidate = tuple(int(v) for v in config_idx)
    history = tuple(int(v) for v in record_idx)
    ordinal = sum(
        DISTANCE_WEIGHT * abs(candidate[dim] - history[dim]) / ORDINAL_DENOMINATORS[dim]
        for dim in range(4)
    )
    embedding = DISTANCE_WEIGHT * (0.0 if candidate[4] == history[4] else 1.0)
    return float(ordinal + embedding)


def config_distance(left: SearchConfig, right: SearchConfig) -> float:
    return record_distance(left.param_idx, right.param_idx)


def shared_space_filling_batch(candidates: list[SearchConfig], count: int = 5) -> list[SearchConfig]:
    """A deterministic greedy maximin design, shared by every paired trajectory."""
    if count > len(candidates):
        raise ValueError("space-filling batch exceeds candidate count")
    ordered = sorted(candidates, key=lambda item: item.config_id)
    selected = [ordered[0]]
    available = {item.config_id: item for item in ordered[1:]}
    while len(selected) < count:
        # Lexicographic config id resolves exact distance ties reproducibly.
        chosen = max(
            available.values(),
            key=lambda cfg: (min(config_distance(cfg, old) for old in selected), cfg.config_id),
        )
        selected.append(chosen)
        del available[chosen.config_id]
    return selected


class MemoryV2:
    """Run-local episodic records and fully re-distilled semantic rules."""

    def __init__(self) -> None:
        self.episodic: list[dict[str, Any]] = []
        self.semantic_rules: list[dict[str, Any]] = []

    def add(self, config: SearchConfig, score: float, round_number: int) -> dict[str, Any]:
        row = {
            "config_id": config.config_id,
            "params": config.params,
            "param_idx": list(config.param_idx),
            "score": float(score),
            "round": int(round_number),
            "domain_id": DOMAIN_ID,
        }
        self.episodic.append(row)
        return row

    def distill(self, round_number: int) -> list[dict[str, Any]]:
        if not self.episodic:
            self.semantic_rules = []
            return self.semantic_rules
        baseline = float(np.mean([float(row["score"]) for row in self.episodic]))
        rules: list[dict[str, Any]] = []
        for dim_name in PARAM_NAMES:
            grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for row in self.episodic:
                grouped[row["params"][dim_name]].append(row)
            for value, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
                support = len(rows)
                if support < RULE_PROMOTION_SUPPORT:
                    continue
                values = [float(row["score"]) for row in rows]
                effect = float(np.mean(values) - baseline)
                if effect > 0:
                    refute = sum(score < baseline for score in values)
                elif effect < 0:
                    refute = sum(score > baseline for score in values)
                else:
                    refute = 0
                confidence = float(support / (support + refute + CONFIDENCE_EPSILON))
                display_value = _weight_label(value) if dim_name == "text_weight" else str(value)
                rules.append(
                    {
                        "rule_id": f"{dim_name}={display_value}",
                        "scope": {dim_name: [value, value]},
                        "scope_dim_count": 1,
                        "baseline_mean": baseline,
                        "effect_size": effect,
                        "support": support,
                        "refute": int(refute),
                        "confidence": confidence,
                        "last_updated_round": int(round_number),
                    }
                )
        self.semantic_rules = rules
        return rules

    def score(self, config: SearchConfig) -> dict[str, float]:
        if not self.episodic:
            raise RuntimeError("JIT score needs at least one episodic record")
        distances = sorted(
            [
                (record_distance(config.param_idx, row["param_idx"]), row)
                for row in self.episodic
            ],
            key=lambda item: (item[0], str(item[1]["config_id"])),
        )
        nearest = distances[: min(K_NEIGHBORS, len(distances))]
        weights = [math.exp(-distance / KNN_TAU) for distance, _ in nearest]
        prediction = float(
            sum(weight * float(row["score"]) for weight, (_, row) in zip(weights, nearest, strict=True))
            / sum(weights)
        )
        adjustment = 0.0
        for rule in self.semantic_rules:
            scope = rule["scope"]
            dim_name = next(iter(scope))
            if config.params[dim_name] == scope[dim_name][0]:
                adjustment += float(rule["confidence"]) * float(rule["effect_size"])
        explore = float(nearest[0][0])
        utility = prediction + adjustment + EXPLORE_LAMBDA * explore
        return {
            "knn_prediction": prediction,
            "rule_adjustment": float(adjustment),
            "explore": explore,
            "utility": float(utility),
        }

    def select_jit(
        self,
        remaining: list[SearchConfig],
        rng: random.Random,
    ) -> tuple[list[SearchConfig], dict[str, dict[str, Any]]]:
        if len(remaining) < 5:
            raise ValueError("need at least five remaining configurations")
        scored: dict[str, dict[str, float]] = {
            cfg.config_id: self.score(cfg) for cfg in remaining
        }
        exploitation: list[SearchConfig] = []
        per_dim_counts: list[dict[Any, int]] = [defaultdict(int) for _ in PARAM_NAMES]
        for cfg in sorted(
            remaining, key=lambda item: (-scored[item.config_id]["utility"], item.config_id)
        ):
            # The cap is imposed only for the three exploitation slots.
            if any(
                per_dim_counts[dim][cfg.params[PARAM_NAMES[dim]]] >= 2
                for dim in range(len(PARAM_NAMES))
            ):
                continue
            exploitation.append(cfg)
            for dim, name in enumerate(PARAM_NAMES):
                per_dim_counts[dim][cfg.params[name]] += 1
            if len(exploitation) == 3:
                break
        if len(exploitation) != 3:
            raise RuntimeError("diversity cap could not select three exploitation configs")
        remaining_after_exploitation = [
            cfg for cfg in remaining if cfg.config_id not in {item.config_id for item in exploitation}
        ]
        exploration = max(
            remaining_after_exploitation,
            key=lambda cfg: (scored[cfg.config_id]["explore"], cfg.config_id),
        )
        remaining_after_exploration = [
            cfg
            for cfg in remaining_after_exploitation
            if cfg.config_id != exploration.config_id
        ]
        safety = rng.choice(sorted(remaining_after_exploration, key=lambda item: item.config_id))
        selected = exploitation + [exploration, safety]
        diagnostics: dict[str, dict[str, Any]] = {}
        for cfg in exploitation:
            diagnostics[cfg.config_id] = {"selection": "exploitation", **scored[cfg.config_id]}
        diagnostics[exploration.config_id] = {
            "selection": "exploration",
            **scored[exploration.config_id],
        }
        diagnostics[safety.config_id] = {
            "selection": "safety_random",
            **scored[safety.config_id],
        }
        return selected, diagnostics


class QueryEmbeddingCache:
    """Run-local 200-query vectors; this is retrieval infrastructure, not memory."""

    def __init__(self, root: Path, qa: list[dict[str, Any]], device: str, batch_size: int) -> None:
        self.root = root
        self.qa = qa
        self.device = device
        self.batch_size = batch_size
        self._vectors: dict[str, dict[str, np.ndarray]] = {}
        self.keys = [self.query_key(item) for item in qa]

    @staticmethod
    def query_key(item: dict[str, Any]) -> str:
        return sha256_json(
            {
                "id": str(item.get("id", "")),
                "query": str(item.get("query", "")),
            }
        )

    def _path(self, backend: str) -> Path:
        return self.root / "query_embeddings" / f"{backend}.npz"

    def ensure(self, backend: str) -> None:
        if backend in self._vectors:
            return
        path = self._path(backend)
        expected_dim = int(BACKENDS[backend]["dims"])
        if path.is_file():
            loaded = np.load(path, allow_pickle=False)
            saved_keys = [str(value) for value in loaded["keys"].tolist()]
            vectors = np.asarray(loaded["vectors"], dtype="float32")
            if saved_keys == self.keys and vectors.shape == (len(self.qa), expected_dim):
                self._vectors[backend] = {
                    key: vectors[index] for index, key in enumerate(saved_keys)
                }
                LOGGER.info("reuse query vectors backend=%s queries=%s", backend, len(saved_keys))
                return
            LOGGER.warning("ignore incompatible run-local vector cache path=%s", path)
        LOGGER.info("encoding all dataset queries backend=%s queries=%s", backend, len(self.qa))
        queries = [str(item.get("query", "")) for item in self.qa]
        vectors = self._encode(backend, queries)
        if vectors.shape != (len(queries), expected_dim):
            raise RuntimeError(
                f"embedding shape mismatch backend={backend}: {vectors.shape}, "
                f"expected={(len(queries), expected_dim)}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.asarray(self.keys), vectors=vectors.astype("float32"))
        self._vectors[backend] = {
            key: vectors[index] for index, key in enumerate(self.keys)
        }
        LOGGER.info("query vector cache saved backend=%s path=%s", backend, path)

    def vector(self, backend: str, item: dict[str, Any]) -> np.ndarray:
        self.ensure(backend)
        return self._vectors[backend][self.query_key(item)]

    def _encode(self, backend: str, queries: list[str]) -> np.ndarray:
        if backend.startswith("jina_"):
            return self._encode_jina(backend, queries)
        if backend == "qwen3_vl_2b":
            return self._encode_qwen(queries)
        raise ValueError(f"unknown embedding backend: {backend}")

    def _encode_jina(self, backend: str, queries: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        import torch

        model_path = str(BACKENDS[backend]["model_path"])
        model = SentenceTransformer(
            model_path,
            device=self.device,
            trust_remote_code=True,
            # Match the existing v2 index builder: the retrieval adapter and
            # vision+text subset (not the larger default audio-inclusive model).
            model_kwargs={"default_task": "retrieval", "modality": "vision"},
        )
        try:
            # Jina's retrieval model cards require query/document-specific APIs.
            encoded = model.encode_query(
                queries,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.asarray(encoded, dtype="float32")
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_qwen(self, queries: list[str]) -> np.ndarray:
        import torch

        model_path = Path(str(BACKENDS["qwen3_vl_2b"]["model_path"]))
        helper_path = model_path / "scripts" / "qwen3_vl_embedding.py"
        if not helper_path.is_file():
            raise RuntimeError(f"missing Qwen embedding helper: {helper_path}")
        module_name = "rq1_memory_v2_qwen_helper"
        spec = importlib.util.spec_from_file_location(module_name, helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load Qwen embedding helper: {helper_path}")
        # The model helper imports qwen-vl-utils at module import time, although
        # this experiment only embeds text queries.  The project runtime has the
        # matching Transformers/Qwen stack but intentionally does not install
        # qwen-vl-utils (which is needed only to decode image/video inputs).
        # Supply the exact text-only return shape instead of silently accepting
        # vision input or replacing a query with a placeholder.
        try:
            import qwen_vl_utils.vision_process  # noqa: F401
        except ModuleNotFoundError:
            def _text_only_vision_info(*_args: Any, **_kwargs: Any) -> tuple[None, None, dict[str, bool]]:
                return None, None, {"do_sample_frames": False}

            qwen_utils = types.ModuleType("qwen_vl_utils")
            vision_process = types.ModuleType("qwen_vl_utils.vision_process")
            vision_process.process_vision_info = _text_only_vision_info
            qwen_utils.vision_process = vision_process
            sys.modules["qwen_vl_utils"] = qwen_utils
            sys.modules["qwen_vl_utils.vision_process"] = vision_process
            LOGGER.info("using text-only qwen_vl_utils compatibility shim")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        embedder = module.Qwen3VLEmbedder(
            str(model_path),
            max_length=settings.IMAGE_EMBEDDING_MAX_LENGTH,
            max_pixels=settings.IMAGE_EMBEDDING_MAX_PIXELS,
        )
        vectors: list[np.ndarray] = []
        try:
            for start in range(0, len(queries), self.batch_size):
                batch = queries[start : start + self.batch_size]
                LOGGER.info(
                    "qwen query embeddings %s-%s/%s",
                    start + 1,
                    start + len(batch),
                    len(queries),
                )
                tensor = embedder.process([{"text": query} for query in batch], normalize=True)
                # Qwen returns bfloat16 on some CPU-capable builds; NumPy does
                # not support it, while ES expects ordinary float vectors.
                vectors.append(tensor.detach().float().cpu().numpy().astype("float32", copy=False))
            return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 2048), dtype="float32")
        finally:
            del embedder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class RetrievalCache:
    """Per-seed raw ES results, which are invariant to controller and top-k slices."""

    def __init__(self, path: Path, es: Elasticsearch, embedding_cache: QueryEmbeddingCache) -> None:
        self.path = path
        self.es = es
        self.embedding_cache = embedding_cache
        self.rows: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if loaded.get("schema") == SEARCH_SPACE_VERSION:
                    self.rows = dict(loaded.get("entries", {}))
                    LOGGER.info("reuse retrieval cache entries=%s path=%s", len(self.rows), path)
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("ignoring unreadable retrieval cache path=%s", path)

    def _key(self, backend: str, item: dict[str, Any]) -> str:
        return f"{backend}:{self.embedding_cache.query_key(item)}"

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

    def raw(self, backend: str, item: dict[str, Any]) -> dict[str, Any]:
        key = self._key(backend, item)
        if key in self.rows:
            return self.rows[key]
        query = str(item.get("query", ""))
        vector = self.embedding_cache.vector(backend, item)
        info = BACKENDS[backend]
        vector_response = self.es.search(
            index=info["text_index"],
            knn={
                "field": "embedding",
                "query_vector": vector.tolist(),
                "k": TEXT_RAW_TOP_K,
                "num_candidates": max(TEXT_RAW_TOP_K, 100),
            },
            size=TEXT_RAW_TOP_K,
            source=["content", "metadata"],
        )
        lexical_response = self.es.search(
            index=info["text_index"],
            size=TEXT_RAW_TOP_K,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "metadata.source_id"],
                    "type": "best_fields",
                }
            },
            source=["content", "metadata"],
        )
        image_response = self.es.search(
            index=info["image_index"],
            knn={
                "field": "embedding",
                "query_vector": vector.tolist(),
                "k": IMAGE_RAW_TOP_K,
                "num_candidates": max(IMAGE_RAW_TOP_K, 100),
            },
            size=IMAGE_RAW_TOP_K,
            source=["source_id", "image_path"],
        )
        image_rows = [
            {
                "rank": rank,
                "id": str((hit.get("_source") or {}).get("source_id") or hit.get("_id") or ""),
                "image_path": str((hit.get("_source") or {}).get("image_path") or ""),
                "score": float(hit.get("_score") or 0.0),
            }
            for rank, hit in enumerate(image_response["hits"]["hits"], start=1)
        ]
        value = {
            "vector": self._text_rows(vector_response, "vector"),
            "lexical": self._text_rows(lexical_response, "lexical"),
            "image": image_rows,
        }
        self.rows[key] = value
        return value

    def evidence(self, config: SearchConfig, item: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw = self.raw(config.embedding_backend, item)
        merged: dict[str, dict[str, Any]] = {}
        for route, rows, route_weight in (
            ("vector", raw["vector"], 1.0),
            ("lexical", raw["lexical"], config.text_weight),
        ):
            for rank, row in enumerate(rows, start=1):
                target = merged.setdefault(
                    row["id"],
                    {
                        **row,
                        "rrf_score": 0.0,
                        "routes": [],
                    },
                )
                target["rrf_score"] += route_weight / (config.rrf_k + rank)
                target["routes"].append(
                    {"route": route, "rank": rank, "score": float(row["score"])}
                )
        ranked = sorted(
            merged.values(), key=lambda row: (-float(row["rrf_score"]), str(row["id"]))
        )
        text_rows = [
            {
                **row,
                "rank": rank,
                "score": round(float(row["rrf_score"]), 8),
            }
            for rank, row in enumerate(ranked[: config.text_top_k], start=1)
        ]
        image_rows = [] if config.image_top_k == 0 else raw["image"][: config.image_top_k]
        return text_rows, image_rows

    def flush(self) -> None:
        # 快照后再序列化：json.dump 在 Python 层遍历字典，并行配置的线程若此刻
        # 插入新键会抛 `dictionary changed size during iteration`。
        atomic_json(
            self.path,
            {
                "schema": SEARCH_SPACE_VERSION,
                "raw_text_top_k": TEXT_RAW_TOP_K,
                "raw_image_top_k": IMAGE_RAW_TOP_K,
                "entries": dict(self.rows),
            },
        )


class JudgeCache:
    """Evidence-fingerprint cache scoped to one paired seed, never a Memory record."""

    def __init__(self, path: Path, model: str) -> None:
        self.path = path
        self.model = model
        self.scores: dict[str, float] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if (
                    loaded.get("schema") == JUDGE_CACHE_SCHEMA
                    and loaded.get("metric") == METRIC_SCHEMA
                    and loaded.get("model") == model
                ):
                    self.scores = {
                        str(key): float(value) for key, value in loaded.get("scores", {}).items()
                    }
                    LOGGER.info("reuse judge cache entries=%s path=%s", len(self.scores), path)
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("ignoring unreadable judge cache path=%s", path)

    def fingerprint(
        self,
        query: str,
        reference: str,
        text_rows: list[dict[str, Any]],
        image_rows: list[dict[str, Any]],
    ) -> str:
        return sha256_json(
            {
                "schema": JUDGE_CACHE_SCHEMA,
                "metric": METRIC_SCHEMA,
                "judge_model": self.model,
                "query": query,
                "reference": reference,
                "text_evidence": [
                    {
                        "rank": int(row["rank"]),
                        "source_id": str(row.get("metadata", {}).get("source_id", "")),
                        "text": str(row.get("text", "")),
                    }
                    for row in text_rows
                ],
                "image_evidence": [
                    {
                        "rank": int(row["rank"]),
                        "source_id": str(row.get("id", "")),
                        "image_path": str(row.get("image_path", "")),
                    }
                    for row in image_rows
                ],
            }
        )

    def flush(self) -> None:
        atomic_json(
            self.path,
            {
                "schema": JUDGE_CACHE_SCHEMA,
                "metric": METRIC_SCHEMA,
                "model": self.model,
                # 同样需要快照，理由见 RetrievalCache.flush
                "scores": dict(self.scores),
            },
        )


def reference_for(item: dict[str, Any]) -> str:
    references = item.get("references") or [item.get("answer", "")]
    return str(references[0] if isinstance(references, list) else references)


class ConfigEvaluator:
    def __init__(
        self,
        queries: list[dict[str, Any]],
        retrieval: RetrievalCache,
        judge_cache: JudgeCache,
        judge_model: str,
        concurrency: int,
    ) -> None:
        self.queries = queries
        self.retrieval = retrieval
        self.judge_cache = judge_cache
        self.judge_model = judge_model
        self.concurrency = concurrency

    def evaluate(self, config: SearchConfig) -> dict[str, Any]:
        prepared: list[dict[str, Any]] = []
        pending: dict[str, dict[str, Any]] = {}
        cache_hits = 0
        parse_counts = {"json": 0, "regex": 0, "failed": 0}
        response_path: Path | None = None
        for item in self.queries:
            query = str(item.get("query", ""))
            reference = reference_for(item)
            text_rows, image_rows = self.retrieval.evidence(config, item)
            fingerprint = self.judge_cache.fingerprint(query, reference, text_rows, image_rows)
            row = {
                "id": str(item.get("id", "")),
                "query": query,
                "reference": reference,
                "text_evidence": text_rows,
                "image_evidence": image_rows,
                "fingerprint": fingerprint,
                "text_hit_at_k": int(
                    any(
                        str(text.get("metadata", {}).get("source_id")) == str(item.get("id", ""))
                        for text in text_rows
                    )
                ),
                "image_hit_at_k": int(
                    any(str(image.get("id")) == str(item.get("id", "")) for image in image_rows)
                ),
            }
            prepared.append(row)
            if fingerprint not in self.judge_cache.scores:
                pending.setdefault(fingerprint, row)
            else:
                cache_hits += 1
        if pending:
            LOGGER.info(
                "judge config=%s queries=%s cache_hits=%s unique_misses=%s",
                config.config_id,
                len(prepared),
                cache_hits,
                len(pending),
            )
            def judge(row: dict[str, Any]) -> RetrievalJudgeResult:
                # A separate client per worker avoids implicit shared mutable request state.
                llm = MultimodalLLM()
                return llm.judge_retrieval_with_details(
                    row["query"],
                    row["reference"],
                    row["text_evidence"],
                    row["image_evidence"],
                    model=self.judge_model,
                )

            responses: list[dict[str, Any]] = []
            errors: list[Exception] = []
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(judge, row): fingerprint for fingerprint, row in pending.items()
                }
                for future in as_completed(futures):
                    fingerprint = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        errors.append(exc)
                        continue
                    self.judge_cache.scores[fingerprint] = float(result.score)
                    parse_counts[result.parse_status] += 1
                    responses.append(
                        {
                            "query_id": pending[fingerprint]["id"],
                            "fingerprint": fingerprint,
                            "raw_response": result.raw_response,
                            "parse_status": result.parse_status,
                            "score": float(result.score),
                        }
                    )
                    if result.parse_status == "failed":
                        LOGGER.warning(
                            "judge score parse failed config=%s query_id=%s fingerprint=%s",
                            config.config_id,
                            pending[fingerprint]["id"],
                            fingerprint,
                        )
            response_path = (
                self.judge_cache.path.parent / "judge_responses" / f"{config.config_id}.json"
            )
            atomic_json(
                response_path,
                {
                    "config_id": config.config_id,
                    "judge_model": self.judge_model,
                    "requested_count": len(pending),
                    "response_count": len(responses),
                    "parse_counts": parse_counts,
                    "responses": responses,
                },
            )
            self.judge_cache.flush()
            if errors:
                raise errors[0]
        else:
            LOGGER.info(
                "judge config=%s queries=%s cache_hits=%s unique_misses=0",
                config.config_id,
                len(prepared),
                cache_hits,
            )
        evaluations = [
            {
                "id": row["id"],
                "retrieval_support_score": float(self.judge_cache.scores[row["fingerprint"]]),
                "text_hit_at_k": row["text_hit_at_k"],
                "image_hit_at_k": row["image_hit_at_k"],
                "judge_fingerprint": row["fingerprint"],
            }
            for row in prepared
        ]
        self.retrieval.flush()
        return {
            "config_id": config.config_id,
            "mean_retrieval_support": float(
                np.mean([row["retrieval_support_score"] for row in evaluations])
            ),
            "mean_text_hit_at_k": float(np.mean([row["text_hit_at_k"] for row in evaluations])),
            "mean_image_hit_at_k": float(np.mean([row["image_hit_at_k"] for row in evaluations])),
            "query_count": len(evaluations),
            "judge_cache_hits": cache_hits,
            "judge_unique_misses": len(pending),
            "judge_new_response_parse_counts": parse_counts,
            "judge_parse_failure_count": parse_counts["failed"],
            "judge_response_file": str(response_path) if response_path else None,
            "evaluations": evaluations,
        }


def seeded_query_sample(qa: list[dict[str, Any]], seed: int, count: int) -> list[dict[str, Any]]:
    if count > len(qa):
        raise ValueError(f"sample size {count} exceeds dataset size {len(qa)}")
    return random.Random(seed).sample(qa, count)


def clean_evaluation_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key != "evaluations"}


def aggregate_parse_counts(trajectories: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"json": 0, "regex": 0, "failed": 0}
    for row in trajectories:
        for status, count in row["evaluation"]["judge_new_response_parse_counts"].items():
            counts[status] += int(count)
    return counts


def run_jit(
    root: Path,
    candidates: list[SearchConfig],
    initial_batch: list[SearchConfig],
    evaluator: ConfigEvaluator,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    controller_root = root / "jit"
    controller_root.mkdir(parents=True, exist_ok=False)
    memory = MemoryV2()
    rng = random.Random(seed)
    tried: set[str] = set()
    trajectories: list[dict[str, Any]] = []
    rounds = budget // 5
    for round_number in range(1, rounds + 1):
        remaining = [cfg for cfg in candidates if cfg.config_id not in tried]
        if round_number == 1:
            batch = list(initial_batch)
            diagnostics = {
                cfg.config_id: {"selection": "shared_space_filling"} for cfg in batch
            }
        else:
            batch, diagnostics = memory.select_jit(remaining, rng)
        if len(batch) != 5 or len({cfg.config_id for cfg in batch}) != 5:
            raise RuntimeError("JIT batch is not five unique configs")
        round_rows: list[dict[str, Any]] = []
        for slot, cfg in enumerate(batch, start=1):
            if cfg.config_id in tried:
                raise RuntimeError(f"JIT selected duplicate config {cfg.config_id}")
            LOGGER.info(
                "jit round=%s/%s slot=%s/5 selection=%s config=%s",
                round_number,
                rounds,
                slot,
                diagnostics[cfg.config_id]["selection"],
                cfg.config_id,
            )
            evaluation = evaluator.evaluate(cfg)
            episodic = memory.add(
                cfg, float(evaluation["mean_retrieval_support"]), round_number
            )
            append_jsonl(controller_root / "episodic_memory.jsonl", episodic)
            row = {
                "round": round_number,
                "slot": slot,
                "config": cfg.as_dict(),
                "selection": diagnostics[cfg.config_id],
                "evaluation": clean_evaluation_summary(evaluation),
            }
            append_jsonl(controller_root / "trajectory.jsonl", row)
            trajectories.append(row)
            round_rows.append({**row, "query_evaluations": evaluation["evaluations"]})
            tried.add(cfg.config_id)
        rules = memory.distill(round_number)
        atomic_json(controller_root / "semantic_memory.json", rules)
        atomic_json(
            controller_root / "rounds" / f"round_{round_number:02d}.json",
            {
                "round": round_number,
                "controller": "jit",
                "semantic_rule_count": len(rules),
                "batch": round_rows,
            },
        )
    best = max(
        trajectories,
        key=lambda row: float(row["evaluation"]["mean_retrieval_support"]),
    )
    parse_counts = aggregate_parse_counts(trajectories)
    summary = {
        "controller": "jit",
        "selection_policy": "memory_v2_absolute_scope",
        "seed": seed,
        "budget": budget,
        "rounds": rounds,
        "configs_evaluated": len(trajectories),
        "unique_configs_evaluated": len(tried),
        "best_config": best["config"],
        "best_retrieval_support": best["evaluation"]["mean_retrieval_support"],
        "best_round": best["round"],
        "final_semantic_rule_count": len(memory.semantic_rules),
        "judge_new_response_parse_counts": parse_counts,
        "judge_parse_failure_count": parse_counts["failed"],
    }
    atomic_json(controller_root / "summary.json", summary)
    return summary


def run_random(
    root: Path,
    candidates: list[SearchConfig],
    initial_batch: list[SearchConfig],
    evaluator: ConfigEvaluator,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    controller_root = root / "random"
    controller_root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    tried: set[str] = set()
    trajectories: list[dict[str, Any]] = []
    rounds = budget // 5
    for round_number in range(1, rounds + 1):
        remaining = [cfg for cfg in candidates if cfg.config_id not in tried]
        if round_number == 1:
            batch = list(initial_batch)
            selection = "shared_space_filling"
        else:
            batch = rng.sample(sorted(remaining, key=lambda item: item.config_id), 5)
            selection = "seeded_random"
        if len(batch) != 5 or len({cfg.config_id for cfg in batch}) != 5:
            raise RuntimeError("Random batch is not five unique configs")
        round_rows: list[dict[str, Any]] = []
        for slot, cfg in enumerate(batch, start=1):
            if cfg.config_id in tried:
                raise RuntimeError(f"Random selected duplicate config {cfg.config_id}")
            LOGGER.info(
                "random round=%s/%s slot=%s/5 selection=%s config=%s",
                round_number,
                rounds,
                slot,
                selection,
                cfg.config_id,
            )
            evaluation = evaluator.evaluate(cfg)
            row = {
                "round": round_number,
                "slot": slot,
                "config": cfg.as_dict(),
                "selection": {"selection": selection},
                "evaluation": clean_evaluation_summary(evaluation),
            }
            append_jsonl(controller_root / "trajectory.jsonl", row)
            trajectories.append(row)
            round_rows.append({**row, "query_evaluations": evaluation["evaluations"]})
            tried.add(cfg.config_id)
        atomic_json(
            controller_root / "rounds" / f"round_{round_number:02d}.json",
            {
                "round": round_number,
                "controller": "random",
                "batch": round_rows,
            },
        )
    best = max(
        trajectories,
        key=lambda row: float(row["evaluation"]["mean_retrieval_support"]),
    )
    parse_counts = aggregate_parse_counts(trajectories)
    summary = {
        "controller": "random",
        "selection_policy": "seeded_random_without_replacement",
        "seed": seed,
        "budget": budget,
        "rounds": rounds,
        "configs_evaluated": len(trajectories),
        "unique_configs_evaluated": len(tried),
        "best_config": best["config"],
        "best_retrieval_support": best["evaluation"]["mean_retrieval_support"],
        "best_round": best["round"],
        "judge_new_response_parse_counts": parse_counts,
        "judge_parse_failure_count": parse_counts["failed"],
    }
    atomic_json(controller_root / "summary.json", summary)
    return summary


def run_llm(
    root: Path,
    candidates: list[SearchConfig],
    initial_batch: list[SearchConfig],
    evaluator: ConfigEvaluator,
    budget: int,
    seed: int,
    model: str,
) -> dict[str, Any]:
    controller_root = root / "llm"
    controller_root.mkdir(parents=True, exist_ok=False)
    controller = LLMController(seed=seed, model=model)
    tried: set[str] = set()
    trajectories: list[dict[str, Any]] = []
    selection_seconds = 0.0
    fallback_count = 0
    rounds = budget // 5
    for round_number in range(1, rounds + 1):
        remaining = [cfg for cfg in candidates if cfg.config_id not in tried]
        if round_number == 1:
            batch = list(initial_batch)
            diagnostics = {
                cfg.config_id: {"selection": "shared_space_filling"} for cfg in batch
            }
        else:
            batch, diagnostics, decision = controller.select_batch(
                remaining=remaining,
                observations=trajectories,
                round_number=round_number,
            )
            selection_seconds += float(decision["selection_seconds"])
            fallback_count += int(decision["fallback_count"])
            atomic_json(
                controller_root / "decisions" / f"round_{round_number:02d}.json",
                decision,
            )
        if len(batch) != 5 or len({cfg.config_id for cfg in batch}) != 5:
            raise RuntimeError("LLM batch is not five unique configs")
        round_rows: list[dict[str, Any]] = []
        for slot, cfg in enumerate(batch, start=1):
            if cfg.config_id in tried:
                raise RuntimeError(f"LLM selected duplicate config {cfg.config_id}")
            LOGGER.info(
                "llm round=%s/%s slot=%s/5 selection=%s config=%s",
                round_number,
                rounds,
                slot,
                diagnostics[cfg.config_id]["selection"],
                cfg.config_id,
            )
            evaluation = evaluator.evaluate(cfg)
            row = {
                "round": round_number,
                "slot": slot,
                "config": cfg.as_dict(),
                "selection": diagnostics[cfg.config_id],
                "evaluation": clean_evaluation_summary(evaluation),
            }
            append_jsonl(controller_root / "trajectory.jsonl", row)
            trajectories.append(row)
            round_rows.append({**row, "query_evaluations": evaluation["evaluations"]})
            tried.add(cfg.config_id)
        atomic_json(
            controller_root / "rounds" / f"round_{round_number:02d}.json",
            {"round": round_number, "controller": "llm", "batch": round_rows},
        )
    best = max(
        trajectories,
        key=lambda row: float(row["evaluation"]["mean_retrieval_support"]),
    )
    parse_counts = aggregate_parse_counts(trajectories)
    summary = {
        "controller": "llm",
        "selection_policy": "llm_complete_config_batch_from_own_history",
        "decision_model": model,
        "seed": seed,
        "budget": budget,
        "rounds": rounds,
        "configs_evaluated": len(trajectories),
        "unique_configs_evaluated": len(tried),
        "best_config": best["config"],
        "best_retrieval_support": best["evaluation"]["mean_retrieval_support"],
        "best_round": best["round"],
        "selection_seconds": selection_seconds,
        "fallback_config_count": fallback_count,
        "judge_new_response_parse_counts": parse_counts,
        "judge_parse_failure_count": parse_counts["failed"],
    }
    atomic_json(controller_root / "summary.json", summary)
    return summary


def es_client(es_url: str) -> Elasticsearch:
    kwargs: dict[str, Any] = {
        "hosts": [es_url],
        "verify_certs": settings.ES_VERIFY_CERTS,
    }
    if settings.ES_USER and settings.ES_PASSWORD:
        kwargs["basic_auth"] = (settings.ES_USER, settings.ES_PASSWORD)
    return Elasticsearch(**kwargs)


def validate_indices(es: Elasticsearch) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for backend, info in BACKENDS.items():
        for kind in ("text_index", "image_index"):
            index_name = str(info[kind])
            if not es.indices.exists(index=index_name):
                raise RuntimeError(f"missing required {kind} for {backend}: {index_name}")
            mappings = es.indices.get_mapping(index=index_name)
            props = mappings[index_name]["mappings"].get("properties", {})
            dims = int(props.get("embedding", {}).get("dims", -1))
            if dims != int(info["dims"]):
                raise RuntimeError(
                    f"embedding dimension mismatch {index_name}: {dims} != {info['dims']}"
                )
            count = int(es.count(index=index_name)["count"])
            if count != 3944:
                raise RuntimeError(f"unexpected document count {index_name}: {count} != 3944")
            checked[f"{backend}.{kind}"] = {"index": index_name, "dims": dims, "count": count}
    return checked


def static_validate(candidates: list[SearchConfig], budget: int, seeds: list[int]) -> dict[str, Any]:
    if len(candidates) != 864:
        raise RuntimeError(f"search space should contain 864 configs, got {len(candidates)}")
    if budget < 5 or budget % 5 != 0 or budget > len(candidates):
        raise ValueError("budget must be a multiple of 5 between 5 and 864")
    initial = shared_space_filling_batch(candidates)
    if len({cfg.config_id for cfg in initial}) != 5:
        raise RuntimeError("shared space-filling batch is not unique")
    probe: dict[str, Any] = {}
    for seed in seeds:
        memory = MemoryV2()
        jit_rng = random.Random(seed)
        random_rng = random.Random(seed)
        jit_tried = {cfg.config_id for cfg in initial}
        random_tried = {cfg.config_id for cfg in initial}
        for cfg in initial:
            # Deterministic absolute scores exercise all post-first-round JIT paths.
            synthetic_score = (
                0.4
                + 0.03 * cfg.param_idx[0]
                - 0.02 * cfg.param_idx[1]
                + 0.01 * cfg.param_idx[2]
                + 0.005 * cfg.param_idx[3]
                + 0.015 * cfg.param_idx[4]
            )
            memory.add(cfg, synthetic_score, 1)
        memory.distill(1)
        for round_number in range(2, budget // 5 + 1):
            jit_remaining = [cfg for cfg in candidates if cfg.config_id not in jit_tried]
            jit_batch, _ = memory.select_jit(jit_remaining, jit_rng)
            if len(jit_batch) != 5 or any(cfg.config_id in jit_tried for cfg in jit_batch):
                raise RuntimeError("static JIT no-replacement check failed")
            for cfg in jit_batch:
                synthetic_score = 0.5 + 0.01 * ((sum(cfg.param_idx) + round_number) % 11)
                memory.add(cfg, synthetic_score, round_number)
                jit_tried.add(cfg.config_id)
            memory.distill(round_number)
            random_remaining = [cfg for cfg in candidates if cfg.config_id not in random_tried]
            random_batch = random_rng.sample(
                sorted(random_remaining, key=lambda item: item.config_id), 5
            )
            if len(random_batch) != 5 or any(cfg.config_id in random_tried for cfg in random_batch):
                raise RuntimeError("static random no-replacement check failed")
            random_tried.update(cfg.config_id for cfg in random_batch)
        if len(jit_tried) != budget or len(random_tried) != budget:
            raise RuntimeError("static trajectory budget check failed")
        probe[str(seed)] = {
            "jit_unique_configs": len(jit_tried),
            "random_unique_configs": len(random_tried),
            "semantic_rule_count": len(memory.semantic_rules),
        }
    return {
        "search_space_count": len(candidates),
        "budget": budget,
        "rounds": budget // 5,
        "shared_space_filling_batch": [cfg.as_dict() for cfg in initial],
        "seed_checks": probe,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent Memory-v2 JIT versus seeded Random RQ1 search"
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/train_0828",
        help="directory containing qa_search.json and corpus.json",
    )
    parser.add_argument("--qa-file", default="qa_search.json")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--es-url", default="http://127.0.0.1:19200")
    parser.add_argument("--judge-model", default=settings.SEARCH_JUDGE_MODEL)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--llm-model", default=os.getenv("AO_LLM_CONTROLLER_MODEL", "qwen3.8-max"))
    parser.add_argument("--eval-concurrency", type=int, default=settings.EVAL_CONCURRENCY)
    parser.add_argument("--embedding-device", default=os.getenv("AO_EMBEDDING_DEVICE", "cuda"))
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--query-embeddings-from",
        default=None,
        help="copy compatible cached 200-query vectors from this directory before evaluation",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="perform static no-replacement checks and validate all six ES indexes, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform static no-replacement checks without ES or model/LLM calls, then exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_concurrency < 1 or args.embedding_batch_size < 1:
        raise ValueError("concurrency and embedding batch size must be positive")
    candidates = all_configs()
    static = static_validate(candidates, args.budget, args.seeds)
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", **static}, ensure_ascii=False, indent=2))
        return
    es = es_client(args.es_url)
    try:
        indices = validate_indices(es)
        if args.validate:
            print(
                json.dumps(
                    {"mode": "validate", **static, "indices": indices},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        qa, _ = load_dataset(args.dataset_dir, args.qa_file)
        if len(qa) != 200:
            raise RuntimeError(
                f"this protocol requires exactly 200 source queries, found {len(qa)}"
            )
        if args.query_count != 100:
            raise ValueError("this protocol fixes --query-count at 100")
        run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_rq1_memory_v2_budget50"
        root = Path(args.runs_dir) / run_id
        if root.exists():
            raise FileExistsError(
                f"refusing to write into existing run directory: {root}; choose another --run-id"
            )
        root.mkdir(parents=True)
        if args.query_embeddings_from:
            source_dir = Path(args.query_embeddings_from).resolve()
            destination_dir = root / "query_embeddings"
            destination_dir.mkdir(parents=True, exist_ok=False)
            for backend in EMBEDDING_VALUES:
                source = source_dir / f"{backend}.npz"
                if not source.is_file():
                    raise FileNotFoundError(f"missing query embedding cache: {source}")
                shutil.copyfile(source, destination_dir / source.name)
        initial = shared_space_filling_batch(candidates)
        controllers = ["jit", "random"] + (["llm"] if args.include_llm else [])
        manifest = {
            "experiment": "RQ1 Memory-v2 JIT versus seeded Random" + (" versus LLM" if args.include_llm else ""),
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "qa_file": args.qa_file,
            "source_query_count": len(qa),
            "query_count_per_config": args.query_count,
            "seeds": args.seeds,
            "controllers": controllers,
            "budget_per_controller": args.budget,
            "rounds_per_controller": args.budget // 5,
            "batch_size": 5,
            "candidate_count": len(candidates),
            "search_space_version": SEARCH_SPACE_VERSION,
            "metric_schema": METRIC_SCHEMA,
            "judge_cache_schema": JUDGE_CACHE_SCHEMA,
            "judge_model": args.judge_model,
            "llm_controller_model": args.llm_model if args.include_llm else None,
            "judge_response_files": "seed_<seed>/judge_responses/<config_id>.json",
            "query_embeddings_source": str(source_dir) if args.query_embeddings_from else None,
            "es_url": args.es_url,
            "indices": indices,
            "shared_space_filling_batch": [cfg.as_dict() for cfg in initial],
            "jit_memory_isolated_per_seed": True,
            "random_memory": "none",
            "notes": (
                "Each controller samples only from its own remaining configurations. "
                "The controllers share query samples, the first five "
                "space-filling configs, raw retrieval cache, and evidence-fingerprint "
                "judge cache; none of these are controller memory. "
                "The LLM observes only its own evaluated configurations."
            ),
        }
        atomic_json(root / "manifest.json", manifest)
        embedding_cache = QueryEmbeddingCache(
            root=root,
            qa=qa,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
        # Compute vectors once for the 200-source-query universe.  Samples select
        # from this run-local immutable cache; model objects are unloaded between backends.
        for backend in EMBEDDING_VALUES:
            embedding_cache.ensure(backend)
        paired: list[dict[str, Any]] = []
        for seed in args.seeds:
            seed_root = root / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=False)
            sample = seeded_query_sample(qa, seed, args.query_count)
            atomic_json(
                seed_root / "query_sample.json",
                {
                    "seed": seed,
                    "sample_size": len(sample),
                    "queries": [
                        {
                            "id": str(item.get("id", "")),
                            "query_key": embedding_cache.query_key(item),
                            "query": str(item.get("query", "")),
                            "reference": reference_for(item),
                        }
                        for item in sample
                    ],
                },
            )
            retrieval = RetrievalCache(seed_root / "retrieval_cache.json", es, embedding_cache)
            judge_cache = JudgeCache(seed_root / "judge_cache.json", args.judge_model)
            evaluator = ConfigEvaluator(
                queries=sample,
                retrieval=retrieval,
                judge_cache=judge_cache,
                judge_model=args.judge_model,
                concurrency=args.eval_concurrency,
            )
            # Both executions begin from a clean controller directory.  JIT only
            # observes its local episodic rows; Random never receives memory.
            jit = run_jit(
                seed_root,
                candidates,
                initial,
                evaluator,
                args.budget,
                seed,
            )
            random_summary = run_random(
                seed_root,
                candidates,
                initial,
                evaluator,
                args.budget,
                seed,
            )
            llm_summary = (
                run_llm(
                    seed_root,
                    candidates,
                    initial,
                    evaluator,
                    args.budget,
                    seed,
                    args.llm_model,
                )
                if args.include_llm else None
            )
            pair = {
                "seed": seed,
                "query_count": len(sample),
                "shared_first_round": [cfg.config_id for cfg in initial],
                "jit": jit,
                "random": random_summary,
                "best_support_difference_jit_minus_random": float(
                    jit["best_retrieval_support"] - random_summary["best_retrieval_support"]
                ),
            }
            if llm_summary is not None:
                pair["llm"] = llm_summary
                pair["best_support_difference_jit_minus_llm"] = float(
                    jit["best_retrieval_support"] - llm_summary["best_retrieval_support"]
                )
            atomic_json(seed_root / "paired_summary.json", pair)
            paired.append(pair)
        aggregate = {
            "run_id": run_id,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "budget_per_controller": args.budget,
            "query_count_per_config": args.query_count,
            "seeds": args.seeds,
            "pairs": paired,
            "judge_new_response_parse_counts": {
                status: sum(
                    pair[controller]["judge_new_response_parse_counts"][status]
                    for pair in paired
                    for controller in controllers
                )
                for status in ("json", "regex", "failed")
            },
            "judge_parse_failure_count": sum(
                pair[controller]["judge_parse_failure_count"]
                for pair in paired
                for controller in controllers
            ),
            "mean_best_support_difference_jit_minus_random": float(
                np.mean(
                    [
                        pair["best_support_difference_jit_minus_random"]
                        for pair in paired
                    ]
                )
            ),
        }
        if args.include_llm:
            aggregate["mean_best_support_difference_jit_minus_llm"] = float(
                np.mean([pair["best_support_difference_jit_minus_llm"] for pair in paired])
            )
        atomic_json(root / "summary.json", aggregate)
        LOGGER.info("completed run_id=%s summary=%s", run_id, root / "summary.json")
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    finally:
        es.close()


if __name__ == "__main__":
    main()
