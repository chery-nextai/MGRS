"""全局配置：全部项都由环境变量提供，仓库内不写死任何机器相关路径或密钥。

完整变量清单与示例见仓库根的 .env.example。判题（DQCR）端点不在此处，
见 mgrs/judge_client.py。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.getenv("AO_EXPLORE_RUNS_DIR", str(PROJECT_ROOT / "runs")))
MEMORY_PATH = Path(os.getenv("AO_EXPLORE_MEMORY_PATH", str(PROJECT_ROOT / "memory" / "experiences.jsonl")))
SUPPORT_CACHE_PATH = Path(os.getenv("AO_EXPLORE_SUPPORT_CACHE_PATH", str(PROJECT_ROOT / "memory" / "support_scores.json")))
ES_URL = os.getenv("AO_EXPLORE_ES_URL", os.getenv("ES_URL", "http://127.0.0.1:9200"))
ES_USER = os.getenv("ES_USER", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
TEXT_INDEX_NAME = os.getenv("AO_EXPLORE_TEXT_INDEX", "aoexplore_scienceqa_text_v1")
IMAGE_INDEX_NAME = os.getenv("AO_EXPLORE_IMAGE_INDEX", "aoexplore_scienceqa_image_v1")
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "true").lower() in {"1", "true", "yes", "on"}
EMBEDDING_MODEL = os.getenv("AO_EMBEDDING_MODEL", "")
EMBEDDING_DEVICE = os.getenv("AO_EMBEDDING_DEVICE", "cpu")
CLIP_MODEL = os.getenv("AO_CLIP_MODEL", "")
CLIP_DEVICE = os.getenv("AO_CLIP_DEVICE", os.getenv("AO_EMBEDDING_DEVICE", "cpu"))
IMAGE_EMBEDDING_BACKEND = os.getenv("AO_IMAGE_EMBEDDING_BACKEND", "qwen3_vl")
IMAGE_EMBEDDING_MODEL = os.getenv("AO_IMAGE_EMBEDDING_MODEL", "qwen3-vl-embedding")
IMAGE_EMBEDDING_MAX_LENGTH = max(1, int(os.getenv("AO_IMAGE_EMBEDDING_MAX_LENGTH", "8192")))
IMAGE_EMBEDDING_MAX_PIXELS = max(32 * 32, int(os.getenv("AO_IMAGE_EMBEDDING_MAX_PIXELS", str(1800 * 32 * 32))))
LLM_API_BASE = os.getenv("LLM_API_BASE", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("AO_JUDGE_MODEL", "")
SEARCH_JUDGE_MODEL = os.getenv("AO_SEARCH_JUDGE_MODEL", "qwen3.7-flash")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
LLM_MAX_ATTEMPTS = max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "3")))
LLM_RETRY_BACKOFF = max(0.0, float(os.getenv("LLM_RETRY_BACKOFF", "2")))
EVAL_CONCURRENCY = max(1, int(os.getenv("AO_EXPLORE_EVAL_CONCURRENCY", "3")))
FINAL_CHECK_TOP_N = max(1, int(os.getenv("AO_EXPLORE_FINAL_CHECK_TOP_N", "3")))
SCIENCEQA_IMAGE_ROOT = os.getenv(
    "SCIENCEQA_IMAGE_ROOT",
    str(PROJECT_ROOT / "datasets" / "data" / "train-0825"),
)

SEARCH_SPACE_VERSION = "mm_rag_v2_topk_weight"
METRIC_VERSION = "retrieval_support_ground_truth_mm_v2"


@dataclass(frozen=True)
class PipelineConfig:
    config_id: str
    text_final_top_k: int
    image_top_k: int
    chunk_size: int = 512
    chunk_overlap: int = 64
    vector_top_k: int = 20
    es_top_k: int = 20
    rrf_k: int = 60
    text_weight: float = 1.2


def candidate_configs() -> list[PipelineConfig]:
    rows: list[PipelineConfig] = []
    for text_k in (5, 10, 20, 40):
        for image_k in (0, 1, 3, 5, 10, 20):
            for text_weight in (0.8, 1.2, 1.8):
                weight_id = str(text_weight).replace(".", "p")
                rows.append(
                    PipelineConfig(
                        config_id=f"cfg_text{text_k}_image{image_k}_weight{weight_id}",
                        text_final_top_k=text_k,
                        image_top_k=image_k,
                        text_weight=text_weight,
                    )
                )
    return rows


def ensure_dirs() -> None:
    for path in (RUNS_DIR, MEMORY_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)

DATASET_ID = os.getenv('AO_EXPLORE_DATASET_ID', 'scienceqa')
IMAGE_ROOT = os.getenv('AO_EXPLORE_IMAGE_ROOT', SCIENCEQA_IMAGE_ROOT)
REWARD_NORMALIZATION = os.getenv('AO_REWARD_NORMALIZATION', 'zscore')

ALLOW_COMPAT_IMAGE_INDEX = os.getenv("AO_EXPLORE_ALLOW_COMPAT_IMAGE_INDEX", "true").lower() in {"1", "true", "yes", "on"}
