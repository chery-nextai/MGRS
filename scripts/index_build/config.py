"""离线建库（写 ES 索引）所需的配置。

只保留建库链路真正用到的字段，全部由环境变量提供，仓库内不写死机器相关路径或密钥。
注意：这份配置与主包 ``mgrs/config.py``（查询/搜索侧）**不是同一份**——建库侧的
``PipelineConfig`` 带 ``embedding_backend`` 维度，两者字段已经分叉，不要合并。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# scripts/index_build/config.py → 仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Elasticsearch ---
ES_URL = os.getenv("AO_EXPLORE_ES_URL", os.getenv("ES_URL", "http://127.0.0.1:9200"))
ES_USER = os.getenv("ES_USER", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "true").lower() in {"1", "true", "yes", "on"}
# 建库脚本会把「索引前缀」写进这两个变量：
#   <prefix>_text_embedding_v2  /  <prefix>_image_embedding_v2
TEXT_INDEX_NAME = os.getenv("AO_EXPLORE_TEXT_INDEX", "")
IMAGE_INDEX_NAME = os.getenv("AO_EXPLORE_IMAGE_INDEX", "")

# --- 编码器 ---
EMBEDDING_MODEL = os.getenv("AO_EMBEDDING_MODEL", "")
EMBEDDING_DEVICE = os.getenv("AO_EMBEDDING_DEVICE", "cpu")
IMAGE_EMBEDDING_BACKEND = os.getenv("AO_IMAGE_EMBEDDING_BACKEND", "qwen3_vl")
IMAGE_EMBEDDING_MODEL = os.getenv("AO_IMAGE_EMBEDDING_MODEL", "qwen3-vl-embedding")
IMAGE_EMBEDDING_MAX_LENGTH = max(1, int(os.getenv("AO_IMAGE_EMBEDDING_MAX_LENGTH", "8192")))
IMAGE_EMBEDDING_MAX_PIXELS = max(
    32 * 32, int(os.getenv("AO_IMAGE_EMBEDDING_MAX_PIXELS", str(1800 * 32 * 32)))
)
# 旧 ScienceQA 语料的图片根目录（AO 格式语料自带图像，用不到这一项）。
SCIENCEQA_IMAGE_ROOT = os.getenv(
    "SCIENCEQA_IMAGE_ROOT",
    str(PROJECT_ROOT / "datasets" / "data" / "train-0825"),
)

# 各后端的编码器权重目录（本地离线加载）。默认在仓库根的 models/ 下按名字查找，
# 可用环境变量覆盖：
#   MGRS_EMBED_MODEL_ROOT            权重公共根目录，默认 <repo>/models
#   MGRS_EMBED_MODEL_DIR_<BACKEND>   单个后端的权重目录（后端名大写）
_EMBED_MODEL_ROOT = Path(
    os.getenv("MGRS_EMBED_MODEL_ROOT", "").strip() or (PROJECT_ROOT / "models")
)


def _embed_model_dir(backend: str, default_name: str) -> str:
    override = os.getenv(f"MGRS_EMBED_MODEL_DIR_{backend.upper()}", "").strip()
    return override or str(_EMBED_MODEL_ROOT / default_name)


EMBEDDING_BACKENDS = {
    "jina_nano": _embed_model_dir("jina_nano", "jinaai--jina-embeddings-v5-omni-nano"),
    "jina_small": _embed_model_dir("jina_small", "jinaai--jina-embeddings-v5-omni-small"),
    "qwen3_vl_2b": _embed_model_dir("qwen3_vl_2b", "Qwen3-VL-Embedding-2B"),
}


@dataclass(frozen=True)
class PipelineConfig:
    config_id: str
    text_final_top_k: int
    image_top_k: int
    embedding_backend: str = "jina_nano"
    chunk_size: int = 512
    chunk_overlap: int = 64
    vector_top_k: int = 20
    es_top_k: int = 20
    rrf_k: int = 60
    text_weight: float = 1.2


def candidate_configs(embedding_backends: list[str] | None = None) -> list[PipelineConfig]:
    """建库侧只用于校验配置命名口径；搜索侧的空间定义见 mgrs/pipeline/rq1_memory_v2_jit_random.py。"""
    backends = embedding_backends or list(EMBEDDING_BACKENDS)
    rows: list[PipelineConfig] = []
    for backend in backends:
        for text_k in (5, 10, 20, 40):
            for image_k in (1, 3, 5, 10, 20):
                for text_weight in (0.8, 1.2, 1.8):
                    weight_id = str(text_weight).replace(".", "p")
                    rows.append(
                        PipelineConfig(
                            config_id=f"cfg_{backend}_text{text_k}_image{image_k}_weight{weight_id}",
                            text_final_top_k=text_k,
                            image_top_k=image_k,
                            embedding_backend=backend,
                            text_weight=text_weight,
                        )
                    )
    return rows
