from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch, helpers
from PIL import Image

from index_build import config as settings
from index_build.rag_pipeline.multimodal_encoder import MultimodalEncoder

logger = logging.getLogger("ao_explore.image_retrieval")


def resolve_image_path(value: str, sample_id: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if settings.SCIENCEQA_IMAGE_ROOT:
        root = Path(settings.SCIENCEQA_IMAGE_ROOT)
        candidates = [root / "train" / sample_id / path.name, root / sample_id / path.name, root / path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"找不到图片 {value}；请设置 SCIENCEQA_IMAGE_ROOT，或修正 corpus.json 的 image_path。"
    )


class ImageRetriever:
    """
    图像检索器：使用本地多模态模型编码图像。
    支持 Jina v5 omni 和 Qwen3-VL-2B 等多模态模型。
    """

    def __init__(self, corpus: list[dict[str, Any]], embedding_backend: str = "jina_nano"):
        self.embedding_backend = embedding_backend
        self.rows: list[dict[str, Any]] = []

        # 加载图像
        for row in corpus:
            path = resolve_image_path(str(row.get("image_path") or ""), str(row["id"]))
            image = Image.open(path).convert("RGB")
            self.rows.append({"id": str(row["id"]), "path": str(path), "image": image})

        # 初始化多模态 encoder
        self.encoder = MultimodalEncoder(embedding_backend, device=settings.EMBEDDING_DEVICE)

        # ES 连接
        # ES 偶发重启/连接重置会让整轮搜索前功尽弃，这里加大重试与超时。
        kwargs: dict[str, Any] = {"hosts": [settings.ES_URL], "verify_certs": settings.ES_VERIFY_CERTS, "max_retries": 5, "retry_on_timeout": True, "request_timeout": 60}
        if settings.ES_USER and settings.ES_PASSWORD:
            kwargs["basic_auth"] = (settings.ES_USER, settings.ES_PASSWORD)
        self.es = Elasticsearch(**kwargs)

        # 使用带 embedding_backend 标识的 index 名
        self.index_name = f"{settings.IMAGE_INDEX_NAME}_{embedding_backend}"

        logger.info("loaded images count=%s backend=%s index=%s", len(self.rows), embedding_backend, self.index_name)
        self.vectors = None
        self._ensure_index()
        # 图像只在建库时需要；多 backend 并存时保留 PIL 对象会成倍占用内存。
        self._release_images()

    def _release_images(self) -> None:
        for row in self.rows:
            row.pop("image", None)
        self.vectors = None

    def _ensure_index(self) -> None:
        """确保 ES index 存在并编码图像"""
        embedding_dim = self.encoder.get_embedding_dimension()
        meta = {
            "schema_version": "image_dense_v2_multimodal",
            "embedding_backend": self.embedding_backend,
            "document_count": len(self.rows),
        }

        if self.es.indices.exists(index=self.index_name):
            mapping = self.es.indices.get_mapping(index=self.index_name)[self.index_name]["mappings"]
            if mapping.get("properties", {}).get("embedding", {}).get("type") != "dense_vector" or mapping.get("_meta") != meta:
                raise RuntimeError(f"图片 index {self.index_name} 已存在但 schema/元信息不匹配；请更换 index 名称")
            count = self.es.count(index=self.index_name)["count"]
            if count != len(self.rows):
                raise RuntimeError(f"图片 index {self.index_name} 文档数 {count} != {len(self.rows)}；请更换 index 名称")
            self.vectors = None
            logger.info("reuse image index=%s documents=%s", self.index_name, count)
            return

        # 编码所有图像
        logger.info("encoding %s images with backend=%s", len(self.rows), self.embedding_backend)
        images = [row["image"] for row in self.rows]
        self.vectors = self.encoder.encode_image(images, is_query=False)
        logger.info("image encoding complete, shape=%s", self.vectors.shape)

        # 创建 index
        self.es.indices.create(
            index=self.index_name,
            mappings={
                "_meta": meta,
                "properties": {
                    "source_id": {"type": "keyword"},
                    "image_path": {"type": "keyword"},
                    "embedding": {"type": "dense_vector", "dims": embedding_dim, "index": True, "similarity": "cosine"},
                }
            },
        )

        # 写入文档
        actions = [
            {
                "_index": self.index_name,
                "_id": row["id"],
                "_source": {
                    "source_id": row["id"],
                    "image_path": row["path"],
                    "embedding": vector.tolist(),
                },
            }
            for row, vector in zip(self.rows, self.vectors, strict=True)
        ]
        helpers.bulk(self.es, actions, refresh="wait_for")
        logger.info("built image index=%s documents=%s", self.index_name, len(actions))

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """用文本 query 检索图像"""
        if top_k <= 0:
            return []

        # 编码文本 query
        query_vector = self.encoder.encode_text([query], is_query=True)[0]

        # ES KNN 检索
        response = self.es.search(
            index=self.index_name,
            knn={
                "field": "embedding",
                "query_vector": query_vector.tolist(),
                "k": top_k,
                "num_candidates": max(top_k, 20),
            },
            size=top_k,
            source=["source_id", "image_path"],
        )

        return [
            {
                "rank": rank,
                "id": str(hit.get("_source", {}).get("source_id") or hit.get("_id") or ""),
                "image_path": str(hit.get("_source", {}).get("image_path") or ""),
                "score": round(float(hit.get("_score") or 0.0), 8),
            }
            for rank, hit in enumerate(response["hits"]["hits"], start=1)
        ]

    def close(self) -> None:
        self.es.close()
