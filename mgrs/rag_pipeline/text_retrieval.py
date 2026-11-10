from __future__ import annotations

import re
import logging
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from mgrs import config as settings
from mgrs.rag_pipeline.config import PipelineConfig

logger = logging.getLogger("ao_explore.text_retrieval")


def split_text(text: str, chunk_size: int, overlap: int, tokenizer: Any) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max(1, chunk_size - min(overlap, chunk_size - 1))
    chunks = []
    start = 0
    while start < len(ids):
        piece = tokenizer.decode(ids[start : start + chunk_size], skip_special_tokens=True).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(ids):
            break
        start += step
    return chunks


class TextRetriever:
    """AO-style BCE vector + Elasticsearch text retrieval with Weighted RRF."""

    def __init__(self, corpus: list[dict[str, Any]], index_name: str, config: PipelineConfig):
        if not settings.EMBEDDING_MODEL:
            raise RuntimeError("AO_EMBEDDING_MODEL 未配置；请先在 .env 中设置文本编码器路径（见 .env.example）。")
        self.index_name = index_name
        self.config = config
        self.documents: list[dict[str, Any]] = []
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
        tokenizer = self.encoder.tokenizer
        for row in corpus:
            for part_index, text in enumerate(split_text(row.get("content", ""), config.chunk_size, config.chunk_overlap, tokenizer)):
                self.documents.append(
                    {
                        "id": f"{row['id']}_chunk_{part_index}",
                        "text": text,
                        "metadata": {"source_id": str(row["id"]), "image_path": str(row.get("image_path") or "")},
                    }
                )
        if not self.documents:
            raise ValueError("corpus.json 没有可用文本内容")
        logger.info("text corpus_count=%s text_chunk_count=%s chunk_size_tokens=%s chunk_overlap_tokens=%s tokenizer=%s model=%s", len(corpus), len(self.documents), config.chunk_size, config.chunk_overlap, tokenizer.__class__.__name__, settings.EMBEDDING_MODEL)
        kwargs: dict[str, Any] = {"hosts": [settings.ES_URL], "verify_certs": settings.ES_VERIFY_CERTS}
        if settings.ES_USER and settings.ES_PASSWORD:
            kwargs["basic_auth"] = (settings.ES_USER, settings.ES_PASSWORD)
        self.es = Elasticsearch(**kwargs)
        self._ensure_index()

    def _ensure_index(self) -> None:
        meta = {"schema_version": "text_dense_v1", "embedding_model": settings.EMBEDDING_MODEL, "embedding_backend": "sentence_transformers_bce", "chunk_size_tokens": self.config.chunk_size, "chunk_overlap_tokens": self.config.chunk_overlap, "document_count": len(self.documents)}
        if self.es.indices.exists(index=self.index_name):
            mapping = self.es.indices.get_mapping(index=self.index_name)[self.index_name]["mappings"]
            props = mapping.get("properties", {})
            if props.get("embedding", {}).get("type") != "dense_vector" or mapping.get("_meta") != meta:
                raise RuntimeError(f"文本 index {self.index_name} 已存在但 schema/元信息不匹配；请更换 AO_EXPLORE_TEXT_INDEX")
            count = self.es.count(index=self.index_name)["count"]
            if count != len(self.documents):
                raise RuntimeError(f"文本 index {self.index_name} 文档数 {count} != {len(self.documents)}；请更换 AO_EXPLORE_TEXT_INDEX")
            logger.info("reuse text index=%s documents=%s", self.index_name, count)
            return
        self.es.indices.create(
            index=self.index_name,
            mappings={
                "_meta": meta,
                "properties": {
                    "content": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                    "embedding": {"type": "dense_vector", "dims": int(self.encoder.get_sentence_embedding_dimension()), "index": True, "similarity": "cosine"},
                }
            },
        )
        actions = [
            {
                "_index": self.index_name,
                "_id": row["id"],
                "_source": {"content": row["text"], "metadata": row["metadata"], "embedding": vector.tolist()},
            }
            for row, vector in zip(self.documents, self.encoder.encode([row["text"] for row in self.documents], normalize_embeddings=True, show_progress_bar=False), strict=True)
        ]
        helpers.bulk(self.es, actions, refresh="wait_for")

    def retrieve(self, query: str, config: PipelineConfig) -> list[dict[str, Any]]:
        query_vector = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        response = self.es.search(index=self.index_name, knn={"field": "embedding", "query_vector": query_vector, "k": config.vector_top_k, "num_candidates": max(config.vector_top_k, 100)}, size=config.vector_top_k, source=["content", "metadata"])
        vector_rows = [{"id": str(hit.get("_id") or ""), "text": str((hit.get("_source") or {}).get("content") or ""), "metadata": dict((hit.get("_source") or {}).get("metadata") or {}), "score": float(hit.get("_score") or 0.0), "route": "bce_vector"} for hit in response["hits"]["hits"]]
        response = self.es.search(
            index=self.index_name,
            size=config.es_top_k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "metadata.source_id"],
                    "type": "best_fields",
                }
            },
            source=["content", "metadata"],
        )
        text_rows = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source") or {}
            text_rows.append(
                {
                    "id": str(hit.get("_id") or ""),
                    "text": str(source.get("content") or ""),
                    "metadata": dict(source.get("metadata") or {}),
                    "score": float(hit.get("_score") or 0.0),
                    "route": "es_text",
                }
            )
        merged: dict[str, dict[str, Any]] = {}
        for route, rows, weight in (
            ("bce_vector", vector_rows, 1.0),
            ("es_text", text_rows, config.text_weight),
        ):
            for rank, row in enumerate(rows, start=1):
                item = merged.setdefault(row["id"], {**row, "rrf_score": 0.0, "routes": []})
                item["rrf_score"] += weight / (config.rrf_k + rank)
                item["routes"].append({"route": route, "rank": rank, "score": row["score"]})
        ranked = sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)
        return [
            {**row, "rank": rank, "score": round(float(row["rrf_score"]), 8)}
            for rank, row in enumerate(ranked[: config.text_final_top_k], 1)
        ]

    def close(self) -> None:
        self.es.close()
