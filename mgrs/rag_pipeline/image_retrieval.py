from __future__ import annotations

import base64
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from elasticsearch import Elasticsearch, helpers
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from mgrs import config as settings

logger = logging.getLogger("ao_explore.image_retrieval")


def resolve_image_path(value: str, sample_id: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if settings.IMAGE_ROOT:
        root = Path(settings.IMAGE_ROOT)
        candidates = [root / "train" / sample_id / path.name, root / sample_id / path.name, root / path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"找不到图片 {value}；请设置 AO_EXPLORE_IMAGE_ROOT，或修正 corpus.json 的 image_path。"
    )


class ImageRetriever:
    """Text-to-image dense retrieval with a pluggable embedding backend.

    v0 defaults to the DashScope Qwen3-VL-Embedding endpoint.  CLIP remains
    available as an explicit fallback for ablations.
    """

    def __init__(self, corpus: list[dict[str, Any]]):
        self.backend = settings.IMAGE_EMBEDDING_BACKEND.lower()
        self.rows: list[dict[str, Any]] = []
        for row in corpus:
            try:
                path = resolve_image_path(str(row.get("image_path") or ""), str(row["id"]))
            except FileNotFoundError:
                logger.warning("skip corpus image without local file id=%s path=%s", row.get("id"), row.get("image_path"))
                continue
            image = Image.open(path).convert("RGB")
            self.rows.append({"id": str(row["id"]), "path": str(path), "image": image})
        kwargs: dict[str, Any] = {"hosts": [settings.ES_URL], "verify_certs": settings.ES_VERIFY_CERTS}
        if settings.ES_USER and settings.ES_PASSWORD:
            kwargs["basic_auth"] = (settings.ES_USER, settings.ES_PASSWORD)
        self.es = Elasticsearch(**kwargs)
        logger.info("loaded images count=%s backend=%s model=%s index=%s", len(self.rows), self.backend, settings.IMAGE_EMBEDDING_MODEL if self.backend.startswith("qwen") else settings.CLIP_MODEL, settings.IMAGE_INDEX_NAME)
        if not self.rows:
            self.vectors = None
            self.es = None
            logger.warning("no local corpus images; image retrieval disabled")
            return
        if self.backend in {"disabled", "none", "off"}:
            self.es = None
            self.vectors = None
            logger.warning("image retrieval disabled by AO_IMAGE_EMBEDDING_BACKEND=%s", self.backend)
            return
        if self.es.indices.exists(index=settings.IMAGE_INDEX_NAME):
            if self.backend in {"qwen3_vl", "qwen3-vl", "qwen"}:
                try:
                    from dashscope import MultiModalEmbedding
                except ImportError as exc:
                    raise RuntimeError("Qwen3-VL-Embedding backend 需要安装 dashscope。") from exc
                self.qwen_api = MultiModalEmbedding
            elif self.backend in {"local_qwen3_vl", "local_qwen3-vl", "local_qwen", "qwen3_vl_local"}:
                self._init_local_qwen()
            self._ensure_index()
        else:
            if self.backend in {"qwen3_vl", "qwen3-vl", "qwen"}:
                self._init_qwen()
            elif self.backend in {"local_qwen3_vl", "local_qwen3-vl", "local_qwen", "qwen3_vl_local"}:
                self._init_local_qwen()
            elif self.backend in {"clip", "siglip"}:
                self._init_clip()
            elif self.backend in {"disabled", "none", "off"}:
                return
            else:
                raise ValueError(f"不支持的 AO_IMAGE_EMBEDDING_BACKEND: {settings.IMAGE_EMBEDDING_BACKEND}")
            self._ensure_index()

    def _ensure_index(self) -> None:
        meta = {"schema_version": "image_dense_v1", "embedding_backend": self.backend, "embedding_model": settings.IMAGE_EMBEDDING_MODEL if self.backend.startswith("qwen") else settings.CLIP_MODEL, "document_count": len(self.rows)}
        if self.es.indices.exists(index=settings.IMAGE_INDEX_NAME):
            mapping = self.es.indices.get_mapping(index=settings.IMAGE_INDEX_NAME)[settings.IMAGE_INDEX_NAME]["mappings"]
            props = mapping.get("properties", {})
            existing_meta = mapping.get("_meta") or {}
            count = self.es.count(index=settings.IMAGE_INDEX_NAME)["count"]
            exact_match = props.get("embedding", {}).get("type") == "dense_vector" and existing_meta == meta
            compatible_qwen = (
                settings.ALLOW_COMPAT_IMAGE_INDEX
                and props.get("embedding", {}).get("type") == "dense_vector"
                and str(existing_meta.get("embedding_backend", "")).startswith("qwen3_vl")
                and count == len(self.rows)
            )
            if not exact_match and not compatible_qwen:
                raise RuntimeError("existing image index schema/metadata mismatch")
            if count != len(self.rows):
                raise RuntimeError("existing image index document count mismatch")
            self.vectors = None
            logger.info("reuse image index=%s documents=%s compatible=%s", settings.IMAGE_INDEX_NAME, count, compatible_qwen and not exact_match)
            return
        dims = int(self.vectors.shape[1])
        self.es.indices.create(index=settings.IMAGE_INDEX_NAME, mappings={"_meta": meta, "properties": {"source_id": {"type": "keyword"}, "image_path": {"type": "keyword"}, "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"}}})
        actions = [{"_index": settings.IMAGE_INDEX_NAME, "_id": row["id"], "_source": {"source_id": row["id"], "image_path": row["path"], "embedding": vector.tolist()}} for row, vector in zip(self.rows, self.vectors, strict=True)]
        helpers.bulk(self.es, actions, refresh="wait_for")
        logger.info("built image index=%s documents=%s", settings.IMAGE_INDEX_NAME, len(actions))

    def _init_clip(self) -> None:
        if not settings.CLIP_MODEL:
            raise RuntimeError("使用 CLIP backend 时必须配置 AO_CLIP_MODEL。")
        self.device = torch.device(settings.CLIP_DEVICE)
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL, local_files_only=True)
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL, local_files_only=True).to(self.device).eval()
        with torch.inference_mode():
            inputs = self.processor(images=[row["image"] for row in self.rows], return_tensors="pt", padding=True).to(self.device)
            vectors = self.model.get_image_features(**inputs)
            self.vectors = torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy().astype("float32")

    def _init_qwen(self) -> None:
        if not settings.LLM_API_KEY:
            raise RuntimeError("Qwen3-VL-Embedding 需要 LLM_API_KEY（见 .env.example）。")
        try:
            from dashscope import MultiModalEmbedding
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL-Embedding backend 需要安装 dashscope。") from exc
        self.qwen_api = MultiModalEmbedding
        self.vectors = self._qwen_encode_images([row["image"] for row in self.rows])

    def _init_local_qwen(self) -> None:
        """Load the offline Qwen3-VL embedder bundled with the experiment."""
        model_path = Path(settings.IMAGE_EMBEDDING_MODEL).expanduser()
        script_path = model_path / "scripts" / "qwen3_vl_embedding.py"
        if not script_path.is_file():
            raise RuntimeError(
                "本地 Qwen3-VL 模型目录无 scripts/qwen3_vl_embedding.py: "
                f"{script_path}"
            )
        module_name = "ao_explore_qwen3_vl_embedding"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载本地 Qwen embedding helper: {script_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        self.local_qwen = module.Qwen3VLEmbedder(
            str(model_path),
            max_length=settings.IMAGE_EMBEDDING_MAX_LENGTH,
            max_pixels=settings.IMAGE_EMBEDDING_MAX_PIXELS,
        )
        self.local_qwen_cache: dict[str, np.ndarray] = {}
        logger.info(
            "loaded local Qwen3-VL embedding model path=%s device=%s",
            model_path,
            self.local_qwen.model.device,
        )

    def _local_qwen_encode(self, inputs: list[dict[str, Any]]) -> np.ndarray:
        if not inputs:
            return np.empty((0, 2048), dtype="float32")
        embeddings = self.local_qwen.process(inputs, normalize=True)
        return embeddings.detach().cpu().numpy().astype("float32", copy=False)

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _qwen_call(self, contents: list[dict[str, Any]]) -> np.ndarray:
        last_error: Exception | None = None
        response = None
        for attempt in range(1, settings.LLM_MAX_ATTEMPTS + 1):
            try:
                response = self.qwen_api.call(
                    model=settings.IMAGE_EMBEDDING_MODEL,
                    input=contents,
                    api_key=settings.LLM_API_KEY,
                )
                status = int(getattr(response, "status_code", 200) or 200)
                if status < 300:
                    break
                last_error = RuntimeError(f"Qwen embedding status={status}: {getattr(response, 'message', response)}")
            except Exception as exc:
                last_error = exc
            if attempt < settings.LLM_MAX_ATTEMPTS:
                delay = settings.LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning("qwen embedding failed attempt=%s/%s error=%s; retrying", attempt, settings.LLM_MAX_ATTEMPTS, last_error)
                if delay:
                    time.sleep(delay)
        else:
            raise RuntimeError("Qwen embedding exhausted all attempts") from last_error
        if response is None:
            raise RuntimeError("Qwen embedding returned no response") from last_error
        output = getattr(response, "output", None)
        if isinstance(output, dict):
            items = output.get("embeddings") or output.get("data") or []
        else:
            items = getattr(output, "embeddings", None) or getattr(output, "data", None) or []
        vectors = []
        for item in items:
            if isinstance(item, dict):
                vector = item.get("embedding") or item.get("embeddings")
            else:
                vector = getattr(item, "embedding", None) or getattr(item, "embeddings", None)
            if vector is not None:
                vectors.append(vector)
        if len(vectors) != len(contents):
            raise RuntimeError(f"Qwen embedding vector count mismatch: requested={len(contents)} returned={len(vectors)}")
        array = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)

    def _qwen_encode_images(self, images: list[Image.Image]) -> np.ndarray:
        # DashScope Qwen3-VL-Embedding limits an image request to at most 10
        # images. Keep batching here so corpus size can grow without changing
        # the retriever's public interface.
        vectors: list[np.ndarray] = []
        for start in range(0, len(images), 10):
            batch = images[start : start + 10]
            logger.info("qwen image embedding batch=%s-%s/%s", start + 1, start + len(batch), len(images))
            vectors.append(
                self._qwen_call(
                    [{"image": self._image_data_url(image), "factor": 1.0} for image in batch]
                )
            )
            logger.info("qwen image embedding batch complete=%s-%s/%s", start + 1, start + len(batch), len(images))
        if not vectors:
            return np.empty((0, 0), dtype="float32")
        return np.concatenate(vectors, axis=0)

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not self.rows or self.es is None or self.backend in {"disabled", "none", "off"}:
            return []
        if self.backend in {"local_qwen3_vl", "local_qwen3-vl", "local_qwen", "qwen3_vl_local"}:
            cached = self.local_qwen_cache.get(query)
            if cached is None:
                logger.info("local qwen text embedding query_chars=%s", len(query))
                cached = self._local_qwen_encode([{"text": query}])[0]
                self.local_qwen_cache[query] = cached
            query_vector = cached
        elif self.backend in {"qwen3_vl", "qwen3-vl", "qwen"}:
            logger.info("qwen text embedding query_chars=%s", len(query))
            query_vector = self._qwen_call([{"text": query, "factor": 1.0}])[0]
        else:
            with torch.inference_mode():
                inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(self.device)
                query_vector = self.model.get_text_features(**inputs)
                query_vector = torch.nn.functional.normalize(query_vector, dim=-1).cpu().numpy()[0].astype("float32")
        response = self.es.search(index=settings.IMAGE_INDEX_NAME, knn={"field": "embedding", "query_vector": query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector, "k": top_k, "num_candidates": max(top_k, 20)}, size=top_k, source=["source_id", "image_path"])
        return [{"rank": rank, "id": str(hit.get("_source", {}).get("source_id") or hit.get("_id") or ""), "image_path": str(hit.get("_source", {}).get("image_path") or ""), "score": round(float(hit.get("_score") or 0.0), 8)} for rank, hit in enumerate(response["hits"]["hits"], start=1)]

    def close(self) -> None:
        if self.es is not None:
            self.es.close()
