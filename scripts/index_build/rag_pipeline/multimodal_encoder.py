"""
多模态 Encoder 模块：统一文本和图像的 embedding 编码。
支持 Jina v5 omni 和 Qwen3-VL-2B 等多模态模型。

使用 SentenceTransformer 统一接口，确保跨模态检索的正确对齐。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from index_build import config as settings

logger = logging.getLogger("ao_explore.multimodal_encoder")


class MultimodalEncoder:
    """
    统一的多模态编码器，支持文本和图像在同一向量空间中编码。
    使用 SentenceTransformer 的 encode_query / encode_document 接口。
    """

    _model_cache: dict[str, Any] = {}

    def __init__(self, backend: str, device: str | None = None):
        self.backend = backend
        self.device = device or settings.EMBEDDING_DEVICE
        self.model_path = settings.EMBEDDING_BACKENDS.get(backend)
        if not self.model_path:
            raise ValueError(f"未知的 embedding backend: {backend}")

        cache_key = f"{backend}_{self.device}"
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = self._load_model()
        self.model = self._model_cache[cache_key]

    def _load_model(self) -> SentenceTransformer:
        """加载 SentenceTransformer 模型"""
        if self.backend.startswith("jina"):
            return self._load_jina()
        elif self.backend.startswith("qwen3_vl"):
            return self._load_qwen3_vl()
        else:
            raise ValueError(f"不支持的 backend: {self.backend}")

    def _load_jina(self) -> SentenceTransformer:
        """
        加载 Jina v5 omni 模型。
        使用 SentenceTransformer 的 encode_query / encode_document 接口。
        """
        logger.info("loading Jina v5 omni model: %s (device=%s)", self.model_path, self.device)

        model = SentenceTransformer(
            self.model_path,
            trust_remote_code=True,
            model_kwargs={
                "default_task": "retrieval",
                "modality": "vision",  # 只加载 vision + text，不加载 audio
            },
            device=self.device,
        )

        logger.info("Jina model loaded, embedding dim=%d", model.get_sentence_embedding_dimension())
        return model

    def _load_qwen3_vl(self) -> SentenceTransformer:
        """
        加载 Qwen3-VL-Embedding-2B 模型。
        使用 SentenceTransformer，默认带有 instruction prompt。
        """
        logger.info("loading Qwen3-VL-Embedding model: %s (device=%s)", self.model_path, self.device)

        model = SentenceTransformer(
            self.model_path,
            trust_remote_code=True,
            device=self.device,
        )

        logger.info("Qwen3-VL model loaded, embedding dim=%d", model.get_sentence_embedding_dimension())
        return model

    def encode_text(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        """
        编码文本。
        - is_query=True: 使用 encode_query (Jina) 或带 prompt 的 encode (Qwen)
        - is_query=False: 使用 encode_document (Jina) 或直接 encode (Qwen)
        """
        if self.backend.startswith("jina"):
            if is_query:
                return self.model.encode_query(texts, show_progress_bar=False, normalize_embeddings=True)
            else:
                return self.model.encode_document(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            # Qwen3-VL: 使用 prompt 参数区分 query 和 document
            if is_query:
                return self.model.encode(
                    texts,
                    prompt="Represent the user's input for retrieval.",
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            else:
                return self.model.encode(
                    texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

    def encode_image(self, images: list[Image.Image], is_query: bool = False, image_paths: list[str] | None = None) -> np.ndarray:
        """
        编码图像。
        - is_query=True: 图像作为 query（较少用）
        - is_query=False: 图像作为 document（常见用法）
        - image_paths: 可选的图片路径列表（Qwen3-VL 需要路径而非 PIL.Image）
        """
        if self.backend.startswith("jina"):
            # Jina: encode_query / encode_document 可以直接接受 PIL.Image
            if is_query:
                return self.model.encode_query(images, show_progress_bar=False, normalize_embeddings=True)
            else:
                return self.model.encode_document(images, show_progress_bar=False, normalize_embeddings=True)
        else:
            # Qwen3-VL 图像编码。
            # 注意：jina 的 custom_st 会在 jina 模型加载时把
            # SentenceTransformer.encode 全局 patch 掉；该 patch 的多模态
            # multipart 处理只适用于 jina 自己的 Transformer，而 qwen 的
            # self[0] 是标准 ST Transformer，收到 multipart 特征后会丢弃并以
            # 空参数调用底层 qwen3_vl，报 “input_ids/inputs_embeds 缺失”。
            # 此外 ST 原生的 encode 对 qwen 图片会静默退化成文本编码。
            # 因此这里绕过 patch，直接用 processor + 模型模块编码：
            # 用结构化 image message 让模板插入真实 image token。
            import torch
            from PIL import Image as _Image

            proc = self.model[0].processor
            dev = next(self.model.parameters()).device
            instruction = "Represent the image for retrieval."
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ]}]
            prompt = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            if image_paths:
                pil_images = [_Image.open(p).convert("RGB") for p in image_paths]
            else:
                pil_images = list(images)

            embs = []
            for img in pil_images:
                inp = proc(text=prompt, images=img, return_tensors="pt", truncation=False)
                feats = {k: v.to(dev) for k, v in inp.items() if torch.is_tensor(v)}
                out = self.model[0](feats)
                idx = 1
                while True:
                    try:
                        out = self.model[idx](out)
                    except (IndexError, KeyError):
                        break
                    idx += 1
                vec = out["sentence_embedding"].detach().cpu().to(torch.float32).numpy()[0]
                embs.append(vec)

            vecs = np.stack(embs).astype("float32")
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
            vecs = vecs / (norms + 1e-12)
            return vecs

    def get_embedding_dimension(self) -> int:
        """获取 embedding 维度"""
        return self.model.get_sentence_embedding_dimension()
