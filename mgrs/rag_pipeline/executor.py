from __future__ import annotations

from typing import Any
from mgrs import config as settings

from mgrs import config as settings
from mgrs.rag_pipeline.config import PipelineConfig
from mgrs.rag_pipeline.image_retrieval import ImageRetriever
from mgrs.rag_pipeline.llm import MultimodalLLM
from mgrs.rag_pipeline.text_retrieval import TextRetriever


class MultimodalRAGPipeline:
    def __init__(self, corpus: list[dict[str, Any]], index_name: str, base_config: PipelineConfig):
        self.text = TextRetriever(corpus, settings.TEXT_INDEX_NAME, base_config)
        self.image = ImageRetriever(corpus)
        self.llm = MultimodalLLM()

    def retrieve(self, query: str, reference: str, config: PipelineConfig, sample_id: str = "") -> dict[str, Any]:
        text_rows = self.text.retrieve(query, config)
        image_rows = self.image.retrieve(query, config.image_top_k)
        return {
            "query": query,
            "reference": reference,
            "text_evidence": text_rows,
            "image_evidence": image_rows,
            "text_hit_at_k": int(any(str(row.get("metadata", {}).get("source_id")) == str(sample_id) for row in text_rows)),
            "image_hit_at_k": int(any(str(row.get("id")) == str(sample_id) for row in image_rows)),
        }

    def evaluate_retrieval(self, query: str, reference: str, config: PipelineConfig, sample_id: str = "") -> dict[str, Any]:
        result = self.retrieve(query, reference, config, sample_id)
        result["retrieval_support_score"] = self.llm.judge_retrieval(
            query, reference, result["text_evidence"], result["image_evidence"]
        )
        return result

    def generate_answer(self, result: dict[str, Any]) -> dict[str, Any]:
        result = dict(result)
        result["answer"] = self.llm.generate_answer(result["query"], result["text_evidence"], result["image_evidence"])
        return result

    def check_retrieval(self, result: dict[str, Any]) -> dict[str, Any]:
        result = dict(result)
        result["retrieval_support_score"] = self.llm.judge_retrieval(
            result["query"], result["reference"], result["text_evidence"], result["image_evidence"], model=settings.LLM_MODEL
        )
        return result

    def run(self, query: str, reference: str, config: PipelineConfig, sample_id: str = "") -> dict[str, Any]:
        return self.generate_answer(self.check_retrieval(self.retrieve(query, reference, config, sample_id)))

    def close(self) -> None:
        self.text.close()
        self.image.close()
