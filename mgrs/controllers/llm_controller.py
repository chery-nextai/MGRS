"""LLM batch selector for the independent Memory-v2 RQ1 search protocol.

The policy sees only its own scored configurations and the public parameter
domain.  It proposes ``batch_size`` complete configurations once per round
(default 5, so every existing caller keeps its exact behaviour).  Invalid or
missing proposals are filled by seeded random selection and explicitly logged.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

from mgrs.rag_pipeline.llm import AOChatClient


LOGGER = logging.getLogger("ao_explore.rq1_llm_controller")

SYSTEM_PROMPT = """你是 ScienceQA 多模态 RAG 检索配置搜索的决策器。
目标是在有限评估预算内尽快找到 retrieval support 高的配置。
你只能依据给出的本组已评估配置与分数决策；没有其他组的结果。
每次选 5 个未评估的完整配置，适度探索不同参数组合。
只输出 JSON：{"configs": [{"text_top_k": 5, "image_top_k": 0,
"text_weight": 0.8, "rrf_k": 20, "embedding_backend": "jina_nano"}],
"reason": "简短理由"}。configs 必须恰好有 5 项，不要输出额外文字。"""


class LLMController:
    name = "llm"

    # 决策 JSON 中的参数名及其顺序（子类可覆写以适配不同的搜索空间）。
    PARAM_FIELDS: tuple[str, ...] = (
        "text_top_k",
        "image_top_k",
        "text_weight",
        "rrf_k",
        "embedding_backend",
    )

    def __init__(
        self,
        *,
        seed: int,
        model: str = "qwen3.8-max",
        metric_key: str = "mean_retrieval_support",
        objective: str = "maximize mean retrieval support with few evaluations",
        system_prompt: str = SYSTEM_PROMPT,
        batch_size: int = 5,
        request_extra: dict[str, Any] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.metric_key = metric_key
        self.objective = objective
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        # 可选请求体扩展（如 {"enable_thinking": False}）。默认 None，行为与旧版一致。
        self.request_extra = request_extra
        self.rng = random.Random(seed)
        self.client = AOChatClient()

    @classmethod
    def _key(cls, params: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(params[name] for name in cls.PARAM_FIELDS)

    def _allowed_values(self) -> dict[str, list[Any]]:
        """公开给决策模型的参数取值域（子类可覆写以适配新的搜索空间）。"""
        return {
            "text_top_k": [5, 10, 20, 40],
            "image_top_k": [0, 1, 3, 5, 10, 20],
            "text_weight": [0.8, 1.2, 1.8],
            "rrf_k": [20, 40, 60, 100],
            "embedding_backend": ["jina_nano", "jina_small", "qwen3_vl_2b"],
        }

    @staticmethod
    def _parse(content: str) -> list[Any]:
        raw = (content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw, flags=re.S).strip()
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("configs"), list):
            raise ValueError("reply must contain a configs list")
        return data["configs"]

    def select_batch(
        self,
        *,
        remaining: list[Any],
        observations: list[dict[str, Any]],
        round_number: int,
    ) -> tuple[list[Any], dict[str, dict[str, Any]], dict[str, Any]]:
        if len(remaining) < self.batch_size:
            raise ValueError(f"need at least {self.batch_size} remaining configurations")
        by_params = {self._key(cfg.params): cfg for cfg in remaining}
        history = [
            {
                "config_id": row["config"]["config_id"],
                "params": row["config"]["params"],
                self.metric_key: row["evaluation"][self.metric_key],
            }
            for row in observations
        ]
        payload = {
            "round": round_number,
            "batch_size": self.batch_size,
            "objective": self.objective,
            "allowed_values": self._allowed_values(),
            "remaining_count": len(remaining),
            "own_observations": history,
            "already_evaluated_config_ids": [row["config_id"] for row in history],
        }
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        started = time.perf_counter()
        raw_response: str | None = None
        error: str | None = None
        proposals: list[Any] = []
        try:
            raw_response = self.client.chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                model=self.model,
                extra=self.request_extra,
            )
            proposals = self._parse(raw_response)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("LLM controller round=%s failed: %s", round_number, error)

        selected: list[Any] = []
        diagnostics: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        for index, proposal in enumerate(proposals):
            if len(selected) == self.batch_size:
                break
            if not isinstance(proposal, dict):
                rejected.append({"index": index, "reason": "not an object", "proposal": proposal})
                continue
            try:
                cfg = by_params.get(self._key(proposal))
            except (KeyError, TypeError):
                cfg = None
            if cfg is None or cfg.config_id in diagnostics:
                rejected.append({"index": index, "reason": "unavailable or duplicate", "proposal": proposal})
                continue
            selected.append(cfg)
            diagnostics[cfg.config_id] = {"selection": "llm", "proposal_index": index}

        fallback_count = self.batch_size - len(selected)
        if fallback_count:
            fallback_pool = [cfg for cfg in remaining if cfg.config_id not in diagnostics]
            for cfg in self.rng.sample(sorted(fallback_pool, key=lambda item: item.config_id), fallback_count):
                selected.append(cfg)
                diagnostics[cfg.config_id] = {"selection": "seeded_random_fallback"}
        trace = {
            "round": round_number,
            "model": self.model,
            "request_extra": self.request_extra,
            "system_prompt": self.system_prompt,
            "user_prompt": prompt,
            "raw_response": raw_response,
            "request_or_parse_error": error,
            "proposal_count": len(proposals),
            "rejected_proposals": rejected,
            "fallback_count": fallback_count,
            "selection_seconds": time.perf_counter() - started,
            "selected_config_ids": [cfg.config_id for cfg in selected],
        }
        return selected, diagnostics, trace
