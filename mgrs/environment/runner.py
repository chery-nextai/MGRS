from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mgrs import config as settings
from mgrs.controllers.base import Controller
from mgrs.environment.data import load_dataset
from mgrs.memory.store import ExperienceMemory
from mgrs.metrics.token_f1 import token_f1
from mgrs.rag_pipeline.config import PipelineConfig, candidate_configs
from mgrs.rag_pipeline.executor import MultimodalRAGPipeline

logger = logging.getLogger("ao_explore.runner")


def answer_for_scoring(query: str, answer: str) -> str:
    """Map a numeric multiple-choice answer back to its option text."""
    options = {int(index): text.strip() for index, text in re.findall(r"(?m)^\s*(\d+)\.\s*(.+?)\s*$", str(query))}
    raw = str(answer).strip()
    match = re.fullmatch(r"(?:(?:answer|答案)\s*[:：]?\s*)?(?:(?:option|选项)\s*)?[\[\(]?\s*(\d+)\s*[\]\).。]?", raw, re.IGNORECASE)
    if match:
        return options.get(int(match.group(1)), raw)
    return raw


def action_family(config: PipelineConfig, incumbent: PipelineConfig | None) -> str:
    if incumbent is None:
        return "initial"
    if config.image_top_k > incumbent.image_top_k:
        return "increase_image_top_k"
    if config.image_top_k < incumbent.image_top_k:
        return "decrease_image_top_k"
    if config.text_final_top_k > incumbent.text_final_top_k:
        return "increase_text_top_k"
    if config.text_final_top_k < incumbent.text_final_top_k:
        return "decrease_text_top_k"
    if config.text_weight > incumbent.text_weight:
        return "increase_text_weight"
    if config.text_weight < incumbent.text_weight:
        return "decrease_text_weight"
    return "same_retrieval_depth"


def config_params(config: PipelineConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "text_top_k": config.text_final_top_k,
        "image_top_k": config.image_top_k,
        "text_weight": config.text_weight,
    }


def config_delta(config: PipelineConfig, incumbent: PipelineConfig | None) -> dict[str, Any] | None:
    if incumbent is None:
        return None
    return {
        "text_top_k": config.text_final_top_k - incumbent.text_final_top_k,
        "image_top_k": config.image_top_k - incumbent.image_top_k,
        "text_weight": config.text_weight - incumbent.text_weight,
    }


class ExperimentRunner:
    def __init__(
        self,
        controller: Controller,
        dataset_dir: str | Path,
        budget: int,
        run_id: str,
        seed: int,
        qa_file: str = "qa_search.json",
        eval_qa_file: str = "qa_eval.json",
        search_only: bool = False,
        episode: int = 1,
        memory_path: str | Path | None = None,
        trajectory_path: str | Path | None = None,
    ):
        self.controller, self.dataset_dir, self.budget, self.run_id, self.seed = controller, Path(dataset_dir), budget, run_id, seed
        self.episode = int(episode)
        self.memory_path = Path(memory_path) if memory_path is not None else None
        self.qa_file = qa_file
        self.eval_qa_file = eval_qa_file
        self.search_only = search_only
        self.qa, self.corpus = load_dataset(self.dataset_dir, qa_file)
        self.eval_qa = [] if self.search_only else load_dataset(self.dataset_dir, eval_qa_file)[0]
        self.candidates = candidate_configs()
        self.run_dir = settings.RUNS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = Path(trajectory_path) if trajectory_path is not None else self.run_dir / "trajectory.jsonl"
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.support_cache_path = settings.SUPPORT_CACHE_PATH
        try:
            self.support_cache = json.loads(self.support_cache_path.read_text(encoding="utf-8")) if self.support_cache_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            self.support_cache = {}

    def _cache_key(self, config_id: str, item: dict[str, Any], model: str, qa_file: str | None = None) -> str:
        return f"{settings.SEARCH_SPACE_VERSION}|{model}|{qa_file or self.qa_file}|{config_id}|{item.get('id', '')}"

    def _save_support_cache(self) -> None:
        self.support_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.support_cache_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.support_cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.support_cache_path)

    def _state(self, step: int, incumbent: PipelineConfig | None, rewards: list[float], tried: set[str]) -> dict[str, Any]:
        return {
            "step": step,
            "budget": self.budget,
            "incumbent": incumbent.__dict__ if incumbent else None,
            "incumbent_params": config_params(incumbent),
            "recent_rewards": rewards[-5:],
            "action_family_counts": {},
            "remaining_count": len(self.candidates) - len(tried),
            "tested_config_ids": sorted(tried),
            "search_space_version": settings.SEARCH_SPACE_VERSION,
            "failure_count": 0,
            "dataset_id": settings.DATASET_ID,
            "reward_normalization": settings.REWARD_NORMALIZATION,
        }

    def _evaluate(self, pipeline: MultimodalRAGPipeline, cfg: PipelineConfig) -> dict[str, Any]:
        def evaluate_item(item: dict[str, Any]) -> dict[str, Any]:
            refs = item.get("references") or [item.get("answer", "")]
            reference = str(refs[0] if isinstance(refs, list) else refs)
            result = pipeline.retrieve(str(item.get("query", "")), reference, cfg, str(item.get("id", "")))
            key = self._cache_key(cfg.config_id, item, settings.SEARCH_JUDGE_MODEL)
            if key in self.support_cache:
                result["retrieval_support_score"] = float(self.support_cache[key])
            else:
                result["retrieval_support_score"] = pipeline.llm.judge_retrieval(
                    result["query"], result["reference"], result["text_evidence"], result["image_evidence"], model=settings.SEARCH_JUDGE_MODEL
                )
            result["id"] = str(item.get("id", ""))
            result["dataset_id"] = item.get("dataset_id", settings.DATASET_ID)
            result["domain_id"] = item.get("domain_id")
            return result

        logger.info("evaluate config=%s queries=%s", cfg.config_id, len(self.qa))
        with ThreadPoolExecutor(max_workers=settings.EVAL_CONCURRENCY) as executor:
            evaluations = list(executor.map(evaluate_item, self.qa))
        for query_index, result in enumerate(evaluations, start=1):
            logger.info("config=%s query=%s/%s id=%s support=%.4f", cfg.config_id, query_index, len(self.qa), result["id"], result["retrieval_support_score"])
        support = sum(float(row["retrieval_support_score"]) for row in evaluations) / max(1, len(evaluations))
        text_hit = sum(float(row["text_hit_at_k"]) for row in evaluations) / max(1, len(evaluations))
        image_hit = sum(float(row["image_hit_at_k"]) for row in evaluations) / max(1, len(evaluations))
        for item in evaluations:
            self.support_cache[self._cache_key(cfg.config_id, item, settings.SEARCH_JUDGE_MODEL)] = item["retrieval_support_score"]
        self._save_support_cache()
        return {"config_id": cfg.config_id, "mean_retrieval_support": support, "mean_text_hit_at_k": text_hit, "mean_image_hit_at_k": image_hit, "mean_token_f1": None, "choice_accuracy": None, "query_count": len(evaluations), "evaluations": evaluations}

    def _final_answer_evaluation(self, pipeline: MultimodalRAGPipeline, evaluation: dict[str, Any]) -> dict[str, Any]:
        def answer_item(item: dict[str, Any]) -> dict[str, Any]:
            result = pipeline.generate_answer(item)
            result["scored_answer"] = answer_for_scoring(result["query"], result["answer"])
            result["token_f1"] = token_f1(result["scored_answer"], result["reference"])
            result["choice_exact_match"] = int(result["scored_answer"].strip().casefold() == result["reference"].strip().casefold())
            return result

        with ThreadPoolExecutor(max_workers=settings.EVAL_CONCURRENCY) as executor:
            evaluations = list(executor.map(answer_item, evaluation["evaluations"]))
        result = {k: v for k, v in evaluation.items() if k != "evaluations"}
        result["mean_token_f1"] = sum(float(row["token_f1"]) for row in evaluations) / max(1, len(evaluations))
        result["choice_accuracy"] = sum(float(row["choice_exact_match"]) for row in evaluations) / max(1, len(evaluations))
        result["evaluations"] = evaluations
        return result

    def _retrieve_final_evaluation(self, pipeline: MultimodalRAGPipeline, cfg: PipelineConfig) -> dict[str, Any]:
        def retrieve_item(item: dict[str, Any]) -> dict[str, Any]:
            refs = item.get("references") or [item.get("answer", "")]
            reference = str(refs[0] if isinstance(refs, list) else refs)
            result = pipeline.retrieve(str(item.get("query", "")), reference, cfg, str(item.get("id", "")))
            result["id"] = str(item.get("id", ""))
            result["dataset_id"] = item.get("dataset_id", settings.DATASET_ID)
            result["domain_id"] = item.get("domain_id")
            return result

        logger.info("final evaluation config=%s qa_file=%s queries=%s", cfg.config_id, self.eval_qa_file, len(self.eval_qa))
        with ThreadPoolExecutor(max_workers=settings.EVAL_CONCURRENCY) as executor:
            evaluations = list(executor.map(retrieve_item, self.eval_qa))
        text_hit = sum(float(row["text_hit_at_k"]) for row in evaluations) / max(1, len(evaluations))
        image_hit = sum(float(row["image_hit_at_k"]) for row in evaluations) / max(1, len(evaluations))
        return {
            "config_id": cfg.config_id,
            "mean_retrieval_support": None,
            "mean_text_hit_at_k": text_hit,
            "mean_image_hit_at_k": image_hit,
            "mean_token_f1": None,
            "choice_accuracy": None,
            "query_count": len(evaluations),
            "evaluations": evaluations,
        }

    def _final_check(self, pipeline: MultimodalRAGPipeline, rows: list[dict[str, Any]], top_n: int = 3) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ranked = sorted(rows, key=lambda row: float(row["evaluation"]["mean_retrieval_support"]), reverse=True)[:top_n]
        def check_row(row: dict[str, Any]) -> dict[str, Any]:
            checked = []
            uncached = []
            for item in row["evaluation"]["evaluations"]:
                key = self._cache_key(row["action"]["action_id"], item, settings.LLM_MODEL)
                if key in self.support_cache:
                    cached = dict(item)
                    cached["retrieval_support_score"] = float(self.support_cache[key])
                    checked.append(cached)
                else:
                    uncached.append(item)
            with ThreadPoolExecutor(max_workers=settings.EVAL_CONCURRENCY) as executor:
                checked.extend(executor.map(pipeline.check_retrieval, uncached))
            for item in checked:
                self.support_cache[self._cache_key(row["action"]["action_id"], item, settings.LLM_MODEL)] = item["retrieval_support_score"]
            self._save_support_cache()
            support = sum(float(item["retrieval_support_score"]) for item in checked) / max(1, len(checked))
            return {"config_id": row["action"]["action_id"], "mean_retrieval_support": support, "query_count": len(checked), "evaluations": checked}
        # Keep candidate configurations sequential; only the 20 questions within
        # each candidate are evaluated concurrently.
        checks = [check_row(row) for row in ranked]
        best_index = max(range(len(checks)), key=lambda i: checks[i]["mean_retrieval_support"])
        return ranked[best_index], checks

    def run(self) -> dict[str, Any]:
        settings.ensure_dirs()
        logger.info("dataset search_qa=%s qa_file=%s eval_qa=%s eval_qa_file=%s corpus=%s text_index=%s image_index=%s run_dir=%s", len(self.qa), self.qa_file, len(self.eval_qa), self.eval_qa_file, len(self.corpus), settings.TEXT_INDEX_NAME, settings.IMAGE_INDEX_NAME, self.run_dir)
        pipeline = MultimodalRAGPipeline(self.corpus, settings.TEXT_INDEX_NAME, self.candidates[0])
        logger.info("pipeline initialized; persistent ES indices ready")
        # Search-only experiments use an isolated per-run memory so paired
        # seeds/controllers cannot contaminate one another. Existing runs keep
        # the historical default memory behavior.
        memory_file = self.memory_path or (self.run_dir / "memory.jsonl" if self.search_only else settings.MEMORY_PATH)
        memory = ExperienceMemory(memory_file)
        incumbent: PipelineConfig | None = None
        best_support = 0.0
        rewards: list[float] = []
        tried: set[str] = set()
        rows: list[dict[str, Any]] = []
        try:
            for step in range(self.budget):
                remaining = [cfg for cfg in self.candidates if cfg.config_id not in tried]
                if not remaining:
                    break
                state = self._state(step, incumbent, rewards, tried)
                options = [{"action_id": cfg.config_id, "action_family": action_family(cfg, incumbent), "complete_config": cfg.__dict__, "params": config_params(cfg), "delta": config_delta(cfg, incumbent)} for cfg in remaining]
                selected = self.controller.select(state, options, memory)
                cfg = next(item for item in remaining if item.config_id == selected["action_id"])
                logger.info("step=%s/%s selected=%s family=%s remaining=%s", step + 1, self.budget, cfg.config_id, selected.get("action_family"), len(remaining))
                evaluation = self._evaluate(pipeline, cfg)
                reward = float(evaluation["mean_retrieval_support"] - best_support)
                if evaluation["mean_retrieval_support"] > best_support:
                    best_support, incumbent = evaluation["mean_retrieval_support"], cfg
                rewards.append(reward)
                tried.add(cfg.config_id)
                evaluation_summary = {k: v for k, v in evaluation.items() if k != "evaluations"}
                current_support = float(evaluation["mean_retrieval_support"])
                previous_best = max((float(row.get("best_so_far_support", 0.0)) for row in rows), default=0.0)
                previous_best_step = min((int(row.get("best_step", step + 1)) for row in rows), default=step + 1)
                best_so_far = max(previous_best, current_support)
                best_step = step + 1 if current_support > previous_best else previous_best_step
                transition = {"run_id": self.run_id, "episode": self.episode, "step": step, "state": state, "action": selected, "reward": reward, "return": None, "evaluation": evaluation, "best_so_far_support": best_so_far, "best_step": best_step, "search_space_version": settings.SEARCH_SPACE_VERSION, "metric_version": settings.METRIC_VERSION, "search_judge_model": settings.SEARCH_JUDGE_MODEL}
                # Keep per-query evidence in the run trajectory, but avoid duplicating the large list in global Memory.
                memory.append({**transition, "evaluation": evaluation_summary})
                self.controller.observe(transition)
                rows.append(transition)
                with self.trajectory_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(transition, ensure_ascii=False) + "\n")
                logger.info("step=%s complete config=%s reward=%.6f support=%.6f text_hit=%.6f image_hit=%.6f", step + 1, cfg.config_id, reward, evaluation["mean_retrieval_support"], evaluation["mean_text_hit_at_k"], evaluation["mean_image_hit_at_k"])
            search_best_row = max(rows, key=lambda row: float(row.get("evaluation", {}).get("mean_retrieval_support", 0.0)), default={})
            final_best_row, final_checks = (({}, []) if self.search_only else (self._final_check(pipeline, rows, top_n=min(settings.FINAL_CHECK_TOP_N, len(rows))) if rows else ({}, [])))
            if final_best_row and not self.search_only:
                final_cfg_obj = next(cfg for cfg in self.candidates if cfg.config_id == final_best_row["action"]["action_id"])
                final_eval = self._final_answer_evaluation(pipeline, self._retrieve_final_evaluation(pipeline, final_cfg_obj))
            else:
                final_eval = {}
        finally:
            pipeline.close()
        final_cfg = (search_best_row if self.search_only else final_best_row).get("action", {}).get("complete_config")
        best_check = max(final_checks, key=lambda check: float(check["mean_retrieval_support"]), default={})
        result = {"run_id": self.run_id, "episode": self.episode, "controller": self.controller.name, "seed": self.seed, "search_only": self.search_only, "qa_file": self.qa_file, "query_count": len(self.qa), "eval_qa_file": None if self.search_only else self.eval_qa_file, "eval_query_count": 0 if self.search_only else len(self.eval_qa), "corpus_count": len(self.corpus), "budget": self.budget, "steps_completed": len(rows), "best_config": final_cfg, "best_retrieval_support": best_check.get("mean_retrieval_support", best_support), "best_step": search_best_row.get("best_step"), "best_text_hit_at_k": final_eval.get("mean_text_hit_at_k", 0.0), "best_image_hit_at_k": final_eval.get("mean_image_hit_at_k", 0.0), "best_token_f1": final_eval.get("mean_token_f1", 0.0), "choice_accuracy": final_eval.get("choice_accuracy", 0.0), "search_best_config": search_best_row.get("action", {}).get("complete_config"), "search_best_retrieval_support": best_support, "final_check_top_n": len(final_checks), "final_check": [{k: v for k, v in check.items() if k != "evaluations"} for check in final_checks], "eval_concurrency": settings.EVAL_CONCURRENCY, "return": None, "search_space_version": settings.SEARCH_SPACE_VERSION, "metric_version": settings.METRIC_VERSION, "search_judge_model": settings.SEARCH_JUDGE_MODEL, "final_check_model": None if self.search_only else settings.LLM_MODEL}
        final_output = {"selected_config": final_cfg, "search_only": self.search_only, "search_qa_file": self.qa_file, "eval_qa_file": None if self.search_only else self.eval_qa_file, "search_best_config": search_best_row.get("action", {}).get("complete_config"), "search_best_step": search_best_row.get("best_step"), "best_so_far_support": search_best_row.get("best_so_far_support"), "checks": final_checks, "answer_evaluation": final_eval, "search_judge_model": settings.SEARCH_JUDGE_MODEL, "final_check_model": None if self.search_only else settings.LLM_MODEL}
        (self.run_dir / "final_evaluation.json").write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("result written steps_completed=%s", len(rows))
        return result
