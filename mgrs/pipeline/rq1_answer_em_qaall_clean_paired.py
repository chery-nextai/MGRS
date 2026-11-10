#!/usr/bin/env python3
"""Resumable QA-all Exact Match experiment with disjoint dev and final seeds.

The source qa_all.json and 3944-item corpus are never changed. Exact text
queries with conflicting references are excluded at sampling time; a repeated
query with one reference contributes at most one representative per run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mgrs.controllers.llm_controller import LLMController, SYSTEM_PROMPT
from mgrs.environment.data import load_dataset
from mgrs.pipeline import rq1_fixed_eval_paired5 as paired
from mgrs.pipeline import rq1_memory_v2_jit_random as search
from mgrs.pipeline.rq1_answer_f1_smoke import (
    ANSWER_CACHE_SCHEMA,
    ANSWER_PROMPT_VERSION,
    ANSWER_SCORING_VERSION,
    AnswerF1Evaluator,
)


LOGGER = logging.getLogger("ao_explore.rq1_answer_em_qaall_clean")
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "datasets/train_0828"
DEV_SEEDS = (51, 52)
FINAL_SEEDS = (61, 62, 63, 64, 65)
SAMPLE_SEED = 20261009
QUERY_COUNT = 50
REWARD_KEY = "mean_answer_exact_match"
REWARD_SCHEMA = "answer_exact_match_qaall_unique_query_v1"
LLM_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "retrieval support 高", "最终答案 Exact Match 准确率高"
)
PILOT_CONFIGS = (
    "t20_i3_w1p8_r40_ejina_nano",
    "t5_i3_w1p8_r40_ejina_nano",
)
DEV_POLICIES = ("jit_baseline", "jit_no_rules", "jit_no_cap", "random")
FINAL_POLICIES = ("jit", "random", "llm")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_once(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"saved artifact changed: {path}")
        return
    search.atomic_json(path, value)


def id_order(row: dict[str, Any]) -> tuple[int, str]:
    value = str(row["id"])
    return (int(value), value)


def build_samples() -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    qa, corpus = load_dataset(DATA, "qa_all.json")
    if len(qa) != 3944 or len(corpus) != 3944:
        raise RuntimeError("qa_all/corpus size changed")
    eval_rows = read_json(DATA / "qa_eval.json")
    eval_ids = {str(row["id"]) for row in eval_rows}
    eval_queries = {str(row["query"]) for row in eval_rows}
    all_ids = {str(row["id"]) for row in qa}
    if len(eval_ids) != 200 or not eval_ids.issubset(all_ids):
        raise RuntimeError("held-out 200 questions no longer match qa_all")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qa:
        groups[str(row["query"])].append(row)
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    conflicting = [
        rows for rows in groups.values()
        if len({tuple(str(v) for v in row["references"]) for row in rows}) > 1
    ]
    conflicting_queries = {str(rows[0]["query"]) for rows in conflicting}
    consistent_queries = {query for query in groups if query not in conflicting_queries}
    eval_query_overlap = consistent_queries & eval_queries
    representatives = [
        min(rows, key=id_order)
        for query, rows in groups.items()
        if query not in conflicting_queries and query not in eval_queries
    ]
    representatives.sort(key=lambda row: str(row["query"]))
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(representatives)

    if (
        len(groups) != 2816 or len(repeated) != 164
        or sum(map(len, repeated)) != 1292 or len(conflicting) != 73
        or sum(map(len, conflicting)) != 1049 or len(consistent_queries) != 2743
        or len(eval_query_overlap) != 137 or len(representatives) != 2606
    ):
        raise RuntimeError("the pre-audited QA grouping changed")

    samples: dict[int, list[dict[str, Any]]] = {}
    for index, seed in enumerate((*DEV_SEEDS, *FINAL_SEEDS)):
        sample = representatives[index * QUERY_COUNT:(index + 1) * QUERY_COUNT]
        if len(sample) != QUERY_COUNT:
            raise RuntimeError("insufficient distinct QA samples")
        samples[seed] = sample
    selected = [row for sample in samples.values() for row in sample]
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise RuntimeError("dev/final question IDs overlap")
    if len({str(row["query"]) for row in selected}) != len(selected):
        raise RuntimeError("dev/final query strings overlap")
    if any(str(row["id"]) in eval_ids or str(row["query"]) in eval_queries for row in selected):
        raise RuntimeError("selected questions overlap the held-out set")

    audit = {
        "source_qa_ids": len(qa),
        "source_corpus_items": len(corpus),
        "excluded_heldout_ids": len(eval_ids),
        "distinct_query_strings": len(groups),
        "repeated_query_groups": len(repeated),
        "repeated_query_ids": sum(map(len, repeated)),
        "conflicting_query_groups": len(conflicting),
        "conflicting_query_ids": sum(map(len, conflicting)),
        "consistent_query_groups_before_eval_overlap": len(consistent_queries),
        "excluded_heldout_exact_query_groups": len(eval_query_overlap),
        "eligible_query_groups": len(representatives),
        "conflicting_groups": [
            {"query": rows[0]["query"], "ids": [str(r["id"]) for r in rows],
             "references": [r["references"] for r in rows]}
            for rows in sorted(conflicting, key=lambda rows: rows[0]["query"])
        ],
    }
    dataset = {
        "qa_all_sha256": file_sha256(DATA / "qa_all.json"),
        "corpus_sha256": file_sha256(DATA / "corpus.json"),
        "qa_eval_sha256": file_sha256(DATA / "qa_eval.json"),
        "sample_seed": SAMPLE_SEED,
        "dev_seeds": list(DEV_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "query_count_per_seed": QUERY_COUNT,
        "seed_query_hashes": {
            str(seed): search.sha256_json(
                [search.QueryEmbeddingCache.query_key(row) for row in sample]
            ) for seed, sample in samples.items()
        },
        "seed_ids": {
            str(seed): [str(row["id"]) for row in sample]
            for seed, sample in samples.items()
        },
    }
    return samples, audit, dataset


class AnswerEMEvaluator:
    def __init__(
        self, queries: list[dict[str, Any]], retrieval: search.RetrievalCache,
        cache_path: Path, model: str, concurrency: int,
    ) -> None:
        self.base = AnswerF1Evaluator(queries, retrieval, cache_path, model, concurrency)

    def evaluate(self, config: search.SearchConfig) -> dict[str, Any]:
        result = self.base.evaluate(config)
        result[REWARD_KEY] = result["choice_accuracy"]
        result["metric"] = REWARD_KEY
        return result


class FrozenConfigEvaluator:
    """One durable answer score per configuration and seed, shared by policies."""

    def __init__(
        self, evaluator: AnswerEMEvaluator, score_dir: Path, sample_hash: str,
        answer_model: str,
    ) -> None:
        self.evaluator = evaluator
        self.score_dir = score_dir
        self.score_dir.mkdir(parents=True, exist_ok=True)
        self.sample_hash = sample_hash
        self.answer_model = answer_model

    def evaluate(self, config: search.SearchConfig) -> dict[str, Any]:
        path = self.score_dir / f"{config.config_id}.json"
        if path.exists():
            result = read_json(path)
            if (
                result.get("config_id") != config.config_id
                or result.get("sample_hash") != self.sample_hash
                or result.get("answer_model") != self.answer_model
                or result.get("reward_schema") != REWARD_SCHEMA
                or result.get("query_count") != QUERY_COUNT
            ):
                raise RuntimeError(f"incompatible config score: {path}")
            result["answer_new_calls"] = 0
            result["answer_cache_hits"] = QUERY_COUNT
            result["score_oracle_hit"] = True
            return result
        result = self.evaluator.evaluate(config)
        if result["query_count"] != QUERY_COUNT:
            raise RuntimeError("answer evaluator used an unexpected query count")
        result["sample_hash"] = self.sample_hash
        result["answer_model"] = self.answer_model
        result["reward_schema"] = REWARD_SCHEMA
        search.atomic_json(path, result)
        result["score_oracle_hit"] = False
        return result


class NoRulesMemory(search.MemoryV2):
    def distill(self, round_number: int) -> list[dict[str, Any]]:
        self.semantic_rules = []
        return []


class NoCapMemory(search.MemoryV2):
    def select_jit(
        self, remaining: list[search.SearchConfig], rng: random.Random,
    ) -> tuple[list[search.SearchConfig], dict[str, dict[str, Any]]]:
        scored = {cfg.config_id: self.score(cfg) for cfg in remaining}
        ordered = sorted(
            remaining, key=lambda cfg: (-scored[cfg.config_id]["utility"], cfg.config_id)
        )
        exploitation = ordered[:3]
        rest = [cfg for cfg in remaining if cfg not in exploitation]
        exploration = max(rest, key=lambda cfg: (scored[cfg.config_id]["explore"], cfg.config_id))
        rest = [cfg for cfg in rest if cfg != exploration]
        safety = rng.choice(sorted(rest, key=lambda cfg: cfg.config_id))
        selected = exploitation + [exploration, safety]
        diagnostics: dict[str, dict[str, Any]] = {}
        for cfg in exploitation:
            diagnostics[cfg.config_id] = {"selection": "exploitation_no_cap", **scored[cfg.config_id]}
        diagnostics[exploration.config_id] = {"selection": "exploration", **scored[exploration.config_id]}
        diagnostics[safety.config_id] = {"selection": "safety_random", **scored[safety.config_id]}
        return selected, diagnostics


MEMORY_CLASSES: dict[str, type[search.MemoryV2]] = {
    "jit_baseline": search.MemoryV2,
    "jit_no_rules": NoRulesMemory,
    "jit_no_cap": NoCapMemory,
}


def trajectory_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    saved: dict[tuple[int, int], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (int(row["round"]), int(row["slot"]))
            if key in saved:
                raise RuntimeError(f"duplicate trajectory row: {path} {key}")
            saved[key] = row
    return saved


def best_so_far_mean(rows: list[dict[str, Any]]) -> float:
    best = 0.0
    values = []
    for row in rows:
        best = max(best, float(row["evaluation"][REWARD_KEY]))
        values.append(best)
    return float(np.mean(values))


def run_policy(
    root: Path, policy: str, candidates: list[search.SearchConfig],
    initial: list[search.SearchConfig], evaluator: FrozenConfigEvaluator,
    seed: int, decision_model: str, progress_path: Path,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    saved = trajectory_rows(root / "trajectory.jsonl")
    by_id = {cfg.config_id: cfg for cfg in candidates}
    memory_cls = MEMORY_CLASSES.get(policy)
    memory = memory_cls() if memory_cls else None
    tried: set[str] = set()
    trajectory: list[dict[str, Any]] = []
    for round_number in (1, 2, 3):
        plan_path = root / "plans" / f"round_{round_number:02d}.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            batch = [by_id[config_id] for config_id in plan["config_ids"]]
            diagnostics = plan["diagnostics"]
        else:
            remaining = [cfg for cfg in candidates if cfg.config_id not in tried]
            if round_number == 1:
                batch = list(initial)
                diagnostics = {cfg.config_id: {"selection": "shared_space_filling"} for cfg in batch}
            elif memory is not None:
                batch, diagnostics = memory.select_jit(
                    remaining, random.Random(seed * 1000 + round_number)
                )
            elif policy == "random":
                batch = random.Random(seed * 1000 + round_number).sample(
                    sorted(remaining, key=lambda cfg: cfg.config_id), 5
                )
                diagnostics = {cfg.config_id: {"selection": "seeded_random"} for cfg in batch}
            elif policy == "llm":
                controller = LLMController(
                    seed=seed * 1000 + round_number,
                    model=decision_model,
                    metric_key=REWARD_KEY,
                    objective="maximize mean final-answer Exact Match with few configuration evaluations",
                    system_prompt=LLM_SYSTEM_PROMPT,
                )
                batch, diagnostics, decision = controller.select_batch(
                    remaining=remaining, observations=trajectory, round_number=round_number
                )
                search.atomic_json(root / "decisions" / f"round_{round_number:02d}.json", decision)
            else:
                raise ValueError(f"unknown policy: {policy}")
            if len(batch) != 5 or len({cfg.config_id for cfg in batch}) != 5:
                raise RuntimeError(f"{policy} selected a non-unique or wrong-sized batch")
            search.atomic_json(plan_path, {
                "round": round_number,
                "config_ids": [cfg.config_id for cfg in batch],
                "diagnostics": diagnostics,
            })

        for slot, cfg in enumerate(batch, start=1):
            if cfg.config_id in tried:
                raise RuntimeError(f"{policy} selected a previously tried config: {cfg.config_id}")
            key = (round_number, slot)
            if key in saved:
                row = saved[key]
                if row["config"]["config_id"] != cfg.config_id:
                    raise RuntimeError(f"trajectory and plan disagree: {root} {key}")
            else:
                LOGGER.info("seed=%s policy=%s round=%s slot=%s config=%s",
                            seed, policy, round_number, slot, cfg.config_id)
                started = time.perf_counter()
                evaluation = evaluator.evaluate(cfg)
                row = {
                    "round": round_number,
                    "slot": slot,
                    "evaluation_position": len(trajectory) + 1,
                    "config": cfg.as_dict(),
                    "selection": diagnostics[cfg.config_id],
                    "evaluation": search.clean_evaluation_summary(evaluation),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                search.append_jsonl(root / "trajectory.jsonl", row)
                saved[key] = row
            tried.add(cfg.config_id)
            trajectory.append(row)
            if memory is not None:
                memory.add(cfg, float(row["evaluation"][REWARD_KEY]), round_number)
            search.atomic_json(progress_path, {
                "seed": seed, "policy": policy, "round": round_number,
                "slot": slot, "configs_this_policy": len(trajectory),
                "score": row["evaluation"][REWARD_KEY],
                "last_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        if memory is not None:
            rules = memory.distill(round_number)
            search.atomic_json(root / "semantic_memory.json", rules)
        round_rows = []
        for row in trajectory[-5:]:
            config_id = row["config"]["config_id"]
            evaluation = read_json(evaluator.score_dir / f"{config_id}.json")
            round_rows.append({**row, "query_evaluations": evaluation["evaluations"]})
        search.atomic_json(root / "rounds" / f"round_{round_number:02d}.json", {
            "round": round_number, "policy": policy, "batch": round_rows,
        })

    best = max(trajectory, key=lambda row: float(row["evaluation"][REWARD_KEY]))
    summary = {
        "policy": policy,
        "seed": seed,
        "budget": 15,
        "configs_evaluated": len(trajectory),
        "best_config": best["config"],
        "best_answer_exact_match": best["evaluation"][REWARD_KEY],
        "best_step": best["evaluation_position"],
        "best_so_far_mean_15": best_so_far_mean(trajectory),
        "score_oracle_hits": sum(bool(row["evaluation"]["score_oracle_hit"]) for row in trajectory),
        "answer_new_calls": sum(int(row["evaluation"]["answer_new_calls"]) for row in trajectory),
    }
    search.atomic_json(summary_path, summary)
    return summary


def run_repeat_pilot(
    root: Path, sample: list[dict[str, Any]], retrieval: search.RetrievalCache,
    by_id: dict[str, search.SearchConfig], model: str, concurrency: int,
) -> dict[str, Any]:
    pilot_root = root / "repeat_pilot"
    summary_path = pilot_root / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    summary: dict[str, Any] = {"seed": DEV_SEEDS[0], "query_count": QUERY_COUNT, "configs": {}}
    for config_id in PILOT_CONFIGS:
        config = by_id[config_id]
        repeats = []
        for repeat in (1, 2, 3):
            result_path = pilot_root / config_id / f"repeat_{repeat}.json"
            if result_path.exists():
                result = read_json(result_path)
            else:
                LOGGER.info("repeat pilot config=%s repeat=%s", config_id, repeat)
                evaluator = AnswerEMEvaluator(
                    sample, retrieval,
                    pilot_root / config_id / f"answer_cache_repeat_{repeat}.json",
                    model, concurrency,
                )
                result = evaluator.evaluate(config)
                search.atomic_json(result_path, result)
            repeats.append(result)
        scores = [float(result[REWARD_KEY]) for result in repeats]
        per_query = [
            [int(result["evaluations"][index]["exact_match"]) for result in repeats]
            for index in range(QUERY_COUNT)
        ]
        summary["configs"][config_id] = {
            "scores": scores,
            "score_range": max(scores) - min(scores),
            "per_query_flip_count": sum(len(set(values)) > 1 for values in per_query),
            "result_paths": [str(pilot_root / config_id / f"repeat_{i}.json") for i in (1, 2, 3)],
        }
    search.atomic_json(summary_path, summary)
    return summary


def rankdata(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = np.zeros(len(values), dtype="float64")
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        for index in order[start:stop]:
            ranks[index] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def retrospective_ranking(
    seed_root: Path, policy: str, candidates: list[search.SearchConfig],
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (seed_root / policy / "trajectory.jsonl").read_text().splitlines()]
    scored_files = list((seed_root / "config_scores").glob("*.json"))
    observed = {path.stem: float(read_json(path)[REWARD_KEY]) for path in scored_files}
    results = []
    for step in (5, 10):
        memory = MEMORY_CLASSES[policy]()
        for row in rows[:step]:
            cfg = next(cfg for cfg in candidates if cfg.config_id == row["config"]["config_id"])
            memory.add(cfg, float(row["evaluation"][REWARD_KEY]), int(row["round"]))
        memory.distill((step + 4) // 5)
        tried = {row["config"]["config_id"] for row in rows[:step]}
        untried = [cfg for cfg in candidates if cfg.config_id not in tried]
        predicted = {cfg.config_id: memory.score(cfg)["utility"] for cfg in untried}
        assessed = [cfg.config_id for cfg in untried if cfg.config_id in observed]
        p = [predicted[config_id] for config_id in assessed]
        y = [observed[config_id] for config_id in assessed]
        pr, yr = rankdata(p), rankdata(y)
        correlation = float(np.corrcoef(pr, yr)[0, 1]) if len(assessed) > 2 and np.std(pr) and np.std(yr) else None
        best_observed = max(assessed, key=lambda cid: (observed[cid], cid)) if assessed else None
        rank = (1 + sum(value > predicted[best_observed] for value in predicted.values())) if best_observed else None
        results.append({
            "after_step": step,
            "retrospectively_assessed_untried_count": len(assessed),
            "spearman_on_assessed_configs": correlation,
            "best_assessed_config": best_observed,
            "best_assessed_score": observed[best_observed] if best_observed else None,
            "best_assessed_predicted_rank_among_all_untried": rank,
            "note": "Assessed configurations were selected by the pilot policies; this is not an unbiased full-space ranking test.",
        })
    return results


def select_variant(dev_summaries: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    baseline = "jit_baseline"
    candidates = []
    for policy in ("jit_no_rules", "jit_no_cap"):
        improvements = [
            dev_summaries[seed][policy]["best_so_far_mean_15"]
            - dev_summaries[seed][baseline]["best_so_far_mean_15"]
            for seed in DEV_SEEDS
        ]
        if all(value > 0.0 for value in improvements) and float(np.mean(improvements)) >= 0.005:
            candidates.append((float(np.mean(improvements)), policy, improvements))
    chosen = max(candidates)[1] if candidates else baseline
    return {
        "selected_dev_policy": chosen,
        "selected_memory_class": chosen,
        "selection_rule": "Prefer original JIT unless a variant improves best-so-far mean on both dev seeds and by at least 0.005 on average.",
        "qualifying_variants": [
            {"policy": policy, "mean_improvement": improvement, "per_seed_improvements": values}
            for improvement, policy, values in candidates
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--answer-model", default="qwen3.7-flash")
    parser.add_argument("--decision-model", default="qwen3.8-max")
    parser.add_argument("--eval-concurrency", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_concurrency < 1:
        raise ValueError("eval concurrency must be positive")
    samples, audit, dataset = build_samples()
    candidates = search.all_configs()
    if len(candidates) != 864:
        raise RuntimeError("search space changed")
    by_id = {cfg.config_id: cfg for cfg in candidates}
    initial = {seed: paired.seeded_initial_batch(candidates, seed) for seed in samples}
    manifest = {
        "experiment": "QA-all unique-query answer-EM repeat/dev/final paired search",
        "run_id": args.run_id,
        "source": dataset,
        "audit_counts": {key: value for key, value in audit.items() if key != "conflicting_groups"},
        "sample_protocol": "Remove exact query strings with conflicting references and exact query overlap with held-out qa_eval; choose lowest numeric ID from each consistent group; shuffle with SAMPLE_SEED; allocate disjoint 50-query blocks to dev then final seeds.",
        "reward_key": REWARD_KEY,
        "reward_schema": REWARD_SCHEMA,
        "answer_model": args.answer_model,
        "answer_temperature": 0.0,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "answer_scoring_version": ANSWER_SCORING_VERSION,
        "answer_cache_schema": ANSWER_CACHE_SCHEMA,
        "decision_model": args.decision_model,
        "pilot_configs": list(PILOT_CONFIGS),
        "pilot_repeats": 3,
        "dev_policies": list(DEV_POLICIES),
        "final_policies": list(FINAL_POLICIES),
        "budget_per_policy": 15,
        "rounds_per_policy": 3,
        "configs_per_round": 5,
        "candidate_count": len(candidates),
        "search_space_version": search.SEARCH_SPACE_VERSION,
        "seed_initial_batches": {
            str(seed): [cfg.config_id for cfg in batch] for seed, batch in initial.items()
        },
        "embedding_device": "cpu",
        "eval_concurrency": args.eval_concurrency,
        "config_score_cache": "one score per seed and config, shared across dev/final policies within the seed",
    }
    if args.dry_run:
        print(json.dumps({"manifest": manifest, "audit_counts": manifest["audit_counts"]}, ensure_ascii=False, indent=2))
        return

    root = PROJECT / "runs" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    save_once(root / "manifest.json", manifest)
    save_once(root / "dataset_audit.json", audit)
    for seed, sample in samples.items():
        stage = "dev" if seed in DEV_SEEDS else "final"
        sample_path = root / stage / f"seed_{seed}" / "query_sample.json"
        save_once(sample_path, {
            "seed": seed,
            "stage": stage,
            "query_count": QUERY_COUNT,
            "query_keys_sha256": dataset["seed_query_hashes"][str(seed)],
            "queries": [
                {"id": str(row["id"]), "query": str(row["query"]),
                 "reference": search.reference_for(row)} for row in sample
            ],
        })

    all_queries = [row for seed in (*DEV_SEEDS, *FINAL_SEEDS) for row in samples[seed]]
    embedding_cache = search.QueryEmbeddingCache(root, all_queries, "cpu", 8)
    for backend in search.EMBEDDING_VALUES:
        embedding_cache.ensure(backend)
    es = search.es_client("http://127.0.0.1:19200")
    try:
        search.validate_indices(es)
        retrieval = search.RetrievalCache(root / "retrieval_cache.json", es, embedding_cache)
        LOGGER.info("stage=repeat_pilot")
        pilot = run_repeat_pilot(
            root, samples[DEV_SEEDS[0]], retrieval, by_id,
            args.answer_model, args.eval_concurrency,
        )
        LOGGER.info("repeat pilot scores=%s", {
            config_id: data["scores"] for config_id, data in pilot["configs"].items()
        })

        dev_summaries: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in DEV_SEEDS:
            LOGGER.info("stage=dev seed=%s", seed)
            seed_root = root / "dev" / f"seed_{seed}"
            evaluator = FrozenConfigEvaluator(
                AnswerEMEvaluator(
                    samples[seed], retrieval, seed_root / "answer_cache.json",
                    args.answer_model, args.eval_concurrency,
                ),
                seed_root / "config_scores", dataset["seed_query_hashes"][str(seed)],
                args.answer_model,
            )
            dev_summaries[seed] = {}
            for policy in DEV_POLICIES:
                dev_summaries[seed][policy] = run_policy(
                    seed_root / policy, policy, candidates, initial[seed],
                    evaluator, seed, args.decision_model, root / "progress.json",
                )
            rankings = {
                policy: retrospective_ranking(seed_root, policy, candidates)
                for policy in DEV_POLICIES if policy.startswith("jit_")
            }
            search.atomic_json(seed_root / "ranking_diagnostics.json", rankings)
        selection = select_variant(dev_summaries)
        save_once(root / "dev" / "selected_variant.json", selection)
        search.atomic_json(root / "dev" / "summary.json", {
            "seeds": list(DEV_SEEDS), "policies": list(DEV_POLICIES),
            "per_seed": dev_summaries, "selected_variant": selection,
        })
        LOGGER.info("stage=final selected_variant=%s", selection["selected_dev_policy"])

        final_pairs = []
        for seed in FINAL_SEEDS:
            seed_root = root / "final" / f"seed_{seed}"
            evaluator = FrozenConfigEvaluator(
                AnswerEMEvaluator(
                    samples[seed], retrieval, seed_root / "answer_cache.json",
                    args.answer_model, args.eval_concurrency,
                ),
                seed_root / "config_scores", dataset["seed_query_hashes"][str(seed)],
                args.answer_model,
            )
            summaries = {}
            for policy in FINAL_POLICIES:
                effective_policy = selection["selected_dev_policy"] if policy == "jit" else policy
                summaries[policy] = run_policy(
                    seed_root / policy, effective_policy, candidates, initial[seed],
                    evaluator, seed, args.decision_model, root / "progress.json",
                )
            pair = {
                "seed": seed,
                "query_keys_sha256": dataset["seed_query_hashes"][str(seed)],
                "shared_initial_config_ids": [cfg.config_id for cfg in initial[seed]],
                "selected_jit_variant": selection["selected_dev_policy"],
                **summaries,
                "best_em_difference_jit_minus_random": summaries["jit"]["best_answer_exact_match"] - summaries["random"]["best_answer_exact_match"],
                "best_em_difference_jit_minus_llm": summaries["jit"]["best_answer_exact_match"] - summaries["llm"]["best_answer_exact_match"],
                "auc_difference_jit_minus_random": summaries["jit"]["best_so_far_mean_15"] - summaries["random"]["best_so_far_mean_15"],
                "auc_difference_jit_minus_llm": summaries["jit"]["best_so_far_mean_15"] - summaries["llm"]["best_so_far_mean_15"],
            }
            save_once(seed_root / "paired_summary.json", pair)
            final_pairs.append(pair)
            LOGGER.info("finished final seed=%s jit=%.4f random=%.4f llm=%.4f",
                        seed, summaries["jit"]["best_answer_exact_match"],
                        summaries["random"]["best_answer_exact_match"],
                        summaries["llm"]["best_answer_exact_match"])
        search.atomic_json(root / "summary.json", {
            "run_id": args.run_id,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reward_schema": REWARD_SCHEMA,
            "dataset_hashes": {key: dataset[key] for key in ("qa_all_sha256", "corpus_sha256", "qa_eval_sha256")},
            "selected_jit_variant": selection["selected_dev_policy"],
            "repeat_pilot": pilot,
            "dev_summary_path": str(root / "dev" / "summary.json"),
            "pairs": final_pairs,
            "mean_best_em_difference_jit_minus_random": float(np.mean([p["best_em_difference_jit_minus_random"] for p in final_pairs])),
            "mean_best_em_difference_jit_minus_llm": float(np.mean([p["best_em_difference_jit_minus_llm"] for p in final_pairs])),
            "mean_auc_difference_jit_minus_random": float(np.mean([p["auc_difference_jit_minus_random"] for p in final_pairs])),
            "mean_auc_difference_jit_minus_llm": float(np.mean([p["auc_difference_jit_minus_llm"] for p in final_pairs])),
        })
        LOGGER.info("experiment complete: %s", root / "summary.json")
    finally:
        es.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
