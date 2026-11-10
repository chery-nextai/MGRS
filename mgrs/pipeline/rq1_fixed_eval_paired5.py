#!/usr/bin/env python3
"""Five paired RQ1 searches on one fixed query set and one score cache.

This entry point reuses the existing JIT, Random, and LLM controllers without
changing their implementations or any previous run.  Search seeds affect the
shared initial batch and each controller's choices, never the query sample.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from mgrs import config as settings
from mgrs.environment.data import load_dataset
from mgrs.pipeline import rq1_memory_v2_jit_random as experiment


LOGGER = logging.getLogger("ao_explore.rq1_fixed_eval_paired5")
PROJECT = Path(__file__).resolve().parents[2]
SOURCE_RUN = PROJECT / "runs/20260928_rq1_budget15_jit_random_llm_s24_26_cache"
EMBEDDING_SOURCE = PROJECT / "runs/20260917_225601_rq1_memory_v2_budget50/query_embeddings"


def seeded_initial_batch(
    candidates: list[experiment.SearchConfig], seed: int
) -> list[experiment.SearchConfig]:
    """Seed the same five-point maximin design for all arms of one pair."""
    ordered = sorted(candidates, key=lambda cfg: cfg.config_id)
    first = random.Random(seed).choice(ordered)
    selected = [first]
    remaining = {cfg.config_id: cfg for cfg in ordered if cfg.config_id != first.config_id}
    while len(selected) < 5:
        chosen = max(
            remaining.values(),
            key=lambda cfg: (
                min(experiment.config_distance(cfg, old) for old in selected),
                cfg.config_id,
            ),
        )
        selected.append(chosen)
        del remaining[chosen.config_id]
    return selected


def load_fixed_sample(
    qa: list[dict[str, Any]], sample_path: Path
) -> tuple[list[dict[str, Any]], str]:
    saved = json.loads(sample_path.read_text(encoding="utf-8"))
    records = saved["queries"]
    by_identity = {
        (str(row.get("id", "")), str(row.get("query", ""))): row for row in qa
    }
    if len(records) != 100 or len(by_identity) != len(qa):
        raise ValueError("expected 100 saved queries and unique source query identities")
    sample: list[dict[str, Any]] = []
    for record in records:
        key = (str(record["id"]), str(record["query"]))
        if key not in by_identity:
            raise ValueError(f"saved query is missing from source data: {key[0]}")
        row = by_identity[key]
        if (
            record["query_key"] != experiment.QueryEmbeddingCache.query_key(row)
            or str(record["reference"]) != experiment.reference_for(row)
        ):
            raise ValueError(f"saved query or reference changed: {key[0]}")
        sample.append(row)
    keys = [experiment.QueryEmbeddingCache.query_key(row) for row in sample]
    if len(set(keys)) != 100:
        raise ValueError("fixed query sample contains duplicates")
    return sample, experiment.sha256_json(keys)


class FrozenConfigEvaluator:
    """One immutable evaluation per config, shared by every arm and search seed."""

    def __init__(self, evaluator: experiment.ConfigEvaluator, score_dir: Path) -> None:
        self.evaluator = evaluator
        self.score_dir = score_dir

    def evaluate(self, config: experiment.SearchConfig) -> dict[str, Any]:
        path = self.score_dir / f"{config.config_id}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["config_id"] != config.config_id or result["query_count"] != 100:
                raise RuntimeError(f"invalid frozen score: {path}")
            # Counters in each controller's summary describe new API work only.
            result["judge_cache_hits"] = 100
            result["judge_unique_misses"] = 0
            result["judge_new_response_parse_counts"] = {
                "json": 0,
                "regex": 0,
                "failed": 0,
            }
            result["judge_parse_failure_count"] = 0
            result["score_oracle_hit"] = True
            LOGGER.info(
                "frozen score config=%s support=%.4f",
                config.config_id,
                result["mean_retrieval_support"],
            )
            return result
        result = self.evaluator.evaluate(config)
        if result["query_count"] != 100:
            raise RuntimeError("evaluator used the wrong query sample")
        experiment.atomic_json(path, result)
        result["score_oracle_hit"] = False
        return result


def copy_and_check_embedding_cache(qa: list[dict[str, Any]], root: Path) -> None:
    expected_keys = [experiment.QueryEmbeddingCache.query_key(row) for row in qa]
    destination = root / "query_embeddings"
    destination.mkdir()
    for backend in experiment.EMBEDDING_VALUES:
        source = EMBEDDING_SOURCE / f"{backend}.npz"
        with np.load(source, allow_pickle=False) as loaded:
            keys = [str(key) for key in loaded["keys"].tolist()]
            shape = loaded["vectors"].shape
        expected_shape = (len(qa), experiment.BACKENDS[backend]["dims"])
        if keys != expected_keys or shape != expected_shape:
            raise ValueError(f"incompatible embedding cache: {source}")
        shutil.copy2(source, destination / source.name)
        LOGGER.info("copied verified query embeddings backend=%s", backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 32, 33, 34, 35])
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--eval-concurrency", type=int, default=3)
    parser.add_argument("--judge-model", default=settings.SEARCH_JUDGE_MODEL)
    parser.add_argument("--llm-model", default="qwen3.8-max")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget != 15 or args.eval_concurrency < 1 or len(args.seeds) != 5:
        raise ValueError("this pilot requires five seeds, budget 15, and positive concurrency")
    if len(set(args.seeds)) != 5:
        raise ValueError("search seeds must be distinct")
    candidates = experiment.all_configs()
    if len(candidates) != 864:
        raise RuntimeError("search space changed")
    initial_batches = {seed: seeded_initial_batch(candidates, seed) for seed in args.seeds}
    initial_sets = [tuple(cfg.config_id for cfg in batch) for batch in initial_batches.values()]
    if len(set(initial_sets)) != 5:
        raise RuntimeError("seeded initial batches must differ across paired runs")

    dataset_dir = PROJECT / "datasets/train_0828"
    qa, _ = load_dataset(str(dataset_dir), "qa_search.json")
    if len(qa) != 200:
        raise RuntimeError(f"expected 200 source queries, found {len(qa)}")
    source_sample = SOURCE_RUN / "seed_25/query_sample.json"
    sample, sample_hash = load_fixed_sample(qa, source_sample)
    manifest = {
        "experiment": "fixed 100-query paired search: JIT, Random, LLM",
        "run_id": args.run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_dir": str(dataset_dir),
        "qa_file": "qa_search.json",
        "fixed_query_sample_source": str(source_sample),
        "fixed_query_keys_sha256": sample_hash,
        "query_count_per_config": len(sample),
        "search_seeds": args.seeds,
        "controllers": ["jit", "random", "llm"],
        "budget_per_controller": args.budget,
        "batch_size": 5,
        "candidate_count": len(candidates),
        "judge_model": args.judge_model,
        "metric_schema": experiment.METRIC_SCHEMA,
        "judge_cache_schema": experiment.JUDGE_CACHE_SCHEMA,
        "llm_controller_model": args.llm_model,
        "eval_concurrency": args.eval_concurrency,
        "embedding_cache_source": str(EMBEDDING_SOURCE),
        "retrieval_cache_source": str(SOURCE_RUN / "seed_25/retrieval_cache.json"),
        "judge_cache_initial": "empty for the new prompt and model",
        "score_oracle": "one persisted result per config, shared across all pairs",
        "seeded_initial_batches": {
            str(seed): [cfg.as_dict() for cfg in batch]
            for seed, batch in initial_batches.items()
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    root = PROJECT / "runs" / args.run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {root}")
    es = experiment.es_client("http://127.0.0.1:19200")
    try:
        manifest["indices"] = experiment.validate_indices(es)
        root.mkdir(parents=True)
        copy_and_check_embedding_cache(qa, root)
        shared = root / "shared_cache"
        shared.mkdir()
        source_cache = SOURCE_RUN / "seed_25"
        shutil.copy2(source_cache / "retrieval_cache.json", shared / "retrieval_cache.json")
        experiment.atomic_json(root / "fixed_query_sample.json", {
            "source": str(source_sample),
            "query_keys_sha256": sample_hash,
            "queries": json.loads(source_sample.read_text(encoding="utf-8"))["queries"],
        })
        experiment.atomic_json(root / "manifest.json", manifest)

        embedding_cache = experiment.QueryEmbeddingCache(root, qa, "cpu", 16)
        for backend in experiment.EMBEDDING_VALUES:
            embedding_cache.ensure(backend)
        retrieval = experiment.RetrievalCache(shared / "retrieval_cache.json", es, embedding_cache)
        judge_cache = experiment.JudgeCache(shared / "judge_cache.json", args.judge_model)
        if not retrieval.rows or judge_cache.scores:
            raise RuntimeError("expected a populated retrieval cache and an empty new judge cache")
        base_evaluator = experiment.ConfigEvaluator(
            queries=sample,
            retrieval=retrieval,
            judge_cache=judge_cache,
            judge_model=args.judge_model,
            concurrency=args.eval_concurrency,
        )
        evaluator = FrozenConfigEvaluator(base_evaluator, root / "frozen_config_scores")
        pairs: list[dict[str, Any]] = []
        for seed in args.seeds:
            pair_root = root / f"pair_{seed}"
            pair_root.mkdir()
            initial = initial_batches[seed]
            LOGGER.info("begin pair seed=%s initial=%s", seed, [cfg.config_id for cfg in initial])
            jit = experiment.run_jit(pair_root, candidates, initial, evaluator, args.budget, seed)
            random_summary = experiment.run_random(
                pair_root, candidates, initial, evaluator, args.budget, seed
            )
            llm = experiment.run_llm(
                pair_root, candidates, initial, evaluator, args.budget, seed, args.llm_model
            )
            pair = {
                "seed": seed,
                "fixed_query_keys_sha256": sample_hash,
                "shared_first_round": [cfg.config_id for cfg in initial],
                "jit": jit,
                "random": random_summary,
                "llm": llm,
                "best_support_difference_jit_minus_random": (
                    jit["best_retrieval_support"] - random_summary["best_retrieval_support"]
                ),
                "best_support_difference_jit_minus_llm": (
                    jit["best_retrieval_support"] - llm["best_retrieval_support"]
                ),
            }
            experiment.atomic_json(pair_root / "paired_summary.json", pair)
            pairs.append(pair)
            LOGGER.info("completed pair seed=%s JIT=%.4f Random=%.4f LLM=%.4f", seed,
                        jit["best_retrieval_support"], random_summary["best_retrieval_support"],
                        llm["best_retrieval_support"])
        aggregate = {
            "run_id": args.run_id,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "fixed_query_keys_sha256": sample_hash,
            "search_seeds": args.seeds,
            "pairs": pairs,
            "unique_config_scores": len(list((root / "frozen_config_scores").glob("*.json"))),
        }
        experiment.atomic_json(root / "summary.json", aggregate)
        LOGGER.info("completed run=%s", root)
    finally:
        es.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
