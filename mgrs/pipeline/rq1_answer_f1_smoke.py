#!/usr/bin/env python3
"""Smoke test: steer a two-round Memory-v2 JIT search with answer token F1.

This is an isolated pilot. It reuses the fixed search queries and retrieval
cache, but never reads or writes retrieval-support judge scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from mgrs.metrics.token_f1 import token_f1
from mgrs.environment.runner import answer_for_scoring
from mgrs.pipeline import rq1_fixed_eval_paired5 as paired
from mgrs.pipeline import rq1_memory_v2_jit_random as search
from mgrs.environment.data import load_dataset
from mgrs.rag_pipeline.llm import MultimodalLLM, _evidence_text


LOGGER = logging.getLogger("ao_explore.answer_f1_smoke")
PROJECT = Path(__file__).resolve().parents[2]
SOURCE_RUN = paired.SOURCE_RUN
DEFAULT_QUERY_IDS = ("10814", "20827", "525", "5537", "12623", "1021")
ANSWER_PROMPT_VERSION = "scienceqa_short_answer_evidence_v1"
ANSWER_CACHE_SCHEMA = "answer_token_f1_evidence_fingerprint_v1"
ANSWER_SCORING_VERSION = "choice_prefix_normalization_v2"


def normalized_answer(query: str, answer: str) -> str:
    """Accept either a choice number or that number followed by its choice text."""
    scored = answer_for_scoring(query, answer)
    match = re.fullmatch(r"\s*(\d+)\s*[.\):。]\s*(.+?)\s*", scored, re.S)
    if not match:
        return scored
    options = {
        index: text.strip()
        for index, text in re.findall(r"(?m)^\s*(\d+)\.\s*(.+?)\s*$", query)
    }
    option = options.get(match.group(1))
    if option and option.casefold().rstrip(".。 ") == match.group(2).casefold().rstrip(".。 "):
        return option
    return scored


def answer_with_evidence(row: dict[str, Any], model: str) -> str:
    llm = MultimodalLLM()
    prompt = (
        f"问题：{row['query']}\n\n文本证据：\n{_evidence_text(row['text_evidence'])}"
        "\n\n请只给出最终简短答案，不要解释。"
    )
    return llm.client.chat(
        llm._messages(
            "你是一个严谨的 ScienceQA 问答助手。只能依据证据作答。",
            prompt,
            row["image_evidence"],
        ),
        temperature=0.0,
        model=model,
    )


class AnswerF1Evaluator:
    def __init__(
        self,
        queries: list[dict[str, Any]],
        retrieval: search.RetrievalCache,
        cache_path: Path,
        model: str,
        concurrency: int,
    ) -> None:
        self.queries = queries
        self.retrieval = retrieval
        self.cache_path = cache_path
        self.model = model
        self.concurrency = concurrency
        self.records: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("schema") != ANSWER_CACHE_SCHEMA or cached.get("model") != model:
                raise RuntimeError(f"incompatible answer cache: {cache_path}")
            self.records = dict(cached["records"])
            query_by_id = {str(item["id"]): str(item["query"]) for item in queries}
            for record in self.records.values():
                query = query_by_id[str(record["id"])]
                scored = normalized_answer(query, record["raw_answer"])
                record["scored_answer"] = scored
                record["token_f1"] = token_f1(scored, record["reference"])
                record["exact_match"] = int(
                    scored.strip().casefold() == record["reference"].strip().casefold()
                )
            self.flush()

    def fingerprint(self, row: dict[str, Any]) -> str:
        return search.sha256_json(
            {
                "schema": ANSWER_CACHE_SCHEMA,
                "prompt_version": ANSWER_PROMPT_VERSION,
                "model": self.model,
                "query": row["query"],
                "reference": row["reference"],
                "text_evidence": [
                    {"rank": r["rank"], "text": r.get("text", "")}
                    for r in row["text_evidence"]
                ],
                "image_evidence": [
                    {"rank": r["rank"], "image_path": r.get("image_path", "")}
                    for r in row["image_evidence"]
                ],
            }
        )

    def flush(self) -> None:
        search.atomic_json(
            self.cache_path,
            {
                "schema": ANSWER_CACHE_SCHEMA,
                "model": self.model,
                "prompt_version": ANSWER_PROMPT_VERSION,
                "scoring_version": ANSWER_SCORING_VERSION,
                "records": self.records,
            },
        )

    def evaluate(self, config: search.SearchConfig) -> dict[str, Any]:
        prepared: list[dict[str, Any]] = []
        pending: dict[str, dict[str, Any]] = {}
        for item in self.queries:
            text_rows, image_rows = self.retrieval.evidence(config, item)
            row = {
                "id": str(item["id"]),
                "query": str(item["query"]),
                "reference": search.reference_for(item),
                "text_evidence": text_rows,
                "image_evidence": image_rows,
                "text_hit_at_k": int(any(
                    str(r.get("metadata", {}).get("source_id")) == str(item["id"])
                    for r in text_rows
                )),
                "image_hit_at_k": int(any(
                    str(r.get("id")) == str(item["id"]) for r in image_rows
                )),
            }
            row["fingerprint"] = self.fingerprint(row)
            prepared.append(row)
            if row["fingerprint"] not in self.records:
                pending.setdefault(row["fingerprint"], row)

        LOGGER.info(
            "answer config=%s queries=%s cache_hits=%s new_calls=%s",
            config.config_id, len(prepared), len(prepared) - len(pending), len(pending),
        )
        if pending:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(answer_with_evidence, row, self.model): fingerprint
                    for fingerprint, row in pending.items()
                }
                for future in as_completed(futures):
                    fingerprint = futures[future]
                    answer = future.result()  # API failures must not become zero reward.
                    row = pending[fingerprint]
                    scored = normalized_answer(row["query"], answer)
                    self.records[fingerprint] = {
                        "id": row["id"],
                        "reference": row["reference"],
                        "raw_answer": answer,
                        "scored_answer": scored,
                        "token_f1": token_f1(scored, row["reference"]),
                        "exact_match": int(
                            scored.strip().casefold() == row["reference"].strip().casefold()
                        ),
                    }
                    self.flush()

        evaluations = [
            {
                **self.records[row["fingerprint"]],
                "answer_fingerprint": row["fingerprint"],
                "text_hit_at_k": row["text_hit_at_k"],
                "image_hit_at_k": row["image_hit_at_k"],
            }
            for row in prepared
        ]
        self.retrieval.flush()
        return {
            "config_id": config.config_id,
            "metric": "mean_token_f1",
            "mean_token_f1": float(np.mean([r["token_f1"] for r in evaluations])),
            "choice_accuracy": float(np.mean([r["exact_match"] for r in evaluations])),
            "mean_text_hit_at_k": float(np.mean([r["text_hit_at_k"] for r in evaluations])),
            "mean_image_hit_at_k": float(np.mean([r["image_hit_at_k"] for r in evaluations])),
            "query_count": len(evaluations),
            "answer_cache_hits": len(prepared) - len(pending),
            "answer_new_calls": len(pending),
            "evaluations": evaluations,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--query-ids", nargs="+", default=list(DEFAULT_QUERY_IDS))
    parser.add_argument("--answer-model", default="qwen3.7-flash")
    parser.add_argument("--eval-concurrency", type=int, default=3)
    parser.add_argument("--answer-cache-from", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_concurrency < 1 or not args.query_ids or len(set(args.query_ids)) != len(args.query_ids):
        raise ValueError("query IDs must be nonempty and unique; concurrency must be positive")
    qa, _ = load_dataset(PROJECT / "datasets/train_0828", "qa_search.json")
    fixed, fixed_hash = paired.load_fixed_sample(
        qa, SOURCE_RUN / "seed_25/query_sample.json"
    )
    wanted = set(args.query_ids)
    sample = [row for row in fixed if str(row["id"]) in wanted]
    if len(sample) != len(wanted):
        raise ValueError("all query IDs must belong to the existing fixed 100-query set")
    candidates = search.all_configs()
    initial = paired.seeded_initial_batch(candidates, args.seed)
    manifest = {
        "experiment": "answer-token-F1-guided two-round JIT smoke test",
        "run_id": args.run_id,
        "metric": "mean_token_f1",
        "answer_model": args.answer_model,
        "temperature": 0.0,
        "prompt_version": ANSWER_PROMPT_VERSION,
        "scoring_version": ANSWER_SCORING_VERSION,
        "answer_cache_from": args.answer_cache_from,
        "query_ids": [str(row["id"]) for row in sample],
        "fixed_100_query_keys_sha256": fixed_hash,
        "selected_query_keys_sha256": search.sha256_json(
            [search.QueryEmbeddingCache.query_key(row) for row in sample]
        ),
        "seed": args.seed,
        "rounds": 2,
        "configs_per_round": 5,
        "search_space_count": len(candidates),
        "first_batch": [cfg.as_dict() for cfg in initial],
        "eval_concurrency": args.eval_concurrency,
        "retrieval_cache_source": str(SOURCE_RUN / "seed_25/retrieval_cache.json"),
        "embedding_cache_source": str(paired.EMBEDDING_SOURCE),
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    root = PROJECT / "runs" / args.run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {root}")
    started = time.perf_counter()
    root.mkdir(parents=True)
    search.atomic_json(root / "manifest.json", manifest)
    paired.copy_and_check_embedding_cache(qa, root)
    shutil.copy2(SOURCE_RUN / "seed_25/retrieval_cache.json", root / "retrieval_cache.json")
    if args.answer_cache_from:
        shutil.copy2(Path(args.answer_cache_from), root / "answer_cache.json")
    embedding_cache = search.QueryEmbeddingCache(root, qa, "cpu", 16)
    for backend in search.EMBEDDING_VALUES:
        embedding_cache.ensure(backend)
    es = search.es_client("http://127.0.0.1:19200")
    try:
        retrieval = search.RetrievalCache(root / "retrieval_cache.json", es, embedding_cache)
        evaluator = AnswerF1Evaluator(
            sample, retrieval, root / "answer_cache.json", args.answer_model,
            args.eval_concurrency,
        )
        memory = search.MemoryV2()
        rng = random.Random(args.seed)
        tried: set[str] = set()
        trajectory: list[dict[str, Any]] = []
        for round_number in (1, 2):
            if round_number == 1:
                batch = initial
                diagnostics = {
                    cfg.config_id: {"selection": "shared_space_filling"} for cfg in batch
                }
            else:
                remaining = [cfg for cfg in candidates if cfg.config_id not in tried]
                batch, diagnostics = memory.select_jit(remaining, rng)
            round_rows: list[dict[str, Any]] = []
            for slot, cfg in enumerate(batch, start=1):
                if cfg.config_id in tried:
                    raise RuntimeError(f"JIT selected duplicate config: {cfg.config_id}")
                t0 = time.perf_counter()
                evaluation = evaluator.evaluate(cfg)
                episodic = memory.add(cfg, evaluation["mean_token_f1"], round_number)
                search.append_jsonl(root / "episodic_memory.jsonl", episodic)
                row = {
                    "round": round_number,
                    "slot": slot,
                    "config": cfg.as_dict(),
                    "selection": diagnostics[cfg.config_id],
                    "evaluation": search.clean_evaluation_summary(evaluation),
                    "elapsed_seconds": round(time.perf_counter() - t0, 3),
                }
                search.append_jsonl(root / "trajectory.jsonl", row)
                round_rows.append({**row, "query_evaluations": evaluation["evaluations"]})
                trajectory.append(row)
                tried.add(cfg.config_id)
            rules = memory.distill(round_number)
            search.atomic_json(root / f"round_{round_number:02d}.json", {
                "round": round_number, "batch": round_rows, "semantic_rule_count": len(rules)
            })
            search.atomic_json(root / "semantic_memory.json", rules)
        best = max(trajectory, key=lambda row: row["evaluation"]["mean_token_f1"])
        summary = {
            "run_id": args.run_id,
            "metric": "mean_token_f1",
            "configs_evaluated": len(trajectory),
            "unique_configs_evaluated": len(tried),
            "rounds_completed": 2,
            "query_count_per_config": len(sample),
            "best_config": best["config"],
            "best_mean_token_f1": best["evaluation"]["mean_token_f1"],
            "best_round": best["round"],
            "total_answer_api_calls": sum(row["evaluation"]["answer_new_calls"] for row in trajectory),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        search.atomic_json(root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        es.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
