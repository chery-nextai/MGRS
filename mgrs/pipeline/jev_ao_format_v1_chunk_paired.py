#!/usr/bin/env python3
"""AO 格式包上的 chunk_size 搜索维度实验（288 空间）。

与 `jev_ao_format_v1_paired.py` 的**唯一差异是搜索空间**，其余（题集、六策略、
奖励口径、初始批次设计、冻结评分缓存、run 产物布局）全部保持一致，因此两者可比。

搜索空间
========
    text_top_k(4) × image_top_k(6) × chunk_size(4) × embedding_backend(3) = 288

原空间（864）里的 text_weight / rrf_k 被固定为 pharma(chunk=512) 上的中位取值，
换出 chunk_size 这一维。**这是一个假设**：两个旋钮原先正是因为在 chunk=512 下
实测 η²≈0.003/0.008 才被判为摆设，而 chunk 变细会改变 RRF 的排名分布。

实现方式
========
`rq1_memory_v2_chunk_dim.install()` 在进程内替换 `search` 模块的属性（以及 `six`
绑定死的 LLMController / SYSTEM_PROMPT）。不修改任何共享模块，scienceqa 的 864
那条线保持字节级可复现，随时可回退。

索引
====
文本索引随 chunk_size 变化，图像索引与切分无关（跨 chunk 复用，不重复打图像索引）：

    chunk=512 → <prefix>_text_embedding_v2_<backend>          （既有索引，无 _c512 后缀）
    chunk=其他 → <prefix>_c<chunk>_text_embedding_v2_<backend>
    图像       → <prefix>_image_embedding_v2_<backend>

检索缓存的文本两路按键 `{backend}:c{chunk}:{query_key}` 隔离（否则新的 chunk
配置会静默复用 chunk=512 的文本排名）；图像一路按 `{backend}:image:{query_key}`
缓存。schema 升版后旧缓存自动失效。

注意：本脚本会消耗付费 JEV 额度。`--plan-only` 与 `--validate-only` 均不调 JEV。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from mgrs.pipeline import jev_initial_batch_repeat6 as six
from mgrs.pipeline import run_layout as layout
from mgrs.pipeline import rq1_memory_v2_chunk_dim as chunk
from mgrs.pipeline import rq1_memory_v2_jit_random as search
from mgrs.pipeline.jev_ao_format_v1_paired import (
    DATASETS,
    DEFAULT_DATASETS_ROOT,
    DEFAULT_RELAY_TOKEN_FROM,
    ensure_relay_token,
    load_queries,
    resolve_dataset,
)

LOG = logging.getLogger("jev_ao_format_v1_chunk_paired")
PROJECT = Path(__file__).resolve().parents[2]

# 图像转写（Qwen）只依赖图片内容，与搜索配置无关，可跨「同数据集」的 run 复用。
# 本实验的图片集与 20261020 paired(seed67) 一致（实测 874 张里 873 张命中），
# 若不预置就会冷启动把 ~1088 张图重转一遍（约 90 分钟），故默认从该 run 继承。
DEFAULT_VISION_CACHE_FROM = "20261020_jev_ao_paired_pharmaceuticals_q50_budget30_seed67"


def _resolve_run_dir(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else PROJECT / "runs" / path


def seed_vision_cache(run: Path, source: str) -> int:
    """把同数据集 run 的 vision_facts.jsonl 并入本 run（只补缺，不覆盖）。

    只追加目标里没有的 image_sha256，避免同哈希不同描述触发 load_jsonl 的
    冲突校验；model / prompt_version 与当前常量不符的行直接跳过。
    """
    if not source.strip():
        return 0
    src_dir = _resolve_run_dir(source)
    src = src_dir / "vision_facts.jsonl"
    target = run / "vision_facts.jsonl"
    if not src.is_file():
        LOG.warning("vision 缓存源缺 vision_facts.jsonl，跳过预置: %s", src)
        return 0
    have: dict[str, Any] = {}
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                have[str(row["image_sha256"])] = row
    added = skipped = 0
    with target.open("a", encoding="utf-8") as handle:
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            digest = str(row["image_sha256"])
            if digest in have:
                continue
            if (row.get("model") != six.VISION_MODEL
                    or row.get("prompt_version") != six.VISION_PROMPT_VERSION):
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            have[digest] = row
            added += 1
        handle.flush()
        os.fsync(handle.fileno())
    LOG.info("vision 缓存预置 源=%s 追加=%s 跳过=%s 合计=%s",
             src_dir.name, added, skipped, len(have))
    return added


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AO 格式包 chunk_size 搜索维度实验（JEV 冻结奖励，288 空间）"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                        help="预置数据集；与 --dataset-dir/--index-prefix 二选一")
    parser.add_argument("--datasets-root", default=str(DEFAULT_DATASETS_ROOT))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--index-prefix", default=None)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default=str(PROJECT / "runs"))
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--initial-seed-offsets", default="0,1,2,3,4,5")
    parser.add_argument("--policies", default=",".join(six.ALL_POLICIES))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--per-round", type=int, default=5)
    parser.add_argument("--es-url", default="http://127.0.0.1:19200")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--jev-api", action="store_true",
                        help="直连判题网关（Bearer key），不走本机回环 relay")
    parser.add_argument("--jev-api-url", default=six.JEV_API_URL)
    parser.add_argument("--jev-api-key-env", default=six.JEV_API_KEY_ENV)
    parser.add_argument("--relay-token-from", default=DEFAULT_RELAY_TOKEN_FROM)
    parser.add_argument("--vision-cache-from", default=DEFAULT_VISION_CACHE_FROM,
                        help="同数据集 run（目录名或绝对路径），从中补齐 "
                             "vision_facts.jsonl 转写缓存；置空则冷启动从头转写")
    parser.add_argument("--chunk-values", default="",
                        help="收窄 chunk 维度（逗号分隔，须是 %s 的子集），"
                             "用于小规模探针" % ",".join(str(c) for c in chunk.ALL_BUILT_CHUNKS))
    parser.add_argument("--validate-only", action="store_true",
                        help="只校验搜索空间与索引，不建 run、不调 JEV，然后退出")
    parser.add_argument("--plan-only", action="store_true",
                        help="写 manifest/preflight 并打印初始批次，不调 JEV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 必须在任何 search.all_configs() / six.initial_batches() 之前装空间。
    if args.chunk_values.strip():
        selected = chunk.set_chunk_values(
            [int(x) for x in args.chunk_values.split(",") if x.strip() != ""]
        )
        LOG.info("chunk 维度收窄为 %s", list(selected))
    space = chunk.install(six_module=six)
    space_check = chunk.verify_space()
    LOG.info("搜索空间 %s 候选数=%s chunk=%s",
             space["space_version"], space["candidate_count"], list(chunk.CHUNK_VALUES))

    spec = resolve_dataset(args)
    dataset_dir: Path = spec["dataset_dir"]
    prefix: str = spec["index_prefix"]

    # 文本索引随 chunk 变化，这里同时把 BACKENDS 的占位名改写到目标数据集前缀。
    text_indices = chunk.apply_backends(prefix)
    qa, corpus = load_queries(dataset_dir, spec["qa_file"])
    query_count = len(qa)

    candidates = search.all_configs()
    if len(candidates) != int(space["candidate_count"]):
        raise RuntimeError(
            f"搜索空间变化: {len(candidates)} != {space['candidate_count']}"
        )

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for policy in policies:
        if policy not in six.ALL_POLICIES:
            raise ValueError(f"未知策略: {policy}")
    offsets = [int(x) for x in args.initial_seed_offsets.split(",") if x.strip() != ""]
    if len(offsets) != args.repeats:
        raise ValueError("--initial-seed-offsets 数量必须等于 --repeats")
    if query_count < args.per_round:
        raise RuntimeError(f"题数 {query_count} 少于每轮配置数 {args.per_round}")

    es = search.es_client(args.es_url)
    try:
        indices = chunk.validate_indices(es, prefix)
        LOG.info("索引校验通过: %s 个索引", len(indices))
        base = {
            "dataset": spec["name"],
            "dataset_dir": str(dataset_dir),
            "index_prefix": prefix,
            "corpus_pages": len(corpus),
            "query_count": query_count,
            "search_space": space,
            "space_check": space_check,
            "index_doc_counts": {k: v["count"] for k, v in indices.items()},
        }
        if args.validate_only:
            print(json.dumps({"mode": "validate_only", **base},
                             ensure_ascii=False, indent=2))
            return

        init_seeds, batches = six.initial_batches(
            candidates, args.seed, args.repeats, offsets)
        sample_hash = search.sha256_json(
            [search.QueryEmbeddingCache.query_key(row) for row in qa])

        budget = args.rounds * args.per_round
        narrowed = tuple(chunk.CHUNK_VALUES) != tuple(chunk.ALL_BUILT_CHUNKS)
        chunk_tag = ("_c" + "-".join(str(c) for c in chunk.CHUNK_VALUES)) if narrowed else ""
        run_id = args.run_id or (
            f"{time.strftime('%Y%m%d')}_jev_ao_chunk_paired_{spec['name']}"
            f"_q{query_count}_budget{budget}{chunk_tag}"
        )
        run = Path(args.runs_dir) / run_id
        run.mkdir(parents=True, exist_ok=True)

        access_mode = "direct_api" if args.jev_api else "relay"
        api_key = ""
        if args.jev_api:
            api_key = os.environ.get(args.jev_api_key_env, "").strip()
            if not api_key:
                raise RuntimeError(
                    f"{args.jev_api_key_env} 未在环境中提供；直连模式需要 JEV API key")
        else:
            ensure_relay_token(run, args.relay_token_from)

        manifest = {
            "experiment": (
                "AO 格式包（pharmaceuticals）chunk_size 搜索维度实验："
                "搜索空间换成 text_top_k × image_top_k × chunk_size × embedding_backend，"
                "同一 seed、六次重复、初始五步不同，JEV 冻结奖励"
            ),
            "run_id": run_id,
            "dataset": spec["name"],
            "dataset_dir": str(dataset_dir),
            "qa_file": spec["qa_file"],
            "index_prefix": prefix,
            "source": {
                "qa_sha256": six.em.file_sha256(dataset_dir / spec["qa_file"]),
                "corpus_sha256": six.em.file_sha256(dataset_dir / "corpus.json"),
                "corpus_pages": len(corpus),
                "query_count": query_count,
                "sample_hash": sample_hash,
            },
            "search_space": space,
            "chunk_values": list(chunk.CHUNK_VALUES),
            "text_indices": text_indices,
            "image_indices": {
                backend: chunk.image_index_name(prefix, backend)
                for backend in chunk.EMBEDDING_VALUES
            },
            "retrieval_cache": {
                "schema": chunk.SPACE_VERSION,
                "text_key_template": "{backend}:c{chunk}:{query_key}",
                "image_key_template": "{backend}:image:{query_key}",
                "image_reused_across_chunks": True,
            },
            "base_seed": args.seed,
            "repeats": args.repeats,
            "initial_batch_seeds": init_seeds,
            "initial_batch_config_ids": {
                f"rep_{i + 1:02d}": [cfg.config_id for cfg in batch]
                for i, batch in enumerate(batches)
            },
            "initial_batch_design": (
                "six.initial_batches: first point drawn from random.Random(init_seed), "
                "remaining four by maximin space filling"
            ),
            "shared_within_repeat": True,
            "policies": policies,
            "configs_per_round": args.per_round,
            "rounds": args.rounds,
            "budget_per_policy": budget,
            "candidate_count": len(candidates),
            "reward_key": six.REWARD_KEY,
            "reward_schema": six.REWARD_SCHEMA,
            "reward_scheme": six.REWARD_SCHEME,
            "reward_definition": six.REWARD_DEFINITION,
            "vision_model": six.VISION_MODEL,
            "vision_prompt_version": six.VISION_PROMPT_VERSION,
            "jev_model": six.JEV_MODEL,
            "decision_model": six.DECISION_MODEL,
            "frozen_config_scores_shared_across_repeats": True,
            "indices": indices,
            "cache_reused_from": None,
            "notes": (
                "题集为数据集自带 qa_search.json 全量；多个 references 取首条作候选答案。"
                "JevEvaluator 的 text_hit_at_k/image_hit_at_k 沿用 source_id == query_id "
                "判据，在本数据集的 q:<n> / p:<n> 命名空间下恒为 0，不具召回含义，"
                "请勿当作 Recall 使用。"
                f"本实验用 chunk_size 替换了 text_weight / rrf_k 两个维度："
                f"text_weight={chunk.FIXED_TEXT_WEIGHT}、rrf_k={chunk.FIXED_RRF_K} 被固定为 "
                "pharma(chunk=512) 上的中位取值，这是一个**假设**——两个旋钮原先正是在 "
                "chunk=512 下被实测判定为摆设的，chunk 变细会改变 RRF 的排名分布，"
                "因此跨 chunk 的比较应理解为「该设置下」的效应。"
                "chunk=512 的文本索引沿用既有命名（无 _c512 后缀）。"
            ),
        }
        # plan-only 预演必须纯只读：若先写了 manifest/preflight/query_sample，
        # 之后正式跑会因 save_once 的字节级幂等校验不同而直接失败。
        if args.plan_only:
            print(json.dumps(
                {"mode": "plan_only", **base, "run": str(run),
                 "initial_batch_config_ids": manifest["initial_batch_config_ids"]},
                ensure_ascii=False, indent=2))
            return
        six.em.save_once(run / "manifest.json", manifest)
        search.atomic_json(
            run / "run_info.json",
            {
                "jev_access_mode": "direct_api" if args.jev_api else "relay",
                "jev_url": args.jev_api_url if args.jev_api else six.RELAY_URL,
                "jev_long_context_header": six.JEV_LONG_CONTEXT_HEADER,
                "https_proxy": os.environ.get("HTTPS_PROXY", ""),
                "no_proxy": os.environ.get("NO_PROXY", ""),
                "es_url": args.es_url,
                "embedding_device": args.embedding_device,
                "space_version": chunk.SPACE_VERSION,
                "chunk_values": list(chunk.CHUNK_VALUES),
                "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        preflight = {
            "base_seed": args.seed,
            "repeats": args.repeats,
            "policies": policies,
            "budget_per_policy": budget,
            "search_space": space,
            "initial_batch_seeds": init_seeds,
            "repeats_distinct": True,
            "initial_batch_config_ids": manifest["initial_batch_config_ids"],
        }
        six.em.save_once(run / "preflight.json", preflight)
        six.em.save_once(
            layout.cache_dir(run) / "query_sample.json",
            {
                "seed": args.seed,
                "sample_size": query_count,
                "sample_hash": sample_hash,
                "queries": [
                    {
                        "id": str(row["id"]),
                        "query_key": search.QueryEmbeddingCache.query_key(row),
                        "query": str(row["query"]),
                        "reference": search.reference_for(row),
                    }
                    for row in qa
                ],
            },
        )
        transport: Any = (
            six.make_transport(args.jev_api_url, api_key)
            if args.jev_api
            else six.RelayTransport(
                six.RELAY_URL,
                (run / "relay_token").read_text(encoding="utf-8").strip(),
            )
        )
        reachable, detail = transport.probe()
        if not reachable:
            raise RuntimeError(
                f"JEV {transport.mode} 不可达 {transport.url}: {detail}")
        LOG.info("jev access mode=%s url=%s probe=%s",
                 transport.mode, transport.url, detail)

        embedding = search.QueryEmbeddingCache(
            run, qa, args.embedding_device, args.embedding_batch_size)
        for backend in chunk.EMBEDDING_VALUES:
            embedding.ensure(backend)
        retrieval = chunk.ChunkRetrievalCache(
            run / "retrieval_cache.json", es, embedding, prefix)
        seed_vision_cache(run, args.vision_cache_from)
        vision = six.VisionCache(run)
        cache_root = layout.cache_dir(run)
        evaluator = six.FrozenConfigEvaluator(
            six.JevEvaluator(
                qa, retrieval, vision, transport,
                cache_root / "jev_responses.jsonl",
            ),
            cache_root / "config_scores",
            sample_hash,
            query_count,
        )
        pairs = []
        for repeat in range(1, args.repeats + 1):
            initial = batches[repeat - 1]
            summaries = {}
            for policy in policies:
                summaries[policy] = six.run_policy(
                    args.seed, repeat, policy, candidates, initial, evaluator,
                    run, args.rounds, args.per_round,
                )
            pair = {
                "seed": args.seed,
                "repeat": repeat,
                "initial_batch_seed": init_seeds[repeat - 1],
                "initial_config_ids": [cfg.config_id for cfg in initial],
                "policies": summaries,
            }
            search.atomic_json(
                layout.seed_dir(run, repeat - 1) / "repeat_summary.json", pair)
            pairs.append(pair)
            LOG.info("seed=%s rep=%02d complete best=%s", args.seed, repeat,
                     {p: round(s["best_mean_jev_support"], 4)
                      for p, s in summaries.items()})
        aggregate = {}
        for policy in policies:
            bests = [pair["policies"][policy]["best_mean_jev_support"] for pair in pairs]
            steps = [pair["policies"][policy]["best_step"] for pair in pairs]
            aggregate[policy] = {
                "mean_best": float(np.mean(bests)),
                "std_best": float(np.std(bests)),
                "min_best": float(np.min(bests)),
                "max_best": float(np.max(bests)),
                "mean_best_step": float(np.mean(steps)),
                "best_per_repeat": bests,
                "best_step_per_repeat": steps,
            }
        search.atomic_json(
            run / "summary.json",
            {
                "run_id": run_id,
                "dataset": spec["name"],
                "dataset_dir": str(dataset_dir),
                "index_prefix": prefix,
                "query_count": query_count,
                "base_seed": args.seed,
                "repeats": args.repeats,
                "policies": policies,
                "search_space": space,
                "chunk_values": list(chunk.CHUNK_VALUES),
                "reward_key": six.REWARD_KEY,
                "reward_schema": six.REWARD_SCHEMA,
                "initial_batch_seeds": init_seeds,
                "initial_batch_config_ids": manifest["initial_batch_config_ids"],
                "aggregate": aggregate,
                "pairs": pairs,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        LOG.info("complete summary=%s", run / "summary.json")
    finally:
        es.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()