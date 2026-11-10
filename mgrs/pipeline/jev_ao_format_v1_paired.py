#!/usr/bin/env python3
"""AO 格式包（pharmaceuticals / industrial_subset_v1）上的六策略配对对比实验。

复用 jev_initial_batch_repeat6.py 的全部策略实现与 JEV 奖励层，只替换三处
数据集相关的写死项：

1. 题集改用数据集自带的 qa_search.json（上游官方划分），不重新抽样，也不使用
   ScienceQA 的数字 ID 抽样器。多个 references 时取首条作候选答案，与
   search.reference_for 的既有语义一致。
2. 运行时改写 search.BACKENDS 的索引名到
   aoexplore_<name>_p<pages>_{text,image}_embedding_v2_<backend>，
   不动共享模块（因此 validate_indices 的 3944 写死不受影响 —— 本脚本不调用它）。
3. 不复制 cache_source 的 ScienceQA 缓存（retrieval_cache.json /
   vision_facts.jsonl / query_embeddings / config_scores / jev_responses），
   新数据集从零重建；只复用 relay_token。

奖励口径不变：JEV Jev_AO 对固定题的 P(supported) 均值，图像先由
Qwen3.7 Flash 在不知道问题的情况下转写。同一 seed 的六次重复共用同一份
冻结配置评分缓存，跨重复差异只来自初始批次。

注意：本脚本会消耗付费 JEV 额度。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from mgrs.pipeline import jev_initial_batch_repeat6 as six
from mgrs.pipeline import run_layout as layout
from mgrs.pipeline import rq1_memory_v2_jit_random as search

LOG = logging.getLogger("jev_ao_format_v1_paired")
PROJECT = Path(__file__).resolve().parents[2]
# AO 格式语料根目录（数据集正文/图像所在目录）。默认 <repo>/datasets/ao_ready_v1，
# 可用 MGRS_AO_CORPUS_ROOT 覆盖；单个数据集再用 MGRS_AO_CORPUS_DIR_<NAME> 指定，
# 例如 MGRS_AO_CORPUS_DIR_TECHSLIDES=/data/ao_ready_v1/techslides/data。
DEFAULT_DATASETS_ROOT = Path(
    os.environ.get("MGRS_AO_CORPUS_ROOT", "").strip()
    or (PROJECT / "datasets" / "ao_ready_v1")
)
# 历史 relay 部署的 token 来源 run；默认不使用（判题走 --jev-api 直连）。
DEFAULT_RELAY_TOKEN_FROM = os.environ.get("MGRS_RELAY_TOKEN_FROM", "").strip()


def _corpus_dir(name: str, relative: str) -> str:
    """数据集语料目录：优先取 per-dataset 环境变量，否则拼在语料根目录下。"""
    override = os.environ.get(f"MGRS_AO_CORPUS_DIR_{name.upper()}", "").strip()
    return override or str(DEFAULT_DATASETS_ROOT / relative)


def _index_prefix(name: str, default: str) -> str:
    """ES 索引前缀：自建索引时用 MGRS_INDEX_PREFIX_<NAME> 覆盖数据集自带前缀。"""
    return os.environ.get(f"MGRS_INDEX_PREFIX_{name.upper()}", "").strip() or default

# 论文定版用的三个知识库。每个条目：
#   dir          语料目录（相对语料根目录，或用 MGRS_AO_CORPUS_DIR_<NAME> 给绝对路径）
#   index_prefix ES 索引前缀；apply_backends() 据此拼出
#                <prefix>_text_embedding_v2_<backend> / <prefix>_image_embedding_v2_<backend>
#   pages        语料页数（用于启动前的索引体量自检）
#   qa_file      语料目录内的默认题集；定版实验用 --qa-file 显式指定精选 50 题（见 scripts/）
# 注：pharmaceuticals 的语料来自另一批交接包（目录内直接是 corpus.json，无 data/ 子目录），
# 与 finance_en / techslides 不同根，通常需要单独设 MGRS_AO_CORPUS_DIR_PHARMACEUTICALS。
DATASETS: dict[str, dict[str, Any]] = {
    "pharmaceuticals": {
        "dir": _corpus_dir("pharmaceuticals", "pharmaceuticals"),
        "index_prefix": _index_prefix("pharmaceuticals", "aoexplore_pharmaceuticals_p2313"),
        "pages": 2313,
        "qa_file": "qa_search.json",
    },
    "finance_en": {
        "dir": _corpus_dir("finance_en", "finance_en/data"),
        "index_prefix": _index_prefix("finance_en", "mgrs_ao_finance_en_b36d2b08b6fc_handoff_v1"),
        "pages": 2942,
        "qa_file": "qa_search.json",
    },
    "techslides": {
        "dir": _corpus_dir("techslides", "techslides/data"),
        "index_prefix": _index_prefix("techslides", "mgrs_ao_techslides_b86435175991_handoff_v1"),
        "pages": 1963,
        "qa_file": "qa_search.json",
    },
}


def apply_backends(prefix: str) -> None:
    """把 search.BACKENDS 的索引名改写到目标数据集前缀（dims/model_path 不变）。"""
    for backend, info in search.BACKENDS.items():
        info["text_index"] = f"{prefix}_text_embedding_v2_{backend}"
        info["image_index"] = f"{prefix}_image_embedding_v2_{backend}"


def validate_dataset_indices(es: Any, prefix: str) -> dict[str, Any]:
    """数据集无关校验：索引存在 + 非空 + dims 与 BACKENDS 声明一致。

    刻意不校验精确文档数：文本索引按各 backend 的分词器切 chunk，文档数本就
    不同（pharma 实测 3118/3182/3182），图像索引等于页数。
    """
    checked: dict[str, Any] = {}
    for backend, info in search.BACKENDS.items():
        for kind in ("text_index", "image_index"):
            name = str(info[kind])
            if not es.indices.exists(index=name):
                raise RuntimeError(
                    f"缺少索引 {kind}[{backend}]: {name}（确认已建库且前缀正确）"
                )
            count = int(es.count(index=name)["count"])
            if count <= 0:
                raise RuntimeError(f"索引为空 {kind}[{backend}]: {name}")
            dims = int(
                es.indices.get_mapping(index=name)[name]["mappings"]["properties"][
                    "embedding"
                ]["dims"]
            )
            if dims != int(info["dims"]):
                raise RuntimeError(f"dims 不一致 {name}: {dims} != {info['dims']}")
            checked[name] = {
                "backend": backend,
                "kind": kind,
                "count": count,
                "dims": dims,
            }
    return checked


def load_queries(
    dataset_dir: Path, qa_file: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读官方划分的 qa_search.json 与 corpus.json，并做最小结构校验。"""
    qa_path = dataset_dir / qa_file
    corpus_path = dataset_dir / "corpus.json"
    if not qa_path.is_file():
        raise RuntimeError(f"缺少 QA 文件: {qa_path}")
    if not corpus_path.is_file():
        raise RuntimeError(f"缺少语料文件: {corpus_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(qa, list) or not isinstance(corpus, list):
        raise ValueError("qa/corpus 顶层必须是 list")
    for row in qa:
        if not str(row.get("id") or "").strip():
            raise ValueError(f"QA 行缺少 id: {row}")
        if not str(row.get("query") or "").strip():
            raise ValueError(f"QA 行缺少 query: {row.get('id')}")
        if not search.reference_for(row).strip():
            raise ValueError(f"QA 行缺少参考答案: {row.get('id')}")
    return qa, corpus


def ensure_relay_token(run: Path, source: str) -> None:
    target = run / "relay_token"
    if target.exists():
        return
    src = Path(source)
    if not src.is_absolute():
        src = PROJECT / "runs" / source
    tried = [src, src / "relay_token"] if src.is_dir() else [src]
    for candidate in tried:
        if candidate.is_file():
            shutil.copy2(candidate, target)
            LOG.info("relay_token 复制自 %s", candidate)
            return
    raise RuntimeError(
        f"找不到 relay_token，已尝试 {[str(t) for t in tried]}；"
        "用 --relay-token-from 指定来源，或改用 --jev-api 直连模式"
    )


def resolve_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset:
        spec = dict(DATASETS[args.dataset])
        spec["name"] = args.dataset
        spec["dataset_dir"] = Path(args.datasets_root) / spec["dir"]
        if args.qa_file:
            spec["qa_file"] = args.qa_file
    else:
        if not args.dataset_dir or not args.index_prefix:
            raise SystemExit("需要 --dataset，或同时提供 --dataset-dir 与 --index-prefix")
        spec = {
            "name": Path(args.dataset_dir).name,
            "dataset_dir": Path(args.dataset_dir),
            "index_prefix": args.index_prefix,
            "qa_file": args.qa_file or "qa_search.json",
            "pages": None,
        }
    if not Path(spec["dataset_dir"]).is_dir():
        raise SystemExit(f"数据集目录不存在: {spec['dataset_dir']}")
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AO 格式包六策略配对对比实验（JEV 冻结奖励）"
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
    parser.add_argument("--validate-only", action="store_true",
                        help="只校验索引与题集，不建 run、不调 JEV，然后退出")
    parser.add_argument("--plan-only", action="store_true",
                        help="写 manifest/preflight 并打印六批初始配置，不调 JEV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = resolve_dataset(args)
    dataset_dir: Path = spec["dataset_dir"]
    prefix: str = spec["index_prefix"]

    apply_backends(prefix)
    qa, corpus = load_queries(dataset_dir, spec["qa_file"])
    query_count = len(qa)

    candidates = search.all_configs()
    if len(candidates) != 864:
        raise RuntimeError(f"搜索空间变化: {len(candidates)} != 864")

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
        indices = validate_dataset_indices(es, prefix)
        LOG.info("索引校验通过: %s", json.dumps(
            {k: v["count"] for k, v in indices.items()}, ensure_ascii=False))
        base = {
            "dataset": spec["name"],
            "dataset_dir": str(dataset_dir),
            "index_prefix": prefix,
            "corpus_pages": len(corpus),
            "query_count": query_count,
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
        run_id = args.run_id or (
            f"{time.strftime('%Y%m%d')}_jev_ao_paired_{spec['name']}"
            f"_q{query_count}_budget{budget}"
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
                "AO 格式包（pharmaceuticals / industrial_subset_v1）六策略配对对比："
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
            ),
        }
        # plan-only 预演必须纯只读：若先写了 manifest/preflight/query_sample，
        # 之后正式跑会因 save_once 的字节级幂等校验不同而直接失败（本 run 已两次踩到）。
        if args.plan_only:
            print(json.dumps(
                {"mode": "plan_only", **base, "run": str(run),
                 "initial_batch_config_ids": manifest["initial_batch_config_ids"]},
                ensure_ascii=False, indent=2))
            return
        six.em.save_once(run / "manifest.json", manifest)
        # 访问方式/代理属于运行环境，会随隧道或出口变化；
        # 若放进 manifest，save_once 会因字节差异直接报错，故单列一个可覆盖文件。
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
                "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        preflight = {
            "base_seed": args.seed,
            "repeats": args.repeats,
            "policies": policies,
            "budget_per_policy": budget,
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
        for backend in search.EMBEDDING_VALUES:
            embedding.ensure(backend)
        retrieval = search.RetrievalCache(
            run / "retrieval_cache.json", es, embedding)
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