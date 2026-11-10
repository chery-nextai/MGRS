#!/usr/bin/env python3
"""单点决策版 chunk 实验：共享初始 5 步（前置阶段）+ 之后每轮只决策 1 个配置。

与 `jev_ao_format_v1_chunk_paired.py` 的关系
===========================================
空间（288）、题集、六策略、奖励口径、六次重复、初始批次设计、冻结评分缓存全部相同，
**只有决策粒度**从「每轮 5 个」改成「每轮 1 个」：

    budget = 5（共享初始，前置阶段）+ 25（单点决策）= 30
    round 1     = 初始批次前置阶段（5 个配置，不算控制器步）
    round 2..26 = 控制器第 1..25 步，每步 1 个配置
    映射：控制器步 k ↔ evaluation_position k+5 ↔ round k+1

初始批次前置的实现方式：**预写** `plans/round_01.json`。`six.run_policy` 读到已存在的
计划文件时直接采用其中的 config_ids（该分支不做 `len(batch) == per_round` 校验），因此
不必改共享模块就能把「5 步初始批次」与「25 次单点决策」分开。预写内容与共享代码在
round 1 自行生成的内容逐字段一致（selection=shared_seeded_maximin、decision_seconds=0.0），
只是提前落盘，从而绕开批量校验。

缓存复用
========
`--cache-from` 指向同一空间的既有 run 时，沿用它的图像转写（vision_facts.jsonl）、
冻结配置评分（cache/config_scores/）、ES 原始检索结果（retrieval_cache.json）、
query embeddings（query_embeddings/*.npz）与 JEV 响应（cache/jev_responses.jsonl）。
复用前校验源 run 的 sample_hash / 空间版本 / chunk 取值 / index prefix，任一不符则
**整体拒绝复用**（fail-closed，绝不静默混用）。已评过的配置沿用同一分数，因此与源 run
严格可比，且只为新采到的配置付费。

注意：本脚本会消耗付费 JEV 额度。`--plan-only` 与 `--validate-only` 均不调 JEV。
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
from mgrs.pipeline import rq1_memory_v2_chunk_dim as chunk
from mgrs.pipeline import rq1_memory_v2_chunk_single as single
from mgrs.pipeline import rq1_memory_v2_jit_random as search
from mgrs.pipeline.jev_ao_format_v1_chunk_paired import seed_vision_cache
from mgrs.pipeline.jev_ao_format_v1_paired import (
    DATASETS,
    DEFAULT_DATASETS_ROOT,
    DEFAULT_RELAY_TOKEN_FROM,
    ensure_relay_token,
    load_queries,
    resolve_dataset,
)

LOG = logging.getLogger("jev_ao_format_v1_chunk_single")
PROJECT = Path(__file__).resolve().parents[2]

# six.initial_batches 固定返回 5 个配置（与批量版共享同一批起点）。
INITIAL_SIZE = 5
DEFAULT_CACHE_FROM = "20261022_jev_ao_chunk_paired_pharmaceuticals_q50_budget30"
BAND_EPSILON = 0.01

# 定版口径的默认值（scripts/run.sh 只改数据集/种子数，其余都走这里）：
# 每个 repeat 的初始 seed 偏移表，--repeats 取前 N 项；finance_en 的偏移顺序与另两个不同。
DEFAULT_SEED_OFFSETS = {
    "techslides": "0,1,2,3,4,5,-67,-66,-65,-25,710,1957,56,-54,32,33,247,8,9,13",
    "pharmaceuticals": "0,1,2,3,4,5,-67,-66,-65,-25,710,1957,56,-54,32,33,247,8,9,13",
    "finance_en": "0,1,2,3,4,5,8,9,13,-67,-66,-65,-25,710,1957,56,-54,32,33,247",
}
# 本地 ES 端口按数据集分开（其余数据集走 29200）。
DEFAULT_ES_URLS = {"pharmaceuticals": "http://127.0.0.1:19200"}
FALLBACK_ES_URL = "http://127.0.0.1:29200"
# 定版题集子集目录：<repo>/datasets/ao_ready_v1_subsets/<dataset>/qa_screened50_from_*.json
QA_SUBSET_DIR = "ao_ready_v1_subsets"


def _resolve_run_dir(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else PROJECT / "runs" / path


def cache_source_ok(src_dir: Path, seed: int, sample_hash: str, prefix: str) -> tuple[bool, str]:
    """复用前的 fail-closed 校验：题集 / 空间版本 / chunk 取值 / 索引前缀必须一致。"""
    sample_path = layout.cache_path(src_dir, "query_sample.json", legacy_seed=seed)
    if not sample_path.is_file():
        return False, f"源 run 缺 {sample_path.name}"
    if json.loads(sample_path.read_text(encoding="utf-8")).get("sample_hash") != sample_hash:
        return False, "源 run 的题集 sample_hash 不一致"
    manifest_path = src_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, "源 run 缺 manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("search_space") or {}).get("space_version") != chunk.SPACE_VERSION:
        return False, "源 run 的搜索空间版本不一致"
    if tuple(manifest.get("chunk_values") or ()) != tuple(chunk.CHUNK_VALUES):
        return False, "源 run 的 chunk 取值不一致"
    if manifest.get("index_prefix") != prefix:
        return False, f"源 run 的 index_prefix 不一致（{manifest.get('index_prefix')} != {prefix}）"
    return True, "校验通过"


def seed_caches(
    run: Path, source: str, seed: int, sample_hash: str, prefix: str, query_count: int
) -> dict[str, Any]:
    """把源 run 的转写/评分/检索/向量/JEV 响应缓存并入本 run（只补缺，不覆盖）。"""
    report: dict[str, Any] = {
        "source": None, "reused": False, "reason": "", "vision_added": 0,
        "scores_reused": 0, "scores_present": 0, "scores_incompatible": 0,
        "retrieval_reused": False, "embeddings_reused": [], "jev_responses_reused": False,
    }
    if not source.strip():
        report["reason"] = "cache_from 为空：冷启动，全部重新计算"
        return report
    src_dir = _resolve_run_dir(source)
    report["source"] = src_dir.name
    if not src_dir.is_dir():
        report["reason"] = f"源 run 目录不存在: {src_dir}"
        LOG.warning("cache 源不存在，冷启动: %s", src_dir)
        return report
    ok, reason = cache_source_ok(src_dir, seed, sample_hash, prefix)
    report["reason"] = reason
    if not ok:
        LOG.warning("拒绝复用缓存（fail-closed）源=%s 原因=%s", src_dir.name, reason)
        return report
    report["reused"] = True
    report["vision_added"] = seed_vision_cache(run, str(src_dir))
    dst_scores = layout.cache_dir(run) / "config_scores"
    dst_scores.mkdir(parents=True, exist_ok=True)
    src_scores = layout.cache_dir(src_dir) / "config_scores"
    if not src_scores.is_dir():
        src_scores = src_dir / f"seed_{seed}" / "config_scores"
    for path in sorted(src_scores.glob("*.json")):
        dst = dst_scores / path.name
        if dst.exists():
            report["scores_present"] += 1
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("sample_hash") != sample_hash
            or row.get("reward_schema") != six.REWARD_SCHEMA
            or int(row.get("query_count", -1)) != query_count
        ):
            report["scores_incompatible"] += 1
            continue
        shutil.copy2(path, dst)
        report["scores_reused"] += 1
    src_rc = src_dir / "retrieval_cache.json"
    dst_rc = run / "retrieval_cache.json"
    if src_rc.is_file() and not dst_rc.exists():
        cached = json.loads(src_rc.read_text(encoding="utf-8"))
        if cached.get("schema") == chunk.SPACE_VERSION and str(cached.get("prefix")) == prefix:
            shutil.copy2(src_rc, dst_rc)
            report["retrieval_reused"] = True
        else:
            LOG.warning("检索缓存 schema/prefix 不符，忽略: %s", src_rc)
    for backend in chunk.EMBEDDING_VALUES:
        src = src_dir / "query_embeddings" / f"{backend}.npz"
        dst = run / "query_embeddings" / f"{backend}.npz"
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            report["embeddings_reused"].append(backend)
    src_jr = layout.cache_path(src_dir, "jev_responses.jsonl", legacy_seed=seed)
    dst_jr = layout.cache_dir(run) / "jev_responses.jsonl"
    if src_jr.is_file() and not dst_jr.exists():
        dst_jr.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_jr, dst_jr)
        report["jev_responses_reused"] = True
    LOG.info(
        "缓存复用 源=%s 评分新增=%s(已有%s/不兼容%s) 转写追加=%s 检索=%s 向量=%s JEV响应=%s",
        src_dir.name, report["scores_reused"], report["scores_present"],
        report["scores_incompatible"], report["vision_added"], report["retrieval_reused"],
        report["embeddings_reused"], report["jev_responses_reused"],
    )
    return report


def write_initial_plan(
    run: Path, repeat: int, policy: str, initial: list[search.SearchConfig]
) -> Path:
    """预写 round 1 计划：让 `six.run_policy` 走计划缓存分支，跳过 per_round 数量校验。

    内容与共享代码在 round 1 自行生成的内容逐字段一致（初始批次是共享的 seeded
    maximin 设计、不计决策时间），因此「共享初始 5 步」不会被误当成一次批量决策。
    """
    root = layout.seed_dir(run, repeat - 1) / policy
    plan_path = root / "plans" / "round_01.json"
    if plan_path.exists():
        return plan_path
    search.atomic_json(
        plan_path,
        {
            "round": 1,
            "config_ids": [cfg.config_id for cfg in initial],
            "diagnostics": {
                cfg.config_id: {"selection": "shared_seeded_maximin"} for cfg in initial
            },
            "decision_seconds": 0.0,
            "note": "初始批次前置阶段（入口预写，非控制器决策）",
        },
    )
    return plan_path


def policy_step_stats(run: Path, repeat: int, policy: str) -> dict[str, Any]:
    """派生两种口径的步数指标：含初始 5 步的 evaluation_position 与只看控制器步。"""
    root = layout.seed_dir(run, repeat - 1) / policy
    rows = sorted(
        six.em.trajectory_rows(root / "trajectory.jsonl").values(),
        key=lambda row: int(row["evaluation_position"]),
    )
    scores = [float(row["evaluation"][six.REWARD_KEY]) for row in rows]
    if len(scores) <= INITIAL_SIZE:
        raise RuntimeError(f"轨迹只有 {len(scores)} 行，没有控制器步可统计")
    initial, controller = scores[:INITIAL_SIZE], scores[INITIAL_SIZE:]
    best = max(scores)
    first_hit = scores.index(best) + 1
    first_band = next(index + 1 for index, value in enumerate(scores) if value >= best - BAND_EPSILON)

    def controller_step(position: int) -> int | None:
        # 位置 <= 5 表示最优就出现在共享初始批次里，此时控制器步数无定义。
        return int(position - INITIAL_SIZE) if position > INITIAL_SIZE else None

    return {
        "initial_batch_size": INITIAL_SIZE,
        "initial_batch_best": float(max(initial)),
        "initial_batch_mean": float(np.mean(initial)),
        "controller_best": float(max(controller)) if controller else None,
        "controller_steps_evaluated": len(controller),
        "final_best_step": int(first_hit),
        "final_best_controller_step": controller_step(first_hit),
        "first_band_step": int(first_band),
        "first_band_controller_step": controller_step(first_band),
        "band_epsilon": BAND_EPSILON,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"AO 格式包 chunk 空间单点决策实验（JEV 冻结奖励，288 空间，"
                    f"{INITIAL_SIZE} 步共享初始 + 单点决策）"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                        help="预置数据集；与 --dataset-dir/--index-prefix 二选一")
    parser.add_argument("--datasets-root", default=str(DEFAULT_DATASETS_ROOT))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--index-prefix", default=None)
    parser.add_argument("--qa-file", default=None,
                        help="题集（相对数据集目录或绝对路径）；留空优先用"
                             " datasets/ao_ready_v1_subsets/<数据集>/qa_screened50_from_*.json，"
                             "没有再退回该数据集自带的 QA 文件")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default=str(PROJECT / "runs"))
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--repeats", type=int, default=20,
                        help="独立搜索过程数量（定版 20；--initial-seed-offsets 不足时取默认表前 N 项）")
    parser.add_argument("--initial-seed-offsets", default=None,
                        help="每个 repeat 的初始 seed 偏移（逗号分隔）；留空按数据集的定版表取前 --repeats 项")
    parser.add_argument("--policies", default=",".join(six.ALL_POLICIES))
    parser.add_argument("--controller-steps", type=int, default=25,
                        help=f"初始 {INITIAL_SIZE} 步之后的单点决策步数（每步 1 个配置）")
    parser.add_argument("--no-thinking", action="store_true",
                        help="决策模型关闭思考链（同一模型 + enable_thinking=false）")
    parser.add_argument("--es-url", default=None,
                        help="ES 地址；留空按数据集取本地端口（pharmaceuticals 19200，其余 29200）")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--jev-api", action="store_true",
                        help="直连判题网关（Bearer key），不走本机回环 relay")
    parser.add_argument("--jev-api-url", default=six.JEV_API_URL)
    parser.add_argument("--jev-api-key-env", default=six.JEV_API_KEY_ENV)
    parser.add_argument("--relay-token-from", default=DEFAULT_RELAY_TOKEN_FROM)
    parser.add_argument("--cache-from", default=DEFAULT_CACHE_FROM,
                        help="同空间既有 run（目录名或绝对路径），复用其转写/评分/检索/"
                             "向量/JEV 响应缓存；置空则冷启动全部重算")
    parser.add_argument("--source-scores", default="",
                        help="惰性派生源 run（目录名或绝对路径）：当前题集是其子集时，该 run 里"
                             "已评过的配置直接切子集重算均值（零调用、不写源）；"
                             "只有源里没有的配置才真实调用 JEV")
    parser.add_argument("--vision-from", default="",
                        help="额外只并入该 run 的 vision_facts.jsonl（按 image_sha256 补缺、"
                             "跳过 model/prompt_version 不符的行）。图片转写与题集无关，"
                             "而 seed_caches 的 fail-closed 校验要求 sample_hash 一致，"
                             "跨题集（如 qa_search → qa_test）无法经 --cache-from 复用，"
                             "这个口子专门省掉重复转写")
    parser.add_argument("--chunk-values", default="",
                        help="收窄 chunk 维度（逗号分隔，须是 %s 的子集），"
                             "用于小规模探针" % ",".join(str(c) for c in chunk.ALL_BUILT_CHUNKS))
    parser.add_argument("--text-values", default="",
                        help="收窄 text_top_k 维度（逗号分隔，须是 %s 的子集）"
                             % ",".join(str(v) for v in chunk.ALL_TEXT_VALUES))
    parser.add_argument("--image-values", default="",
                        help="收窄 image_top_k 维度（逗号分隔，须是 %s 的子集）"
                             % ",".join(str(v) for v in chunk.ALL_IMAGE_VALUES))
    parser.add_argument("--embedding-values", default="",
                        help="收窄检索后端维度（逗号分隔，须是 %s 的子集）"
                             % ",".join(chunk.ALL_EMBEDDING_VALUES))
    parser.add_argument("--validate-only", action="store_true",
                        help="只校验搜索空间与索引，不建 run、不调 JEV，然后退出")
    parser.add_argument("--plan-only", action="store_true",
                        help="打印初始批次/步数映射/缓存源，不建 run、不调 JEV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.controller_steps < 1:
        raise ValueError("--controller-steps 必须 >= 1")

    # 必须在任何 search.all_configs() / six.initial_batches() 之前装空间与控制器。
    try:
        if args.text_values.strip():
            LOG.info("text_top_k 维度收窄为 %s", list(chunk.set_text_values(
                [int(x) for x in args.text_values.split(",") if x.strip() != ""])))
        if args.image_values.strip():
            LOG.info("image_top_k 维度收窄为 %s", list(chunk.set_image_values(
                [int(x) for x in args.image_values.split(",") if x.strip() != ""])))
        if args.chunk_values.strip():
            selected = chunk.set_chunk_values(
                [int(x) for x in args.chunk_values.split(",") if x.strip() != ""]
            )
            LOG.info("chunk 维度收窄为 %s", list(selected))
        if args.embedding_values.strip():
            LOG.info("检索后端收窄为 %s", list(chunk.set_embedding_values(
                [x.strip() for x in args.embedding_values.split(",") if x.strip() != ""])))
    except ValueError as exc:
        raise SystemExit(f"[abort] 维度取值非法：{exc}") from exc

    space = single.install(six_module=six, decision_thinking=not args.no_thinking)
    space_check = chunk.verify_space()
    LOG.info("搜索空间 %s 候选数=%s chunk=%s 决策粒度=1",
             space["space_version"], space["candidate_count"], list(chunk.CHUNK_VALUES))
    if space["candidate_count"] < INITIAL_SIZE:
        raise SystemExit(
            f"[abort] 收窄后只剩 {space['candidate_count']} 个候选，少于初始批的 "
            f"{INITIAL_SIZE} 个，无法起跑；请放宽 text/image/chunk/backend 任一维度")
    # 初始批的第一个点由 init_seed 决定，首点不同则批次必然不同，
    # 因此 repeat 数不能超过候选数（超了就无法凑出互不相同的初始批）。
    if args.repeats > space["candidate_count"]:
        raise SystemExit(
            f"[abort] 只有 {space['candidate_count']} 个候选，最多支持 "
            f"{space['candidate_count']} 个互不相同的初始批，--repeats={args.repeats} 超出；"
            f"请调小 SEEDS 或放宽 text/image/chunk/backend 维度")

    spec = resolve_dataset(args)
    dataset_dir: Path = spec["dataset_dir"]
    prefix: str = spec["index_prefix"]

    # 题集留空时优先用定版子集（qa_screened50_from_*.json），没有再退回数据集自带 QA。
    if not args.qa_file:
        subset_dir = PROJECT / "datasets" / QA_SUBSET_DIR / spec["name"]
        subsets = sorted(subset_dir.glob("qa_screened50_from_*.json")) if subset_dir.is_dir() else []
        if subsets:
            spec["qa_file"] = str(subsets[0])
            LOG.info("题集取自定版子集 %s", subsets[0])

    # 文本索引随 chunk 变化，这里同时把 BACKENDS 的占位名改写到目标数据集前缀。
    text_indices = chunk.apply_backends(prefix)
    qa, corpus = load_queries(dataset_dir, spec["qa_file"])
    query_count = len(qa)

    candidates = search.all_configs()
    if len(candidates) != int(space["candidate_count"]):
        raise RuntimeError(f"搜索空间变化: {len(candidates)} != {space['candidate_count']}")

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for policy in policies:
        if policy not in six.ALL_POLICIES:
            raise ValueError(f"未知策略: {policy}")
    raw_offsets = args.initial_seed_offsets or DEFAULT_SEED_OFFSETS.get(spec["name"], "")
    offsets = [int(x) for x in raw_offsets.split(",") if x.strip() != ""]
    if args.initial_seed_offsets:
        if len(offsets) != args.repeats:
            raise ValueError("--initial-seed-offsets 数量必须等于 --repeats")
    elif args.repeats > len(offsets):
        raise SystemExit(
            f"--repeats={args.repeats} 超过 {spec['name']} 的默认偏移表（{len(offsets)} 项），"
            f"请显式传 --initial-seed-offsets")
    else:
        offsets = offsets[:args.repeats]

    es_url = args.es_url or DEFAULT_ES_URLS.get(spec["name"], FALLBACK_ES_URL)
    es = search.es_client(es_url)
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

        try:
            init_seeds, batches = six.initial_batches(
                candidates, args.seed, args.repeats, offsets)
        except RuntimeError as exc:
            raise SystemExit(
                f"[abort] {exc}：收窄后的候选空间（{len(candidates)} 个）太小，"
                f"多个 repeat 抽到了相同的初始五步；请放宽 text/image/chunk/backend 维度，"
                f"或调小 SEEDS / 换 BASE_SEED") from exc
        for index, batch in enumerate(batches):
            if len(batch) != INITIAL_SIZE:
                raise RuntimeError(
                    f"rep_{index + 1:02d} 初始批次 {len(batch)} 个 != {INITIAL_SIZE}")
        sample_hash = search.sha256_json(
            [search.QueryEmbeddingCache.query_key(row) for row in qa])

        controller_steps = args.controller_steps
        per_round = 1
        rounds = controller_steps + 1  # round 1 = 初始批次前置阶段
        budget = INITIAL_SIZE + controller_steps
        narrowed = tuple(chunk.CHUNK_VALUES) != tuple(chunk.ALL_BUILT_CHUNKS)
        chunk_tag = ("_c" + "-".join(str(c) for c in chunk.CHUNK_VALUES)) if narrowed else ""
        run_id = args.run_id or (
            f"{time.strftime('%Y%m%d')}_jev_ao_chunk_single_{spec['name']}"
            f"_q{query_count}_budget{budget}_n{args.repeats}{chunk_tag}"
        )
        run = Path(args.runs_dir) / run_id

        step_mapping = (
            f"round 1 = 初始批次前置阶段（{INITIAL_SIZE} 个配置，非控制器决策）；"
            f"round 2..{rounds} = 控制器第 1..{controller_steps} 步，每步 {per_round} 个；"
            f"控制器步 k ↔ evaluation_position k+{INITIAL_SIZE} ↔ round k+1"
        )
        if args.validate_only:
            print(json.dumps(
                {"mode": "validate_only", **base, "controller_steps": controller_steps,
                 "budget_per_policy": budget, "step_mapping": step_mapping},
                ensure_ascii=False, indent=2))
            return

        if args.plan_only:
            # plan-only 预演必须纯只读：不建 run、不写 manifest/preflight/query_sample，
            # 否则正式跑会因 save_once 的字节级幂等校验不同而直接失败。
            print(json.dumps(
                {
                    "mode": "plan_only",
                    **base,
                    "run": str(Path(args.runs_dir) / run_id),
                    "controller_steps": controller_steps,
                    "configs_per_round": per_round,
                    "rounds": rounds,
                    "budget_per_policy": budget,
                    "step_mapping": step_mapping,
                    "cache_from": args.cache_from,
                    "jit_selection": space.get("jit_selection"),
                    "initial_batch_config_ids": {
                        f"rep_{i + 1:02d}": [cfg.config_id for cfg in batch]
                        for i, batch in enumerate(batches)
                    },
                },
                ensure_ascii=False, indent=2))
            return

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
                "AO 格式包（pharmaceuticals）chunk 空间单点决策实验："
                "与批量版 chunk 实验同空间、同题集、同六策略、同初始批次设计，"
                "但控制器每轮只决策 1 个配置（步数即决策数）"
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
            "controller_rng_seeds": init_seeds,
            "controller_rng_per_repeat": (
                "run_policy(rng_seed=init_seed)：臂内 RNG（random 臂/jit 平局/TPE/"
                "llm 兜底）用各 repeat 的 init_seed，2026-10-28 起 repeats 为"
                "独立搜索过程；评分缓存仍按配置×题跨 repeat 共享"),
            "initial_batch_size": INITIAL_SIZE,
            "initial_batch_stage": "pre-phase：入口预写 plans/round_01.json，不占控制器步",
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
            "decision_granularity": per_round,
            "configs_per_round": per_round,
            "controller_steps_per_repeat": controller_steps,
            "rounds": rounds,
            "budget_per_policy": budget,
            "step_mapping": step_mapping,
            "jit_selection": space.get("jit_selection"),
            "llm_batch_size": space.get("llm_batch_size"),
            "decision_thinking": space.get("decision_thinking"),
            "candidate_count": len(candidates),
            "reward_key": six.REWARD_KEY,
            # reward_schema 含聚合参数（换 γ 即成新口径，旧配置分数会重算）；
            # reward_fingerprint_schema 不含 γ，是真正写进 JEV 请求指纹的那一个。
            "reward_schema": six.REWARD_SCHEMA,
            "reward_fingerprint_schema": six.FINGERPRINT_SCHEMA,
            "reward_scheme": six.REWARD_SCHEME,
            "reward_question_names": sorted(six.JEV_QUESTIONS),
            "redundancy_gamma": six.REDUNDANCY_GAMMA,
            "redundancy_mode": six.REDUNDANCY_MODE,
            "reward_definition": six.REWARD_DEFINITION,
            "vision_model": six.VISION_MODEL,
            "vision_prompt_version": six.VISION_PROMPT_VERSION,
            "jev_model": six.JEV_MODEL,
            "decision_model": six.DECISION_MODEL,
            "frozen_config_scores_shared_across_repeats": True,
            "indices": indices,
            "cache_reused_from": args.cache_from.strip() or None,
            "notes": (
                "题集为数据集自带 qa_search.json 全量；多个 references 取首条作候选答案。"
                "JevEvaluator 的 text_hit_at_k/image_hit_at_k 沿用 source_id == query_id "
                "判据，在本数据集的 q:<n> / p:<n> 命名空间下恒为 0，不具召回含义，"
                "请勿当作 Recall 使用。"
                f"空间沿用批量版 chunk 实验（chunk_size 替换 text_weight / rrf_k，"
                f"text_weight={chunk.FIXED_TEXT_WEIGHT}、rrf_k={chunk.FIXED_RRF_K} 固定）。"
                f"决策粒度改为每轮 1 个：{step_mapping}。"
                "单点 JIT 口径：每步取 utility 最高者，原 5 槽里的探索槽（最大化到最近"
                "观测点距离）与安全随机槽在单点模式下不存在，探索性只剩 utility 内的 "
                "EXPLORE_LAMBDA 距离项——jit 与 jit_no_rules 的差异因此被压缩到只剩语义"
                "规则修正项，解读时必须按此口径说明。"
                "坐标下降/Tpe 的批量语义随之变化：每步只走一个近邻 / 每步一次 TPE 采样，"
                "但两臂的配置总数与随机起步份数不变。"
                "缓存复用：同 sample_hash / 同空间版本的配置评分直接沿用同一分数，"
                "因此与源 run 严格可比，且只为新采到的配置付费。"
                f"llm 臂决策模型 {six.DECISION_MODEL}：批量版为思考链开启，本版为思考链"
                "关闭（enable_thinking=false，同一模型、同一 prompt、同一参数域）——"
                "开启时单次决策实测 20–160s 且偶发网关 180s 流超时，25 步 × 6 重复不可承受；"
                "解读 llm 行时必须声明这一口径差异，其余 5 个策略与批量版严格同口径。"
            ),
        }
        six.em.save_once(run / "manifest.json", manifest)
        search.atomic_json(
            run / "run_info.json",
            {
                "jev_access_mode": access_mode,
                "jev_url": args.jev_api_url if args.jev_api else six.RELAY_URL,
                "jev_long_context_header": six.JEV_LONG_CONTEXT_HEADER,
                "https_proxy": os.environ.get("HTTPS_PROXY", ""),
                "no_proxy": os.environ.get("NO_PROXY", ""),
                "es_url": args.es_url,
                "embedding_device": args.embedding_device,
                "space_version": chunk.SPACE_VERSION,
                "chunk_values": list(chunk.CHUNK_VALUES),
                "decision_granularity": per_round,
                "controller_steps_per_repeat": controller_steps,
                "cache_reused_from": args.cache_from.strip() or None,
                "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        preflight = {
            "base_seed": args.seed,
            "repeats": args.repeats,
            "policies": policies,
            "budget_per_policy": budget,
            "controller_steps_per_repeat": controller_steps,
            "configs_per_round": per_round,
            "rounds": rounds,
            "step_mapping": step_mapping,
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
            raise RuntimeError(f"JEV {transport.mode} 不可达 {transport.url}: {detail}")
        LOG.info("jev access mode=%s url=%s probe=%s", transport.mode, transport.url, detail)

        cache_report = seed_caches(
            run, args.cache_from, args.seed, sample_hash, prefix, query_count)
        if args.vision_from.strip():
            cache_report["vision_from"] = args.vision_from.strip()
            cache_report["vision_added_from"] = seed_vision_cache(
                run, args.vision_from.strip())
            LOG.info("vision 额外并入 %s 条（源 %s）",
                     cache_report["vision_added_from"], args.vision_from.strip())
        embedding = search.QueryEmbeddingCache(
            run, qa, args.embedding_device, args.embedding_batch_size)
        for backend in chunk.EMBEDDING_VALUES:
            embedding.ensure(backend)
        retrieval = chunk.ChunkRetrievalCache(
            run / "retrieval_cache.json", es, embedding, prefix)
        vision = six.VisionCache(run)
        cache_root = layout.cache_dir(run)
        source_score_dir = None
        if args.source_scores.strip():
            source_score_dir = (
                layout.cache_path(_resolve_run_dir(args.source_scores), "config_scores", legacy_seed=args.seed)
            )
            if not source_score_dir.is_dir():
                raise RuntimeError(f"--source-scores 目录不存在: {source_score_dir}")
            LOG.info("惰性派生源: %s（命中即切子集，零调用）", source_score_dir)
        evaluator = six.FrozenConfigEvaluator(
            six.JevEvaluator(
                qa, retrieval, vision, transport,
                cache_root / "jev_responses.jsonl",
            ),
            cache_root / "config_scores",
            sample_hash,
            query_count,
            source_score_dir=source_score_dir,
        )
        pairs = []
        for repeat in range(1, args.repeats + 1):
            initial = batches[repeat - 1]
            layout.write_seed_meta(
                run, repeat - 1,
                seed=init_seeds[repeat - 1],
                base_seed=args.seed,
                repeat=repeat,
                initial_seed_offset=offsets[repeat - 1],
            )
            summaries = {}
            for policy in policies:
                # 预写 round 1 计划 → 初始批次成为前置阶段，共享代码不再对其做批量校验。
                write_initial_plan(run, repeat, policy, initial)
                summary = six.run_policy(
                    args.seed, repeat, policy, candidates, initial, evaluator,
                    run, rounds, per_round,
                    rng_seed=init_seeds[repeat - 1],
                )
                summaries[policy] = {**summary, **policy_step_stats(run, repeat, policy)}
            pair = {
                "seed": args.seed,
                "repeat": repeat,
                "initial_batch_seed": init_seeds[repeat - 1],
                "controller_rng_seed": init_seeds[repeat - 1],
                "initial_config_ids": [cfg.config_id for cfg in initial],
                "initial_batch_size": INITIAL_SIZE,
                "controller_steps": controller_steps,
                "policies": summaries,
            }
            search.atomic_json(layout.seed_dir(run, repeat - 1) / "repeat_summary.json", pair)
            pairs.append(pair)
            LOG.info("seed=%s rep=%02d complete best=%s", args.seed, repeat,
                     {p: round(s["best_mean_jev_support"], 4)
                      for p, s in summaries.items()})
        aggregate = {}
        for policy in policies:
            rows = [pair["policies"][policy] for pair in pairs]
            bests = [row["best_mean_jev_support"] for row in rows]
            steps = [row["best_step"] for row in rows]
            final_steps = [row["final_best_controller_step"] for row in rows
                           if row["final_best_controller_step"] is not None]
            band_steps = [row["first_band_controller_step"] for row in rows
                          if row["first_band_controller_step"] is not None]
            controller_bests = [row["controller_best"] for row in rows
                                if row["controller_best"] is not None]
            aggregate[policy] = {
                "mean_best": float(np.mean(bests)),
                "std_best": float(np.std(bests)),
                "min_best": float(np.min(bests)),
                "max_best": float(np.max(bests)),
                "mean_best_step": float(np.mean(steps)),
                "best_per_repeat": bests,
                "best_step_per_repeat": steps,
                "mean_best_controller_step": float(np.mean(steps)) - INITIAL_SIZE,
                "mean_final_best_controller_step": (
                    float(np.mean(final_steps)) if final_steps else None),
                "final_best_in_controller_steps": len(final_steps),
                "mean_first_band_controller_step": (
                    float(np.mean(band_steps)) if band_steps else None),
                "first_band_in_controller_steps": len(band_steps),
                "band_epsilon": BAND_EPSILON,
                "mean_initial_batch_best": float(
                    np.mean([row["initial_batch_best"] for row in rows])),
                "mean_controller_best": (
                    float(np.mean(controller_bests)) if controller_bests else None),
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
                "decision_granularity": per_round,
                "initial_batch_size": INITIAL_SIZE,
                "controller_steps_per_repeat": controller_steps,
                "budget_per_policy": budget,
                "step_mapping": step_mapping,
                "cache_reused_from": args.cache_from.strip() or None,
                "cache_reuse": cache_report,
                "reward_key": six.REWARD_KEY,
                "reward_schema": six.REWARD_SCHEMA,
                "reward_fingerprint_schema": six.FINGERPRINT_SCHEMA,
                "redundancy_gamma": six.REDUNDANCY_GAMMA,
                "redundancy_mode": six.REDUNDANCY_MODE,
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