#!/usr/bin/env python3
"""同一 seed、六次重复的初始批次敏感性流水线（JEV 冻结奖励）。

固定协议（运行前冻结，勿在中途修改）：

- **基准 seed**（``--seed``，默认 61）同时决定固定查询样本和全部 controller 的搜索随机性；
  六次重复使用**完全相同**的基准 seed。
- 每次重复的**初始 5 个配置**由 ``paired.seeded_initial_batch(candidates, init_seed)`` 生成，
  沿用现有 seeded maximin 设计：第一个点由该 seed 随机抽取，其余 4 个做 maximin 空间填充。
  ``init_seed`` 依次取 ``base_seed + offset``（默认 offset 0..5），
  因此**六次重复的初始五步互不相同**。
- 每次重复内部，各策略**共享同一批初始 5 个配置**，从第 6 个配置起各自独立选点。
- 奖励：JEV ``Jev_AO`` 对固定题的 ``P(supported)`` 均值；图像先由 Qwen3.7 Flash
  在不知道问题的情况下转写。同一 seed 的六次重复**共用同一份冻结配置评分缓存**，
  因此跨重复的差异只来自初始批次。
- 每策略预算 ``rounds × per_round``（默认 30）个配置，每轮先选 5 个再读取该轮反馈。

结论口径：这是「初始批次敏感性」实验，不是新的 seed 对照；报告必须写明
「同一 seed、六次重复、初始五步不同」。

JEV 通过仅本机回环的 SSH 反向隧道访问本地 relay；本脚本会消耗付费 JEV 额度。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

from mgrs import judge_client
from mgrs.controllers.llm_controller import LLMController, SYSTEM_PROMPT
from mgrs.controllers.raise_baseline_controllers import (
    CoordinateDescentController,
    TPEController,
)
from mgrs.pipeline import dqcr_reward as reward
from mgrs.pipeline import run_layout as layout
from mgrs.pipeline import rq1_answer_em_qaall_clean_paired as em
from mgrs.pipeline import rq1_fixed_eval_paired5 as paired
from mgrs.pipeline import rq1_memory_v2_jit_random as search
from mgrs.rag_pipeline.llm import MultimodalLLM

LOG = logging.getLogger("jev_initial_batch_repeat6")
PROJECT = Path(__file__).resolve().parents[2]

DEFAULT_RUN_ID = "20261015_jev_initial_batch_repeat6_q50_budget30"
DEFAULT_CACHE_SOURCE = "20261012_jev_support_6seed_no_rules_llm_q50_budget30"
ALL_POLICIES = ("jit", "jit_no_rules", "random", "llm", "coordinate_descent", "tpe")
REWARD_KEY = "mean_jev_support"
# 评分口径为 DQCR（双题复合判题奖励），题面与聚合见 mgrs/pipeline/dqcr_reward.py。
# 两个 schema 有明确分工：
#   FINGERPRINT_SCHEMA —— 进 JEV 请求指纹（JevEvaluator.fingerprint）。只描述"发出去的请求"，
#     所以**不含 γ/mode**：换聚合参数不改请求，应答缓存全命中，γ 扫描零调用。
#   REWARD_SCHEMA —— 写进 config_scores 并经 seed_caches 校验（seed_caches 比 reward_schema），
#     含 γ/mode：换聚合参数即成新口径，旧配置分数判 incompatible 并重算。
_RESOLVED_REWARD = reward.resolve()
REWARD_SCHEME = reward.SCHEME
REWARD_SCHEMA = _RESOLVED_REWARD.schema
FINGERPRINT_SCHEMA = _RESOLVED_REWARD.fingerprint_schema
JEV_QUESTIONS = _RESOLVED_REWARD.questions
REWARD_DEFINITION = _RESOLVED_REWARD.definition
REDUNDANCY_GAMMA = _RESOLVED_REWARD.gamma
REDUNDANCY_MODE = _RESOLVED_REWARD.mode
VISION_MODEL = (os.environ.get("MGRS_VISION_MODEL") or "qwen3.7-flash").strip()
DECISION_MODEL = (os.environ.get("MGRS_DECISION_MODEL") or "qwen3.8-max").strip()
# 判题后端（DQCR 的判题器）全部走环境变量，仓库内不留内置端点或密钥：
#   MGRS_JUDGE_BACKEND       jev（默认，原生网关协议）| openai_chat（OpenAI 兼容端点）
#   MGRS_JUDGE_API_URL       判题端点（旧名 JEV_API_URL 仍可用）
#   MGRS_JUDGE_MODEL         判题模型名（旧名 JEV_MODEL 仍可用）
#   $MGRS_JUDGE_API_KEY_ENV  密钥所在环境变量名，默认 JEV_API_KEY
# 详见 mgrs/judge_client.py 与 .env.example。
# 注意：JEV_MODEL 会写进 manifest，并由 parse_jev_response 与应答的 model 字段严格比对。
# 换判题模型即等于换评分基线，旧 run 的 config_scores 不可与新 run 混算；同时必须一起换
# MGRS_JUDGE_SCHEMA_TAG（请求指纹不含 URL 与模型名，否则新旧后端缓存会互相静默命中）。
JEV_MODEL = judge_client.JUDGE_MODEL
# 长上下文开关头：判题证据可达 6 万+ 字符，部分网关需显式放开窗口才不报
# max_tokens_exceeded。未知请求头会被其他网关忽略，故默认始终携带。
JEV_LONG_CONTEXT_HEADER = judge_client.JUDGE_LONG_CONTEXT_HEADER
VISION_PROMPT = (
    "Describe only what is visible in this image in English. Transcribe all legible labels, "
    "numbers, table cells, and chart values. Describe relevant spatial and graphical "
    "relationships. Do not answer a question or infer facts outside the image."
)
VISION_PROMPT_VERSION = "image_facts_no_question_v1"
# JEV_QUESTIONS 由 DQCR 口径注入：一次调用两道题（support + redundancy），
# **但聚合后仍只有一个标量出口**——REWARD_KEY 仍是 `mean_jev_support`，
# 控制器、轨迹、manifest、下游统计链路一字不改。
RELAY_URL = (
    os.environ.get("MGRS_RELAY_URL")
    or os.environ.get("RELAY_URL")
    or "http://127.0.0.1:17654/jev"
).strip()
JEV_API_URL = judge_client.JUDGE_API_URL
JEV_API_KEY_ENV = judge_client.JUDGE_API_KEY_ENV


# 判题传输层统一收敛到 mgrs/judge_client.py（原生网关 jev / OpenAI 兼容 openai_chat）。
# 这里保留旧类名做别名，使既有构造点 DirectAPITransport(url, key) 与 RelayTransport(url, token) 不变。
RelayTransport = judge_client.RelayTransport
DirectAPITransport = judge_client.JevGatewayTransport


def make_transport(url: str = "", key: str = "") -> Any:
    """按 MGRS_JUDGE_BACKEND 构造判题传输层。

    ``jev``（默认）走原生网关协议；``openai_chat`` 走任意 OpenAI 兼容
    ``/chat/completions`` 端点（提示词渲染与应答映射见 mgrs/judge_client.py）。
    两者产出的应答结构一致，因此指纹缓存与 γ 聚合链路完全复用。
    """
    endpoint = (url or JEV_API_URL).strip()
    backend = judge_client.JUDGE_BACKEND
    missing = [
        label
        for label, value in (
            ("MGRS_JUDGE_API_URL（或 --jev-api-url）", endpoint),
            (f"${JEV_API_KEY_ENV}", key),
            ("MGRS_JUDGE_MODEL", JEV_MODEL),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "判题后端未配置，缺少 " + "、".join(missing) + "；参考 .env.example 设置后重跑"
        )
    if backend in {"openai_chat", "openai", "chat"}:
        return judge_client.OpenAIChatTransport(endpoint, key, JEV_MODEL)
    return DirectAPITransport(endpoint, key, JEV_MODEL)


class NoRulesMemory(search.MemoryV2):
    """保留 JIT 选择规则，但禁用蒸馏出的语义规则（去规则 JIT 臂）。"""

    def distill(self, round_number: int) -> list[dict[str, Any]]:
        self.semantic_rules = []
        return []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8192)
def image_hash(path: str) -> str:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(path)
    return sha256_file(file)


def load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            value = str(row[key])
            if value in rows and rows[value] != row:
                raise RuntimeError(f"conflicting cached row: {path} {value}")
            rows[value] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class VisionCache:
    def __init__(self, run: Path) -> None:
        self.path = run / "vision_facts.jsonl"
        self.rows = load_jsonl(self.path, "image_sha256")
        # ensure() 会被并行配置的线程共用：缺锁时每个配置都会把别的配置
        # 刚转写完、尚未进 self.rows 的图再转一遍；而 qwen3.7-flash 在
        # temperature=0 下也不是逐字确定的（2026-10-26 实测：216 张重复图
        # 里 213 张文本不一致），冲突行会让下次 load_jsonl fail-closed 报错。
        self._lock = threading.Lock()

    @staticmethod
    def describe(path: str) -> str:
        llm = MultimodalLLM()
        messages = llm._messages(
            "You are a precise image transcription assistant.",
            VISION_PROMPT,
            [{"image_path": path}],
        )
        if len(messages[1]["content"]) != 2:
            raise RuntimeError(f"vision payload omitted image: {path}")
        return llm.client.chat(messages, temperature=0.0, model=VISION_MODEL)

    def _describe_retry(self, path: str, attempts: int = 6) -> str:
        """网关 429(quota exceeded) 属分钟级抖动，指数退避重试；非限流错误立即抛。"""
        delay = 10.0
        for attempt in range(1, attempts + 1):
            try:
                return self.describe(path)
            except Exception as exc:  # noqa: BLE001
                text = f"{exc}".lower()
                limited = "429" in text or "quota" in text or "too many requests" in text
                if attempt == attempts or not limited:
                    raise
                LOG.warning(
                    "vision 转写被限流，退避 %.0fs 后重试（%s/%s）%s",
                    delay, attempt, attempts, Path(path).name,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 120.0)
        raise RuntimeError("unreachable")

    def ensure(self, prepared: list[dict[str, Any]]) -> None:
        with self._lock:
            self._ensure_locked(prepared)

    def _ensure_locked(self, prepared: list[dict[str, Any]]) -> None:
        missing: dict[str, str] = {}
        for row in prepared:
            for image in row["image_evidence"]:
                path = str(image.get("image_path") or "")
                digest = image_hash(path)
                if digest not in self.rows:
                    missing[digest] = path
        if not missing:
            return
        LOG.info("vision missing=%s cached=%s", len(missing), len(self.rows))
        errors = []
        with ThreadPoolExecutor(
            max_workers=int(os.environ.get("VISION_WORKERS", "6"))
        ) as pool:
            futures = {
                pool.submit(self._describe_retry, path): (digest, path)
                for digest, path in missing.items()
            }
            for future in as_completed(futures):
                digest, path = futures[future]
                try:
                    description = future.result()
                    if not description.strip():
                        raise RuntimeError("empty image description")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{digest}: {type(exc).__name__}: {exc}")
                    continue
                record = {
                    "image_sha256": digest,
                    "image_path": path,
                    "model": VISION_MODEL,
                    "prompt_version": VISION_PROMPT_VERSION,
                    "description": description,
                }
                append_jsonl(self.path, record)
                self.rows[digest] = record  # 同进程内后续 ensure 视为已缓存
                self.rows[digest] = record
                if len(self.rows) % 25 == 0:
                    LOG.info("vision cached=%s", len(self.rows))
        if errors:
            raise RuntimeError(f"vision failures={len(errors)} first={errors[0]}")

    def facts(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "rank": int(image["rank"]),
                "source_id": str(image["id"]),
                "description": self.rows[image_hash(str(image["image_path"]))]["description"],
            }
            for image in images
        ]


def parse_jev_response(
    result: dict[str, Any],
    questions: dict[str, Any],
    expected_model: str = JEV_MODEL,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """校验并抽取 JEV 应答（纯函数，便于离线单测）。

    返回 ({题名: {选项: 概率}}, {题名: choice})。
    对 model、题型、选项集合、概率取值域全部 fail-closed —— 多选题时任一题不符即整条请求判失败，
    避免"support 正常但 redundancy 缺项"被静默当成单题档记下来。
    """
    if result.get("model") != expected_model:
        raise RuntimeError("unexpected Jev model")
    answers_raw = result.get("answers") or {}
    answers: dict[str, dict[str, float]] = {}
    choices: dict[str, str] = {}
    for question_name, spec in questions.items():
        if question_name not in answers_raw:
            raise RuntimeError(f"Jev response missing answer for question {question_name!r}")
        answer = answers_raw.get(question_name)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise RuntimeError(f"unexpected Jev answer type for question {question_name!r}")
        probabilities = {
            str(key): float(value)
            for key, value in (answer.get("probabilities") or {}).items()
        }
        if set(probabilities) != set(spec["criteria"]):
            raise RuntimeError(
                f"unexpected Jev options for question {question_name!r}: "
                f"{sorted(probabilities)}"
            )
        if not all(0.0 <= value <= 1.0 for value in probabilities.values()):
            raise RuntimeError(f"Jev probability out of range for question {question_name!r}")
        answers[question_name] = probabilities
        choices[question_name] = str(answer.get("choice", ""))
    return answers, choices


class JevEvaluator:
    def __init__(
        self,
        queries: list[dict[str, Any]],
        retrieval: search.RetrievalCache,
        vision: VisionCache,
        transport: RelayTransport | DirectAPITransport,
        cache_path: Path,
    ) -> None:
        self.queries = queries
        self.retrieval = retrieval
        self.vision = vision
        self.transport = transport
        self.cache_path = cache_path
        self.records = load_jsonl(cache_path, "fingerprint")
        self.billing_failure = threading.Event()

    @staticmethod
    def fingerprint(row: dict[str, Any]) -> str:
        # 用 FINGERPRINT_SCHEMA（只描述请求本身，不含 γ/mode）：
        # 换 γ 重跑时请求逐字节不变 → 全部命中 jev_responses.jsonl。
        return search.sha256_json({
            "schema": FINGERPRINT_SCHEMA,
            "query": row["query"],
            "reference": row["reference"],
            "text": [{"rank": x["rank"], "text": x.get("text", "")} for x in row["text_evidence"]],
            "images": [
                {"rank": x["rank"], "sha256": image_hash(str(x["image_path"]))}
                for x in row["image_evidence"]
            ],
        })

    def call_jev(self, row: dict[str, Any]) -> dict[str, Any]:
        state = {
            "question": row["query"],
            "candidate_answer": row["reference"],
            "retrieved_text": [
                {
                    "rank": x["rank"],
                    "source_id": str(x.get("metadata", {}).get("source_id") or x.get("id") or ""),
                    "text": x.get("text", ""),
                }
                for x in row["text_evidence"]
            ],
            "image_observations": self.vision.facts(row["image_evidence"]),
        }
        payload = {"model": JEV_MODEL, "state": state, "questions": JEV_QUESTIONS}
        # 判题端点在网络抖动时可能中断数分钟：保持同一份冻结请求并等待重连。
        # 具体怎么发由传输层决定（原生网关 / OpenAI 兼容端点，见 mgrs/judge_client.py）。
        response = None
        for attempt in range(30):
            if self.billing_failure.is_set():
                raise RuntimeError("Jev billing failure already observed; stop this batch")
            try:
                response = self.transport.post(payload)
                if response.ok:
                    result = response.json()
                    answers, choices = parse_jev_response(result, JEV_QUESTIONS)
                    # γ=0 只取原始分量：**记录必须由请求唯一决定**。
                    # 若把 γ/mode 或折减后的分写进记录，同一指纹在换 γ 重跑时会与旧行
                    # 冲突（load_jsonl 同键不同值直接 raise），或更糟——静默沿用旧 γ 的分。
                    raw = reward.combine(answers, 0.0)
                    return {
                        "fingerprint": row["fingerprint"],
                        "query_id": row["id"],
                        "support_score": raw["support_score"],
                        "redundancy_expected": raw["redundancy_expected"],
                        "answers": answers,
                        # 兼容字段：choice / probabilities / score 始终指 support 题
                        "choice": choices["support"],
                        "probabilities": answers["support"],
                        "choices": choices,
                        "score": raw["support_score"],
                        "model": result["model"],
                        "usage": result.get("usage") or {},
                    }
                if response.status_code == 402:
                    self.billing_failure.set()
                    (self.cache_path.parent.parent / "PAUSED_JUDGE_402.txt").write_text(
                        "JEV API returned HTTP 402; stop before further paid evaluation.\n",
                        encoding="utf-8",
                    )
                    raise RuntimeError(f"Jev {self.transport.mode} HTTP 402 billing failure")
                if response.status_code in {400, 413, 422}:
                    # 确定性请求错误（如 max_tokens_exceeded）：重试永远不会成功。
                    # 必须留标记快速停下，否则 launcher 会无限重试空转数小时。
                    (
                        self.cache_path.parent.parent / "PAUSED_JEV_REQUEST_ERROR.txt"
                    ).write_text(
                        f"JEV HTTP {response.status_code}: {response.text[:800]}\n"
                        "确定性请求错误，重试不会成功；请检查证据文本长度与"
                        "JEV_LONG_CONTEXT_HEADER 设置，排除后删除本文件再续跑。\n",
                        encoding="utf-8",
                    )
                    raise RuntimeError(
                        f"Jev {self.transport.mode} HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                if response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"Jev {self.transport.mode} HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
            except requests.RequestException:
                if attempt == 29:
                    raise
            if attempt == 29:
                raise RuntimeError(f"Jev {self.transport.mode} retries exhausted")
            time.sleep(min(2 ** min(attempt, 5), 30))
        raise AssertionError("unreachable")

    def evaluate(self, config: search.SearchConfig) -> dict[str, Any]:
        prepared = []
        for item in self.queries:
            text, images = self.retrieval.evidence(config, item)
            row = {
                "id": str(item["id"]),
                "query": str(item["query"]),
                "reference": search.reference_for(item),
                "text_evidence": text,
                "image_evidence": images,
                "text_hit_at_k": int(
                    any(str(t.get("metadata", {}).get("source_id")) == str(item["id"]) for t in text)
                ),
                "image_hit_at_k": int(
                    any(str(im.get("id")) == str(item["id"]) for im in images)
                ),
            }
            row["fingerprint"] = self.fingerprint(row)
            prepared.append(row)
        self.retrieval.flush()
        self.vision.ensure(prepared)
        pending = {
            row["fingerprint"]: row
            for row in prepared
            if row["fingerprint"] not in self.records
        }
        LOG.info(
            "Jev config=%s queries=%s cache_hits=%s pending=%s",
            config.config_id, len(prepared), len(prepared) - len(pending), len(pending),
        )
        errors = []
        if pending:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(self.call_jev, row): fingerprint
                    for fingerprint, row in pending.items()
                }
                for future in as_completed(futures):
                    fingerprint = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{fingerprint}: {type(exc).__name__}: {exc}")
                        continue
                    append_jsonl(self.cache_path, record)
                    self.records[fingerprint] = record
            if errors:
                raise RuntimeError(f"Jev failures={len(errors)} first={errors[0]}")
        evaluations = []
        for row in prepared:
            record = self.records[row["fingerprint"]]
            aggregated = self.aggregate(record)
            evaluations.append(
                {
                    "id": row["id"],
                    # jev_support_score 是**聚合出口**：单题档 = P(supported)，
                    # 双题档 = P(support) ⊙ γ·E[冗余]。字段名不变，下游无需改动。
                    "jev_support_score": aggregated["score"],
                    # 两个未折减的分量随查询存下：换 γ 可离线重聚合，零调用。
                    "jev_support_raw": aggregated["support_score"],
                    "jev_redundancy": aggregated["redundancy_expected"],
                    "jev_choice": record["choice"],
                    "jev_fingerprint": row["fingerprint"],
                    "text_hit_at_k": row["text_hit_at_k"],
                    "image_hit_at_k": row["image_hit_at_k"],
                }
            )
        return {
            "config_id": config.config_id,
            "metric": REWARD_KEY,
            REWARD_KEY: float(np.mean([x["jev_support_score"] for x in evaluations])),
            "mean_jev_support_raw": float(np.mean([x["jev_support_raw"] for x in evaluations])),
            "mean_jev_redundancy": float(np.mean([x["jev_redundancy"] for x in evaluations])),
            "redundancy_gamma": REDUNDANCY_GAMMA,
            "redundancy_mode": REDUNDANCY_MODE,
            "mean_text_hit_at_k": float(np.mean([x["text_hit_at_k"] for x in evaluations])),
            "mean_image_hit_at_k": float(np.mean([x["image_hit_at_k"] for x in evaluations])),
            "query_count": len(evaluations),
            "jev_cache_hits": len(prepared) - len(pending),
            "jev_new_calls": len(pending),
            "jev_choice_counts": dict(
                Counter(self.records[row["fingerprint"]]["choice"] for row in prepared)
            ),
            "evaluations": evaluations,
        }

    @staticmethod
    def aggregate(record: dict[str, Any]) -> dict[str, float]:
        """从缓存记录算聚合奖励（纯函数，读时才折叠 γ）。

        新记录存原始多题概率 `answers`（与 γ 无关）；更早期的记录只有单题分数，
        直接平移到聚合口径。
        """
        answers = record.get("answers")
        if isinstance(answers, dict):
            return reward.combine(answers, REDUNDANCY_GAMMA, REDUNDANCY_MODE)
        score = float(record["score"])
        return {"score": score, "support_score": score, "redundancy_expected": 0.0}


class FrozenConfigEvaluator:
    """同一 seed 下跨重复共享的冻结配置评分：同一配置永远得到同一分数。"""

    def __init__(
        self,
        evaluator: JevEvaluator,
        score_dir: Path,
        sample_hash: str,
        query_count: int,
        source_score_dir: Path | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.score_dir = score_dir
        self.sample_hash = sample_hash
        self.query_count = query_count
        # 惰性派生源：当前题集是源 run 题集的子集时，直接把源的全量评分切成子集重算均值，
        # 零调用。只在源里确实有该配置时才生效，源里没有才走真实评。
        self.source_score_dir = source_score_dir
        self.derived_count = 0
        score_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, config: search.SearchConfig) -> dict[str, Any]:
        path = self.score_dir / f"{config.config_id}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if (
                result.get("config_id") != config.config_id
                or result.get("sample_hash") != self.sample_hash
                or result.get("reward_schema") != REWARD_SCHEMA
                or result.get("query_count") != self.query_count
            ):
                raise RuntimeError(f"incompatible frozen config score: {path}")
            result["score_oracle_hit"] = True
            return result
        derived = self.derive_from_source(config, path)
        if derived is not None:
            return derived
        result = self.evaluator.evaluate(config)
        if result["query_count"] != self.query_count:
            raise RuntimeError("wrong query count")
        result["sample_hash"] = self.sample_hash
        result["reward_schema"] = REWARD_SCHEMA
        search.atomic_json(path, result)
        result["score_oracle_hit"] = False
        return result

    @staticmethod
    def _fold(raw: float, redundancy: float) -> float:
        """按当前 γ/mode 把原始分量折成聚合分（与 dqcr_reward.combine 同式）。"""
        if REDUNDANCY_GAMMA == 0.0 or not redundancy:
            return float(raw)
        if REDUNDANCY_MODE == "scale":
            return float(raw) * (1.0 - REDUNDANCY_GAMMA * float(redundancy))
        if REDUNDANCY_MODE == "subtract":
            return float(raw) - REDUNDANCY_GAMMA * float(redundancy)
        raise RuntimeError(f"未知冗余聚合模式: {REDUNDANCY_MODE!r}")

    def derive_from_source(
        self, config: search.SearchConfig, path: Path
    ) -> dict[str, Any] | None:
        """源 run 已评过该配置时，切出当前题集的子集重算均值，零调用。"""
        if self.source_score_dir is None:
            return None
        source_path = self.source_score_dir / f"{config.config_id}.json"
        if not source_path.is_file():
            return None
        record = json.loads(source_path.read_text(encoding="utf-8"))
        wanted = {str(row["id"]) for row in self.evaluator.queries}
        rows = [e for e in record.get("evaluations", []) if str(e["id"]) in wanted]
        if len(rows) != len(wanted) or len(rows) != self.query_count:
            # 题集不是源题集的子集（或源记录不完整）→ 不能派生，老实重评。
            return None
        evaluations = [
            {
                "id": str(row["id"]),
                "jev_support_score": self._fold(
                    row.get("jev_support_raw", row["jev_support_score"]),
                    row.get("jev_redundancy", 0.0),
                ),
                "jev_support_raw": float(row.get("jev_support_raw", row["jev_support_score"])),
                "jev_redundancy": float(row.get("jev_redundancy", 0.0)),
                "jev_choice": row.get("jev_choice"),
                "jev_fingerprint": row.get("jev_fingerprint"),
                "text_hit_at_k": int(row.get("text_hit_at_k", 0)),
                "image_hit_at_k": int(row.get("image_hit_at_k", 0)),
            }
            for row in rows
        ]
        result = {
            "config_id": config.config_id,
            "metric": REWARD_KEY,
            REWARD_KEY: float(np.mean([x["jev_support_score"] for x in evaluations])),
            "mean_jev_support_raw": float(np.mean([x["jev_support_raw"] for x in evaluations])),
            "mean_jev_redundancy": float(np.mean([x["jev_redundancy"] for x in evaluations])),
            "redundancy_gamma": REDUNDANCY_GAMMA,
            "redundancy_mode": REDUNDANCY_MODE,
            "mean_text_hit_at_k": float(np.mean([x["text_hit_at_k"] for x in evaluations])),
            "mean_image_hit_at_k": float(np.mean([x["image_hit_at_k"] for x in evaluations])),
            "query_count": len(evaluations),
            "jev_cache_hits": 0,
            "jev_new_calls": 0,
            "jev_choice_counts": dict(
                Counter(str(x["jev_choice"]) for x in evaluations)
            ),
            "evaluations": evaluations,
            "sample_hash": self.sample_hash,
            "reward_schema": REWARD_SCHEMA,
            # 溯源：这份分数不是本次调用得到的，而是源 run 全量评分的子集折叠。
            "derived_from": {
                "run": self.source_score_dir.parent.parent.name,
                "source_query_count": record.get("query_count"),
                "source_sample_hash": record.get("sample_hash"),
                "source_reward_schema": record.get("reward_schema"),
                "source_redundancy": [
                    record.get("redundancy_gamma"),
                    record.get("redundancy_mode"),
                ],
            },
        }
        search.atomic_json(path, result)
        self.derived_count += 1
        result["score_oracle_hit"] = True
        return result


def initial_batches(
    candidates: list[search.SearchConfig],
    base_seed: int,
    repeats: int,
    offsets: list[int],
) -> tuple[list[int], list[list[search.SearchConfig]]]:
    """六次重复各自的初始五步：沿用 seeded maximin，但用不同的 init_seed。

    收窄搜索空间（小规模探针）时，不同 init_seed 可能抽到同一组初始五步
    （首点相同则整批相同，批次由首点唯一决定），此时按 seed 递增重抽。
    默认全量空间下不会触发，因此与旧版结果逐字节一致。
    """
    if len(offsets) != repeats:
        raise ValueError("offsets length must equal repeats")
    init_seeds: list[int] = []
    batches: list[list[search.SearchConfig]] = []
    used: set[tuple[str, ...]] = set()
    for offset in offsets:
        seed = base_seed + offset
        for attempt in range(len(candidates) + 1):
            batch = paired.seeded_initial_batch(candidates, seed + attempt)
            key = tuple(cfg.config_id for cfg in batch)
            if key not in used:
                break
        else:
            raise RuntimeError("initial batches are not pairwise distinct")
        used.add(key)
        init_seeds.append(seed + attempt)
        batches.append(batch)
    for batch in batches:
        if len(batch) != 5 or len({cfg.config_id for cfg in batch}) != 5:
            raise RuntimeError("initial batch must contain 5 distinct configs")
    return init_seeds, batches


def run_policy(
    seed: int,
    repeat: int,
    policy: str,
    candidates: list[search.SearchConfig],
    initial: list[search.SearchConfig],
    evaluator: FrozenConfigEvaluator,
    run: Path,
    rounds: int,
    per_round: int,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """rng_seed：本 repeat 的搜索 RNG 种子（2026-10-28 口径：传各 repeat 的 init_seed，
    使臂内随机性与初始批一起随 repeat 独立；缺省 None 回落共 seed 的旧行为）。"""
    rng = seed if rng_seed is None else rng_seed
    root = layout.seed_dir(run, repeat - 1) / policy
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    saved = em.trajectory_rows(root / "trajectory.jsonl")
    by_id = {cfg.config_id: cfg for cfg in candidates}
    memory: search.MemoryV2 | None = None
    if policy == "jit":
        memory = search.MemoryV2()
    elif policy == "jit_no_rules":
        memory = NoRulesMemory()
    controller: Any = None
    if policy == "coordinate_descent":
        controller = CoordinateDescentController(REWARD_KEY)
    elif policy == "tpe":
        controller = TPEController(REWARD_KEY, candidates, rng)
    trajectory: list[dict[str, Any]] = []
    tried: set[str] = set()
    decision_seconds_total = 0.0
    for round_number in range(1, rounds + 1):
        plan_path = root / "plans" / f"round_{round_number:02d}.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            batch = [by_id[cid] for cid in plan["config_ids"]]
            diagnostics = plan["diagnostics"]
            decision_seconds_total += float(plan.get("decision_seconds", 0.0))
        else:
            remaining = [cfg for cfg in candidates if cfg.config_id not in tried]
            if round_number == 1:
                # 同一重复内所有策略共享这一批初始配置。
                batch = list(initial)
                diagnostics = {
                    cfg.config_id: {"selection": "shared_seeded_maximin"} for cfg in batch
                }
                decision_seconds = 0.0
            elif policy in ("jit", "jit_no_rules"):
                assert memory is not None
                started = time.perf_counter()
                batch, diagnostics = memory.select_jit(
                    remaining, random.Random(rng * 1000 + round_number)
                )
                decision_seconds = time.perf_counter() - started
            elif policy == "random":
                batch = random.Random(rng * 1000 + round_number).sample(
                    sorted(remaining, key=lambda c: c.config_id), per_round
                )
                diagnostics = {cfg.config_id: {"selection": "seeded_random"} for cfg in batch}
                decision_seconds = 0.0
            elif policy == "llm":
                started = time.perf_counter()
                llm_controller = LLMController(
                    seed=rng * 1000 + round_number,
                    model=DECISION_MODEL,
                    metric_key=REWARD_KEY,
                    objective=(
                        "maximize mean Jev probability that retrieved evidence "
                        "supports the reference answer"
                    ),
                    system_prompt=SYSTEM_PROMPT.replace(
                        "retrieval support 高", "JEV 检索证据支持概率高"
                    ),
                )
                batch, diagnostics, decision = llm_controller.select_batch(
                    remaining=remaining,
                    observations=trajectory,
                    round_number=round_number,
                )
                decision_seconds = time.perf_counter() - started
                search.atomic_json(root / "decisions" / f"round_{round_number:02d}.json", decision)
            elif policy in ("coordinate_descent", "tpe"):
                started = time.perf_counter()
                batch, diagnostics = controller.select_batch(
                    remaining=remaining,
                    observations=trajectory,
                    round_number=round_number,
                    batch_size=per_round,
                )
                decision_seconds = time.perf_counter() - started
            else:
                raise ValueError(policy)
            if len(batch) != per_round or len({c.config_id for c in batch}) != per_round:
                raise RuntimeError("invalid batch")
            if any(c.config_id in tried for c in batch):
                raise RuntimeError("controller repeated an evaluated configuration")
            decision_seconds_total += decision_seconds
            search.atomic_json(
                plan_path,
                {
                    "round": round_number,
                    "config_ids": [c.config_id for c in batch],
                    "diagnostics": diagnostics,
                    "decision_seconds": decision_seconds,
                },
            )
        for slot, cfg in enumerate(batch, start=1):
            if cfg.config_id in tried:
                raise RuntimeError("config repeated within policy")
            key = (round_number, slot)
            if key in saved:
                row = saved[key]
                if row["config"]["config_id"] != cfg.config_id:
                    raise RuntimeError("plan/trajectory mismatch")
            else:
                LOG.info(
                    "seed=%s rep=%02d policy=%s round=%s slot=%s config=%s",
                    seed, repeat, policy, round_number, slot, cfg.config_id,
                )
                start = time.perf_counter()
                evaluation = evaluator.evaluate(cfg)
                row = {
                    "round": round_number,
                    "slot": slot,
                    "evaluation_position": len(trajectory) + 1,
                    "config": cfg.as_dict(),
                    "selection": diagnostics[cfg.config_id],
                    "evaluation": search.clean_evaluation_summary(evaluation),
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                }
                append_jsonl(root / "trajectory.jsonl", row)
                saved[key] = row
            trajectory.append(row)
            tried.add(cfg.config_id)
            if memory is not None:
                memory.add(cfg, float(row["evaluation"][REWARD_KEY]), round_number)
            search.atomic_json(
                run / "progress.json",
                {
                    "stage": "search",
                    "seed": seed,
                    "repeat": repeat,
                    "policy": policy,
                    "round": round_number,
                    "slot": slot,
                    "config_id": cfg.config_id,
                    "score": row["evaluation"][REWARD_KEY],
                    "configs_this_policy": len(trajectory),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
        if memory is not None:
            search.atomic_json(root / "semantic_memory.json", memory.distill(round_number))
    best = max(trajectory, key=lambda row: float(row["evaluation"][REWARD_KEY]))
    best_running = []
    current = 0.0
    for row in trajectory:
        current = max(current, float(row["evaluation"][REWARD_KEY]))
        best_running.append(current)
    result = {
        "seed": seed,
        "repeat": repeat,
        "policy": policy,
        "budget": len(trajectory),
        "initial_config_ids": [cfg.config_id for cfg in initial],
        "best_config": best["config"],
        "best_mean_jev_support": best["evaluation"][REWARD_KEY],
        "best_step": best["evaluation_position"],
        "best_so_far_mean_30": float(np.mean(best_running)),
        "score_oracle_hits": sum(
            bool(row["evaluation"].get("score_oracle_hit")) for row in trajectory
        ),
        "decision_seconds_total": decision_seconds_total,
    }
    search.atomic_json(summary_path, result)
    return result


def bootstrap(
    run: Path,
    samples: dict[int, list[dict[str, Any]]],
    dataset: dict[str, Any],
    candidates: list[search.SearchConfig],
    cache_source: Path,
    base_seed: int,
    repeats: int,
    policies: list[str],
    rounds: int,
    per_round: int,
    query_count: int,
    init_seeds: list[int],
    batches: list[list[search.SearchConfig]],
    access_mode: str = "relay",
) -> dict[str, Any]:
    run.mkdir(parents=True, exist_ok=True)
    if access_mode == "relay" and not (run / "relay_token").exists():
        token_src = cache_source / "relay_token"
        if not token_src.is_file():
            raise RuntimeError(f"relay token missing: {token_src}")
        shutil.copy2(token_src, run / "relay_token")
    manifest = {
        "experiment": (
            "single-seed initial-batch sensitivity: six repeats with the same search seed "
            "but pairwise-distinct seeded-maximin initial five, Jev reward, budget 30"
        ),
        "run_id": run.name,
        "cache_source_run_id": cache_source.name,
        "source": {
            k: dataset[k]
            for k in (
                "qa_all_sha256",
                "corpus_sha256",
                "qa_eval_sha256",
                "sample_seed",
                "query_count_per_seed",
                "seed_query_hashes",
            )
            if k in dataset
        },
        "base_seed": base_seed,
        "repeats": repeats,
        "initial_batch_seeds": init_seeds,
        "initial_batch_config_ids": {
            f"rep_{i + 1:02d}": [cfg.config_id for cfg in batch]
            for i, batch in enumerate(batches)
        },
        "initial_batch_design": (
            "paired.seeded_initial_batch: first point drawn from random.Random(init_seed), "
            "remaining four by maximin space filling"
        ),
        "shared_within_repeat": True,
        "policies": list(policies),
        "query_count": query_count,
        "configs_per_round": per_round,
        "rounds": rounds,
        "budget_per_policy": rounds * per_round,
        "candidate_count": len(candidates),
        "reward_key": REWARD_KEY,
        "reward_schema": REWARD_SCHEMA,
        "reward_fingerprint_schema": FINGERPRINT_SCHEMA,
        "jev_question_names": sorted(JEV_QUESTIONS),
        "redundancy_gamma": REDUNDANCY_GAMMA,
        "redundancy_mode": REDUNDANCY_MODE,
        "reward_scheme": REWARD_SCHEME,
        "reward_definition": REWARD_DEFINITION,
        "vision_model": VISION_MODEL,
        "vision_prompt_version": VISION_PROMPT_VERSION,
        "jev_model": JEV_MODEL,
        "decision_model": DECISION_MODEL,
        "frozen_config_scores_shared_across_repeats": True,
        "jev_access": (
            "loopback relay"
            if access_mode == "relay"
            else f"direct judge API ({judge_client.JUDGE_API_URL}) with bearer key from env"
        ),
    }
    em.save_once(run / "manifest.json", manifest)
    for name in ("retrieval_cache.json", "vision_facts.jsonl"):
        target = run / name
        source = cache_source / name
        if not target.exists() and source.is_file():
            shutil.copy2(source, target)
    target = run / "query_embeddings"
    if not target.exists() and (cache_source / "query_embeddings").is_dir():
        shutil.copytree(cache_source / "query_embeddings", target)
    cache_root = layout.cache_dir(run)
    src_seed_root = layout.cache_dir(cache_source)
    if not src_seed_root.is_dir():
        src_seed_root = cache_source / f"seed_{base_seed}"
    if src_seed_root.is_dir():
        for name in ("jev_responses.jsonl",):
            target = cache_root / name
            source = src_seed_root / name
            if not target.exists() and source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        # 逐文件补齐而不是 `if not target.exists()`：入口在 evaluate 之前就会建好
        # cache/config_scores 目录，旧写法在目录已存在时静默跳过整批继承，
        # 导致 --cache-from 的 run 重复付费重评源 run 已经评过的格子。
        target = cache_root / "config_scores"
        source = src_seed_root / "config_scores"
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for src_file in sorted(source.glob("*.json")):
                dst_file = target / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
    em.save_once(
        cache_root / "query_sample.json",
        {
            "seed": base_seed,
            "query_count": query_count,
            "query_keys_sha256": dataset["seed_query_hashes"][str(base_seed)],
            "queries": [
                {
                    "id": str(row["id"]),
                    "query": str(row["query"]),
                    "reference": search.reference_for(row),
                }
                for row in samples[base_seed]
            ],
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同一 seed 六次重复的初始批次敏感性流水线")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--cache-source",
        default=DEFAULT_CACHE_SOURCE,
        help="缓存源运行目录名（相对 PROJECT/runs），或绝对路径以直接引用原项目",
    )
    parser.add_argument("--seed", type=int, default=61, help="六次重复共用的基准 seed")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--initial-seed-offsets",
        default="0,1,2,3,4,5",
        help="每次重复的初始批次 seed = base_seed + offset，逗号分隔，数量须等于 repeats",
    )
    parser.add_argument("--policies", default=",".join(ALL_POLICIES))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--per-round", type=int, default=5)
    parser.add_argument("--query-count", type=int, default=50)
    parser.add_argument("--es-url", default="http://127.0.0.1:19200")
    parser.add_argument(
        "--jev-api",
        action="store_true",
        help="直连判题网关（Bearer key），不走本机回环 relay",
    )
    parser.add_argument("--jev-api-url", default=JEV_API_URL)
    parser.add_argument(
        "--jev-api-key-env",
        default="JEV_API_KEY",
        help="直连模式下 API key 所在的环境变量名（由启动器 source .env 提供）",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只写 manifest/preflight 并打印六批初始配置，不调用 JEV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = PROJECT / "runs" / args.run_id
    cache_source = Path(args.cache_source)
    if not cache_source.is_absolute():
        cache_source = PROJECT / "runs" / cache_source
    access_mode = "direct_api" if args.jev_api else "relay"
    api_key = ""
    if args.jev_api:
        api_key = os.environ.get(args.jev_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{args.jev_api_key_env} missing from environment; the launcher must source "
                ".env with JEV_MODE=api"
            )
    if not cache_source.is_dir():
        raise RuntimeError(f"cache source not found: {cache_source}")
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    for policy in policies:
        if policy not in ALL_POLICIES:
            raise ValueError(f"unknown policy: {policy}")
    offsets = [int(item) for item in args.initial_seed_offsets.split(",") if item.strip() != ""]
    if len(offsets) != args.repeats:
        raise ValueError("--initial-seed-offsets must contain exactly --repeats entries")

    if args.seed not in em.DEV_SEEDS and args.seed not in em.FINAL_SEEDS:
        em.FINAL_SEEDS = (*em.FINAL_SEEDS, args.seed)
    samples, _, dataset = em.build_samples()
    if len(samples[args.seed]) != args.query_count:
        raise RuntimeError(
            f"seed {args.seed} query sample has {len(samples[args.seed])} rows, "
            f"expected {args.query_count}"
        )
    candidates = search.all_configs()
    if len(candidates) != 864:
        raise RuntimeError(f"search space changed: {len(candidates)}")
    init_seeds, batches = initial_batches(candidates, args.seed, args.repeats, offsets)
    bootstrap(
        run, samples, dataset, candidates, cache_source, args.seed, args.repeats,
        policies, args.rounds, args.per_round, args.query_count, init_seeds, batches,
        access_mode,
    )
    search.atomic_json(
        run / "preflight.json",
        {
            "base_seed": args.seed,
            "repeats": args.repeats,
            "policies": policies,
            "budget_per_policy": args.rounds * args.per_round,
            "initial_batch_seeds": init_seeds,
            "repeats_distinct": True,
            "initial_batch_config_ids": {
                f"rep_{i + 1:02d}": [cfg.config_id for cfg in batch]
                for i, batch in enumerate(batches)
            },
        },
    )
    for i, batch in enumerate(batches, start=1):
        LOG.info("rep_%02d init_seed=%s initial=%s", i, init_seeds[i - 1],
                 [cfg.config_id for cfg in batch])
    if args.plan_only:
        LOG.info("plan-only: manifest/preflight written, no Jev calls")
        return

    if args.jev_api:
        transport: RelayTransport | DirectAPITransport = make_transport(
            args.jev_api_url, api_key
        )
    else:
        transport = RelayTransport(
            RELAY_URL, (run / "relay_token").read_text(encoding="utf-8").strip()
        )
    reachable, detail = transport.probe()
    if not reachable:
        raise RuntimeError(f"JEV {transport.mode} unreachable at {transport.url}: {detail}")
    LOG.info("jev access mode=%s url=%s probe=%s", transport.mode, transport.url, detail)

    all_queries = [row for seed in (*em.DEV_SEEDS, *em.FINAL_SEEDS) for row in samples[seed]]
    embedding = search.QueryEmbeddingCache(run, all_queries, "cpu", 8)
    for backend in search.EMBEDDING_VALUES:
        embedding.ensure(backend)
    es = search.es_client(args.es_url)
    try:
        search.validate_indices(es)
        retrieval = search.RetrievalCache(run / "retrieval_cache.json", es, embedding)
        vision = VisionCache(run)
        cache_root = layout.cache_dir(run)
        evaluator = FrozenConfigEvaluator(
            JevEvaluator(
                samples[args.seed], retrieval, vision, transport,
                cache_root / "jev_responses.jsonl",
            ),
            cache_root / "config_scores",
            dataset["seed_query_hashes"][str(args.seed)],
            args.query_count,
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
                summaries[policy] = run_policy(
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
            search.atomic_json(layout.seed_dir(run, repeat - 1) / "repeat_summary.json", pair)
            pairs.append(pair)
            LOG.info(
                "seed=%s rep=%02d complete best=%s",
                args.seed, repeat,
                {p: round(s["best_mean_jev_support"], 4) for p, s in summaries.items()},
            )
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
                "run_id": run.name,
                "base_seed": args.seed,
                "repeats": args.repeats,
                "policies": policies,
                "reward_key": REWARD_KEY,
                "reward_schema": REWARD_SCHEMA,
                "initial_batch_seeds": init_seeds,
                "initial_batch_config_ids": {
                    f"rep_{i + 1:02d}": [cfg.config_id for cfg in batch]
                    for i, batch in enumerate(batches)
                },
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