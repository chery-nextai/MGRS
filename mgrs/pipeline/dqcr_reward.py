"""DQCR（Dual-Question Composite Reward）：双题复合判题奖励的题面、聚合与请求指纹。

一次判题调用同时回答两道独立选择题：

    support     检索证据是否确立候选答案           3 档：supported / insufficient / conflict
    redundancy  证据集去重后还剩多少未重复的事实   4 档：all_distinct … mostly_repeated

两题的概率各自独立，聚合后仍只有一个标量出口：

    E[R]  = Σ w · P(redundancy)
    score = P(support) · (1 − γ · E[R])      # mode=scale，恒在 [0,1]
    score = P(support) − γ · E[R]            # mode=subtract

拆成两道题的原因：单题写法会把「支持度」和「冗余度」挤进同一个 P(supported)，
冗余只能以否决条件的形式起作用——要么完全无视它，要么让它接管判定并反转排序，
中间强度不可达。分成两题后支持度通道被解耦，冗余度成为可连续调节的独立项。

**γ 只进 REWARD_SCHEMA、不进 FINGERPRINT_SCHEMA**：换 γ 不改请求，判题应答缓存全命中，
因此 γ 扫描是零调用的离线重聚合。
"""
from __future__ import annotations

import os
from typing import Any, NamedTuple

# 口径标识：写进 manifest 的 reward_scheme 字段。
SCHEME = "dqcr"

# schema 前缀必须带判题模型标识：`JevEvaluator.fingerprint()` 只含本 schema 串与请求
# state，**不含 model 名或 URL**，所以换判题模型/后端而前缀不动时，旧应答会被静默命中
# 复用（表现为"换了模型分数一点没变"）。
# 默认 `jev_ao`（定版实验所用前缀，写在各 run 的 manifest.reward_fingerprint_schema 里）。
# 换判题后端时请一并改前缀，例如 MGRS_JUDGE_SCHEMA_TAG=gpt4o_judge。
MODEL_SCHEMA_TAG = (
    os.environ.get("MGRS_JUDGE_SCHEMA_TAG")
    or os.environ.get("JEV_SCHEMA_MODEL_TAG")
    or "jev_ao"
).strip()

# 聚合参数（只影响聚合，不影响请求）：γ 越大冗余折减越强。
DEFAULT_GAMMA = 0.5
DEFAULT_MODE = "scale"

# 请求指纹用的 schema：不含 γ / mode，因此换聚合参数不改请求。
FINGERPRINT_SCHEMA = (
    MODEL_SCHEMA_TAG + "_dual_support_plus_redundancy_qwen37_image_facts"
)

# 两个 criteria 共用的收尾约束：题面与候选答案都是待验证的断言，不能当证据用。
_TAIL = (
    "The question and candidate answer are claims to verify, not evidence. "
    "Do not use outside facts or guess from answer choices. "
    "If an image description is absent, do not imagine the image."
)

# ---- support 题：三档，只判证据本身是否确立候选答案 ------------------------------
SUPPORT_QUESTION: dict[str, Any] = {
    "type": "choice",
    "criteria": {
        "supported": (
            "The retrieved evidence itself establishes the candidate answer with enough "
            "specific and consistent information."
        ),
        "insufficient": (
            "The retrieved evidence does not establish the candidate answer; required facts, "
            "figures, or relevant image content are missing."
        ),
        "conflict": (
            "The retrieved evidence clearly supports an answer that contradicts the candidate "
            "answer."
        ),
    },
    "instructions": (
        "Classify whether the retrieved evidence alone supports the candidate answer. "
        + _TAIL
    ),
}

# ---- redundancy 题：4 档分级，映射到 [0,1] 权重后取期望 --------------------------
# 只问"去重后新增了多少事实"，不问"有多少条重复"——后者容易被集合体量代理。
REDUNDANCY_QUESTION: dict[str, Any] = {
    "type": "choice",
    "criteria": {
        "all_distinct": (
            "Every retrieved item supplies facts that no other item in the set already states."
        ),
        "mostly_distinct": (
            "Most items supply facts not stated elsewhere; only a small share of the set "
            "restates facts that already appear in other items."
        ),
        "half_repeated": (
            "About half of the items restate facts already stated elsewhere in the set; the "
            "remaining items supply facts not stated elsewhere."
        ),
        "mostly_repeated": (
            "Most items restate facts already stated elsewhere in the set, so the set supplies "
            "little beyond a few distinct facts."
        ),
    },
    "instructions": (
        "Judge how much of the retrieved evidence set supplies facts that are not already "
        "stated elsewhere in the same set. Compare the items with each other, and count a fact "
        "once no matter how many items state it. Judge repetition of facts, not the size of the "
        "set: a long set whose items each add new facts is not repetitive. " + _TAIL
    ),
}

REDUNDANCY_WEIGHTS: dict[str, float] = {
    "all_distinct": 0.0,
    "mostly_distinct": 0.25,
    "half_repeated": 0.5,
    "mostly_repeated": 1.0,
}

# 发给判题器的题集：两道题在同一次调用里完成。
QUESTIONS: dict[str, Any] = {
    "support": SUPPORT_QUESTION,
    "redundancy": REDUNDANCY_QUESTION,
}

DEFINITION = (
    "mean Jev support discounted by a gamma-weighted redundancy expectation over fixed "
    "queries, where 'support' asks whether the retrieved evidence establishes the candidate "
    "answer and 'redundancy' is a separate 4-level choice question grading how many distinct "
    "facts the set supplies beyond what other items already state, with Qwen image-only "
    "transcriptions"
)


class ResolvedReward(NamedTuple):
    """一次解析出的完整口径。下游入口逐字段绑定为模块级常量。"""

    schema: str  # 写进 config_scores / 冻结校验（含 γ 与 mode）
    fingerprint_schema: str  # 写进 JEV 请求指纹（不含 γ 与 mode）
    questions: dict[str, Any]  # 实际发出的 JEV_QUESTIONS
    definition: str
    gamma: float
    mode: str


def questions() -> dict[str, Any]:
    """返回判题题集（深拷贝一层，避免调用方改动模块常量）。"""
    return {name: dict(spec) for name, spec in QUESTIONS.items()}


def schema(gamma: float | None = None, mode: str | None = None) -> str:
    """config_scores 的口径标识：把 γ / mode 一并写入，换聚合参数即成新口径。"""
    g = DEFAULT_GAMMA if gamma is None else float(gamma)
    m = mode or DEFAULT_MODE
    return f"{FINGERPRINT_SCHEMA}_g{g:g}_{m}_agg"


def combine(
    answers: dict[str, dict[str, float]],
    gamma: float = 0.0,
    mode: str = DEFAULT_MODE,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """把 JEV 多题应答折成单一奖励标量（纯函数，便于离线重聚合与单测）。

    answers: {题名: {选项: 概率}}
    返回 {"score", "support_score", "redundancy_expected"}。
    """
    if "support" not in answers:
        raise ValueError("JEV 应答缺少 support 题")
    support = float(answers["support"].get("supported", 0.0))
    table = REDUNDANCY_WEIGHTS if weights is None else weights
    redundancy = 0.0
    if "redundancy" in answers:
        probs = answers["redundancy"]
        unknown = set(probs) - set(table)
        if unknown:
            raise ValueError(f"冗余题出现未知选项: {sorted(unknown)}")
        redundancy = float(sum(table[key] * float(value) for key, value in probs.items()))
    if "redundancy" not in answers or gamma == 0.0:
        score = support
    elif mode == "scale":
        score = support * (1.0 - gamma * redundancy)
    elif mode == "subtract":
        score = support - gamma * redundancy
    else:
        raise ValueError(f"未知冗余聚合模式: {mode!r}")
    return {
        "score": float(score),
        "support_score": support,
        "redundancy_expected": redundancy,
    }


def _resolve_gamma() -> float:
    raw = os.environ.get("REDUNDANCY_GAMMA", str(DEFAULT_GAMMA)).strip()
    try:
        gamma = float(raw)
    except ValueError:
        raise SystemExit(f"REDUNDANCY_GAMMA={raw!r} 不是数字") from None
    if not 0.0 <= gamma <= 1.0:
        raise SystemExit(f"REDUNDANCY_GAMMA={gamma} 超出 [0,1]")
    return gamma


def _resolve_mode() -> str:
    mode = (os.environ.get("REDUNDANCY_MODE") or DEFAULT_MODE).strip().lower()
    if mode not in {"scale", "subtract"}:
        raise SystemExit(f"REDUNDANCY_MODE={mode!r}，可选：scale, subtract")
    return mode


def resolve() -> ResolvedReward:
    """解析聚合参数。只从环境变量来（REDUNDANCY_GAMMA / REDUNDANCY_MODE），非法值 fail-closed。"""
    gamma = _resolve_gamma()
    mode = _resolve_mode()
    return ResolvedReward(
        schema=schema(gamma, mode),
        fingerprint_schema=FINGERPRINT_SCHEMA,
        questions=questions(),
        definition=DEFINITION,
        gamma=gamma,
        mode=mode,
    )


if __name__ == "__main__":
    import hashlib
    import json

    resolved = resolve()
    digest = hashlib.sha256(
        json.dumps(resolved.questions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    print(f"scheme             = {SCHEME}")
    print(f"reward_schema      = {resolved.schema}")
    print(f"fingerprint_schema = {resolved.fingerprint_schema}")
    print(f"gamma / mode       = {resolved.gamma:g} / {resolved.mode}")
    print(f"questions          = {sorted(resolved.questions)}")
    print(f"support options    = {list(resolved.questions['support']['criteria'])}")
    print(f"redundancy options = {list(resolved.questions['redundancy']['criteria'])}")
    print(f"prompt_sha256      = {digest}")