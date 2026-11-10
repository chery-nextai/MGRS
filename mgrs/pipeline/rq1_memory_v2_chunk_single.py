#!/usr/bin/env python3
"""chunk 288 空间的「单点决策」变体：JIT 每步 top-1 + LLM 每次只决策 1 个配置。

与 `rq1_memory_v2_chunk_dim.py` 的差异**只有决策粒度**：那个版本一次决策 5 个
（选完再读反馈），本版本一次只决策 1 个（每步用得上上一步的反馈）。空间、距离函数、
奖励口径、题集、索引一律相同。

为什么必须单独一个模块
======================
`six.run_policy` 有两处硬编码的「每轮 5 个」：
  1. `MemoryV2.select_jit(remaining, rng)` 固定返回 3 利用 + 1 探索 + 1 安全随机；
  2. `LLMController.select_batch` 固定请求 5 个（prompt 文案 / payload / 兜底都写死 5）。
沿用仓库既有做法：只在进程内替换 `search` / `six` 的模块属性，不修改共享模块，
864 空间那条线与批量版 chunk 实验保持字节级可复现。

单点 JIT 的语义（一处刻意偏离，必须写进报告）
============================================
原 5 槽 = 3 利用 + 1 探索（最大化到最近观测点的距离）+ 1 安全随机。单点模式按既定
口径改为**每步取 utility 最高者**：探索槽与安全随机槽都不存在，探索性只剩 utility 里的
`EXPLORE_LAMBDA * explore` 距离项。jit 与 jit_no_rules 的差异被压缩到只剩语义规则
修正项，解读时必须按此口径说明。
"""

from __future__ import annotations

import logging
from typing import Any

from mgrs.pipeline import rq1_memory_v2_chunk_dim as chunk
from mgrs.pipeline import rq1_memory_v2_jit_random as search


LOGGER = logging.getLogger("ao_explore.rq1_memory_v2_chunk_single")


class TopUtilityMemoryMixin:
    """单点 JIT：每步只取 utility 最高的未评配置（无探索槽、无安全随机槽）。"""

    def select_jit(
        self,
        remaining: list[search.SearchConfig],
        rng: Any,
    ) -> tuple[list[search.SearchConfig], dict[str, dict[str, Any]]]:
        del rng
        if not remaining:
            raise ValueError("no remaining configurations")
        scored = {cfg.config_id: self.score(cfg) for cfg in remaining}
        # 与批量版 exploit 排序同口径：utility 降序，并列时 config_id 升序。
        chosen = min(remaining, key=lambda cfg: (-scored[cfg.config_id]["utility"], cfg.config_id))
        return [chosen], {chosen.config_id: {"selection": "utility_top1", **scored[chosen.config_id]}}


class SingleStepMemory(TopUtilityMemoryMixin, search.MemoryV2):
    """`jit` 臂：JIT 打分（kNN 预测 + 规则修正 + 探索项）+ 每步 top-1。"""


class SingleStepNoRulesMemory(TopUtilityMemoryMixin, search.MemoryV2):
    """`jit_no_rules` 臂：禁用语义规则蒸馏（同 `six.NoRulesMemory`）+ 每步 top-1。"""

    def distill(self, round_number: int) -> list[dict[str, Any]]:
        del round_number
        self.semantic_rules = []
        return []


def _single_step_prompt(prompt: str) -> str:
    """把批量 prompt 的「5 个」改成「1 个」；模板漂移时报错，而不是静默按 5 个决策。"""
    adapted = prompt.replace("每次选 5 个未评估的完整配置", "每次只选 1 个未评估的完整配置")
    adapted = adapted.replace("configs 必须恰好有 5 项", "configs 必须恰好有 1 项")
    if adapted == prompt:
        raise RuntimeError("system prompt 模板已变化，单点改写失败（拒绝静默按 5 个决策）")
    return adapted


class SingleStepLLMController(chunk.ChunkLLMController):
    """chunk 版 LLM 决策器，但每次只决策 1 个配置。"""

    # 请求体扩展（由 install() 按决策口径设定）：None = 原始思考链。
    request_extra: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        kwargs["system_prompt"] = _single_step_prompt(str(kwargs["system_prompt"]))
        kwargs.setdefault("request_extra", type(self).request_extra)
        super().__init__(batch_size=1, **kwargs)


def _assert_no_rules_shape(base: type) -> None:
    """`six.NoRulesMemory` 只允许覆写 distill；多出成员说明本变体需要同步。"""
    extra = {name for name in vars(base) if not name.startswith("__")} - {"distill"}
    if extra:
        raise RuntimeError(f"six.NoRulesMemory 新增了 {sorted(extra)}，单点变体需同步")


def install(*, six_module: Any = None, decision_thinking: bool = True) -> dict[str, Any]:
    """把 chunk 空间 + 单点控制器装进 `search` / `six`；进程内全局生效，幂等。

    `decision_thinking=False` 时给决策请求带上 `{"enable_thinking": False}`：同一模型、
    同一 prompt，只是不带思考链（实测单次决策由 20–160s 降到 ~2s，且不再撞网关 180s
    流超时）。默认 True = 原始行为，其他调用方不受影响。
    """
    space = chunk.install(six_module=six_module)
    SingleStepLLMController.request_extra = (
        None if decision_thinking else {"enable_thinking": False}
    )
    # six.run_policy：jit 走 `search.MemoryV2()`，jit_no_rules 走模块级
    # `NoRulesMemory()`，所以补丁要分别打在 search 与 six 上。
    search.MemoryV2 = SingleStepMemory
    if six_module is not None:
        _assert_no_rules_shape(six_module.NoRulesMemory)
        six_module.MemoryV2 = SingleStepMemory
        six_module.NoRulesMemory = SingleStepNoRulesMemory
        six_module.LLMController = SingleStepLLMController
        # SYSTEM_PROMPT 保持 chunk 版（仍写「5 个」），由 SingleStepLLMController 改写。
    return {
        **space,
        "decision_granularity": 1,
        "jit_selection": "utility_top1_no_slots",
        "llm_batch_size": 1,
        "decision_thinking": decision_thinking,
    }