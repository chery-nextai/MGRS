"""Batched Coordinate Descent and categorical TPE for the 864-config RAG search.

Adapted from the selection rules in RAISE (arXiv:2605.30029), rather than
RAISE's end-to-end RAG pipeline.  Controllers see only completed observations;
all five proposals in a batch are fixed before any of them is evaluated.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any


def _observed(observations: list[dict[str, Any]], reward_key: str) -> list[tuple[float, int, str, tuple[int, ...]]]:
    rows = []
    for row in observations:
        rows.append((
            float(row["evaluation"][reward_key]),
            int(row["evaluation_position"]),
            str(row["config"]["config_id"]),
            tuple(int(x) for x in row["config"]["param_idx"]),
        ))
    return rows


class CoordinateDescentController:
    """Examine one-parameter neighbors of the best observed configurations.

RAISE evaluates coordinate choices sequentially.  Here each batch is proposed
    from the observations available at its start; coordinate order rotates so
    every dimension receives evaluations within a 30-trial budget.
    """

    name = "coordinate_descent"

    def __init__(self, reward_key: str) -> None:
        self.reward_key = reward_key

    def select_batch(self, remaining: list[Any], observations: list[dict[str, Any]],
                     round_number: int, batch_size: int = 5) -> tuple[list[Any], dict[str, dict[str, Any]]]:
        if not observations or len(remaining) < batch_size:
            raise ValueError("coordinate descent requires observed scores and enough candidates")
        by_idx = {tuple(c.param_idx): c for c in remaining}
        if len(by_idx) != len(remaining):
            raise ValueError("duplicate parameter tuples")
        history = sorted(_observed(observations, self.reward_key), key=lambda x: (-x[0], x[1], x[2]))
        dims = len(history[0][3])
        order = [((round_number - 2 + offset) % dims) for offset in range(dims)]
        selected: list[Any] = []
        diagnostics: dict[str, dict[str, Any]] = {}
        for score, _, pivot_id, pivot in history:
            for dim in order:
                options = []
                for idx, cfg in by_idx.items():
                    if cfg.config_id in diagnostics:
                        continue
                    if sum(left != right for left, right in zip(idx, pivot)) == 1 and idx[dim] != pivot[dim]:
                        options.append(cfg)
                options.sort(key=lambda c: (abs(c.param_idx[dim] - pivot[dim]), c.config_id))
                for cfg in options:
                    selected.append(cfg)
                    diagnostics[cfg.config_id] = {
                        "selection": "coordinate_descent", "pivot_config_id": pivot_id,
                        "pivot_score": score, "dimension_index": dim,
                        "old_index": pivot[dim], "new_index": cfg.param_idx[dim],
                    }
                    if len(selected) == batch_size:
                        return selected, diagnostics
        # This is only reachable after one-hop neighborhoods have been exhausted.
        for cfg in sorted(remaining, key=lambda c: c.config_id):
            if cfg.config_id not in diagnostics:
                selected.append(cfg)
                diagnostics[cfg.config_id] = {"selection": "coordinate_descent_exhausted_fallback"}
                if len(selected) == batch_size:
                    return selected, diagnostics
        raise RuntimeError("coordinate descent could not fill batch")


class TPEController:
    """RAISE-style categorical good/bad density ratio with batch feedback."""

    name = "tpe"

    def __init__(self, reward_key: str, candidates: list[Any], seed: int,
                 startup_trials: int = 10, gamma: float = 0.2,
                 alpha: float = 1.0, candidate_pool_size: int = 24) -> None:
        if not 0 < gamma < 1 or alpha <= 0 or startup_trials < 1 or candidate_pool_size < 1:
            raise ValueError("invalid TPE settings")
        self.reward_key = reward_key
        self.seed = seed
        self.startup_trials = startup_trials
        self.gamma = gamma
        self.alpha = alpha
        self.candidate_pool_size = candidate_pool_size
        self.dimensions = len(candidates[0].param_idx)
        self.choices = [sorted({int(c.param_idx[d]) for c in candidates}) for d in range(self.dimensions)]

    def select_batch(self, remaining: list[Any], observations: list[dict[str, Any]],
                     round_number: int, batch_size: int = 5) -> tuple[list[Any], dict[str, dict[str, Any]]]:
        if len(remaining) < batch_size:
            raise ValueError("not enough candidates")
        rng = random.Random(self.seed * 1000 + round_number)
        if len(observations) < self.startup_trials:
            batch = rng.sample(sorted(remaining, key=lambda c: c.config_id), batch_size)
            return batch, {c.config_id: {
                "selection": "tpe_startup_random", "startup_trials": self.startup_trials,
            } for c in batch}

        history = sorted(_observed(observations, self.reward_key), key=lambda x: (-x[0], x[1], x[2]))
        n_good = max(1, int(len(history) * self.gamma))
        good, bad = history[:n_good], history[n_good:]
        good_counts = [Counter(row[3][d] for row in good) for d in range(self.dimensions)]
        bad_counts = [Counter(row[3][d] for row in bad) for d in range(self.dimensions)]
        ratios = []
        for d, values in enumerate(self.choices):
            ratios.append({value: (
                (good_counts[d][value] + self.alpha) / (len(good) + self.alpha * len(values))
            ) / (
                (bad_counts[d][value] + self.alpha) / (len(bad) + self.alpha * len(values))
            ) for value in values})
        by_idx = {tuple(c.param_idx): c for c in remaining}
        selected: list[Any] = []
        diagnostics: dict[str, dict[str, Any]] = {}

        def surrogate(cfg: Any) -> float:
            return sum(math.log(ratios[d][int(value)]) for d, value in enumerate(cfg.param_idx))

        for _ in range(batch_size):
            proposals: dict[str, Any] = {}
            for _ in range(self.candidate_pool_size * 20):
                idx = tuple(rng.choices(values, weights=[ratios[d][v] for v in values], k=1)[0]
                            for d, values in enumerate(self.choices))
                cfg = by_idx.get(idx)
                if cfg is None or cfg.config_id in diagnostics:
                    continue
                proposals[cfg.config_id] = cfg
                if len(proposals) >= self.candidate_pool_size:
                    break
            if proposals:
                cfg = max(proposals.values(), key=lambda c: (surrogate(c), c.config_id))
                source = "tpe_density_ratio"
            else:
                pool = [c for c in remaining if c.config_id not in diagnostics]
                cfg = max(pool, key=lambda c: (surrogate(c), c.config_id))
                source = "tpe_pool_exhausted_fallback"
            selected.append(cfg)
            diagnostics[cfg.config_id] = {
                "selection": source, "surrogate_log_ratio": surrogate(cfg),
                "n_good": len(good), "n_bad": len(bad),
                "gamma": self.gamma, "alpha": self.alpha,
                "candidate_pool_size": self.candidate_pool_size,
            }
        return selected, diagnostics
