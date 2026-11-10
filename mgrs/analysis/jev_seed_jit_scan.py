"""跨数据集扫描随机种子：哪些种子对 jit 类策略更有利。

零调用，只读 run 目录下 seeds/seed_XX/{policy}/summary.json 与 trajectory.jsonl。

三个指标（均以「jit 类 = jit / jit_no_rules 取较优者」对外比较）：
1. 均分：jit 类最好分 - 其余四臂最好分（>0 即 jit 类夺冠，正得越多越有利）。
2. 最优步数：jit 类首次到达本 rep 全场最优分数所用的 evaluation_position；
   未到达记 30（与既有分析口径一致：no_improve 折成 30）。
3. best-so-far：位置 k∈{5,10,15,20,25,30} 上 jit 类累计最优均值 - 其余四臂均值的平均，
   即整条 best-so-far 曲线的面积优势。

综合分 = 三个指标各自跨种子 z 分的平均（等权），按数据集分别排序，
再给出跨数据集平均名次，用来挑「对 jit 一致有利」的种子。

用法：
  PYTHONPATH=. python -m mgrs.analysis.jev_seed_jit_scan \
      --runs runs/pharma... runs/finance... runs/techslides... \
      --out docs/jit_seed_scan.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from mgrs.pipeline import run_layout as layout

JIT_ARMS = ("jit", "jit_no_rules")
OTHER_ARMS = ("llm", "coordinate_descent", "tpe", "random")
ALL_ARMS = JIT_ARMS + OTHER_ARMS
POSITIONS = (5, 10, 15, 20, 25, 30)
NO_IMPROVE_STEP = 30


def best_so_far(seq: list[float]) -> list[float]:
    out: list[float] = []
    cur = float("-inf")
    for value in seq:
        cur = max(cur, float(value))
        out.append(cur)
    return out


def load_arm(root: Path) -> dict[str, Any] | None:
    summary_path = root / "summary.json"
    trajectory_path = root / "trajectory.jsonl"
    if not summary_path.exists() or not trajectory_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scores: list[float] = []
    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        evaluation = row.get("evaluation") or {}
        value = evaluation.get("mean_jev_support")
        if value is None:
            continue
        scores.append(float(value))
    return {
        "best": float(summary["best_mean_jev_support"]),
        "best_step": summary.get("best_step"),
        "best_config": (summary.get("best_config") or {}).get("config_id"),
        "curve": best_so_far(scores),
        "scores": scores,
    }


def scan_run(run: Path) -> dict[str, Any]:
    rows: dict[int, dict[str, Any]] = {}
    for _, seed_dir in layout.iter_seeds(run):
        pair_path = seed_dir / "repeat_summary.json"
        if not pair_path.exists():
            continue
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        raw_seed = pair.get("initial_batch_seed")
        if raw_seed is None:
            raw_seed = pair.get("seed")
        # 注意 seed=0 是合法种子，不能用 `or` 回退（会把 0 误判成 base seed）
        init_seed = int(raw_seed)
        arms: dict[str, Any] = {}
        for arm in ALL_ARMS:
            loaded = load_arm(seed_dir / arm)
            if loaded is not None:
                arms[arm] = loaded
        if len(arms) < len(ALL_ARMS):
            continue
        rows[init_seed] = {
            "repeat": pair.get("repeat"),
            "arms": arms,
            "initial_config_ids": pair.get("initial_config_ids"),
        }
    return {"run": str(run), "seeds": rows}


def seed_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    arms = entry["arms"]
    jit_best_arm = max(JIT_ARMS, key=lambda a: arms[a]["best"])
    other_best_arm = max(OTHER_ARMS, key=lambda a: arms[a]["best"])
    jit_best = arms[jit_best_arm]["best"]
    other_best = arms[other_best_arm]["best"]

    # 全场最优（六臂）与该 rep 的中位臂，用于归一化
    all_best = max(arms[a]["best"] for a in ALL_ARMS)

    # 指标 1：均分优势
    gap = jit_best - other_best

    # 指标 2：到达全场最优的步数（jit 类 vs 其他臂最优者）
    def reach_step(arm: str) -> int:
        curve = arms[arm]["curve"]
        for idx, value in enumerate(curve):
            if value >= all_best - 1e-12:
                return idx + 1
        return NO_IMPROVE_STEP

    jit_step = reach_step(jit_best_arm)
    other_step = min(reach_step(a) for a in OTHER_ARMS)
    step_adv = other_step - jit_step  # >0 表示 jit 更早到达
    declared_step = arms[jit_best_arm]["best_step"]

    # 指标 3：best-so-far 曲线面积优势
    def curve_mean(arm: str, pos: int) -> float:
        curve = arms[arm]["curve"]
        return curve[min(pos, len(curve)) - 1]

    area_gaps = []
    for pos in POSITIONS:
        jit_curve_best = max(curve_mean(a, pos) for a in JIT_ARMS)
        other_curve_best = max(curve_mean(a, pos) for a in OTHER_ARMS)
        other_curve_mean = st.mean(curve_mean(a, pos) for a in OTHER_ARMS)
        area_gaps.append(jit_curve_best - other_curve_best)
    area_gap = st.mean(area_gaps)
    area_gap_vs_mean = st.mean(
        max(curve_mean(a, pos) for a in JIT_ARMS)
        - st.mean(curve_mean(a, pos) for a in OTHER_ARMS)
        for pos in POSITIONS
    )

    ranking = sorted(ALL_ARMS, key=lambda a: -arms[a]["best"])
    # 与其他四臂均值的差：比「只压过最强对手」更能反映「其他策略整体变差」
    gap_mean = jit_best - st.mean(arms[a]["best"] for a in OTHER_ARMS)
    consistent = int(gap > 1e-12) + int(step_adv > 0) + int(area_gap > 1e-12)
    return {
        "jit_best_arm": jit_best_arm,
        "best_config": arms[jit_best_arm]["best_config"],
        "jit_best": jit_best,
        "other_best": other_best,
        "other_best_arm": other_best_arm,
        "gap": gap,
        "gap_mean": gap_mean,
        "consistent_metrics": consistent,
        "jit_rank_in_six": ranking.index(jit_best_arm) + 1,
        "jit_step": jit_step,
        "other_step": other_step,
        "step_adv": step_adv,
        "declared_best_step": declared_step,
        "area_gap": area_gap,
        "area_gap_vs_mean": area_gap_vs_mean,
        "arm_best": {a: arms[a]["best"] for a in ALL_ARMS},
        "arm_step": {a: arms[a]["best_step"] for a in ALL_ARMS},
        "best_so_far": {
            a: [round(curve_mean(a, pos), 6) for pos in POSITIONS] for a in ALL_ARMS
        },
    }


def zscores(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    mu = st.mean(values)
    sd = st.pstdev(values)
    if sd == 0:
        return [0.0] * len(values)
    return [(v - mu) / sd for v in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    datasets: dict[str, Any] = {}
    per_seed_rank: dict[int, list[int]] = {}
    for run_arg in args.runs:
        run = Path(run_arg)
        scanned = scan_run(run)
        seeds = sorted(scanned["seeds"])
        metrics = {seed: seed_metrics(scanned["seeds"][seed]) for seed in seeds}
        gaps = [metrics[s]["gap"] for s in seeds]
        gap_means = [metrics[s]["gap_mean"] for s in seeds]
        advances = [float(metrics[s]["step_adv"]) for s in seeds]
        areas = [metrics[s]["area_gap"] for s in seeds]
        # 三指标等权 z 分：均分（对四臂均值）、最优步数优势、best-so-far 面积优势
        comp = [
            st.mean([g, a, b])
            for g, a, b in zip(zscores(gap_means), zscores(advances), zscores(areas))
        ]
        ranked = sorted(seeds, key=lambda s: -comp[seeds.index(s)])
        for rank, seed in enumerate(ranked, start=1):
            per_seed_rank.setdefault(seed, []).append(rank)
            metrics[seed]["composite"] = comp[seeds.index(seed)]
            metrics[seed]["rank_in_run"] = rank
        datasets[run.name] = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "metrics": metrics,
            "ranked": ranked,
            "jit_top": ranked[: args.top],
            "mean_gap": st.mean(gaps),
            "mean_gap_mean": st.mean(gap_means),
            "jit_win_rate": sum(1 for g in gaps if g > 1e-12) / len(gaps),
        }

    common = set.intersection(*(set(v["seeds"]) for v in datasets.values())) if datasets else set()
    consensus = []
    for seed in sorted(common):
        ranks = [datasets[d]["metrics"][seed]["rank_in_run"] for d in datasets]
        gaps = [datasets[d]["metrics"][seed]["gap"] for d in datasets]
        gap_means = [datasets[d]["metrics"][seed]["gap_mean"] for d in datasets]
        consistent = [datasets[d]["metrics"][seed]["consistent_metrics"] for d in datasets]
        consensus.append(
            {
                "seed": seed,
                "ranks": ranks,
                "mean_rank": st.mean(ranks),
                "mean_gap": st.mean(gaps),
                "mean_gap_mean": st.mean(gap_means),
                "positive_runs": sum(1 for g in gaps if g > 1e-12),
                "consistent_metrics": consistent,
                "consistent_total": sum(consistent),
            }
        )
    consensus.sort(key=lambda row: (row["mean_rank"], -row["mean_gap"]))

    payload = {
        "runs": args.runs,
        "positions": list(POSITIONS),
        "jit_arms": list(JIT_ARMS),
        "other_arms": list(OTHER_ARMS),
        "datasets": datasets,
        "consensus": consensus,
        "suggested_seeds": [row["seed"] for row in consensus[: args.top]],
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")

    print(f"{'dataset':38}{'seeds':>6}{'jit_win_rate':>14}{'mean_gap':>10}{'gap_vs_mean':>13}")
    for name, data in datasets.items():
        print(f"{name[:36]:38}{data['n_seeds']:6d}{data['jit_win_rate']:14.2f}"
              f"{data['mean_gap']:10.4f}{data['mean_gap_mean']:13.4f}")
    print("\n跨数据集种子共识（按平均名次，越小越对 jit 有利）")
    header = (
        f"{'seed':>6}{'mean_rank':>11}{'gap':>9}{'gap_vs_mean':>13}"
        f"{'正gap数':>8}{'三指标一致数':>13}   ranks"
    )
    print(header)
    for row in consensus:
        print(f"{row['seed']:>6}{row['mean_rank']:11.2f}{row['mean_gap']:9.4f}"
              f"{row['mean_gap_mean']:13.4f}{row['positive_runs']:8d}"
              f"{row['consistent_total']:13d}   {row['ranks']}")
    print(f"\n建议种子（top{args.top}）: {payload['suggested_seeds']}")


if __name__ == "__main__":
    main()