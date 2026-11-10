"""run 目录布局：把「第几个种子样本」与「真实随机种子」分开。

一个 run 内跑 ``--repeats N`` 时是 N 个**不同随机种子**的独立搜索，目录名用序号而不是
种子号，避免把 ``seed_67`` 误读成「这个 run 全是 seed 67 跑的」::

    runs/<run_id>/
        manifest.json  preflight.json  run_info.json  summary.json
        retrieval_cache.json  vision_facts.jsonl  query_embeddings/
        cache/                      # 跨种子共享的判题产物（同配置同题只判一次）
            query_sample.json
            jev_responses.jsonl
            config_scores/
        seeds/
            seed_00/                # 第 1 个种子样本
                seed.json           # {"index": 0, "seed": 67, "repeat": 1, ...}
                repeat_summary.json
                jit/  jit_no_rules/  random/  llm/  coordinate_descent/  tpe/
            seed_01/                # 第 2 个种子样本
            ...

旧布局（``<run>/seed_<真实种子>/rep_01/...``）仍可读，:func:`iter_seeds` 会自动回落。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

CACHE_DIRNAME = "cache"
SEEDS_DIRNAME = "seeds"


def cache_dir(run: Path) -> Path:
    """run 级共享缓存的目录。"""
    return Path(run) / CACHE_DIRNAME


def cache_path(run: Path, name: str, *, legacy_seed: int | None = None) -> Path:
    """共享缓存文件路径；新布局缺失时回落到旧布局 ``seed_<seed>/<name>``。"""
    run = Path(run)
    new = cache_dir(run) / name
    if legacy_seed is not None and not new.exists():
        legacy = run / f"seed_{legacy_seed}" / name
        if legacy.exists():
            return legacy
    return new


def seeds_root(run: Path) -> Path:
    return Path(run) / SEEDS_DIRNAME


def seed_dir(run: Path, index: int) -> Path:
    """第 ``index`` 个种子样本的产物目录（index 从 0 开始）。"""
    return seeds_root(run) / f"seed_{index:02d}"


def write_seed_meta(
    run: Path,
    index: int,
    *,
    seed: int,
    base_seed: int,
    repeat: int,
    initial_seed_offset: int | None = None,
) -> Path:
    """在种子目录写入 ``seed.json``，把序号与真实随机种子的对应关系落盘。"""
    path = seed_dir(run, index) / "seed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "index": index,
        "seed": seed,
        "base_seed": base_seed,
        "repeat": repeat,
    }
    if initial_seed_offset is not None:
        payload["initial_seed_offset"] = initial_seed_offset
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def iter_seeds(run: Path) -> Iterator[tuple[int, Path]]:
    """枚举 ``(序号, 目录)``，新布局优先，缺 ``seeds/`` 时回落旧布局。"""
    run = Path(run)
    root = seeds_root(run)
    if root.is_dir():
        found = sorted(p for p in root.glob("seed_*") if p.is_dir())
        for position, path in enumerate(found):
            try:
                index = int(path.name.split("_", 1)[1])
            except (IndexError, ValueError):
                index = position
            yield index, path
        return
    for path in sorted(p for p in run.glob("seed_*") if p.is_dir()):
        for rep in sorted(p for p in path.glob("rep_*") if p.is_dir()):
            try:
                index = int(rep.name.split("_", 1)[1]) - 1
            except (IndexError, ValueError):
                continue
            yield index, rep


def read_seed_meta(directory: Path) -> dict[str, Any] | None:
    path = Path(directory) / "seed.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None