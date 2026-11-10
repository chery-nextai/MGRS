from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scienceqa(
    dataset_dir: str | Path,
    qa_file: str = "qa_search.json",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(dataset_dir)
    qa_path = Path(qa_file)
    if not qa_path.is_absolute():
        qa_path = root / qa_path
    with qa_path.open(encoding="utf-8") as handle:
        qa = json.load(handle)
    with (root / "corpus.json").open(encoding="utf-8") as handle:
        corpus = json.load(handle)
    if not isinstance(qa, list) or not isinstance(corpus, list):
        raise ValueError("qa.json 和 corpus.json 顶层必须是 list")
    return qa, corpus


def load_dataset(dataset_dir: str | Path, qa_file: str = 'qa_search.json') -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return load_scienceqa(dataset_dir, qa_file)
