from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from mgrs import config as settings


def canonical_state_text(state: dict[str, Any]) -> str:
    return json.dumps({k: state.get(k) for k in ("step", "budget", "incumbent", "incumbent_params", "recent_rewards", "action_family_counts", "tested_config_ids", "remaining_count", "failure_count", "search_space_version")}, sort_keys=True, ensure_ascii=False)


class StateEncoder:
    def __init__(self):
        if not settings.EMBEDDING_MODEL:
            raise RuntimeError("AO_EMBEDDING_MODEL 未配置，无法编码 memory state。")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)

    def encode(self, state: dict[str, Any]) -> list[float]:
        vector = self.model.encode([canonical_state_text(state)], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vector, dtype=np.float32).tolist()


class ExperienceMemory:
    def __init__(self, path: str | Path = settings.MEMORY_PATH, encoder: StateEncoder | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder or StateEncoder()

    def append(self, transition: dict[str, Any]) -> None:
        state = transition.get("state", {})
        row = dict(transition)
        row["state_embedding"] = self.encoder.encode(state)
        row.setdefault("return", None)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def retrieve(self, state: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
        query = np.asarray(self.encoder.encode(state), dtype=np.float32)
        scored = []
        for row in self._rows():
            if row.get("metric_version") != settings.METRIC_VERSION or row.get("search_space_version") != settings.SEARCH_SPACE_VERSION:
                continue
            if row.get("search_judge_model") != settings.SEARCH_JUDGE_MODEL:
                continue
            vector = np.asarray(row.get("state_embedding", []), dtype=np.float32)
            if vector.shape != query.shape:
                continue
            denom = math.sqrt(float(vector @ vector)) * math.sqrt(float(query @ query))
            similarity = float(vector @ query) / denom if denom else 0.0
            scored.append((similarity, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{**row, "state_similarity": score} for score, row in scored[:top_k]]
