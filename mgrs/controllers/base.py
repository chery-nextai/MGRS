from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Controller(ABC):
    name = "base"

    def __init__(self, seed: int = 11):
        self.seed = seed

    @abstractmethod
    def select(self, state: dict[str, Any], candidates: list[dict[str, Any]], memory: Any) -> dict[str, Any]:
        raise NotImplementedError

    def observe(self, transition: dict[str, Any]) -> None:
        del transition
