from __future__ import annotations

import re
from collections import Counter


def tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", str(text).lower())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(tokens(prediction)), Counter(tokens(reference))
    overlap = sum((pred & ref).values())
    if not pred or not ref or overlap == 0:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)
