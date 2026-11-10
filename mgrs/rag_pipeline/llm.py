from __future__ import annotations

import base64
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from mgrs import config as settings


logger = logging.getLogger("ao_explore.llm")
RETRYABLE_STATUS_CODES = {400, 408, 409, 425, 429, 500, 502, 503, 504}
RETRIEVAL_JUDGE_PROMPT_VERSION = "v5_legacy_prompt"


@dataclass(frozen=True)
class RetrievalJudgeResult:
    score: float
    raw_response: str
    parse_status: str


def _image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class AOChatClient:
    """Small OpenAI-compatible client driven entirely by environment variables."""

    def __init__(self) -> None:
        if not settings.LLM_API_BASE or not settings.LLM_API_KEY or not settings.LLM_MODEL:
            raise RuntimeError("LLM_API_BASE、LLM_API_KEY、LLM_MODEL 必须配置（见 .env.example）。")
        self.url = settings.LLM_API_BASE.rstrip("/") + "/chat/completions"

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        model: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if extra:
            # 可选请求体扩展（如 {"enable_thinking": False}）。不传时请求体与旧版逐字段一致。
            payload.update(extra)
        for attempt in range(1, settings.LLM_MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {settings.LLM_API_KEY}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=settings.LLM_TIMEOUT,
                )
            except requests.RequestException as exc:
                if attempt >= settings.LLM_MAX_ATTEMPTS:
                    raise
                logger.warning(
                    "LLM request failed attempt=%s/%s error=%s; retrying",
                    attempt,
                    settings.LLM_MAX_ATTEMPTS,
                    exc,
                )
            else:
                if response.ok:
                    body = response.json()
                    return str(body["choices"][0]["message"]["content"])

                request_id = response.headers.get("x-request-id") or response.headers.get("x-dashscope-request-id") or ""
                error_body = re.sub(r"\s+", " ", response.text).strip()[:1000]
                logger.warning(
                    "LLM API error attempt=%s/%s status=%s request_id=%s body=%s",
                    attempt,
                    settings.LLM_MAX_ATTEMPTS,
                    response.status_code,
                    request_id,
                    error_body,
                )
                if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= settings.LLM_MAX_ATTEMPTS:
                    response.raise_for_status()

            delay = settings.LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
            if delay:
                time.sleep(delay)

        raise RuntimeError("LLM request exhausted all attempts")


def _evidence_text(text_rows: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{row.get('rank')}] {row.get('text', '')}" for row in text_rows)


class MultimodalLLM:
    def __init__(self, client: AOChatClient | None = None) -> None:
        self.client = client or AOChatClient()

    def _messages(self, system: str, user_text: str, image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for row in image_rows:
            path = row.get("image_path")
            if path and Path(path).is_file():
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(str(path))}})
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    def generate_answer(self, query: str, text_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> str:
        prompt = f"问题：{query}\n\n文本证据：\n{_evidence_text(text_rows)}\n\n请只给出最终简短答案，不要解释。"
        return self.client.chat(self._messages("你是一个严谨的 ScienceQA 问答助手。只能依据证据作答。", prompt, image_rows))

    def judge_retrieval(self, query: str, ground_truth_answer: str, text_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]], model: str | None = None) -> float:
        return self.judge_retrieval_with_details(
            query, ground_truth_answer, text_rows, image_rows, model=model
        ).score

    def judge_retrieval_with_details(self, query: str, ground_truth_answer: str, text_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]], model: str | None = None) -> RetrievalJudgeResult:
        prompt = (
            "判断检索到的文本和图片是否包含足够、正确、完整且无冲突的信息，能够支持给定 ground-truth answer。"
            "评分对象只针对检索证据，不要重新生成答案。仅输出 JSON，例如 {\"retrieval_support_score\": 0.75}，分数范围为 0 到 1。\n"
            f"问题：{query}\nGround-truth answer：{ground_truth_answer}\n文本证据：\n{_evidence_text(text_rows)}"
        )
        raw = self.client.chat(self._messages("你是检索证据评审器。", prompt, image_rows), model=model or settings.SEARCH_JUDGE_MODEL)
        score, parse_status = parse_score_with_status(raw)
        return RetrievalJudgeResult(score=score, raw_response=raw, parse_status=parse_status)


def parse_score(raw: str) -> float:
    return parse_score_with_status(raw)[0]


def parse_score_with_status(raw: str) -> tuple[float, str]:
    """Return the score and whether JSON, regex fallback, or neither parsed it."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        obj = None
    if isinstance(obj, dict):
        candidate = obj.get("retrieval_support_score", obj.get("score"))
        try:
            value = float(candidate)
        except (ValueError, TypeError):
            pass
        else:
            if math.isfinite(value):
                return max(0.0, min(1.0, value)), "json"
    match = re.search(r"(?:retrieval_support_score|score)\D{0,20}(0(?:\.\d+)?|1(?:\.0+)?)", raw, re.I)
    if match:
        return float(match.group(1)), "regex"
    return 0.0, "failed"
