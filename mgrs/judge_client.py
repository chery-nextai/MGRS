"""判题后端适配层（DQCR 的可替换判题器）。

DQCR 奖励需要一个「判题模型」对每条 (问题, 候选答案, 检索证据) 给出两道选择题的
概率分布：``support``（三档）与 ``redundancy``（四档）。本模块把「这次请求怎么发」
从评估链路里抽出来，使任何 OpenAI 兼容的对话模型都能充当判题器。

后端由 ``MGRS_JUDGE_BACKEND`` 选择：

``jev``（默认）
    原生 JEV 网关协议：``POST {model, state, questions}`` →
    ``{model, answers: {题名: {type, choice, probabilities}}}``。
    端点/模型/密钥由 ``MGRS_JUDGE_API_URL``（兼容旧名 ``JEV_API_URL``）、
    ``MGRS_JUDGE_MODEL``（旧名 ``JEV_MODEL``）、``MGRS_JUDGE_API_KEY_ENV`` 指定的
    环境变量（默认 ``JEV_API_KEY``）提供。

``openai_chat``
    任意 OpenAI 兼容 ``/chat/completions`` 端点。把 ``questions`` 档位渲染成提示词，
    要求模型返回 JSON（每题一个 choice + 各选项概率），再映射回与原生协议**完全相同**
    的应答结构；因此 ``parse_jev_response`` 的 fail-closed 校验、请求指纹缓存、
    γ 聚合链路全部不用改。

.. warning::
   换判题后端 = 换评分基线，不同后端的分数不可混算。请求指纹只包含
   ``JEV_SCHEMA_MODEL_TAG`` 前缀与 state，**不含模型名或 URL**，所以切后端时务必
   同时换一个 tag（例如 ``MGRS_JUDGE_SCHEMA_TAG=gpt4o_judge``），否则会静默命中
   另一个后端的旧应答缓存。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

__all__ = [
    "JudgeResponse",
    "JevGatewayTransport",
    "OpenAIChatTransport",
    "RelayTransport",
    "build_transport",
    "render_judge_messages",
]


def _env(*names: str, default: str = "") -> str:
    """按优先级读第一个非空环境变量。"""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def _env_float(*names: str, default: float) -> float:
    raw = _env(*names)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


# 判题端点的全部可配置项都走环境变量，仓库内不留任何默认地址或密钥。
JUDGE_BACKEND = _env("MGRS_JUDGE_BACKEND", default="jev").lower()
JUDGE_API_URL = _env("MGRS_JUDGE_API_URL", "JEV_API_URL")
JUDGE_MODEL = _env("MGRS_JUDGE_MODEL", "JEV_MODEL")
JUDGE_API_KEY_ENV = _env("MGRS_JUDGE_API_KEY_ENV", default="JEV_API_KEY")
JUDGE_TIMEOUT = _env_float("MGRS_JUDGE_TIMEOUT", default=110.0)
JUDGE_CHAT_PATH = _env("MGRS_JUDGE_CHAT_PATH", default="/chat/completions")
JUDGE_TEMPERATURE = _env_float("MGRS_JUDGE_TEMPERATURE", default=0.0)
JUDGE_MAX_TOKENS = int(_env_float("MGRS_JUDGE_MAX_TOKENS", default=1024))
# 只有确认端点支持 response_format 时才开；不支持的端点会因未知字段报 400。
JUDGE_JSON_MODE = _env("MGRS_JUDGE_JSON_MODE", default="0").lower() in {"1", "true", "yes", "on"}
# 部分网关要求额外请求头（例如放开长上下文）。用 MGRS_JUDGE_EXTRA_HEADERS 传 JSON，
# 默认空；未知请求头会被大多数网关忽略，但仍建议只在需要时开启。
def _extra_headers() -> dict[str, str]:
    raw = _env("MGRS_JUDGE_EXTRA_HEADERS")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}


JUDGE_LONG_CONTEXT_HEADER = _extra_headers()


@dataclass
class JudgeResponse:
    """统一的判题应答包装，字段与 ``requests.Response`` 的最小子集对齐。"""

    status_code: int
    text: str = ""
    payload: dict[str, Any] | None = None
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict[str, Any]:
        if self.payload is not None:
            return self.payload
        return json.loads(self.text)


def render_judge_messages(
    state: dict[str, Any],
    questions: dict[str, Any],
) -> list[dict[str, str]]:
    """把原生判题请求（state + questions）渲染成对话消息。

    提示词逐字使用档位里的 ``instructions`` 与 ``criteria`` 文本，不做改写，
    以保证与论文中 DQCR 的题面一致。
    """
    system = (
        "You are a strict retrieval judge. You are given a question, a candidate answer, "
        "the retrieved text passages and the image observations. For each judgement task you "
        "must output a probability distribution over exactly the listed options. "
        "Judge only from the supplied evidence. Respond with JSON only, no prose."
    )

    parts: list[str] = ["## Question", str(state.get("question", "")), ""]
    parts += ["## Candidate answer", str(state.get("candidate_answer", "")), ""]

    parts += ["## Retrieved text"]
    text_rows = state.get("retrieved_text") or []
    if not text_rows:
        parts.append("(none)")
    for row in text_rows:
        parts.append(
            f"[rank {row.get('rank')}] source_id={row.get('source_id', '')}\n{row.get('text', '')}"
        )
    parts.append("")

    parts += ["## Image observations"]
    image_rows = state.get("image_observations") or []
    if not image_rows:
        parts.append("(none)")
    for row in image_rows:
        parts.append(
            f"[rank {row.get('rank')}] source_id={row.get('source_id', '')}\n"
            f"{row.get('description', '')}"
        )
    parts.append("")

    schema: dict[str, Any] = {}
    for name, spec in questions.items():
        criteria = spec.get("criteria") or {}
        parts += [f"## Judgement task: {name}", str(spec.get("instructions", "")), "Options:"]
        for option, description in criteria.items():
            parts.append(f"- {option}: {description}")
        parts.append("")
        schema[name] = {
            "choice": "<one of: " + ", ".join(criteria) + ">",
            "probabilities": {option: "<float in [0,1]; all options must sum to 1>" for option in criteria},
        }

    parts += [
        "## Output format",
        "Return a single JSON object with one key per judgement task:",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    """容错解析模型输出：剥掉 ```json 围栏，取第一个完整 JSON 对象。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge output is not a JSON object")
    return parsed


class JevGatewayTransport:
    """直连原生 JEV 网关（``Authorization: Bearer <key>``，POST ``{model,state,questions}``）。

    URL、模型名、密钥全部来自环境变量或构造参数，仓库内无内置端点。若端点受出口地区
    限制（HTTP 451），需自行把 ``HTTPS_PROXY`` 指向可用出口。
    """

    mode = "direct_api"

    def __init__(
        self,
        url: str,
        key: str,
        model: str | None = None,
        timeout: float = JUDGE_TIMEOUT,
    ) -> None:
        self.url = url
        self.key = key
        self.model = model or JUDGE_MODEL
        self.timeout = timeout

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    def probe(self) -> tuple[bool, str]:
        """零成本探活：空 body 拿到 4xx 说明已抵达业务层；451 即地区封锁。"""
        try:
            response = requests.post(
                self.url,
                headers={**self.headers(), "Content-Type": "application/json"},
                json={},
                timeout=15,
            )
        except requests.RequestException as exc:
            return False, f"unreachable: {exc.__class__.__name__}"
        if response.status_code == 451:
            return False, "HTTP 451 endpoint not available in your region（需换出口或配 HTTPS_PROXY）"
        if response.status_code in {401, 403}:
            return False, f"HTTP {response.status_code} 鉴权被拒（key 无效或过期）"
        return True, f"HTTP {response.status_code}"

    def post(self, payload: dict[str, Any], timeout: float | None = None) -> JudgeResponse:
        response = requests.post(
            self.url,
            headers={**self.headers(), "Content-Type": "application/json", **JUDGE_LONG_CONTEXT_HEADER},
            json=payload,
            timeout=timeout or self.timeout,
        )
        return JudgeResponse(status_code=response.status_code, text=response.text)


class RelayTransport:
    """经本机回环 relay 访问判题端点（历史部署方式，默认不使用）。"""

    mode = "relay"

    def __init__(self, url: str, token: str, timeout: float = JUDGE_TIMEOUT) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout

    def headers(self) -> dict[str, str]:
        return {"X-Relay-Token": self.token}

    def probe(self) -> tuple[bool, str]:
        try:
            response = requests.get(self.url, headers=self.headers(), timeout=5)
        except requests.RequestException as exc:
            return False, f"unreachable: {exc.__class__.__name__}"
        return response.status_code < 500, f"HTTP {response.status_code}"

    def post(self, payload: dict[str, Any], timeout: float | None = None) -> JudgeResponse:
        response = requests.post(
            self.url,
            headers={**self.headers(), "Content-Type": "application/json", **JUDGE_LONG_CONTEXT_HEADER},
            json=payload,
            timeout=timeout or self.timeout,
        )
        return JudgeResponse(status_code=response.status_code, text=response.text)


@dataclass
class OpenAIChatTransport:
    """任意 OpenAI 兼容 ``/chat/completions`` 端点作为判题器。

    把原生请求渲染成对话消息，要求模型返回 JSON 概率分布，再映射回原生应答结构，
    使下游校验/缓存/聚合链路无感知。``model`` 字段回填为请求里的标签模型名，
    以便 ``parse_jev_response`` 的模型一致性检查通过。

    模型自报概率与 JEV 的 token 概率口径不同，**分数不可与论文中 JEV 判题的数字直接比较**；
    该后端的用途是在没有 JEV 网关时跑通完整搜索流程。
    """

    base_url: str
    key: str
    model: str
    chat_path: str = JUDGE_CHAT_PATH
    timeout: float = JUDGE_TIMEOUT
    temperature: float = JUDGE_TEMPERATURE
    max_tokens: int = JUDGE_MAX_TOKENS
    json_mode: bool = JUDGE_JSON_MODE
    extra_body: dict[str, Any] = field(default_factory=dict)

    mode = "openai_chat"

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + self.chat_path

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    def probe(self) -> tuple[bool, str]:
        try:
            response = requests.post(
                self.url,
                headers={**self.headers(), "Content-Type": "application/json"},
                json={},
                timeout=15,
            )
        except requests.RequestException as exc:
            return False, f"unreachable: {exc.__class__.__name__}"
        if response.status_code in {401, 403}:
            return False, f"HTTP {response.status_code} 鉴权被拒（key 无效或过期）"
        if response.status_code == 404:
            return False, f"HTTP 404 端点不存在：{self.url}（检查 MGRS_JUDGE_API_URL / MGRS_JUDGE_CHAT_PATH）"
        return True, f"HTTP {response.status_code}"

    def post(self, payload: dict[str, Any], timeout: float | None = None) -> JudgeResponse:
        questions = payload.get("questions") or {}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": render_judge_messages(payload.get("state") or {}, questions),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        response = requests.post(
            self.url,
            headers={**self.headers(), "Content-Type": "application/json"},
            json=body,
            timeout=timeout or self.timeout,
        )
        if not response.ok:
            return JudgeResponse(status_code=response.status_code, text=response.text)
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = _extract_json(content if isinstance(content, str) else json.dumps(content))
            answers = self._to_native_answers(parsed, questions)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # 422 = 确定性请求错误：上游会留标记并停止，不会无意义重试。
            return JudgeResponse(
                status_code=422,
                text=f"judge output unusable: {exc.__class__.__name__}: {exc}",
            )
        return JudgeResponse(
            status_code=200,
            payload={
                "model": payload.get("model") or self.model,
                "answers": answers,
                "usage": data.get("usage") or {},
                "chat_model": data.get("model", self.model),
            },
        )

    @staticmethod
    def _to_native_answers(
        parsed: dict[str, Any],
        questions: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """把模型 JSON 映射成 ``{题名: {type, choice, probabilities}}``。

        选项集合必须与档位 ``criteria`` 完全一致（多、少、改名都判失败）；概率缺失或
        非法时回退到 choice 的 one-hot，保证下游 [0,1] 校验与聚合不会拿到脏值。
        """
        answers: dict[str, dict[str, Any]] = {}
        for name, spec in questions.items():
            criteria = set((spec.get("criteria") or {}).keys())
            entry = parsed.get(name)
            if isinstance(entry, str):
                entry = {"choice": entry}
            if not isinstance(entry, dict):
                raise ValueError(f"missing judgement task {name!r} in judge output")
            raw_probs = entry.get("probabilities") or entry.get("probs") or {}
            if not isinstance(raw_probs, dict):
                raise ValueError(f"task {name!r}: probabilities must be an object")
            probabilities = {str(k): float(v) for k, v in raw_probs.items()}
            choice = str(entry.get("choice") or "")
            if set(probabilities) != criteria:
                if choice in criteria:
                    probabilities = {option: (1.0 if option == choice else 0.0) for option in criteria}
                else:
                    raise ValueError(
                        f"task {name!r}: options {sorted(probabilities)} != expected {sorted(criteria)}"
                    )
            if not all(0.0 <= value <= 1.0 for value in probabilities.values()):
                raise ValueError(f"task {name!r}: probability out of [0,1]")
            if choice not in criteria:
                choice = max(probabilities, key=lambda k: probabilities[k])
            answers[name] = {"type": "choice", "choice": choice, "probabilities": probabilities}
        return answers


def build_transport(
    backend: str | None = None,
    url: str | None = None,
    key: str | None = None,
    model: str | None = None,
) -> JevGatewayTransport | OpenAIChatTransport | RelayTransport:
    """按环境变量（或显式参数）构造判题后端。

    缺少 URL / 密钥 / 模型名时直接报错并给出应设置的变量名，避免跑到一半才失败。
    """
    kind = (backend or JUDGE_BACKEND).lower()
    api_key = key if key is not None else os.environ.get(JUDGE_API_KEY_ENV, "").strip()
    if kind in {"jev", "jev_gateway", "direct_api", "gateway"}:
        endpoint = url or JUDGE_API_URL
        label = model or JUDGE_MODEL
        missing = [
            name
            for name, value in (
                ("MGRS_JUDGE_API_URL", endpoint),
                (f"${JUDGE_API_KEY_ENV}", api_key),
                ("MGRS_JUDGE_MODEL", label),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "判题后端未配置：" + ", ".join(missing) + "（见 .env.example）"
            )
        return JevGatewayTransport(endpoint, api_key, label)
    if kind in {"openai_chat", "openai", "chat"}:
        endpoint = url or JUDGE_API_URL
        label = model or JUDGE_MODEL
        missing = [
            name
            for name, value in (
                ("MGRS_JUDGE_API_URL", endpoint),
                (f"${JUDGE_API_KEY_ENV}", api_key),
                ("MGRS_JUDGE_MODEL", label),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "openai_chat 判题后端未配置：" + ", ".join(missing) + "（见 .env.example）"
            )
        return OpenAIChatTransport(endpoint, api_key, label)
    raise ValueError(
        f"未知判题后端 {kind!r}；可选 jev / openai_chat（MGRS_JUDGE_BACKEND）"
    )
