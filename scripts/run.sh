#!/usr/bin/env bash
# 只改下面几行；其余参数直接透传给 python 入口（--help 看全部选项）。
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${DATASET:-techslides}"   # techslides | pharmaceuticals | finance_en
SEEDS="${SEEDS:-20}"               # 独立搜索过程数量（1..20）
STEPS="${STEPS:-25}"               # 控制器决策步数，每策略总评估 = 5 + STEPS
TEXT_TOPK="${TEXT_TOPK:-}"         # 空=全量 5,10,20,40
IMAGE_TOPK="${IMAGE_TOPK:-}"       # 空=全量 0,1,3,5,10,20
CHUNKS="${CHUNKS:-}"               # 空=全量 64,128,256,512
BACKENDS="${BACKENDS:-}"           # 空=全量 jina_nano,jina_small,qwen3_vl_2b
POLICIES="${POLICIES:-}"           # 空=六策略全跑；或 llm,jit,random,tpe,...

[[ -f .env ]] || { echo "缺少 .env：cp .env.example .env 后填写端点与密钥" >&2; exit 1; }
set -a; . ./.env; set +a                      # 端点 / 密钥 / 模型都在这，已 gitignore
export PYTHONPATH="$PWD"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy   # 判题走内网直连，不吃代理

extra=()
[[ -n "$TEXT_TOPK" ]] && extra+=(--text-values "$TEXT_TOPK")
[[ -n "$IMAGE_TOPK" ]] && extra+=(--image-values "$IMAGE_TOPK")
[[ -n "$CHUNKS" ]] && extra+=(--chunk-values "$CHUNKS")
[[ -n "$BACKENDS" ]] && extra+=(--embedding-values "$BACKENDS")
[[ -n "$POLICIES" ]] && extra+=(--policies "$POLICIES")

mkdir -p logs
exec "${PYTHON:-python3}" -u -m mgrs.pipeline.jev_ao_format_v1_chunk_single \
  --dataset "$DATASET" \
  --repeats "$SEEDS" \
  --controller-steps "$STEPS" \
  --cache-from "" \
  --jev-api \
  --no-thinking \
  ${extra[@]+"${extra[@]}"} \
  "$@" 2>&1 | tee "logs/${DATASET}_$(date +%Y%m%d_%H%M%S).log"