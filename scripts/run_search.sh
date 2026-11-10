#!/usr/bin/env bash
# MGRS 定版搜索：一个知识库 × 六个策略 × 20 个 seed × 30 次评估。
# 奖励口径固定为 DQCR（双题复合：support 三档 + redundancy 四档，γ=0.5，scale 归一）。
#
# 用法：
#   DATASET=techslides bash scripts/run_search.sh                        # 续跑（已完成的 seed/策略自动跳过）
#   DATASET=finance_en STRATEGIES=jit_no_rules bash scripts/run_search.sh # 只跑部分策略
#   DATASET=pharmaceuticals VISION_FROM=<run_id> bash scripts/run_search.sh # 复用已有图像转写
#
# DATASET ∈ techslides | pharmaceuticals | finance_en
# 从零重跑：换一个新的 RUN_ID，或手动删除 runs/<RUN_ID>/{cache/,seeds/,retrieval_cache.json}
#
# 每次评估都是「真实检索 + 真实判题」：不派生旧分数、不复用检索缓存（--cache-from ""）。
# 图像转写可用 VISION_FROM=<run_id> 复用（按 image_sha256 去重）；不设置则现转写。
#
# 需要先在 .env（或环境）里配好判题与 LLM 端点，见 .env.example：
#   MGRS_JUDGE_BACKEND / MGRS_JUDGE_API_URL / MGRS_JUDGE_MODEL / <密钥变量>
#   LLM_API_BASE / LLM_API_KEY / MGRS_VISION_MODEL / MGRS_DECISION_MODEL
#   AO_EXPLORE_ES_URL、MGRS_AO_CORPUS_DIR_<DATASET>、MGRS_EMBED_MODEL_*
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
DATASET="${DATASET:-techslides}"

# ── 单实例硬锁 ───────────────────────────────────────────────────────────────
# 同一 run 目录被两个进程并发写会产生重复 trajectory 行与 conflicting cached row，
# 因此用 mkdir 原子锁：锁已存在直接退出，不会启动 python。
LOCKDIR="/tmp/mgrs_${DATASET}.$(printf '%s' "${RUN_ID:-default}" | md5sum | cut -c1-12).lockdir"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[abort] $(date '+%F %T') 已有实例持锁 $LOCKDIR，本次启动放弃" >&2
  exit 3
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT INT TERM
if pgrep -f "jev_ao_format_v1_chunk_single --dataset ${DATASET}" >/dev/null 2>&1; then
  echo "[abort] 已有 ${DATASET} 的搜索进程在跑（异常残留？），本次启动放弃" >&2
  exit 3
fi

# ── 环境与奖励口径 ───────────────────────────────────────────────────────────
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

export REDUNDANCY_GAMMA="${REDUNDANCY_GAMMA:-0.5}"     # 冗余题权重
export REDUNDANCY_MODE="${REDUNDANCY_MODE:-scale}"     # 概率 → 分数的归一方式
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# 判题/转写都是 HTTP 调用，不走系统代理；ES 与本地编码器必须留在 no_proxy 里。
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="$NO_PROXY"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export TOKENIZERS_PARALLELISM=false
export VISION_WORKERS="${VISION_WORKERS:-4}"
export PYTHON="${PYTHON:-python3}"

# ── 定版参数（三个知识库，20 个 seed 完全一致）─────────────────────────────────
REPEATS="${REPEATS:-20}"
SEED="${SEED:-67}"
STRATEGIES="${STRATEGIES:-}"          # 留空 = 六个策略全跑
QA_SUBDIR="${QA_SUBDIR:-ao_ready_v1_subsets}"
case "$DATASET" in
  techslides)
    RUN_ID="${RUN_ID:-20261103_jev_ao_chunk_single_techslides_screened50_from_200_rep6_nocache}"
    QA_FILE="${QA_FILE:-$ROOT/datasets/$QA_SUBDIR/techslides/qa_screened50_from_200.json}"
    ES_URL="${ES_URL:-${AO_EXPLORE_ES_URL:-http://127.0.0.1:29200}}"
    INITIAL_SEED_OFFSETS="${INITIAL_SEED_OFFSETS:-0,1,2,3,4,5,-67,-66,-65,-25,710,1957,56,-54,32,33,247,8,9,13}"
    ;;
  pharmaceuticals)
    RUN_ID="${RUN_ID:-20261103_jev_ao_chunk_single_pharma_screened50_from_200_rep6_nocache}"
    QA_FILE="${QA_FILE:-$ROOT/datasets/$QA_SUBDIR/pharmaceuticals/qa_screened50_from_200.json}"
    ES_URL="${ES_URL:-${AO_EXPLORE_ES_URL:-http://127.0.0.1:19200}}"
    INITIAL_SEED_OFFSETS="${INITIAL_SEED_OFFSETS:-0,1,2,3,4,5,-67,-66,-65,-25,710,1957,56,-54,32,33,247,8,9,13}"
    ;;
  finance_en)
    RUN_ID="${RUN_ID:-20261103_jev_ao_chunk_single_finance_en_q50_rep6_nocache}"
    QA_FILE="${QA_FILE:-$ROOT/datasets/$QA_SUBDIR/finance_en/qa_screened50_from_179.json}"
    ES_URL="${ES_URL:-${AO_EXPLORE_ES_URL:-http://127.0.0.1:29200}}"
    INITIAL_SEED_OFFSETS="${INITIAL_SEED_OFFSETS:-0,1,2,3,4,5,8,9,13,-67,-66,-65,-25,710,1957,56,-54,32,33,247}"
    ;;
  *)
    echo "[abort] 未知 DATASET=$DATASET（可选 techslides / pharmaceuticals / finance_en）" >&2
    exit 2
    ;;
esac

RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR" logs

# ── 防串基线 ─────────────────────────────────────────────────────────────────
# 判题请求指纹只含 schema tag 与 state，**不含模型名或 URL**：换判题模型后在同一 run
# 目录续跑会静默命中旧模型的应答缓存（表现为「换了模型分数一点没变」）。
# 所以这里把 manifest 里记录的判题模型与当前配置比对，不一致就直接拒绝启动。
MANIFEST="$RUN_DIR/manifest.json"
CURRENT_JUDGE="${MGRS_JUDGE_MODEL:-}"
if [[ -f "$MANIFEST" && -n "$CURRENT_JUDGE" && "${ALLOW_JUDGE_MISMATCH:-0}" != "1" ]]; then
  RECORDED_JUDGE="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8")).get("jev_model",""))' "$MANIFEST")"
  if [[ -n "$RECORDED_JUDGE" && "$RECORDED_JUDGE" != "$CURRENT_JUDGE" ]]; then
    echo "[abort] $MANIFEST 记录判题模型=$RECORDED_JUDGE，当前 MGRS_JUDGE_MODEL=$CURRENT_JUDGE。" >&2
    echo "        换判题模型/后端 = 换评分基线，续跑会静默命中旧应答缓存；请换一个新的 RUN_ID。" >&2
    echo "        确知后果要强行续跑：ALLOW_JUDGE_MISMATCH=1" >&2
    exit 4
  fi
fi

VISION_ARGS=()
if [[ -n "${VISION_FROM:-}" ]]; then
  VISION_ARGS=(--vision-from "$VISION_FROM")
fi
POLICY_ARGS=()
if [[ -n "$STRATEGIES" ]]; then
  POLICY_ARGS=(--policies "$STRATEGIES")
fi

echo "[run] $(date '+%F %T') dataset=$DATASET run_id=$RUN_ID repeats=$REPEATS seed=$SEED"
echo "[run] reward=dqcr gamma=$REDUNDANCY_GAMMA mode=$REDUNDANCY_MODE es=$ES_URL"
echo "[run] judge backend=${MGRS_JUDGE_BACKEND:-jev} model=${CURRENT_JUDGE:-<unset>}"

"$PYTHON" -u -m mgrs.pipeline.jev_ao_format_v1_chunk_single \
  --dataset "$DATASET" \
  --qa-file "$QA_FILE" \
  --run-id "$RUN_ID" \
  --seed "$SEED" \
  --repeats "$REPEATS" \
  --initial-seed-offsets "$INITIAL_SEED_OFFSETS" \
  --cache-from "" \
  --jev-api \
  --no-thinking \
  --es-url "$ES_URL" \
  ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
  ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} \
  2>&1 | tee "logs/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
