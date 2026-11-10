#!/usr/bin/env bash
# 离线复现最终设计报告：读 runs/ 下三个定版 run 的产物，重算全部表格与曲线数据。
# 不调用任何模型或 ES，纯本地统计，几秒完成。
#
# 用法：
#   bash scripts/reproduce_report.sh                 # 覆盖写 docs/20261104_three_dataset_rep20_report.md
#   OUT=/tmp/report.md bash scripts/reproduce_report.sh   # 写到别处，便于与仓库内版本 diff
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${OUT:-docs/20261104_three_dataset_rep20_report.md}"
PYTHON="${PYTHON:-python3}"

for run in \
  runs/20261103_jev_ao_chunk_single_techslides_screened50_from_200_rep6_nocache \
  runs/20261103_jev_ao_chunk_single_pharma_screened50_from_200_rep6_nocache \
  runs/20261103_jev_ao_chunk_single_finance_en_q50_rep6_nocache; do
  [[ -d "$run" ]] || { echo "[abort] 缺少 $run（先跑 scripts/run_search.sh 或恢复仓库自带产物）" >&2; exit 2; }
done

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m mgrs.analysis.three_dataset_rep20_report --out "$OUT"
echo "[ok] 报告已写入 $OUT"
