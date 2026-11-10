# MGRS: Memory-Guided Multimodal Retrieval Search

在**多模态 RAG 的检索配置空间**（文本召回的 `top_k`、图像描述的 `top_k`、分块粒度 `chunk_size`、
视觉-语言编码器后端）里做**在线黑盒配置搜索**。搜索的奖励不是字面匹配分，而是一个
**双题复合判题奖励（DQCR, Dual-Question Composite Reward）**：让判题模型对每条
「问题 + 候选答案 + 检索证据」回答两道选择题——证据是否**支持**该答案（三档）、证据是否**冗余**（四档）——
再合成单个标量分数。

仓库同时包含三个知识库上的**定版实验**（20 个 seed × 六策略 × 每策略 30 次评估）的**完整落盘产物**，
可在**完全不联网、不调用任何模型**的前提下重算出全部报告表格与曲线。

---

## 1. 核心设计

**搜索空间**（288 个候选，4 个维度）

| 维度                  | 取值                                             |
| --------------------- | ------------------------------------------------ |
| `text_top_k`        | 5, 10, 20, 40                                    |
| `image_top_k`       | 0, 1, 3, 5, 10, 20                               |
| `chunk_size`        | 64, 128, 256, 512                                |
| `embedding_backend` | `jina_nano` / `jina_small` / `qwen3_vl_2b` |

**奖励（DQCR 双题复合）**

```
score = P(support) · (1 − γ · E[redundancy]),   γ = 0.5, mode = scale
```

`support` 为三档、`redundancy` 为四档，`E[R] = Σ w·P`。
写入轨迹的键名仍是 `mean_jev_support`，但里面装的是**折减后**的分数；未折减分量见
`mean_jev_support_raw`。γ / mode 只影响读时聚合，**不进请求指纹**，因此改 γ 不会触发重新判题。

**六个对照策略**：`jit`（本文方法）、`jit_no_rules`、`random`、`llm`、`coordinate_descent`、`tpe`。
每个 seed 的六个策略共享同一批 5 个初始配置（1 个随机点 + 4 个 maximin 填充），
之后各走 25 个控制器步，共 30 次评估。

定版参数见 [`configs/definitive_runs.json`](configs/definitive_runs.json)；
完整设计、结果与消融见 [`docs/20261104_three_dataset_rep20_report.md`](docs/20261104_three_dataset_rep20_report.md)。

---

## 2. 仓库结构

```
mgrs/
  judge_client.py              判题后端适配层（jev 原生协议 / 任意 OpenAI 兼容端点）
  config.py                    全局配置（全部走环境变量）
  rag_pipeline/                ES 检索、文本分块、本地编码器、LLM 客户端
  controllers/ metrics/ memory/ environment/ analysis/
  pipeline/                    实验流水线（主入口 + 六策略控制器 + DQCR 奖励层）
scripts/
  run.sh                       日常入口：8 个变量 → 透传给 python 入口
  run_search.sh                复现专用：钉死 20261103 三个 run 的参数（带锁与判题守卫）
  build_ao_dataset_index.py    为语料建 ES 索引（3 后端 × 4 chunk 文本 + 3 图片）
  index_build/                 建库实现（自包含，不依赖其它项目）
  reproduce_report.sh          纯离线复算报告
datasets/ao_ready_v1_subsets/  三个数据集的 50 题子集与筛选统计
runs/                          三个定版实验的完整产物（manifest / cache / seeds / 轨迹）
docs/                          定版设计与结果报告
configs/definitive_runs.json   定版参数
```

依赖：Python ≥ 3.10，`pip install -r requirements.txt`；端点、密钥、语料与权重路径全部走 `.env`
（`cp .env.example .env` 后填，仓库内不含任何地址与密钥）。

---

## 3. 数据

仓库只附带**题集子集**，语料需自行获取：

| 数据集              | 语料                                      | 随附子集                        | 题数 |
| ------------------- | ----------------------------------------- | ------------------------------- | ---- |
| `techslides`      | IBM TechXchange 公开技术幻灯片（1963 页） | `qa_screened50_from_200.json` | 50   |
| `pharmaceuticals` | 制药领域披露文档（2313 页）               | `qa_screened50_from_200.json` | 50   |
| `finance_en`      | 金融领域英文披露文档（2942 页）           | `qa_screened50_from_179.json` | 50   |

* 每个子集配一份 `*_metrics.csv`，记录筛选时的可答性与图像相关性统计。
* 语料形态：一个目录含 `corpus.json`（每页 `{id, text, images:[...]}`）与图片文件。若自有语料，
  按同样结构组织即可，无需改代码。
* **题集默认值**：`--qa-file` 留空时，主入口自动取 `datasets/ao_ready_v1_subsets/<ds>/qa_screened50_from_*.json`；
  找不到才退回数据集目录自带的 `qa_search.json`（全量题集）。`techslides` 目录下另有一份
  `qa_first50_from_qa_test.json`（另一种取前 50 的口径），自动挑选**不会**选它——要用请显式传 `--qa-file`。
* 语料版权归各自发布方所有，本仓库不再分发正文与图像；请从原始来源获取并遵守其许可。

---

## 4. 索引

索引已在本地 ES 上建好，跑实验只需连接，不用重建：

| 数据集 | ES | 索引前缀 |
| --- | --- | --- |
| `techslides` | `127.0.0.1:29200` | `yiwe_ao_techslides_b86435175991_handoff_v1` |
| `finance_en` | `127.0.0.1:29200` | `yiwe_ao_finance_en_b36d2b08b6fc_handoff_v1` |
| `pharmaceuticals` | `127.0.0.1:19200` | `aoexplore_pharmaceuticals_p2313` |

每个前缀下 15 个索引 = 文本 12（4 种 chunk × 3 后端）+ 图片 3；查询侧只读、不建不删：

```
<prefix>_text_embedding_v2_<backend>                # chunk=512，即分块语料本身
<prefix>_c{64,128,256}_text_embedding_v2_<backend>  # 其余三种 chunk
<prefix>_image_embedding_v2_<backend>               # 图片，与 chunk 无关
```

前缀由 `MGRS_INDEX_PREFIX_<DATASET>` 决定，`.env` 里已设好。代码内置的默认前缀是原始机器的
`mgrs_ao_*`，与本机 `yiwe_ao_*` 只差开头这段（哈希部分相同）；`bash scripts/run.sh --validate-only`
会逐个核对这 15 个索引是否存在且非空。

只有换语料、需要自己建库时才动 `scripts/build_ao_dataset_index.py`（`--help` 看参数）。

---

## 5. 跑实验

```bash
bash scripts/run.sh                                        # 默认 techslides，六策略 × 20 seed × 30 次评估
DATASET=finance_en SEEDS=6 bash scripts/run.sh
CHUNKS=512 DATASET=pharmaceuticals bash scripts/run.sh     # 只跑 chunk=512（72 个候选）
CHUNKS=512 BACKENDS=jina_nano SEEDS=6 bash scripts/run.sh  # 小规模探针（24 个候选）
bash scripts/run.sh --plan-only                            # 预演初始批次，不建 run、不调模型
bash scripts/run.sh --validate-only                        # 只校验语料/题集/索引
```

`scripts/run.sh` 只有 8 个变量可改：数据集、种子数、决策步数、策略子集，
以及搜索空间四维的收窄（`TEXT_TOPK` / `IMAGE_TOPK` / `CHUNKS` / `BACKENDS`，留空即全量
`5,10,20,40` / `0,1,3,5,10,20` / `64,128,256,512` / `jina_nano,jina_small,qwen3_vl_2b`）。
其余参数直接透传给 `mgrs/pipeline/jev_ao_format_v1_chunk_single.py`（`--help` 看全部）。
默认值即定版口径，都在入口脚本里：题集自动取 `datasets/ao_ready_v1_subsets/<ds>/qa_screened50_from_*.json`，
seed 偏移表按数据集内置，ES 端口与前缀见 §4。

收窄四维会缩小候选空间，入口会先算候选数并守住两条底线：必须 ≥ 5（初始批大小），
且 ≥ `SEEDS`（每个 repeat 的初始五步要互不相同），不满足直接拒绝并提示放宽哪一维。

端点、密钥、语料路径、编码器权重全在 `.env`（已 `.gitignore`，仓库内不含任何地址与密钥），
配置项与说明见 `.env.example`；换机器只改 `.env`。

`scripts/run_search.sh` 是**复现专用**入口：把 `runs/20261103_*` 三个定版 run 的
`RUN_ID` / `QA_FILE` / `ES_URL` / 初始 seed 偏移表全部钉死，并额外带单实例锁与判题基线守卫。
新实验用 `run.sh`；要逐字复现那三个 run、或用 `VISION_FROM` 复用图像转写时用它。
注意它的 `RUN_ID` 里仍写着 `rep6`，是历史命名，实际 `repeats=20`。

两者共同行为：

* 自动 `source .env`，固定 `γ=0.5`、`mode=scale`（`REDUNDANCY_GAMMA` / `REDUNDANCY_MODE` 可覆盖）。
* 每步都是「真实检索 + 真实判题」，不使用派生分数（`--cache-from ""`）。
* 图像转写可按 `image_sha256` 复用：`run.sh` 传 `--vision-from`，`run_search.sh` 用 `VISION_FROM`。

以下两项**只有 `run_search.sh` 提供**：

* **单实例硬锁**（`mkdir` 原子锁 + `pgrep` 兜底），同一数据集不会并发写同一 run 目录。
  用 `run.sh` 时请自行保证同一 `RUN_ID` 不会被并发启动——并发写会产生重复 trajectory 行与冲突缓存行。
* **判题基线守卫**：若 `runs/<run_id>/manifest.json` 记录的判题模型与当前 `MGRS_JUDGE_MODEL` 不一致，
  直接拒绝启动（换模型必须换 `RUN_ID`；确知后果可用 `ALLOW_JUDGE_MISMATCH=1` 绕过）。
  用 `run.sh` 续跑旧 run 前，请自行核对 `manifest.json` 里的 `jev_model`。

预算参考：三个定版 run 的判题调用量相近——`techslides` 13,473、`pharmaceuticals` 13,608、
`finance_en` 13,673 次（唯一配置 275–282 个 × 50 题，同配置同题只判一次，由 `config_scores/*.json`
的 `jev_new_calls` 累加得到）。墙钟主要取决于 LLM 决策等待与图像转写，定版均为 `--no-thinking`。

---

## 6. 离线复现报告

```bash
bash scripts/reproduce_report.sh                          # 覆盖写 docs/20261104_three_dataset_rep20_report.md
OUT=/tmp/report.md bash scripts/reproduce_report.sh       # 写到别处再 diff
```

不调用任何模型或 ES，读 `runs/` 下的产物重算全部表格，几秒完成，输出与仓库内报告**逐字节一致**。

---

## 7. 换判题后端

`mgrs/judge_client.py` 支持两种后端：

| `MGRS_JUDGE_BACKEND` | 行为                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `jev`                | 原生网关协议 `POST {model, state, questions}` → `{model, answers:{...}}`                               |
| `openai_chat`        | 任意 OpenAI 兼容 `/chat/completions`：把题面渲染成提示词，要求返回 JSON，再映射回**同一套**应答结构 |

```bash
MGRS_JUDGE_BACKEND=openai_chat
MGRS_JUDGE_API_URL=https://api.example.com/v1
MGRS_JUDGE_MODEL=my-judge-model
JEV_API_KEY=sk-...
MGRS_JUDGE_SCHEMA_TAG=myjudge_v1        # 必须换！
```

映射后的应答仍走 `parse_jev_response` 的 fail-closed 校验：题名缺失、选项集不符、概率不满 1 都会直接失败，
不会静默降级。

> **换后端 = 换评分基线。** 请求指纹只含 `MGRS_JUDGE_SCHEMA_TAG` 与 state，**不含 URL 与模型名**；
> 不换 tag 会让新模型静默命中旧基线的应答缓存（表现为「换了模型分数一点没变」）。

---

## 8. 保持口径一致

* **判题模型与 schema tag**：定版为 `Jev_AO` + `jev_ao`，与 `runs/*/manifest.json` 的
  `reward_fingerprint_schema` 对应。跨 run 比较分数前先确认这两项。
* **`decision_thinking`**：定版实验全部 `--no-thinking`。思考链会让每次决策多等 20–160 s，
  `llm` 臂的决策等待会显著推高墙钟（且在长流上更易触发超时兜底）；若打开，须在两个 run 之间显式声明该差异。
* **seed 与 repeat**：重复是**独立搜索过程**（每 repeat 用不同 `init_seed`），但配置评分缓存在同一 run 内
  跨 repeat 共享，所以「同配置同题」只判一次。跨 run 的 `x/20` 只作描述统计，不作为显著性证据。
* **run 目录布局**：`seeds/seed_00 … seed_19` 是**样本序号**（第几个种子），不是随机种子本身；
  真实种子写在每个目录的 `seed.json` 里（`{"index": 0, "seed": 67, "base_seed": 67, "repeat": 1}`）。
  跨种子共享的判题产物放在 run 级 `cache/` 下：
  ```
  runs/<run_id>/
    cache/            query_sample.json  jev_responses.jsonl  config_scores/
    seeds/seed_00/    seed.json  repeat_summary.json  jit/  jit_no_rules/  ...（六策略）
    seeds/seed_01/    ...
  ```
* **图像转写**：`qwen3.7-flash` 在**不知道问题**的条件下转写（`image_facts_no_question_v1`），
  结果按 `image_sha256` 缓存并写进 `vision_facts.jsonl`。

---

## 9. 产物与许可

* `runs/` 下三个定版实验的完整产物（约 200 MB）通过 **Git LFS** 分发：
  ```bash
  git lfs install && git lfs pull
  ```
* `runs/*/manifest.json`、`run_info.json`、`cache/config_scores/*.json` 记录了每次调用的用量与判题口径，
  可用于对账；产物中的端点地址与主机路径已脱敏为占位符，不影响任何数值复算。
* 代码以 [MIT](LICENSE) 许可发布。
