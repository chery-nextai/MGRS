#!/usr/bin/env python
"""为 AO 格式语料构建 3 后端 x (文本/图像) 共 6 个 ES 索引。

背景
----
实验查询侧只读预建索引。本脚本按「索引即配置」契约驱动 scripts/index_build/ 里的
两个 builder（分块 + SentenceTransformer 本地编码 + 批量写入），保证 schema / 索引命名
/ _meta 与查询侧 BACKENDS 完全一致：
  <prefix>_text_embedding_v2_<backend>
  <prefix>_image_embedding_v2_<backend>
查询侧（mgrs/pipeline/jev_ao_format_v1_paired.py 的 DATASETS）只需填同一个 prefix。
编码器权重目录用 MGRS_EMBED_MODEL_ROOT / MGRS_EMBED_MODEL_DIR_<BACKEND> 指定（见 .env.example）。

安全约束
--------
* 两个 builder 的 _ensure_index 只做「存在即校验」（比对 _meta 与文档数），不匹配就抛
  错，**从不删除或覆盖**已存在索引。本脚本同样不删除任何已存在索引。
* 拒绝使用含 train3944 的前缀（历史实验索引的保护标记）。
* --limit-pages 会**自动**给前缀加 _smokeN 后缀，杜绝小样索引污染正式索引名。
* --drop 只删本次前缀下的索引，且同样拒绝 train3944。
* --backends 给出多个时，脚本会为每个后端各起一个子进程（内存与显存干净、可断点续跑）。

用法
----
建库是离线长任务（单后端约 1.5~2h），务必放后台重定向：
  nohup python scripts/build_ao_dataset_index.py ... > build_<prefix>_<backend>.log 2>&1 &

python scripts/build_ao_dataset_index.py \
  --dataset-dir "$MGRS_AO_CORPUS_DIR_PHARMACEUTICALS" \
  --index-prefix ao_pharmaceuticals_p2313 \
  --backends jina_nano,jina_small,qwen3_vl_2b \
  --es-url http://127.0.0.1:19200 --device cuda

小样试跑（索引名自动变成 ..._smoke5_*，试完用 --limit-pages 5 --drop 清掉）
  ... --index-prefix ao_pharmaceuticals_p2313 --limit-pages 5 --backend jina_nano
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
FORBIDDEN_TOKENS = ("train3944",)
BACKEND_CHOICES = ("jina_nano", "jina_small", "qwen3_vl_2b")


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="为 AO 格式语料建 ES 索引（建库实现在 scripts/index_build/）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", required=True, help="数据集目录（含 corpus.json 与图片）")
    p.add_argument("--corpus", default="corpus.json", help="语料文件名")
    p.add_argument("--index-prefix", required=True,
                   help="索引前缀，例如 ao_pharmaceuticals_p2313。"
                        "文本索引=%s_text_embedding_v2_{backend}，图像=%s_image_embedding_v2_{backend}")
    p.add_argument("--backend", default=None, choices=BACKEND_CHOICES,
                   help="只建单个后端（子进程模式使用）")
    p.add_argument("--backends", default="jina_nano,jina_small,qwen3_vl_2b",
                   help="逗号分隔；多于一个时自动为每个后端起子进程")
    p.add_argument("--es-url", default="http://127.0.0.1:19200")
    p.add_argument("--device", default="cuda", help="AO_EMBEDDING_DEVICE，如 cuda / cuda:3 / cpu")
    p.add_argument("--cuda-visible-devices", default=None, help="可选，限定可见 GPU，如 3")
    p.add_argument("--chunk-size", type=int, default=512, help="须与查询侧一致")
    p.add_argument("--chunk-overlap", type=int, default=64, help="须与查询侧一致")
    p.add_argument("--limit-pages", type=int, default=None,
                   help="只取前 N 页做小样试跑；会自动给前缀加 _smokeN 后缀")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--image-only", action="store_true")
    p.add_argument("--drop", action="store_true",
                   help="删除本次前缀下的索引后退出（不建库），用于清理小样")
    return p.parse_args(argv)


def resolve_prefixes(args: argparse.Namespace) -> tuple[str, str]:
    prefix = args.index_prefix.strip().rstrip("_")
    for token in FORBIDDEN_TOKENS:
        if token in prefix:
            raise SystemExit(f"[拒绝] 前缀含受保护标记 {token!r}（现有实验索引）：{prefix}")
    if len(prefix) < 8:
        raise SystemExit(f"[拒绝] 前缀过短、易碰撞：{prefix!r}")
    if args.limit_pages:
        prefix = f"{prefix}_smoke{args.limit_pages}"
    return f"{prefix}_text_embedding_v2", f"{prefix}_image_embedding_v2"


# --------------------------------------------------------------------------- #
# 环境引导：必须在 import index_build 之前完成
# --------------------------------------------------------------------------- #
def bootstrap_env(args: argparse.Namespace, text_prefix: str, image_prefix: str) -> None:
    os.environ["AO_EXPLORE_ES_URL"] = args.es_url
    os.environ["AO_EMBEDDING_DEVICE"] = args.device
    os.environ["AO_EXPLORE_TEXT_INDEX"] = text_prefix
    os.environ["AO_EXPLORE_IMAGE_INDEX"] = image_prefix
    os.environ.setdefault("ES_VERIFY_CERTS", "false")
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    for path in (SCRIPTS_DIR, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def import_index_build():
    """导入建库实现（scripts/index_build/）。"""
    from index_build import config as settings  # noqa: PLC0415
    from index_build.rag_pipeline.image_retrieval import ImageRetriever  # noqa: PLC0415
    from index_build.rag_pipeline.text_retrieval import TextRetriever  # noqa: PLC0415

    return settings, TextRetriever, ImageRetriever


# --------------------------------------------------------------------------- #
# 语料
# --------------------------------------------------------------------------- #
def load_corpus(dataset_dir: Path, corpus_name: str, limit: int | None, need_images: bool):
    path = dataset_dir / corpus_name
    if not path.is_file():
        raise SystemExit(f"[失败] 语料不存在：{path}")
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        raise SystemExit(f"[失败] 语料为空或非列表：{path}")

    total = len(corpus)
    if limit:
        corpus = corpus[:limit]

    if need_images:
        missing = [r.get("id") for r in corpus if not (r.get("image_path") and Path(r["image_path"]).is_file())]
        if missing:
            raise SystemExit(
                f"[失败] {len(missing)}/{len(corpus)} 页图片缺失，例如 {missing[:3]}。"
                " 图像索引要求所有 image_path 可达。"
            )
    empty = sum(1 for r in corpus if not str(r.get("content") or "").strip())
    print(f"[语料] {path}")
    print(f"[语料] 使用 {len(corpus)}/{total} 页，content 为空的页 {empty} 页（文本索引会跳过它们）")
    return corpus, total


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
def preload_and_patch_jina(backend: str) -> None:
    """中和 jina v5 omni 远端代码里的 URL 下载。

    jina 的 custom_st.py 会对每个待编码字符串调用 _is_media_string -> _resolve_input
    -> _download_if_url，而 _download_if_url 在文本以 http:// 或 https:// 开头时会
    直接 urllib.request.urlretrieve 下载。本语料有 318 页含 URL，滑窗切分后必然出现
    「chunk 文本以 URL 开头」，此时 urlparse 出的 host/path 含空格，urlretrieve 抛
    InvalidURL，导致整批 2313 页编码全崩（且索引已建好但为空）。

    本管线只传纯文本或 PIL.Image，从不传 URL，所以把 _download_if_url 换成恒等函数，
    使 _resolve_input 回落到 ("text", x)，语义正确且彻底去掉编码期网络访问。

    注意：必须先加载模型，远端模块此刻才进入 sys.modules，之后才能打补丁。
    """
    from index_build.rag_pipeline.multimodal_encoder import MultimodalEncoder  # noqa: PLC0415

    encoder = MultimodalEncoder(backend)  # 命中 _model_cache，TextRetriever 随后复用同一份

    def _identity(x):
        return x

    patched: list[str] = []
    for name, mod in list(sys.modules.items()):
        attrs = getattr(mod, "__dict__", None)
        if not isinstance(attrs, dict):
            continue
        if callable(attrs.get("_download_if_url")) and callable(attrs.get("_resolve_input")):
            mod._download_if_url = _identity
            patched.append(name)
    if not patched:
        raise SystemExit(f"[失败] {backend} 远端模块未找到 _download_if_url，补丁未生效")

    print(f"[补丁] 已中和 URL 下载，命中模块 {len(patched)} 个")

    # 回归自测：URL 开头的文本必须按文本编码，且不得触发网络访问
    probe = "http://plato httpn broken url /moe Regidor, E. 2004. x"
    vectors = encoder.encode_text([probe], is_query=False)
    if vectors.shape[0] != 1:
        raise SystemExit("[失败] 补丁自测未通过")
    print(f"[补丁] 自测通过：URL 开头文本已按文本编码 dim={vectors.shape[1]}")


def build(args, text_prefix, image_prefix, corpus, total) -> list[dict]:
    settings, TextRetriever, ImageRetriever = import_index_build()
    print(f"[配置] ES={settings.ES_URL}  device={settings.EMBEDDING_DEVICE}  backend={args.backend}")
    print(f"[配置] text_prefix={settings.TEXT_INDEX_NAME}  image_prefix={settings.IMAGE_INDEX_NAME}")
    if settings.ES_URL != args.es_url:
        raise SystemExit(f"[失败] ES 地址未生效：{settings.ES_URL} != {args.es_url}")

    if args.backend.startswith("jina"):
        preload_and_patch_jina(args.backend)

    from index_build.config import PipelineConfig  # noqa: PLC0415

    results: list[dict] = []
    do_text = not args.image_only
    do_image = not args.text_only

    if do_text:
        cfg = PipelineConfig(
            config_id=f"build_{args.backend}",
            text_final_top_k=20,
            image_top_k=20,
            embedding_backend=args.backend,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        t0 = time.time()
        retriever = TextRetriever(corpus, text_prefix, cfg)
        try:
            results.append({
                "kind": "text", "index": retriever.index_name,
                "expected": len(retriever.documents), "seconds": time.time() - t0,
            })
        finally:
            retriever.close()
        print(f"[文本] {retriever.index_name} chunks={len(retriever.documents)} 用时 {time.time()-t0:.1f}s")

    if do_image:
        t0 = time.time()
        retriever = ImageRetriever(corpus, args.backend)
        try:
            results.append({
                "kind": "image", "index": retriever.index_name,
                "expected": len(retriever.rows), "seconds": time.time() - t0,
            })
        finally:
            retriever.close()
        print(f"[图像] {retriever.index_name} docs={len(retriever.rows)} 用时 {time.time()-t0:.1f}s")

    return results


def verify(args, results: list[dict]) -> None:
    """独立复核：ES 实际文档数、mapping 维度、_meta。"""
    from elasticsearch import Elasticsearch  # noqa: PLC0415

    es = Elasticsearch(hosts=[args.es_url], verify_certs=False, request_timeout=60)
    print("\n[复核]")
    ok = True
    for row in results:
        idx = row["index"]
        if not es.indices.exists(index=idx):
            print(f"  {idx}: 不存在 ✗")
            ok = False
            continue
        count = es.count(index=idx)["count"]
        mapping = es.indices.get_mapping(index=idx)[idx]["mappings"]
        props = mapping.get("properties", {})
        dims = props.get("embedding", {}).get("dims")
        meta = mapping.get("_meta", {})
        match = count == row["expected"]
        ok &= match
        print(f"  {idx}")
        print(f"    count={count} (预期 {row['expected']}) {'✓' if match else '✗'}  dims={dims}"
              f"  schema={meta.get('schema_version')}  用时 {row['seconds']:.1f}s")
    if not ok:
        raise SystemExit("[失败] 复核未通过（文档数与预期不符）")
    print("  -- 全部通过 ✓")


def drop_indexes(args, text_prefix, image_prefix) -> None:
    from elasticsearch import Elasticsearch  # noqa: PLC0415

    es = Elasticsearch(hosts=[args.es_url], verify_certs=False, request_timeout=60)
    targets = [f"{text_prefix}_{args.backend}", f"{image_prefix}_{args.backend}"]
    for idx in targets:
        for token in FORBIDDEN_TOKENS:
            if token in idx:
                raise SystemExit(f"[拒绝] 不允许删除含 {token!r} 的索引：{idx}")
        if es.indices.exists(index=idx):
            es.indices.delete(index=idx)
            print(f"[删除] {idx}")
        else:
            print(f"[跳过] {idx} 不存在")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    text_prefix, image_prefix = resolve_prefixes(args)

    if args.drop:
        bootstrap_env(args, text_prefix, image_prefix)
        if args.limit_pages and len(sys.argv) > 1:
            print(f"[删除模式] 目标前缀 text={text_prefix} image={image_prefix}")
        if args.backend:
            drop_indexes(args, text_prefix, image_prefix)
        else:
            for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
                args.backend = backend
                drop_indexes(args, text_prefix, image_prefix)
        return

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not backends:
        raise SystemExit("[失败] --backends 为空")

    # 多后端：每个后端一个子进程，保证显存/内存干净且可断点续跑
    if args.backend is None and len(backends) > 1:
        print(f"[调度] {len(backends)} 个后端将各起一个子进程：{backends}")
        failures = []
        for backend in backends:
            child = [
                sys.executable, str(Path(__file__).resolve()),
                "--dataset-dir", args.dataset_dir, "--corpus", args.corpus,
                "--index-prefix", args.index_prefix, "--backend", backend,
                "--es-url", args.es_url, "--device", args.device,
                "--chunk-size", str(args.chunk_size), "--chunk-overlap", str(args.chunk_overlap),
            ]
            if args.cuda_visible_devices is not None:
                child += ["--cuda-visible-devices", args.cuda_visible_devices]
            if args.limit_pages:
                child += ["--limit-pages", str(args.limit_pages)]
            if args.text_only:
                child.append("--text-only")
            if args.image_only:
                child.append("--image-only")
            print(f"\n{'='*70}\n[子进程] backend={backend}\n{'='*70}", flush=True)
            env = dict(os.environ)
            extra = os.pathsep.join([str(SCRIPTS_DIR), str(REPO_ROOT)])
            env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            rc = subprocess.run(child, env=env, cwd=str(REPO_ROOT)).returncode
            if rc != 0:
                failures.append(backend)
                print(f"[子进程] backend={backend} 失败 rc={rc}")
        if failures:
            raise SystemExit(f"[失败] 以下后端未完成：{failures}（已完成的索引可安全重跑，builder 会复用）")
        print("\n[调度] 全部后端完成 ✓")
        return

    if args.backend is None:
        args.backend = backends[0]

    bootstrap_env(args, text_prefix, image_prefix)
    need_images = not args.text_only
    corpus, total = load_corpus(Path(args.dataset_dir), args.corpus, args.limit_pages, need_images)

    print(f"\n[开始] backend={args.backend}  pages={len(corpus)}"
          f"  chunk={args.chunk_size}/{args.chunk_overlap}  device={args.device}")
    t0 = time.time()
    results = build(args, text_prefix, image_prefix, corpus, total)
    print(f"\n[完成] backend={args.backend} 总用时 {time.time()-t0:.1f}s")
    verify(args, results)


if __name__ == "__main__":
    main()