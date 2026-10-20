"""集中配置：所有后端能力都可在“真组件”和“本地降级”之间切换。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # dotenv 可选
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("ZH_AN_DATA_DIR", BASE_DIR / "data"))
CORPUS_DIR = Path(os.getenv("ZH_AN_CORPUS_DIR", BASE_DIR / "corpus"))
WEB_DIR = BASE_DIR / "web"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    app_name: str = "筑安云"
    app_subtitle: str = "建筑施工安全规范智能问答系统"
    version: str = "1.4.2"
    host: str = os.getenv("ZH_AN_HOST", "127.0.0.1")
    port: int = _i("ZH_AN_PORT", 8011)
    debug: bool = _b("ZH_AN_DEBUG", False)

    # ---------- 存储 ----------
    # sqlite（默认，零依赖）/ mysql（配置 DSN 后自动切换）
    db_backend: str = os.getenv("ZH_AN_DB_BACKEND", "sqlite")
    sqlite_path: Path = DATA_DIR / "zhuan.db"
    mysql_dsn: str = os.getenv("ZH_AN_MYSQL_DSN", "")  # user:pass@host:port/db

    # ---------- 缓存 ----------
    redis_url: str = os.getenv("ZH_AN_REDIS_URL", "")
    cache_ttl: int = _i("ZH_AN_CACHE_TTL", 1800)
    cache_max_items: int = _i("ZH_AN_CACHE_MAX_ITEMS", 512)

    # ---------- 向量库 ----------
    # milvus-lite 默认落地为本地文件，URI 换成 http://host:19530 即接集群
    milvus_uri: str = os.getenv("ZH_AN_MILVUS_URI", str(DATA_DIR / "milvus_zhuan.db"))
    milvus_collection: str = os.getenv("ZH_AN_MILVUS_COLLECTION", "zhuan_clauses")
    milvus_token: str = os.getenv("ZH_AN_MILVUS_TOKEN", "")

    # ---------- 模型 ----------
    embed_model: str = os.getenv("ZH_AN_EMBED_MODEL", "BAAI/bge-m3")
    embed_fallback: str = os.getenv("ZH_AN_EMBED_FALLBACK", "BAAI/bge-small-zh-v1.5")
    embed_device: str = os.getenv("ZH_AN_EMBED_DEVICE", "cpu")
    embed_batch: int = _i("ZH_AN_EMBED_BATCH", 16)
    embed_dim: int = _i("ZH_AN_EMBED_DIM", 1024)
    # 本地 ONNX 模型目录（走 fastembed + onnxruntime，无需 PyTorch；存在则优先使用）
    embed_onnx_path: str = os.getenv("ZH_AN_EMBED_ONNX_PATH", str(BASE_DIR / "models" / "bge-small-zh-v1.5"))
    # 是否允许启动时自动联网下载模型（默认关：演示机走本地模型目录，缺模型就直接降级）
    auto_download: bool = _b("ZH_AN_AUTO_DOWNLOAD", False)
    hf_endpoint: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

    rerank_model: str = os.getenv("ZH_AN_RERANK_MODEL", "BAAI/bge-reranker-large")
    rerank_fallback: str = os.getenv("ZH_AN_RERANK_FALLBACK", "BAAI/bge-reranker-base")
    rerank_onnx_model: str = os.getenv("ZH_AN_RERANK_ONNX_MODEL", "BAAI/bge-reranker-v2-m3-int8")
    rerank_onnx_path: str = os.getenv("ZH_AN_RERANK_ONNX_PATH", str(BASE_DIR / "models" / "bge-reranker-v2-m3-int8"))

    # ---------- 生成 ----------
    llm_base_url: str = os.getenv("ZH_AN_LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("ZH_AN_LLM_API_KEY", "")
    llm_model: str = os.getenv("ZH_AN_LLM_MODEL", "qwen-plus")
    llm_timeout: int = _i("ZH_AN_LLM_TIMEOUT", 45)

    # ---------- 检索参数 ----------
    top_k_dense: int = _i("ZH_AN_TOP_K_DENSE", 20)
    top_k_sparse: int = _i("ZH_AN_TOP_K_SPARSE", 20)
    rrf_k: int = _i("ZH_AN_RRF_K", 60)
    rerank_top_n: int = _i("ZH_AN_RERANK_TOP_N", 8)
    min_relevance: float = _f("ZH_AN_MIN_RELEVANCE", 0.28)
    max_parents: int = _i("ZH_AN_MAX_PARENTS", 4)

    # ---------- 意图 ----------
    intent_model: str = os.getenv("ZH_AN_INTENT_MODEL", "BERT")  # BERT / TFIDF / RULE
    intent_conf_threshold: float = _f("ZH_AN_INTENT_CONF", 0.55)

    categories: list[str] = field(
        default_factory=lambda: ["基坑工程", "脚手架工程", "模板支架", "高处作业", "施工用电", "起重机械", "文明施工与消防"]
    )
    intent_labels: list[str] = field(
        default_factory=lambda: ["规范条文查询", "隐患判定", "整改要求", "系统闲聊"]
    )


settings = Settings()
