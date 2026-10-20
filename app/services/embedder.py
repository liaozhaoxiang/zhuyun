"""向量化：BGE-M3（稠密+稀疏双路）优先，模型不可用时退到本地确定性向量。

降级链：bge-m3 -> bge-small-zh-v1.5 -> hash（零依赖，保证任何机器可跑）
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from pathlib import Path
from typing import Iterable

from app.config import settings

_CJK = re.compile(r"[\u4e00-\u9fa5]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    toks = _WORD.findall(text.lower())
    chars = _CJK.findall(text)
    toks.extend(chars)
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i + 1])
    return toks


def hash_embed(text: str, dim: int = 1024) -> list[float]:
    """确定性哈希向量：字符 1/2-gram + 英文词，带次线性 TF 与 L2 归一化。"""
    vec = [0.0] * dim
    counts: dict[str, int] = {}
    for t in _tokens(text):
        counts[t] = counts.get(t, 0) + 1
    for t, c in counts.items():
        h = hashlib.md5(t.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(c))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def lexical_weights(text: str) -> dict[str, float]:
    """降级稀疏表示：jieba 分词 + 次线性 TF。"""
    try:
        import jieba

        toks = [t.strip() for t in jieba.lcut(text) if len(t.strip()) > 1]
    except Exception:
        toks = _tokens(text)
    counts: dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    return {t: 1.0 + math.log(c) for t, c in counts.items()}


class Embedder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._m3 = None
        self._onnx = None
        self.backend = "hash"
        self.dim = 1024
        self.model_name = "hash-ngram-1024"
        self.device = "cpu"
        self.load_ms = 0
        self.error = ""
        self._tried = False

    # ---------- 加载 ----------
    def ensure_loaded(self) -> None:
        if self._tried:
            return
        with self._lock:
            if self._tried:
                return
            self._tried = True
            t0 = time.time()
            try:
                import os

                os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
                # 优先 PyTorch 后端
                try:
                    from sentence_transformers import SentenceTransformer

                    for name, fallback in ((settings.embed_model, False), (settings.embed_fallback, True)):
                        try:
                            self._model = SentenceTransformer(name, device=settings.embed_device, trust_remote_code=False)
                            self.backend = "bge-fallback" if fallback else "bge-m3"
                            self.model_name = name
                            self.dim = self._model.get_sentence_embedding_dimension() or 1024
                            self.device = settings.embed_device
                            if not fallback:
                                try:
                                    from FlagEmbedding import BGEM3FlagModel

                                    self._m3 = BGEM3FlagModel(name, use_fp16=False)
                                except Exception:
                                    self._m3 = None
                            break
                        except Exception as e:  # 模型下载失败 -> 试下一个
                            self.error = f"{name}: {type(e).__name__}"
                            continue
                except Exception as e:
                    self.error = f"sentence-transformers 不可用: {type(e).__name__}"

                # 其次 ONNX 后端（无需 torch，CPU 上更快，模型体积小）
                if self._model is None:
                    try:
                        from fastembed import TextEmbedding

                        onnx_dir = Path(settings.embed_onnx_path)
                        candidates = []
                        if onnx_dir.exists():
                            candidates.append((settings.embed_fallback, str(onnx_dir)))
                        if settings.auto_download:
                            candidates.append((settings.embed_fallback, None))
                        for name, local_path in candidates:
                            try:
                                kwargs = {"model_name": name}
                                if local_path:
                                    kwargs["specific_model_path"] = local_path
                                self._onnx = TextEmbedding(**kwargs)
                                self.backend = "fastembed-onnx"
                                self.model_name = name + ("(local)" if local_path else "")
                                probe = list(self._onnx.embed(["维度探测"]))[0]
                                self.dim = int(len(probe))
                                self.device = "cpu-onnx"
                                break
                            except Exception as e:
                                self.error = f"{name}: {type(e).__name__}"
                                self._onnx = None
                                continue
                    except Exception as e:
                        self.error = f"fastembed 不可用: {type(e).__name__}"
            except Exception as e:
                self.error = f"模型加载失败: {type(e).__name__}"
            self.load_ms = int((time.time() - t0) * 1000)
            if self._model is None and self._onnx is None:
                self.backend = "hash"
                self.dim = 1024
                self.model_name = "hash-ngram-1024"

    # ---------- 编码 ----------
    def encode_dense(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        self.ensure_loaded()
        if self._model is None and self._onnx is None:
            return [hash_embed(t, self.dim) for t in texts]
        if self._model is None and self._onnx is not None:
            vecs = list(self._onnx.embed(texts))
            out = []
            for v in vecs:
                arr = [float(x) for x in v]
                n = math.sqrt(sum(x * x for x in arr)) or 1.0
                out.append([x / n for x in arr])
            return out
        vecs = self._model.encode(
            texts,
            batch_size=settings.embed_batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        self.ensure_loaded()
        if self.backend == "bge-m3":
            # BGE-M3 检索查询建议加指令前缀（sentence-transformers 下非必需，保留兼容）
            return self.encode_dense([text])[0]
        return self.encode_dense([text])[0]

    def encode_sparse(self, texts: Iterable[str]) -> list[dict[str, float]]:
        texts = list(texts)
        self.ensure_loaded()
        # BGE-M3 的稀疏权重需要通过 FlagEmbedding 的 BGEM3FlagModel 获取；
        # sentence-transformers 只能给稠密向量，因此稀疏路退到词权重倒排。
        if self._m3 is not None:
            try:
                out = self._m3.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
                weights = out.get("lexical_weights") if isinstance(out, dict) else None
                if weights:
                    return [{str(k): float(v) for k, v in w.items()} for w in weights]
            except Exception:
                pass
        return [lexical_weights(t) for t in texts]

    def status(self) -> dict:
        self.ensure_loaded()
        return {
            "backend": self.backend,
            "model": self.model_name,
            "dim": self.dim,
            "device": self.device,
            "load_ms": self.load_ms,
            "error": self.error,
            "real_model": self._model is not None or self._onnx is not None,
        }


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
