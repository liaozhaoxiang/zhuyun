"""精排：BGE-Reranker-Large（CrossEncoder）优先，不可用时退到词面覆盖重排。"""
from __future__ import annotations

import math
import re
import threading
import time
from pathlib import Path

from app.config import settings
from app.services.lexical import tokenize


class Reranker:
    def __init__(self) -> None:
        self._model = None
        self._is_onnx = False
        self.backend = "rule"
        self.model_name = "lexical-coverage"
        self.error = ""
        self.load_ms = 0
        self._tried = False
        self._lock = threading.Lock()

    def ensure_loaded(self) -> None:
        if self._tried:
            return
        with self._lock:
            if self._tried:
                return
            self._tried = True
            t0 = time.time()
            try:
                from sentence_transformers import CrossEncoder

                for name, fb in ((settings.rerank_model, False), (settings.rerank_fallback, True)):
                    try:
                        self._model = CrossEncoder(name, max_length=512, device=settings.embed_device)
                        self.backend = "cross-encoder-fallback" if fb else "cross-encoder"
                        self.model_name = name
                        break
                    except Exception as e:
                        self.error = f"{name}: {type(e).__name__}"
                        continue
            except Exception as e:
                self.error = f"sentence-transformers 不可用: {type(e).__name__}"

            # ONNX 交叉编码器（fastembed，无需 torch）：本地模型目录优先
            if self._model is None:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    onnx_dir = Path(settings.rerank_onnx_path)
                    candidates = []
                    if onnx_dir.exists():
                        candidates.append((settings.rerank_onnx_model, str(onnx_dir)))
                    if settings.auto_download:
                        candidates.append((settings.rerank_onnx_model, None))
                    for name, local_path in candidates:
                        try:
                            kwargs = {"model_name": name}
                            if local_path:
                                kwargs["specific_model_path"] = local_path
                            self._model = TextCrossEncoder(**kwargs)
                            self.backend = "cross-encoder-onnx"
                            self.model_name = name + ("(local)" if local_path else "")
                            self._is_onnx = True
                            break
                        except Exception as e:
                            self.error = f"{name}: {type(e).__name__}"
                            self._model = None
                            continue
                except Exception as e:
                    self.error = f"fastembed 精排不可用: {type(e).__name__}"

            self.load_ms = int((time.time() - t0) * 1000)
            if self._model is None:
                self.backend = "rule"
                self.model_name = "lexical-coverage"

    def score(self, query: str, docs: list[str]) -> list[float]:
        self.ensure_loaded()
        if self._model is not None and docs:
            if self._is_onnx:
                # fastembed 交叉编码器输出原始 logit，转 sigmoid 成 0~1 相关度
                raw = list(self._model.rerank(query, docs))
                return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(x))))) for x in raw]
            raw = self._model.predict([(query, d) for d in docs])
            return [1.0 / (1.0 + math.exp(-float(x))) for x in raw]
        return [self._rule_score(query, d) for d in docs]

    @staticmethod
    def _rule_score(query: str, doc: str) -> float:
        q = set(tokenize(query, expand=True))
        d = set(tokenize(doc))
        if not q:
            return 0.0
        overlap = len(q & d) / len(q)
        # 数字、规范编号命中加成
        bonus = 0.0
        for m in re.findall(r"[0-9]+(?:\.[0-9]+)?", query):
            if m in doc:
                bonus += 0.04
        return min(1.0, 0.15 + 0.8 * overlap + bonus)

    def status(self) -> dict:
        self.ensure_loaded()
        return {
            "backend": self.backend,
            "model": self.model_name,
            "load_ms": self.load_ms,
            "real_model": self._model is not None,
            "error": self.error,
        }


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
