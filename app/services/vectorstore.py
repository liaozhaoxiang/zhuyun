"""向量库：Milvus（milvus-lite 本地文件 / 集群 URI 均可）优先，numpy 内存索引兜底。

集合只存稠密向量与元数据；稀疏路与关键词路走独立倒排索引，三路在 RRF 层融合。
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, settings

_COLLECTION = settings.milvus_collection


class VectorStore:
    def __init__(self, dim: int = 0) -> None:
        self.dim = dim
        self.backend = "numpy"
        self._client = None
        self._lock = threading.Lock()
        self.error = ""
        self._rows: list[dict] = []  # numpy 兜底
        self._mat = None
        self._np_path = DATA_DIR / "vectors.npz"

    # ---------- 初始化 ----------
    def ensure(self, dim: int) -> None:
        if dim == self.dim and (self._client is not None or self.backend == "numpy"):
            if self._client is not None:
                return
        self.dim = dim
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
            if not client.has_collection(_COLLECTION):
                client.create_collection(
                    collection_name=_COLLECTION,
                    dimension=dim,
                    primary_field_name="chunk_id",
                    id_type="string",
                    vector_field_name="dense",
                    metric_type="IP",
                    max_length=160,
                    auto_id=False,
                    enable_dynamic_field=True,
                )
            self._client = client
            self.backend = "milvus"
        except Exception as e:
            self.backend = "numpy"
            self.error = f"{type(e).__name__}: {e}"
            self._load_numpy()

    # ---------- 写入 ----------
    def upsert(self, rows: list[dict]) -> int:
        """rows: [{chunk_id, dense, parent_id, text, std_no, category, clause_no, level, ...}]"""
        if not rows:
            return 0
        if self._client is not None:
            payload = []
            for r in rows:
                payload.append({**r, "dense": [float(x) for x in r["dense"]]})
            self._client.upsert(collection_name=_COLLECTION, data=payload)
            return len(payload)
        # numpy 兜底：内存 + 落盘
        with self._lock:
            ids = {r["chunk_id"]: r for r in self._rows}
            for r in rows:
                ids[r["chunk_id"]] = r
            self._rows = list(ids.values())
            self._rebuild_numpy()
        return len(rows)

    def _rebuild_numpy(self) -> None:
        import numpy as np

        if not self._rows:
            self._mat = None
            return
        self._mat = np.asarray([r["dense"] for r in self._rows], dtype="float32")
        norms = np.linalg.norm(self._mat, axis=1, keepdims=True)
        self._mat = self._mat / np.clip(norms, 1e-9, None)

    def _load_numpy(self) -> None:
        import numpy as np

        if self._np_path.exists():
            try:
                data = np.load(self._np_path, allow_pickle=True)
                self._rows = list(data["rows"])
                if self._rows:
                    self.dim = len(self._rows[0]["dense"])
                self._rebuild_numpy()
            except Exception:
                self._rows = []

    def persist(self) -> None:
        if self._client is None and self._rows:
            import numpy as np

            np.savez_compressed(self._np_path, rows=np.array(self._rows, dtype=object))

    # ---------- 检索 ----------
    def search(self, vector: list[float], top_k: int = 20, category: str | None = None) -> list[dict]:
        if self._client is not None:
            expr = f'category == "{category}"' if category else None
            res = self._client.search(
                collection_name=_COLLECTION,
                data=[[float(x) for x in vector]],
                anns_field="dense",
                limit=top_k,
                output_fields=["parent_id", "text", "std_no", "std_name", "category", "clause_no", "level", "doc_id", "chapter", "page"],
                search_params={"metric_type": "IP"},
                filter=expr or "",
            )
            out = []
            for hits in res:
                for h in hits:
                    ent = h.get("entity", {})
                    out.append(
                        {
                            "chunk_id": h.get("id"),
                            "score": float(h.get("distance", 0.0)),
                            **{k: ent.get(k) for k in ("parent_id", "text", "std_no", "std_name", "category", "clause_no", "level", "doc_id", "chapter", "page")},
                        }
                    )
            return out

        # numpy 兜底
        import numpy as np

        if self._mat is None or not self._rows:
            return []
        q = np.asarray(vector, dtype="float32")
        q = q / max(1e-9, float(np.linalg.norm(q)))
        scores = self._mat @ q
        idx = np.argsort(-scores)
        out = []
        for i in idx:
            r = self._rows[int(i)]
            if category and r.get("category") != category:
                continue
            out.append({"chunk_id": r["chunk_id"], "score": float(scores[int(i)]), **{k: r.get(k) for k in ("parent_id", "text", "std_no", "std_name", "category", "clause_no", "level", "doc_id", "chapter", "page")}})
            if len(out) >= top_k:
                break
        return out

    def count(self) -> int:
        if self._client is not None:
            try:
                st = self._client.get_collection_stats(_COLLECTION)
                return int(st.get("row_count", 0))
            except Exception:
                return 0
        return len(self._rows)

    def reset(self) -> None:
        if self._client is not None:
            try:
                self._client.drop_collection(_COLLECTION)
            except Exception:
                pass
            self._client = None
        self._rows = []
        self._mat = None
        if self._np_path.exists():
            try:
                self._np_path.unlink()
            except BaseException:
                # 沙箱/删除守卫可能拦截，文件留着也会被下一次 persist 覆盖
                pass

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "collection": _COLLECTION,
            "uri": settings.milvus_uri if self.backend == "milvus" else str(self._np_path),
            "dim": self.dim,
            "count": self.count(),
            "error": self.error,
        }


_store: VectorStore | None = None


def get_vector_store(dim: int = 1024) -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(dim)
        _store.ensure(dim)
    return _store
