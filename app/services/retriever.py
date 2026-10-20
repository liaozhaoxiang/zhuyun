"""混合检索：稠密 + 稀疏 + BM25 三路召回，RRF 融合，父块回溯去重，CrossEncoder 精排。"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.chunker import Chunk, build_child_chunks, build_parent_text
from app.services.corpus import get_corpus
from app.services.embedder import get_embedder
from app.services.lexical import BM25Index, SparseIndex
from app.services.reranker import get_reranker
from app.services.vectorstore import get_vector_store


@dataclass
class Hit:
    chunk_id: str
    parent_id: str
    text: str
    meta: dict
    scores: dict = field(default_factory=dict)
    ranks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            **self.meta,
            **{f"{k}_score": round(v, 4) for k, v in self.scores.items()},
            **{f"{k}_rank": v for k, v in self.ranks.items()},
        }


class Retriever:
    def __init__(self) -> None:
        self.corpus = get_corpus()
        self.embedder = get_embedder()
        self.store = get_vector_store()
        self.bm25 = BM25Index()
        self.sparse = SparseIndex()
        self.reranker = get_reranker()
        self.chunks: list[Chunk] = []
        self.cat_map: dict[str, str] = {}
        self.ready = False
        self.build_ms = 0
        self.state = "idle"
        self.progress = 0
        self.message = ""

    # ---------------- 建索引 ----------------
    def build(self) -> dict:
        t0 = time.time()
        self.state = "building"
        self.message = "加载语料"
        corpus = self.corpus
        self.chunks = build_child_chunks(corpus)
        self.progress = 15
        self.message = f"分块完成：{len(self.chunks)} 子块 / {len(corpus.parents)} 父块"

        texts = [c.text for c in self.chunks]
        dense = self.embedder.encode_dense(texts)
        self.progress = 55
        self.message = f"向量化完成：{self.embedder.backend} dim={len(dense[0]) if dense else 0}"

        store = get_vector_store(len(dense[0]) if dense else 1024)
        dim = len(dense[0]) if dense else 1024
        store.ensure(dim)
        # 换了向量模型（维度变化）或条数对不上时，旧向量无意义，清库重建
        if store.dim != dim:
            store.reset()
            store.ensure(dim)
        elif store.count() and store.count() != len(dense):
            store.reset()
        store.ensure(dim)
        rows = []
        for c, v in zip(self.chunks, dense):
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "dense": v,
                    "parent_id": c.parent_id,
                    "text": c.text,
                    **{k: c.meta.get(k) for k in ("doc_id", "std_no", "std_name", "category", "chapter", "clause_no", "level", "page")},
                }
            )
        store.upsert(rows)
        store.persist()
        self.progress = 80
        self.message = "写入 Milvus"

        # 稀疏
        sparse_w = self.embedder.encode_sparse(texts)
        self.sparse.build([(c.chunk_id, c.text, w, c.meta.get("category", "")) for c, w in zip(self.chunks, sparse_w)])
        self.bm25.build([(c.chunk_id, c.text) for c in self.chunks])
        self.cat_map = {c.chunk_id: c.meta.get("category", "") for c in self.chunks}
        self.ready = True
        self.state = "ready"
        self.progress = 100
        self.build_ms = int((time.time() - t0) * 1000)
        self.message = f"索引就绪，耗时 {self.build_ms} ms"
        return {
            "chunks": len(self.chunks),
            "parents": len(self.corpus.parents),
            "build_ms": self.build_ms,
            "embed_backend": self.embedder.backend,
            "vector_backend": store.backend,
        }

    # ---------------- 检索 ----------------
    def _ensure(self) -> None:
        if not self.ready:
            self.build()

    @staticmethod
    def _rrf(rank_lists: dict[str, list[str]], k: int = 60) -> dict[str, float]:
        score: dict[str, float] = {}
        for _, ids in rank_lists.items():
            for r, cid in enumerate(ids, start=1):
                score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
        return score

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int | None = None,
        rerank_n: int | None = None,
        use_sparse: bool = True,
    ) -> dict:
        self._ensure()
        top_k = top_k or settings.top_k_dense
        rerank_n = rerank_n or settings.rerank_top_n
        trace: dict[str, Any] = {"query": query, "category": category, "paths": {}}

        # 1) 稠密
        t = time.perf_counter()
        qv = self.embedder.encode_query(query)
        dense_hits = self.store.search(qv, top_k=top_k, category=category)
        trace["paths"]["dense"] = {"ms": round((time.perf_counter() - t) * 1000, 1), "count": len(dense_hits), "top_score": round(dense_hits[0]["score"], 4) if dense_hits else 0}

        # 2) 稀疏
        sparse_hits = []
        if use_sparse:
            t = time.perf_counter()
            sw = self.embedder.encode_sparse([query])[0]
            sparse_hits = self.sparse.search(sw, top_k=top_k, category=category)
            trace["paths"]["sparse"] = {"ms": round((time.perf_counter() - t) * 1000, 1), "count": len(sparse_hits), "top_score": round(sparse_hits[0][1], 4) if sparse_hits else 0}

        # 3) BM25
        t = time.perf_counter()
        bm_hits = self.bm25.search(query, top_k=top_k, category=category, cat_map=self.cat_map)
        trace["paths"]["bm25"] = {"ms": round((time.perf_counter() - t) * 1000, 1), "count": len(bm_hits), "top_score": round(bm_hits[0][1], 4) if bm_hits else 0}

        # 4) RRF
        rank_lists = {
            "dense": [h["chunk_id"] for h in dense_hits],
            "sparse": [cid for cid, _ in sparse_hits],
            "bm25": [cid for cid, _ in bm_hits],
        }
        fused = self._rrf(rank_lists, settings.rrf_k)
        if not fused:
            return {"hits": [], "parents": [], "trace": {**trace, "empty": True}, "top_score": 0.0}

        # 5) 组装候选
        chunk_map = {c.chunk_id: c for c in self.chunks}
        dense_score = {h["chunk_id"]: h["score"] for h in dense_hits}
        sparse_score = {cid: s for cid, s in sparse_hits}
        bm_score = {cid: s for cid, s in bm_hits}
        rank_dense = {h["chunk_id"]: i + 1 for i, h in enumerate(dense_hits)}
        rank_sparse = {cid: i + 1 for i, (cid, _) in enumerate(sparse_hits)}
        rank_bm = {cid: i + 1 for i, (cid, _) in enumerate(bm_hits)}

        cand = sorted(fused.items(), key=lambda x: -x[1])[: max(rerank_n * 2, 16)]
        cands: list[Hit] = []
        for cid, rrf in cand:
            c = chunk_map.get(cid)
            if not c:
                continue
            cands.append(
                Hit(
                    chunk_id=cid,
                    parent_id=c.parent_id,
                    text=c.text,
                    meta=c.meta,
                    scores={
                        "rrf": rrf,
                        "dense": dense_score.get(cid, 0.0),
                        "sparse": sparse_score.get(cid, 0.0),
                        "bm25": bm_score.get(cid, 0.0),
                    },
                    ranks={
                        "dense": rank_dense.get(cid, 0),
                        "sparse": rank_sparse.get(cid, 0),
                        "bm25": rank_bm.get(cid, 0),
                    },
                )
            )

        # 6) 精排
        t = time.perf_counter()
        scores = self.reranker.score(query, [h.text for h in cands])
        for h, s in zip(cands, scores):
            h.scores["rerank"] = float(s)
        if self.reranker.backend == "rule":
            # 词面覆盖式精排容易出现多条同分（并列后顺序退化），与稠密相似度融合后再排序
            for h in cands:
                h.scores["rerank"] = round(0.45 * h.scores["rerank"] + 0.55 * max(0.0, h.scores.get("dense", 0.0)), 4)
        trace["paths"]["rerank"] = {"ms": round((time.perf_counter() - t) * 1000, 1), "count": len(cands), "backend": self.reranker.backend}

        cands.sort(key=lambda h: (-h.scores.get("rerank", 0.0), -h.scores.get("dense", 0.0), -h.scores.get("rrf", 0.0)))
        top_score = cands[0].scores["rerank"] if cands else 0.0
        kept = [h for h in cands if h.scores.get("rerank", 0) >= settings.min_relevance][:rerank_n]
        if not kept and cands:
            kept = cands[: min(3, len(cands))]
            trace["below_threshold"] = True

        # 7) 父块回溯去重：同一父块只保留最高分子块
        parents: "OrderedDict[str, dict]" = OrderedDict()
        for h in kept:
            pid = h.parent_id
            if pid not in parents:
                p = self.corpus.parents.get(pid, {})
                parents[pid] = {
                    "parent_id": pid,
                    "doc_id": p.get("doc_id", h.meta.get("doc_id")),
                    "std_no": p.get("std_no", h.meta.get("std_no")),
                    "std_name": p.get("std_name", h.meta.get("std_name")),
                    "category": p.get("category", h.meta.get("category")),
                    "chapter": p.get("chapter", h.meta.get("chapter")),
                    "text": p.get("text", h.text),
                    "clause_nos": p.get("clause_nos", []),
                    "best_rerank": h.scores.get("rerank", 0.0),
                    "hits": [],
                }
            if len(parents) > settings.max_parents and pid not in parents:
                continue
            parents[pid]["hits"].append(h.to_dict())
        parents = OrderedDict(list(parents.items())[: settings.max_parents])

        trace["fused_candidates"] = len(fused)
        trace["kept"] = len(kept)
        trace["parents"] = len(parents)
        trace["version"] = {"embed": self.embedder.backend, "rerank": self.reranker.backend, "vector": self.store.backend}
        return {
            "hits": [h.to_dict() for h in kept],
            "parents": list(parents.values()),
            "trace": trace,
            "top_score": round(float(top_score), 4),
        }

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "build_ms": self.build_ms,
            "chunks": len(self.chunks),
            "parents": len(self.corpus.parents),
            "bm25": self.bm25.status(),
            "sparse": self.sparse.status(),
            "vector": self.store.status(),
            "embedder": self.embedder.status(),
            "reranker": self.reranker.status(),
        }

    # ---------------- 增量接入（上传文档） ----------------
    def add_chunks(self, children: list[Chunk], parents: list[dict]) -> int:
        self._ensure()
        if not children:
            return 0
        for p in parents:
            self.corpus.parents.setdefault(p["parent_id"], {
                "parent_id": p["parent_id"],
                "doc_id": p.get("doc_id"),
                "std_no": p.get("std_no", "上传文档"),
                "std_name": p.get("title", "上传文档"),
                "category": p.get("category", "未分类"),
                "chapter": p.get("title", ""),
                "text": p.get("text", ""),
                "clause_nos": [],
            })
        dense = self.embedder.encode_dense([c.text for c in children])
        store = get_vector_store(len(dense[0]) if dense else 1024)
        store.ensure(len(dense[0]) if dense else 1024)
        rows = []
        for c, v in zip(children, dense):
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "dense": v,
                    "parent_id": c.parent_id,
                    "text": c.text,
                    **{k: c.meta.get(k) for k in ("doc_id", "std_no", "std_name", "category", "chapter", "clause_no", "level", "page")},
                }
            )
        store.upsert(rows)
        store.persist()

        self.chunks.extend(children)
        texts = [c.text for c in self.chunks]
        sw = self.embedder.encode_sparse(texts)
        self.sparse.build([(c.chunk_id, c.text, w, c.meta.get("category", "")) for c, w in zip(self.chunks, sw)])
        self.bm25.build([(c.chunk_id, c.text) for c in self.chunks])
        self.cat_map = {c.chunk_id: c.meta.get("category", "") for c in self.chunks}
        return len(children)


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
