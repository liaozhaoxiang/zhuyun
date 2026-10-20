"""问答编排：意图 -> 改写 -> 混合召回 -> 回溯 -> 精排 -> 生成。"""
from __future__ import annotations

import time
from typing import Iterator

from app.config import settings
from app.services.cache import get_cache
from app.services.generator import build_citations, generate_stream, is_refusal, llm_available
from app.services.intent import get_intent_classifier
from app.services.retriever import get_retriever
from app.services.rewriter import backtrack_queries, decompose, hyde_text

CATEGORY_HINTS = {
    "基坑工程": ["基坑", "支护", "锚杆", "土钉", "降水", "冠梁", "排桩", "连续墙", "边坡", "开挖"],
    "脚手架工程": ["脚手架", "立杆", "剪刀撑", "连墙件", "扣件", "悬挑", "盘扣", "碗扣", "爬架", "吊篮", "安全网"],
    "模板支架": ["模板", "支架", "高支模", "顶托", "拆模", "铝模", "爬模", "混凝土浇筑"],
    "高处作业": ["高处", "临边", "洞口", "电梯井", "安全带", "安全帽", "攀登", "悬空", "操作平台", "交叉作业"],
    "施工用电": ["用电", "配电", "漏电", "接零", "接地", "电缆", "开关箱", "电焊机", "行灯", "TN-S"],
    "起重机械": ["塔吊", "塔机", "起重机", "升降机", "物料提升机", "吊装", "吊索具", "汽车吊", "履带吊"],
    "文明施工与消防": ["消防", "动火", "灭火器", "噪声", "扬尘", "渣土", "文明", "宿舍", "食堂", "垃圾"],
}


def infer_category(question: str) -> str | None:
    scores = {c: sum(1 for k in kws if k in question) for c, kws in CATEGORY_HINTS.items()}
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else None


class Pipeline:
    def __init__(self) -> None:
        self.retriever = get_retriever()
        self.intent = get_intent_classifier()
        self.cache = get_cache()

    # ---------- 同步问答（评测 / API 非流式） ----------
    def ask(self, question: str, category: str | None = None, use_cache: bool = True, history: list[dict] | None = None) -> dict:
        t0 = time.perf_counter()
        question = (question or "").strip()
        intent_res = self.intent.predict(question)
        intent = intent_res["intent"]
        cat = category or (infer_category(question) if intent != "系统闲聊" else None)

        if use_cache and not history:
            key = self.cache.key(question, cat or "")
            cached = self.cache.get(key)
            if cached:
                cached["cached"] = True
                cached["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                return cached

        if intent == "系统闲聊":
            answer = "我是筑安云建筑施工安全规范助手，可以回答规范条文、隐患判定与整改要求三类问题。"
            return {
                "question": question,
                "intent": intent_res,
                "category": cat,
                "answer": answer,
                "confidence": 0.9,
                "citations": [],
                "parents": [],
                "trace": {"skipped": "smalltalk"},
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "generator": "rule",
            }

        subs = decompose(question)
        merged: dict[str, dict] = {}
        traces = []
        for sub in subs:
            r = self.retriever.retrieve(sub, category=cat)
            # 回溯：分数偏低时换查询重试
            if (r["top_score"] < 0.35 or not r["parents"]) and cat is None:
                for bq in backtrack_queries(sub, category=cat or ""):
                    r2 = self.retriever.retrieve(bq, category=cat)
                    if r2["top_score"] > r["top_score"]:
                        r = r2
                        r["trace"]["backtrack"] = bq
                        break
            traces.append(r["trace"])
            for p in r["parents"]:
                pid = p["parent_id"]
                if pid not in merged or p["best_rerank"] > merged[pid]["best_rerank"]:
                    merged[pid] = p
        parents = sorted(merged.values(), key=lambda x: -x["best_rerank"])[: settings.max_parents]
        top_score = parents[0]["best_rerank"] if parents else 0.0

        # 首轮为空 -> HyDE 再试一次
        if not parents:
            hy = hyde_text(question, intent)
            r = self.retriever.retrieve(hy, category=cat)
            if r["parents"]:
                parents = r["parents"]
                top_score = r["top_score"]
                traces.append({**r["trace"], "hyde": True})

        cites = build_citations(parents)
        md, conf, gen = "", 0.0, "extractive"
        for ev in generate_stream(question, intent, parents, cites, top_score, history):
            if ev["type"] == "done":
                md = ev["data"]["answer"]
                conf = ev["data"]["confidence"]
                gen = ev["data"]["generator"]
            elif ev["type"] == "citations":
                cites = ev["data"]

        refused = is_refusal(question, parents, cites, top_score)
        if refused:
            # 跨界提问：引用与父块一并清空，避免前端展示误导性溯源
            cites, parents = [], []
        low_conf = (not refused) and top_score < settings.min_relevance

        result = {
            "question": question,
            "intent": intent_res,
            "category": cat,
            "sub_queries": subs,
            "answer": md,
            "confidence": conf,
            "citations": cites,
            "parents": parents,
            "refused": refused,
            "low_confidence": low_conf,
            "trace": {
                "paths": traces,
                "top_score": top_score,
                "threshold": settings.min_relevance,
                "refused": refused,
                "low_confidence": low_conf,
                "version": {
                    "embed": self.retriever.embedder.backend,
                    "rerank": self.retriever.reranker.backend,
                    "vector": self.retriever.store.backend,
                    "generator": gen,
                },
            },
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "generator": gen,
            "cached": False,
        }
        if use_cache and not history:
            self.cache.set(self.cache.key(question, cat or ""), {**result, "cached": True})
        return result

    # ---------- 流式（WebSocket） ----------
    def ask_stream(self, question: str, category: str | None = None, history: list[dict] | None = None) -> Iterator[dict]:
        t0 = time.perf_counter()
        question = (question or "").strip()
        yield {"type": "stage", "stage": "intent", "data": {"state": "running", "label": "意图识别"}}
        intent_res = self.intent.predict(question)
        intent = intent_res["intent"]
        yield {"type": "intent", "data": intent_res}

        cat = category or (infer_category(question) if intent != "系统闲聊" else None)
        yield {"type": "stage", "stage": "intent", "data": {"state": "done", "label": "意图识别", "detail": f"{intent} · {intent_res['confidence']:.2f}", "category": cat}}

        if intent == "系统闲聊":
            text = "我是筑安云建筑施工安全规范助手，可以回答规范条文、隐患判定与整改要求三类问题。例如：脚手架连墙件间距是多少？基坑边堆土 2 米算不算重大隐患？"
            for i in range(0, len(text), 16):
                yield {"type": "token", "data": text[i : i + 16]}
            yield {"type": "citations", "data": []}
            yield {"type": "done", "data": {"question": question, "intent": intent_res, "answer": text, "confidence": 0.9, "citations": [], "latency_ms": int((time.perf_counter() - t0) * 1000), "generator": "rule"}}
            return

        yield {"type": "stage", "stage": "rewrite", "data": {"state": "running", "label": "查询改写"}}
        subs = decompose(question)
        yield {"type": "stage", "stage": "rewrite", "data": {"state": "done", "label": "查询改写", "detail": " → ".join(subs)}}

        yield {"type": "stage", "stage": "retrieve", "data": {"state": "running", "label": "混合召回"}}
        merged: dict[str, dict] = {}
        all_traces = []
        for sub in subs:
            r = self.retriever.retrieve(sub, category=cat)
            all_traces.append(r["trace"])
            for p in r["parents"]:
                pid = p["parent_id"]
                if pid not in merged or p["best_rerank"] > merged[pid]["best_rerank"]:
                    merged[pid] = p
        parents = sorted(merged.values(), key=lambda x: -x["best_rerank"])[: settings.max_parents]
        top_score = parents[0]["best_rerank"] if parents else 0.0

        if (top_score < 0.35 or not parents) and cat is None:
            yield {"type": "stage", "stage": "backtrack", "data": {"state": "running", "label": "回溯扩展"}}
            for bq in backtrack_queries(question):
                r2 = self.retriever.retrieve(bq, category=cat)
                if r2["top_score"] > top_score:
                    parents = r2["parents"]
                    top_score = r2["top_score"]
                    all_traces.append({**r2["trace"], "backtrack": bq})
                    break
            yield {"type": "stage", "stage": "backtrack", "data": {"state": "done", "label": "回溯扩展", "detail": f"top={top_score:.3f}"}}

        if not parents:
            yield {"type": "stage", "stage": "hyde", "data": {"state": "running", "label": "HyDE 假设文档"}}
            r = self.retriever.retrieve(hyde_text(question, intent), category=cat)
            parents, top_score = r["parents"], r["top_score"]
            all_traces.append({**r["trace"], "hyde": True})
            yield {"type": "stage", "stage": "hyde", "data": {"state": "done", "label": "HyDE 假设文档"}}

        trace0 = all_traces[0] if all_traces else {}
        yield {
            "type": "stage",
            "stage": "retrieve",
            "data": {
                "state": "done",
                "label": "混合召回",
                "detail": f"稠密 {trace0.get('paths',{}).get('dense',{}).get('count',0)} / 稀疏 {trace0.get('paths',{}).get('sparse',{}).get('count',0)} / BM25 {trace0.get('paths',{}).get('bm25',{}).get('count',0)} → 父块 {len(parents)}",
                "trace": trace0,
            },
        }

        yield {"type": "stage", "stage": "rerank", "data": {"state": "done", "label": "精排", "detail": f"{self.retriever.reranker.model_name} · top={top_score:.3f}"}}
        cites = build_citations(parents)
        yield {"type": "stage", "stage": "generate", "data": {"state": "running", "label": "生成回答"}}

        answer, conf, gen = "", 0.0, "extractive"
        for ev in generate_stream(question, intent, parents, cites, top_score, history):
            if ev["type"] == "token":
                answer += ev["data"]
                yield ev
            elif ev["type"] == "citations":
                cites = ev["data"]
                yield {"type": "citations", "data": cites}
            elif ev["type"] == "note":
                yield ev
            elif ev["type"] == "done":
                conf = ev["data"]["confidence"]
                gen = ev["data"]["generator"]
                answer = ev["data"]["answer"]

        refused = is_refusal(question, parents, cites, top_score)
        if refused:
            cites, parents = [], []
        low_conf = (not refused) and top_score < settings.min_relevance
        latency = int((time.perf_counter() - t0) * 1000)
        if refused:
            detail = f"未命中知识边界，拒答 · {latency}ms"
        elif low_conf:
            detail = f"低置信度作答（{top_score:.2f} < {settings.min_relevance}） · {latency}ms"
        else:
            detail = f"{gen} · {latency}ms"
        yield {"type": "stage", "stage": "generate", "data": {"state": "done", "label": "生成回答", "detail": detail}}
        yield {
            "type": "done",
            "data": {
                "question": question,
                "intent": intent_res,
                "category": cat,
                "sub_queries": subs,
                "answer": answer,
                "confidence": conf,
                "citations": cites,
                "parents": parents,
                "refused": refused,
                "low_confidence": low_conf,
                "trace": {"paths": all_traces, "top_score": top_score, "threshold": settings.min_relevance, "refused": refused, "low_confidence": low_conf, "version": {"embed": self.retriever.embedder.backend, "rerank": self.retriever.reranker.backend, "vector": self.retriever.store.backend, "generator": gen}},
                "latency_ms": latency,
                "generator": gen,
            },
        }


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
