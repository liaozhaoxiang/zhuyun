"""FastAPI 应用：REST + WebSocket 流式问答 + 知识库 + 评测。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import WEB_DIR, settings
from app.services.cache import get_cache
from app.services.corpus import get_corpus
from app.services.evaluator import get_evaluator
from app.services.ingest import ingest_document
from app.services.pipeline import get_pipeline
from app.services.retriever import get_retriever
from app.services.store import get_store

_started_at = time.time()
_build_state = {"state": "idle", "progress": 0, "message": "", "error": ""}


def _build_index_async() -> None:
    _build_state.update(state="building", progress=5, message="初始化")
    try:
        r = get_retriever()
        info = r.build()
        _build_state.update(state="ready", progress=100, message=f"索引就绪：{info['chunks']} 子块 / {info['build_ms']} ms", error="")
    except Exception as e:
        _build_state.update(state="error", progress=0, message="索引构建失败", error=f"{type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_build_index_async, name="kb-build", daemon=True).start()
    yield


app = FastAPI(title=f"{settings.app_name} API", version=settings.version, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------- 基础 ----------------
@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/health")
def health():
    retr = get_retriever()
    corpus = get_corpus()
    return {
        "app": settings.app_name,
        "version": settings.version,
        "uptime_s": int(time.time() - _started_at),
        "index": {**retr.status(), **_build_state},
        "store": get_store().stats(),
        "cache": get_cache().status(),
        "corpus": corpus.stats(),
        "llm": {"configured": bool(settings.llm_base_url and settings.llm_api_key), "model": settings.llm_model},
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    retr = get_retriever()
    store = get_store().stats()
    cache = get_cache().status()
    lines = [
        "# HELP zhuan_uptime_seconds 服务运行时长",
        "# TYPE zhuan_uptime_seconds gauge",
        f"zhuan_uptime_seconds {int(time.time() - _started_at)}",
        "# HELP zhuan_index_chunks 索引子块数",
        "# TYPE zhuan_index_chunks gauge",
        f"zhuan_index_chunks {retr.status()['chunks']}",
        "# HELP zhuan_qa_total 问答消息数",
        "# TYPE zhuan_qa_total counter",
        f"zhuan_qa_total {store['messages']}",
        "# HELP zhuan_cache_hit_rate 缓存命中率",
        "# TYPE zhuan_cache_hit_rate gauge",
        f"zhuan_cache_hit_rate {cache['hit_rate']}",
    ]
    return "\n".join(lines) + "\n"


# ---------------- 知识库 ----------------
@app.get("/api/corpus/stats")
def corpus_stats():
    corpus = get_corpus()
    return {"stats": corpus.stats(), "categories": corpus.category_summary()}


@app.get("/api/corpus/docs")
def corpus_docs(category: str | None = None, q: str | None = None):
    corpus = get_corpus()
    docs = corpus.docs
    if category:
        docs = [d for d in docs if d["category"] == category]
    if q:
        ql = q.lower()
        docs = [d for d in docs if ql in d["std_name"].lower() or ql in d["std_no"].lower() or any(ql in a.lower() for a in d.get("alias", []))]
    return {"total": len(docs), "items": docs}


@app.get("/api/corpus/docs/{doc_id}")
def corpus_doc(doc_id: str):
    d = get_corpus().doc_detail(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    return d


@app.get("/api/corpus/search")
def corpus_search(q: str = Query(..., min_length=1), limit: int = 20):
    r = get_retriever()
    res = r.retrieve(q, top_k=limit)
    return {"query": q, "hit_count": len(res["hits"]), "hits": res["hits"][:limit], "trace": res["trace"]}


class UploadResp(BaseModel):
    doc_id: str
    title: str
    category: str
    chars: int
    chunks: int
    ocr_used: bool
    elapsed_ms: int


@app.post("/api/kb/upload", response_model=UploadResp)
async def kb_upload(file: UploadFile = File(...), category: str = Form("未分类")):
    raw = await file.read()
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(413, "文件超过 30MB")
    try:
        info = ingest_document(file.filename or "upload.txt", raw, category)
    except Exception as e:
        raise HTTPException(400, f"解析失败：{type(e).__name__}: {e}")
    return info


# ---------------- 问答 ----------------
class AskReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    category: str | None = None
    session_id: str | None = None
    use_cache: bool = True


@app.post("/api/qa")
def qa(req: AskReq):
    pipeline = get_pipeline()
    store = get_store()
    history = None
    if req.session_id:
        history = [{"role": m["role"], "content": m["content"]} for m in store.list_messages(req.session_id)][-6:]
    t0 = time.perf_counter()
    res = pipeline.ask(req.question, category=req.category, use_cache=req.use_cache, history=history)
    latency = int((time.perf_counter() - t0) * 1000)

    sid = req.session_id
    if sid and store.get_session(sid):
        pass
    else:
        title = req.question[:24] + ("…" if len(req.question) > 24 else "")
        sid = store.create_session(title, res.get("category") or "")["id"]
    store.add_message(sid, "user", req.question)
    store.add_message(
        sid,
        "assistant",
        res["answer"],
        intent=res["intent"]["intent"],
        confidence=res["confidence"],
        citations=res["citations"],
        trace=res.get("trace", {}),
        latency_ms=latency,
    )
    store.touch_session(sid)
    return {**res, "session_id": sid, "latency_ms": latency}


@app.websocket("/ws/qa")
async def ws_qa(ws: WebSocket):
    await ws.accept()
    pipeline = get_pipeline()
    store = get_store()
    try:
        while True:
            payload = await ws.receive_text()
            try:
                req = json.loads(payload)
            except Exception:
                await ws.send_json({"type": "error", "data": "消息格式错误"})
                continue
            question = (req.get("question") or "").strip()
            if not question:
                await ws.send_json({"type": "error", "data": "问题不能为空"})
                continue
            category = req.get("category") or None
            sid = req.get("session_id")
            history = None
            if sid and store.get_session(sid):
                history = [{"role": m["role"], "content": m["content"]} for m in store.list_messages(sid)][-6:]

            await ws.send_json({"type": "start", "data": {"question": question, "session_id": sid}})
            t0 = time.perf_counter()
            final = None
            answer = ""
            intent_label = ""
            conf = 0.0
            cites: list = []
            trace: dict = {}
            try:
                for ev in pipeline.ask_stream(question, category=category, history=history):
                    if ev["type"] == "done":
                        final = ev["data"]
                        answer = final.get("answer", answer)
                        intent_label = (final.get("intent") or {}).get("intent", "")
                        conf = final.get("confidence", 0)
                        cites = final.get("citations", [])
                        trace = final.get("trace", {})
                    elif ev["type"] == "intent":
                        intent_label = (ev.get("data") or {}).get("intent", intent_label)
                    await ws.send_json(ev)
                    await asyncio.sleep(0)
            except Exception as e:
                await ws.send_json({"type": "error", "data": f"生成失败：{type(e).__name__}: {e}"})
                continue

            latency = int((time.perf_counter() - t0) * 1000)
            if sid and store.get_session(sid):
                pass
            else:
                title = question[:24] + ("…" if len(question) > 24 else "")
                sid = store.create_session(title, (final or {}).get("category") or "")["id"]
                await ws.send_json({"type": "session", "data": {"session_id": sid, "title": title}})
            store.add_message(sid, "user", question)
            msg_id = store.add_message(sid, "assistant", answer, intent=intent_label, confidence=conf, citations=cites, trace=trace, latency_ms=latency)
            store.touch_session(sid)
            await ws.send_json({"type": "saved", "data": {"session_id": sid, "message_id": msg_id, "latency_ms": latency}})
    except WebSocketDisconnect:
        return
    except Exception:
        return


# ---------------- 会话 ----------------
@app.get("/api/sessions")
def list_sessions(limit: int = 50):
    return {"items": get_store().list_sessions(limit)}


class NewSession(BaseModel):
    title: str = "新会话"
    category: str = ""


@app.post("/api/sessions")
def new_session(body: NewSession):
    return get_store().create_session(body.title, body.category)


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    store = get_store()
    s = store.get_session(sid)
    if not s:
        raise HTTPException(404, "会话不存在")
    return {**s, "messages": store.list_messages(sid)}


@app.delete("/api/sessions/{sid}")
def del_session(sid: str):
    get_store().delete_session(sid)
    return {"ok": True}


class RenameReq(BaseModel):
    title: str


@app.patch("/api/sessions/{sid}")
def rename_session(sid: str, body: RenameReq):
    get_store().rename_session(sid, body.title)
    return {"ok": True}


class FeedbackReq(BaseModel):
    message_id: int
    value: int = Field(..., ge=-1, le=1)


@app.post("/api/feedback")
def feedback(body: FeedbackReq):
    get_store().set_feedback(body.message_id, body.value)
    return {"ok": True}


# ---------------- 评测 ----------------
@app.get("/api/eval/latest")
def eval_latest():
    ev = get_evaluator()
    data = ev.latest()
    if not data:
        return {"available": False, "hint": "尚未运行评测"}
    return {"available": True, **data}


class EvalReq(BaseModel):
    limit: int = 40
    use_llm_judge: bool = False


@app.post("/api/eval/run")
def eval_run(body: EvalReq):
    ev = get_evaluator()
    if ev.running:
        return {"ok": False, "message": "评测进行中"}
    threading.Thread(target=ev.run, kwargs={"limit": body.limit, "use_llm_judge": body.use_llm_judge}, daemon=True).start()
    return {"ok": True, "message": f"评测已启动，样本 {body.limit} 条"}


@app.get("/api/eval/progress")
def eval_progress():
    return get_evaluator().progress()


@app.get("/api/system")
def system():
    retr = get_retriever()
    return {
        "index": retr.status(),
        "build": _build_state,
        "cache": get_cache().status(),
        "store": get_store().stats(),
        "settings": {
            "top_k_dense": settings.top_k_dense,
            "top_k_sparse": settings.top_k_sparse,
            "rrf_k": settings.rrf_k,
            "rerank_top_n": settings.rerank_top_n,
            "min_relevance": settings.min_relevance,
            "max_parents": settings.max_parents,
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
        },
    }
