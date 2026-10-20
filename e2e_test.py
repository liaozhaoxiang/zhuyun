"""端到端验证：真实 HTTP 请求，核对答案与引用条文。"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.request

BASE = "http://127.0.0.1:8011"
OUT = pathlib.Path(__file__).resolve().parent / "data" / "_e2e.json"

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CASES = [
    ("脚手架连墙件间距要求是多少", ["JGJ1302011-6.4.2"], "规范条文查询"),
    ("基坑边堆土 2 米算不算重大隐患", ["JGJ3112013-5.2.3"], "隐患判定"),
    ("配电箱一闸多机怎么整改", ["JGJ462005-8.1.3"], "整改要求"),
    ("安全带应该怎么挂", ["JGJ802016-8.1.1"], "规范条文查询"),
    ("塔吊力矩限制器被短接了严重吗", ["JGJ332012-5.1.7"], "隐患判定"),
    ("基坑监测频率是多少", ["JGJ1202012-8.2.2"], "规范条文查询"),
    ("电梯井口防护门要多高", ["JGJ802016-4.2.3"], "规范条文查询"),
    ("模板支架顶托悬臂长度限制", ["JGJ1622008-6.1.9"], "规范条文查询"),
    ("危大工程专项方案谁编制", ["住建部令第37号-第十条"], "规范条文查询"),
    ("这个隐患整改期限多久", ["QZAJT202401-5.1.3"], "整改要求"),
    ("你好", [], "系统闲聊"),
    ("基坑出现管涌还在挖土", ["JGJ3112013-7.1.2"], "隐患判定"),
]


def ask(q: str, category: str | None = None):
    body = json.dumps({"question": q, "category": category, "use_cache": False}).encode()
    req = urllib.request.Request(BASE + "/api/qa", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with OPENER.open(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    return d, int((time.perf_counter() - t0) * 1000)


def main():
    health = json.loads(OPENER.open(BASE + "/api/health", timeout=20).read().decode())
    rows = []
    for q, gold, expect_intent in CASES:
        try:
            d, ms = ask(q)
        except Exception as e:
            rows.append({"q": q, "error": f"{type(e).__name__}: {e}"})
            continue
        cites = d.get("citations") or []
        ids = [c["clause_id"] for c in cites]
        hit = any(any(i.split("#")[0] == g for g in gold) for i in ids)
        rows.append(
            {
                "q": q,
                "intent": (d.get("intent") or {}).get("intent"),
                "intent_ok": (d.get("intent") or {}).get("intent") == expect_intent,
                "answer_len": len(d.get("answer") or ""),
                "answer_head": (d.get("answer") or "")[:120],
                "cites": len(cites),
                "cite_ids": ids[:5],
                "gold_hit": hit if gold else None,
                "confidence": d.get("confidence"),
                "latency_ms": ms,
            }
        )

    ok = [r for r in rows if "error" not in r]
    summary = {
        "health_index_state": (health.get("index") or {}).get("state"),
        "chunks": (health.get("index") or {}).get("chunks"),
        "embed": ((health.get("index") or {}).get("embedder") or {}).get("backend"),
        "vector": ((health.get("index") or {}).get("vector") or {}).get("backend"),
        "rerank": ((health.get("index") or {}).get("reranker") or {}).get("backend"),
        "cases": len(rows),
        "ok": len(ok),
        "intent_acc": round(sum(1 for r in ok if r["intent_ok"]) / max(1, len(ok)), 3),
        "gold_hit_rate": round(sum(1 for r in ok if r["gold_hit"]) / max(1, sum(1 for r in ok if r["gold_hit"] is not None)), 3),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in ok) / max(1, len(ok))),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
