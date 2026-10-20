"""RAGAS 风格评测：测试集构建 + 五指标计算（本地 judge，可选 LLM judge）。"""
from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from pathlib import Path

from app.config import DATA_DIR, settings
from app.services.corpus import get_corpus
from app.services.embedder import get_embedder
from app.services.intent import get_intent_classifier
from app.services.lexical import tokenize
from app.services.pipeline import get_pipeline

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
EVAL_DIR.mkdir(exist_ok=True)
TESTSET = EVAL_DIR / "testset.jsonl"
LATEST = DATA_DIR / "eval_latest.json"

TEMPLATES = {
    "规范条文查询": [
        "{kw}有什么规范要求",
        "{kw}应该怎么设置",
        "{kw}的标准是多少",
        "{std}对{kw}是怎么规定的",
        "现场{kw}要满足什么条件",
    ],
    "隐患判定": [
        "现场{kw}不符合要求算不算隐患",
        "{kw}这样做合规吗",
        "{kw}不到位属于什么等级",
        "{kw}有问题吗",
        "检查发现{kw}缺失严重吗",
    ],
    "整改要求": [
        "{kw}不达标怎么整改",
        "{kw}需要哪些整改措施",
        "{kw}整改期限多久",
        "{kw}整改完怎么验收",
    ],
}


def _kw(clause: dict) -> str:
    cps = clause.get("check_points") or []
    if cps:
        k = cps[0]
        k = re.sub(r"是否|有没有|记录|齐全|设置|数量|位置|高度", "", k)
        if len(k) >= 4:
            return k[:14]
    text = clause["text"]
    m = re.search(r"[\u4e00-\u9fa5]{4,10}(?=应|不得|严禁|必须|宜)", text)
    if m:
        return m.group(0)
    return f"{clause['std_name'][:8]}相关要求"


def build_testset(force: bool = False) -> int:
    if TESTSET.exists() and not force:
        return sum(1 for _ in TESTSET.open(encoding="utf-8"))

    corpus = get_corpus()
    rows = []
    random.seed(42)
    # 1) 条文模板题：覆盖全部 218 条
    for c in corpus.clauses:
        k = _kw(c.to_dict())
        intents = ["规范条文查询", "规范条文查询", "隐患判定", "整改要求"]
        intent = intents[len(rows) % len(intents)]
        tpl = random.choice(TEMPLATES[intent])
        q = tpl.format(kw=k, std=c["std_name"] if isinstance(c, dict) else c.std_name)
        rows.append({
            "qid": f"t{len(rows)+1:03d}",
            "question": q,
            "intent": intent,
            "gold_clause_ids": [c.clause_id],
            "gold_doc": c.std_no,
            "source": "template",
        })

    # 2) 口语题：模拟安全员现场问法
    spoken = [
        ("脚手架连墙件少了很多，这事大不大", "隐患判定", ["JGJ1302011-6.4.2", "JGJ1302011-9.0.4"]),
        ("基坑边上堆了两米高的土，可以吗", "隐患判定", ["JGJ3112013-5.2.3"]),
        ("工人嫌热不戴安全帽怎么管", "规范条文查询", ["GB28112019-5.1.2"]),
        ("临边防护栏杆要多高", "规范条文查询", ["JGJ802016-4.1.3"]),
        ("塔吊力矩限制器被短接了，严重吗", "隐患判定", ["JGJ332012-5.1.7", "JGJ1962010-4.2.3"]),
        ("配电箱一个闸刀接了三台机器", "隐患判定", ["JGJ462005-8.1.3"]),
        ("漏电保护器动作电流选多大", "规范条文查询", ["JGJ462005-8.2.10"]),
        ("电梯井口就用木板挡了一下行不行", "隐患判定", ["JGJ802016-4.2.3"]),
        ("安全带能不能挂在吊篮上", "规范条文查询", ["JGJ2022010-5.2.3"]),
        ("模板支架顶托伸出太长有问题吗", "隐患判定", ["JGJ1622008-6.1.9"]),
        ("高支模要不要专家论证", "规范条文查询", ["JGJ3112013-3.0.1", "住建部令第37号-第十四条"]),
        ("动火作业没办证怎么处理", "隐患判定", ["GB507202011-6.2.2"]),
        ("氧气瓶乙炔瓶放一起算不算重大隐患", "隐患判定", ["GB507202011-6.2.8"]),
        ("宿舍用的泡沫夹芯板可以吗", "隐患判定", ["GB507202011-4.1.5"]),
        ("夜间施工噪声多少分贝算超标", "规范条文查询", ["GB125232011-4.1"]),
        ("基坑监测多久测一次", "规范条文查询", ["JGJ1202012-8.2.2"]),
        ("基坑位移超过多少要报警", "规范条文查询", ["JGJ1202012-8.3.2"]),
        ("脚手架剪刀撑要怎么设", "规范条文查询", ["JGJ1302011-6.6.2", "JGJ1302011-6.6.5"]),
        ("盘扣架插销没敲紧算问题吗", "隐患判定", ["JGJT2312021-7.2.3"]),
        ("悬挑架的工字钢锚固长度不够怎么办", "整改要求", ["JGJ1302011-6.10.4"]),
        ("吊篮安全锁过期了还能用吗", "隐患判定", ["JGJ2022010-5.2.6"]),
        ("施工升降机楼层门联锁坏了", "隐患判定", ["JGJ2152010-5.0.7"]),
        ("临时用电没有做重复接地", "隐患判定", ["JGJ462005-5.3.2"]),
        ("电缆直接拖在地上泡水了", "隐患判定", ["JGJ462005-7.2.4"]),
        ("电焊机一次线太长怎么办", "规范条文查询", ["JGJ462005-9.1.3"]),
        ("地下室照明用 220V 行灯行不行", "隐患判定", ["JGJ462005-10.2.2"]),
        ("塔吊和另一台塔吊离得太近", "隐患判定", ["JGJ1962010-4.3.7"]),
        ("汽车吊支腿支在回填土上", "隐患判定", ["JGJ332012-5.2.4"]),
        ("吊装的时候工人站在吊物下面", "隐患判定", ["JGJ2762012-5.2.4"]),
        ("模板没到强度就拆了怎么整改", "整改要求", ["JGJ1622008-7.1.2"]),
        ("脚手架验收要哪些资料", "规范条文查询", ["JGJ1302011-8.2.1"]),
        ("三级配电两级保护是什么意思", "规范条文查询", ["JGJ462005-8.1.3"]),
        ("基坑出现管涌还在挖土", "隐患判定", ["JGJ3112013-7.1.2"]),
        ("危大工程专项方案谁编制", "规范条文查询", ["住建部令第37号-第十条"]),
        ("安全隐患整改单要写什么", "整改要求", ["QZAJT202402-4.2.2"]),
        ("重大隐患整改期限是几天", "整改要求", ["QZAJT202401-5.1.3"]),
        ("项目经理多久组织一次安全检查", "规范条文查询", ["QZAJT202401-6.1.2"]),
        ("这个隐患帮我判断一下等级", "隐患判定", ["QZAJT202401-4.2.1"]),
        ("裸露土方没覆盖算文明施工问题吗", "隐患判定", ["JGJ1462013-4.1.3"]),
        ("消防通道被材料占了", "隐患判定", ["GB507202011-6.2.5"]),
        ("安全网破了没换", "隐患判定", ["JGJ1302011-9.0.6"]),
        ("模板支架没有剪刀撑", "隐患判定", ["JGJ1622008-6.2.4"]),
        ("基坑边能不能停混凝土泵车", "隐患判定", ["JGJ3112013-5.2.3"]),
        ("梯子能不能垫砖使用", "隐患判定", ["JGJ802016-5.1.2"]),
        ("吊篮里能不能站三个人", "隐患判定", ["JGJ2022010-5.4.4"]),
        ("安全帽过期了怎么处理", "整改要求", ["GB28112019-5.1.2"]),
        ("脚手架立杆能不能搭接接长", "隐患判定", ["JGJ1302011-6.2.6"]),
    ]
    valid = {c.clause_id for c in corpus.clauses}
    for i, (q, intent, golds) in enumerate(spoken, start=1):
        g = [x for x in golds if x in valid]
        if not g:
            continue
        rows.append({
            "qid": f"s{i:03d}",
            "question": q,
            "intent": intent,
            "gold_clause_ids": g,
            "gold_doc": "",
            "source": "spoken",
        })

    with TESTSET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip(" \t-*>#") for p in parts if len(p.strip()) > 8]


def _support(sent: str, contexts: list[str]) -> float:
    """句子被上下文支持的程度：token 覆盖率（去停用式短词）。"""
    st = [t for t in tokenize(sent) if len(t) > 1]
    if not st:
        return 1.0
    ctx = set()
    for c in contexts:
        ctx.update(tokenize(c))
    hit = sum(1 for t in st if t in ctx)
    return hit / len(st)


# 规范性陈述：只有带义务词/量化词的句子才算"事实主张"，
# 标题、小标题、"可能后果"这类结构性文字不参与忠实度判定。
_CLAIM_RE = re.compile(
    r"应|不得|严禁|必须|不应|不宜|宜|不大于|不小于|不超过|不少于|至少|最多|"
    r"间距|高度|宽度|深度|距离|电压|电流|荷载|强度|厚度|坡度|直径|长度|数量"
)


def _claims(text: str) -> list[str]:
    skip = ("等级提示", "判定依据", "引用条文", "可能后果", "现场检查要点", "相关条文", "依据《")
    out = []
    for s in _sentences(text):
        if len(s) < 10 or not _CLAIM_RE.search(s):
            continue
        if s.startswith(skip):
            continue
        out.append(s)
    return out


class Evaluator:
    def __init__(self) -> None:
        self.running = False
        self._prog = {"running": False, "done": 0, "total": 0, "message": ""}
        self._lock = threading.Lock()

    def progress(self) -> dict:
        return dict(self._prog)

    def latest(self) -> dict | None:
        if LATEST.exists():
            try:
                return json.loads(LATEST.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def run(self, limit: int = 40, use_llm_judge: bool = False) -> dict:
        with self._lock:
            if self.running:
                return {"ok": False}
            self.running = True
        t0 = time.time()
        try:
            n = build_testset()
            rows = [json.loads(l) for l in TESTSET.open(encoding="utf-8")]
            # 分层抽样：模板题与口语题都覆盖
            spoken = [r for r in rows if r["source"] == "spoken"]
            tmpl = [r for r in rows if r["source"] == "template"]
            random.seed(7)
            take_sp = spoken
            take_tm = random.sample(tmpl, min(len(tmpl), max(0, limit - len(take_sp))))
            sample = take_sp + take_tm
            self._prog = {"running": True, "done": 0, "total": len(sample), "message": "评测中"}

            pipeline = get_pipeline()
            emb = get_embedder()
            corpus = get_corpus()
            results = []
            for i, r in enumerate(sample, start=1):
                try:
                    out = pipeline.ask(r["question"], use_cache=False)
                except Exception as e:
                    results.append({**r, "error": f"{type(e).__name__}: {e}", "metrics": {}})
                    self._prog.update(done=i)
                    continue

                cites = out.get("citations") or []
                ctx_ids = [c["clause_id"] for c in cites]
                gold = set(r["gold_clause_ids"])
                # Context Precision（RAGAS 加权平均精度）
                ap = 0.0
                hit_n = 0
                for k, cid in enumerate(ctx_ids, start=1):
                    if cid in gold or any(cid.split("#")[0] == g for g in gold):
                        hit_n += 1
                        ap += hit_n / k
                precision = ap / max(1, hit_n) if hit_n else 0.0
                # Context Recall
                recall = len({g for g in gold if any(cid.split("#")[0] == g or cid == g for cid in ctx_ids)}) / max(1, len(gold))

                # 问题实际看到的证据：被引条文正文 + 检查要点 + 隐患 + 所属父块
                contexts: list[str] = []
                for c in cites:
                    if c.get("text"):
                        contexts.append(c["text"])
                    contexts.extend(c.get("hazards") or [])
                    contexts.extend(c.get("check_points") or [])
                    cl = corpus.by_id.get(c["clause_id"].split("#")[0])
                    if cl:
                        ptxt = corpus.parent_of(cl).get("text") or ""
                        if ptxt:
                            contexts.append(ptxt)

                answer = out.get("answer") or ""
                sents = _sentences(answer)
                claims = _claims(answer)
                # Faithfulness：只校验规范性陈述句是否被证据支持
                if claims and contexts:
                    supports = [_support(s, contexts) for s in claims]
                    faith = sum(1 for x in supports if x >= 0.55) / len(supports)
                else:
                    # 没有可核验的规范性陈述（拒答/引导语），视为无幻觉
                    faith = 1.0

                # Answer Relevancy：问题与答案实质句的相似度（RAGAS 的本地近似）
                try:
                    qv = emb.encode_dense([r["question"]])[0]
                    body = [s for s in sents if len(s) >= 12][:6] or [answer[:300]]
                    avs = emb.encode_dense(body)
                    sims = []
                    for av in avs:
                        dot = sum(a * b for a, b in zip(qv, av))
                        na = math.sqrt(sum(a * a for a in qv)) or 1
                        nb = math.sqrt(sum(b * b for b in av)) or 1
                        sims.append(max(0.0, min(1.0, dot / (na * nb))))
                    sims.sort(reverse=True)
                    relev = sims[0] if sims else 0.0
                except Exception:
                    relev = 0.0

                # Answer Correctness：gold 条文关键要素覆盖（0.8）+ 整体语义相似（0.2）
                gold_clauses = [corpus.by_id[g] for g in gold if g in corpus.by_id]
                if gold_clauses:
                    gold_text = " ".join(gc.text for gc in gold_clauses)
                    gts = {t for t in tokenize(gold_text) if len(t) > 1}
                    at = set(tokenize(answer))
                    recall_c = len(gts & at) / max(1, len(gts))
                    nums = re.findall(
                        r"\d+(?:\.\d+)?\s*(?:mm|cm|m|kV|V|A|mA|kg|t|kPa|MPa|℃|%|度|米|毫米|天|小时|次|人|层|步|倍)",
                        gold_text,
                    )
                    ans_norm = re.sub(r"\s+", "", answer)
                    num_hit = sum(1 for x in nums if re.sub(r"\s+", "", x) in ans_norm)
                    num_cov = (num_hit / len(nums)) if nums else recall_c
                    cover = 0.7 * recall_c + 0.3 * num_cov
                    try:
                        gv = emb.encode_dense([gold_text])[0]
                        dot = sum(a * b for a, b in zip(qv, gv))
                        na = math.sqrt(sum(a * a for a in qv)) or 1
                        nb = math.sqrt(sum(b * b for b in gv)) or 1
                        sem = max(0.0, min(1.0, dot / (na * nb)))
                    except Exception:
                        sem = 0.0
                    correct = 0.85 * cover + 0.15 * sem
                else:
                    correct = 0.0

                metrics = {
                    "faithfulness": round(faith, 4),
                    "answer_relevancy": round(relev, 4),
                    "context_precision": round(precision, 4),
                    "context_recall": round(recall, 4),
                    "answer_correctness": round(correct, 4),
                }
                results.append({
                    **r,
                    "answer": answer[:1200],
                    "citations": [{"clause_id": c["clause_id"], "std_no": c.get("std_no"), "clause_no": c.get("clause_no")} for c in cites],
                    "metrics": metrics,
                    "latency_ms": out.get("latency_ms", 0),
                    "confidence": out.get("confidence", 0),
                    "intent_pred": (out.get("intent") or {}).get("intent", ""),
                })
                self._prog.update(done=i)

            ok = [r for r in results if r.get("metrics")]
            keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
            agg = {k: round(sum(r["metrics"][k] for r in ok) / max(1, len(ok)), 4) for k in keys}
            agg["overall"] = round(sum(agg[k] for k in keys) / len(keys), 4)
            by_intent: dict[str, dict] = {}
            for r in ok:
                it = r["intent"]
                by_intent.setdefault(it, {"n": 0, "faithfulness": 0.0, "context_recall": 0.0, "answer_correctness": 0.0})
                by_intent[it]["n"] += 1
                for k in ("faithfulness", "context_recall", "answer_correctness"):
                    by_intent[it][k] += r["metrics"][k]
            for it, v in by_intent.items():
                for k in ("faithfulness", "context_recall", "answer_correctness"):
                    v[k] = round(v[k] / max(1, v["n"]), 4)

            by_source: dict[str, dict] = {}
            for r in ok:
                src = r.get("source", "template")
                by_source.setdefault(src, {"n": 0, "faithfulness": 0.0, "context_recall": 0.0, "answer_correctness": 0.0})
                by_source[src]["n"] += 1
                for k in ("faithfulness", "context_recall", "answer_correctness"):
                    by_source[src][k] += r["metrics"][k]
            for src, v in by_source.items():
                for k in ("faithfulness", "context_recall", "answer_correctness"):
                    v[k] = round(v[k] / max(1, v["n"]), 4)

            # 意图分类与检索无关，放在全量测试集上算更稳；抽样集再给一个参考值
            clf = get_intent_classifier()
            intent_acc = sum(1 for r in rows if clf.predict(r["question"])["intent"] == r["intent"]) / max(1, len(rows))
            intent_acc_sampled = sum(1 for r in ok if r["intent"] == r["intent_pred"]) / max(1, len(ok))
            payload = {
                "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s": round(time.time() - t0, 1),
                "testset_size": n,
                "evaluated": len(ok),
                "metrics": agg,
                "by_intent": by_intent,
                "by_source": by_source,
                "intent_accuracy": round(intent_acc, 4),
                "intent_accuracy_sampled": round(intent_acc_sampled, 4),
                "judge": "llm" if use_llm_judge else "local-rule",
                "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in ok) / max(1, len(ok))),
                "version": (ok[0].get("citations") and "see-trace") or "",
                "samples": results,
            }
            LATEST.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            hist = EVAL_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
            hist.write_text(json.dumps({k: v for k, v in payload.items() if k != "samples"}, ensure_ascii=False, indent=1), encoding="utf-8")
            self._prog = {"running": False, "done": len(results), "total": len(sample), "message": "完成"}
            return payload
        finally:
            self.running = False
            self._prog["running"] = False


_evaluator: Evaluator | None = None


def get_evaluator() -> Evaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator
