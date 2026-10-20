"""把最近一次评测结果渲染成 docs/eval.md。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "data" / "eval_latest.json"
OUT = ROOT / "docs" / "eval.md"


def main() -> None:
    if not LATEST.exists():
        raise SystemExit("还没有评测结果，先运行 /api/eval/run")
    d = json.loads(LATEST.read_text(encoding="utf-8"))
    m = d["metrics"]
    lines = [
        "# 评测报告",
        "",
        f"- 运行时间：{d['run_at']}",
        f"- 测试集：{d['testset_size']} 条（本次抽样 {d['evaluated']} 条）",
        f"- judge：{d['judge']}",
        f"- 平均延迟：{d['avg_latency_ms']} ms",
        f"- 意图分类准确率：{d['intent_accuracy']}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 分数 |",
        "|---|---|",
        f"| Faithfulness | {m['faithfulness']:.3f} |",
        f"| Answer Relevancy | {m['answer_relevancy']:.3f} |",
        f"| Context Precision | {m['context_precision']:.3f} |",
        f"| Context Recall | {m['context_recall']:.3f} |",
        f"| Answer Correctness | {m['answer_correctness']:.3f} |",
        f"| **Overall** | **{m['overall']:.3f}** |",
        "",
        "## 分意图",
        "",
        "| 意图 | 样本 | Faithfulness | Context Recall | Answer Correctness |",
        "|---|---|---|---|---|",
    ]
    for k, v in d["by_intent"].items():
        lines.append(f"| {k} | {v['n']} | {v['faithfulness']:.3f} | {v['context_recall']:.3f} | {v['answer_correctness']:.3f} |")

    by_source = d.get("by_source") or {}
    if by_source:
        lines += [
            "",
            "## 分题型",
            "",
            "| 来源 | 样本 | Faithfulness | Context Recall | Answer Correctness |",
            "|---|---|---|---|---|",
        ]
        for k, v in by_source.items():
            name = "口语题" if k == "spoken" else "条文模板题"
            lines.append(f"| {name} | {v['n']} | {v['faithfulness']:.3f} | {v['context_recall']:.3f} | {v['answer_correctness']:.3f} |")

    samples = d.get("samples") or []
    bad = sorted([s for s in samples if s.get("metrics")], key=lambda x: x["metrics"]["context_recall"])[:10]
    lines += ["", "## 召回未命中的样例（前 10）", "", "| 问题 | 期望条文 | 实际召回 | Recall |", "|---|---|---|---|"]
    for s in bad:
        got = "、".join(c["clause_id"] for c in (s.get("citations") or [])[:3]) or "无"
        lines.append(f"| {s['question']} | {'、'.join(s['gold_clause_ids'])} | {got} | {s['metrics']['context_recall']:.2f} |")

    lines += [
        "",
        "## 指标口径",
        "",
        "- **Faithfulness**：只取答案里的规范性陈述句（含“应/不得/严禁/必须/不超过”等义务或量化词），逐句与被引条文正文、检查要点、隐患提示及父块做 token 覆盖，≥55% 记为有支撑，取有支撑句占比；标题与结构性文字不参与判定。",
        "- **Answer Relevancy**：问题与答案实质句的最大余弦相似度（本地近似，偏保守）。",
        "- **Context Precision**：按 RAGAS 加权平均精度计算，命中 gold 条文的位置越靠前分越高。",
        "- **Context Recall**：gold 条文中被召回的比例。",
        "- **Answer Correctness**：gold 条文正文的 token 覆盖率（0.7）加数值命中率（0.3），再与整体语义相似度加权（0.85/0.15）。",
        "",
        "> 本地 judge 是近似实现，用于横向比较与回归，不等价于官方 ragas 库结果。",
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
