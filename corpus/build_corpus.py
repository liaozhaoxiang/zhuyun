"""语料构建：合并 raw JSON -> docs.json / clauses.jsonl / 原文 txt / 统计。"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT

CATEGORY_ORDER = ["基坑工程", "脚手架工程", "模板支架", "高处作业", "施工用电", "起重机械", "文明施工与消防"]

REQUIRED = ["std_no", "std_name", "category", "chapter", "clause_no", "level", "text"]


def slug(std_no: str) -> str:
    s = std_no.replace(" ", "").replace("/", "").replace("－", "").replace("-", "")
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]", "", s)
    return s


def load_clauses() -> list[dict]:
    items: list[dict] = []
    for fp in sorted(RAW.glob("*.json")):
        if fp.name == "docs_meta.json":
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for i, c in enumerate(data):
            missing = [k for k in REQUIRED if not c.get(k)]
            if missing:
                raise ValueError(f"{fp.name}[{i}] 缺字段 {missing}")
            items.append(c)
    return items


def main() -> None:
    clauses = load_clauses()
    meta = json.loads((RAW / "docs_meta.json").read_text(encoding="utf-8"))

    # 文档聚合
    docs: dict[str, dict] = {}
    for c in clauses:
        no = c["std_no"]
        if no not in docs:
            m = meta.get(no, {})
            docs[no] = {
                "doc_id": slug(no),
                "std_no": no,
                "std_name": c["std_name"],
                "category": c["category"],
                "publisher": m.get("publisher", ""),
                "issue_date": m.get("issue_date", ""),
                "effective_date": m.get("effective_date", ""),
                "status": m.get("status", "现行"),
                "pages": m.get("pages", 0),
                "scope": m.get("scope", ""),
                "alias": m.get("alias", []),
                "clause_count": 0,
                "doc_type": "企业制度" if no.startswith("Q/") else ("法规" if "令" in no else "标准"),
            }
        docs[no]["clause_count"] += 1

    # 父子块：父块 = 文档 + 章节
    chapters: dict[str, dict] = {}
    out_clauses = []
    for c in clauses:
        d = docs[c["std_no"]]
        chapter_key = f"{d['doc_id']}::{c['chapter']}"
        if chapter_key not in chapters:
            chapters[chapter_key] = {
                "parent_id": f"{d['doc_id']}-ch{len([k for k in chapters if k.startswith(d['doc_id'])]) + 1:02d}",
                "doc_id": d["doc_id"],
                "std_no": c["std_no"],
                "std_name": c["std_name"],
                "category": c["category"],
                "chapter": c["chapter"],
                "clauses": [],
            }
        pid = chapters[chapter_key]["parent_id"]
        chapters[chapter_key]["clauses"].append(c["clause_no"])

        text = c["text"].strip()
        out_clauses.append(
            {
                "clause_id": f"{d['doc_id']}-{c['clause_no']}",
                "parent_id": pid,
                "doc_id": d["doc_id"],
                "std_no": c["std_no"],
                "std_name": c["std_name"],
                "category": c["category"],
                "chapter": c["chapter"],
                "clause_no": c["clause_no"],
                "level": c["level"],
                "text": text,
                "check_points": c.get("check_points", []),
                "hazards": c.get("hazards", []),
                "page": c.get("page", 0),
                "char_len": len(text),
            }
        )

    # 父块正文：章节内全部条文拼接（父块回溯时返回给上游）
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in out_clauses:
        grouped[c["parent_id"]].append(c)
    parents = []
    for key, ch in chapters.items():
        items = sorted(grouped[ch["parent_id"]], key=lambda x: x["page"])
        body = "".join(f"第 {x['clause_no']} 条【{x['level']}】{x['text']}\n" for x in items)
        parents.append(
            {
                "parent_id": ch["parent_id"],
                "doc_id": ch["doc_id"],
                "std_no": ch["std_no"],
                "std_name": ch["std_name"],
                "category": ch["category"],
                "chapter": ch["chapter"],
                "clause_nos": [x["clause_no"] for x in items],
                "text": body,
                "char_len": len(body),
            }
        )

    doc_list = sorted(docs.values(), key=lambda x: (CATEGORY_ORDER.index(x["category"]) if x["category"] in CATEGORY_ORDER else 99, x["std_no"]))
    parents.sort(key=lambda x: x["parent_id"])

    # 写文件
    (OUT / "docs.json").write_text(json.dumps(doc_list, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "parents.json").write_text(json.dumps(parents, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / "clauses.jsonl").open("w", encoding="utf-8") as f:
        for c in out_clauses:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 原文（供文档加载 / OCR 演示，每个文档一份 txt）
    raw_dir = OUT / "documents"
    raw_dir.mkdir(exist_ok=True)
    for no, d in docs.items():
        lines = [f"{d['std_name']}", f"标准编号：{no}", f"发布单位：{d['publisher']}", f"实施日期：{d['effective_date']}", ""]
        cur_ch = None
        for c in [x for x in clauses if x["std_no"] == no]:
            if c["chapter"] != cur_ch:
                cur_ch = c["chapter"]
                lines.append("")
                lines.append(cur_ch)
            lines.append(f"第 {c['clause_no']} 条【{c['level']}】{c['text']}")
            if c.get("check_points"):
                lines.append("检查要点：" + "；".join(c["check_points"]))
        (raw_dir / f"{d['doc_id']}.txt").write_text("\n".join(lines), encoding="utf-8")

    # 统计
    by_cat = Counter(c["category"] for c in out_clauses)
    by_level = Counter(c["level"] for c in out_clauses)
    by_doc = Counter(c["std_no"] for c in out_clauses)
    stats = {
        "documents": len(doc_list),
        "clauses": len(out_clauses),
        "parents": len(parents),
        "avg_chars": round(sum(c["char_len"] for c in out_clauses) / len(out_clauses), 1),
        "by_category": dict(by_cat),
        "by_level": dict(by_level),
        "by_doc": dict(by_doc),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
