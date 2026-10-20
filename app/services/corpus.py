"""语料与知识库服务：文档元数据、条文、父块、统计。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import CORPUS_DIR, settings


@dataclass(frozen=True)
class Clause:
    clause_id: str
    parent_id: str
    doc_id: str
    std_no: str
    std_name: str
    category: str
    chapter: str
    clause_no: str
    level: str
    text: str
    check_points: tuple[str, ...]
    hazards: tuple[str, ...]
    page: int

    @property
    def display_no(self) -> str:
        no = self.clause_no
        if no.startswith("第"):
            return no
        return f"第 {no} 条"

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "std_no": self.std_no,
            "std_name": self.std_name,
            "category": self.category,
            "chapter": self.chapter,
            "clause_no": self.clause_no,
            "level": self.level,
            "text": self.text,
            "check_points": list(self.check_points),
            "hazards": list(self.hazards),
            "page": self.page,
        }


class Corpus:
    """只读语料索引，进程内单例。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or CORPUS_DIR
        self.docs: list[dict] = json.loads((self.root / "docs.json").read_text(encoding="utf-8"))
        self.parents: dict[str, dict] = {
            p["parent_id"]: p for p in json.loads((self.root / "parents.json").read_text(encoding="utf-8"))
        }
        self.clauses: list[Clause] = []
        with (self.root / "clauses.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                self.clauses.append(
                    Clause(
                        clause_id=d["clause_id"],
                        parent_id=d["parent_id"],
                        doc_id=d["doc_id"],
                        std_no=d["std_no"],
                        std_name=d["std_name"],
                        category=d["category"],
                        chapter=d["chapter"],
                        clause_no=d["clause_no"],
                        level=d["level"],
                        text=d["text"],
                        check_points=tuple(d.get("check_points") or []),
                        hazards=tuple(d.get("hazards") or []),
                        page=d.get("page", 0),
                    )
                )
        self.by_id: dict[str, Clause] = {c.clause_id: c for c in self.clauses}
        self.by_doc: dict[str, list[Clause]] = {}
        for c in self.clauses:
            self.by_doc.setdefault(c.doc_id, []).append(c)
        self.docs_by_id = {d["doc_id"]: d for d in self.docs}

    # ---------- 查询 ----------
    def stats(self) -> dict:
        cats = {c: 0 for c in settings.categories}
        levels = {"强制性": 0, "一般": 0, "企业制度": 0}
        for c in self.clauses:
            cats[c.category] = cats.get(c.category, 0) + 1
            levels[c.level] = levels.get(c.level, 0) + 1
        return {
            "documents": len(self.docs),
            "clauses": len(self.clauses),
            "parents": len(self.parents),
            "categories": cats,
            "levels": levels,
            "avg_chars": round(sum(len(c.text) for c in self.clauses) / max(1, len(self.clauses)), 1),
            "standards": sum(1 for d in self.docs if d["doc_type"] == "标准"),
            "regulations": sum(1 for d in self.docs if d["doc_type"] == "法规"),
            "enterprise": sum(1 for d in self.docs if d["doc_type"] == "企业制度"),
        }

    def category_summary(self) -> list[dict]:
        out = []
        for cat in settings.categories:
            docs = [d for d in self.docs if d["category"] == cat]
            n = sum(len(self.by_doc.get(d["doc_id"], [])) for d in docs)
            out.append({"category": cat, "documents": len(docs), "clauses": n})
        return out

    def doc_detail(self, doc_id: str) -> dict | None:
        d = self.docs_by_id.get(doc_id)
        if not d:
            return None
        clauses = sorted(self.by_doc.get(doc_id, []), key=lambda x: x.page)
        chapters: dict[str, list[dict]] = {}
        for c in clauses:
            chapters.setdefault(c.chapter, []).append(c.to_dict())
        return {**d, "chapters": [{"chapter": k, "clauses": v} for k, v in chapters.items()]}

    def parent_of(self, clause: Clause) -> dict:
        return self.parents.get(clause.parent_id, {})


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    return Corpus()
