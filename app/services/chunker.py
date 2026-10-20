"""父子块分块策略。

- 子块：条文级，目标 300 字，超长条文按句切分并保留 50 字重叠；用于向量召回与 BM25。
- 父块：章节级，目标 1200 字，超长章节按条切分；命中子块后回溯父块，保证强制性条文归属清晰。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from app.services.corpus import Clause, Corpus

CHILD_TARGET = 300
CHILD_OVERLAP = 50
PARENT_TARGET = 1200

_SENT = re.compile(r"(?<=[。；！？])")


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str
    title: str
    meta: dict = field(default_factory=dict)


def split_by_sentence(text: str, target: int = CHILD_TARGET, overlap: int = CHILD_OVERLAP) -> list[str]:
    """按句切分到目标长度，带重叠。"""
    if len(text) <= target:
        return [text]
    sents = [s for s in _SENT.split(text) if s.strip()]
    out, buf = [], ""
    for s in sents:
        if len(buf) + len(s) <= target:
            buf += s
            continue
        if buf:
            out.append(buf)
        buf = (buf[-overlap:] if overlap and buf else "") + s
    if buf:
        out.append(buf)
    return out


def build_child_chunks(corpus: Corpus) -> list[Chunk]:
    chunks: list[Chunk] = []
    for c in corpus.clauses:
        parts = split_by_sentence(c.text)
        for i, part in enumerate(parts):
            cid = c.clause_id if len(parts) == 1 else f"{c.clause_id}#{i + 1}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    parent_id=c.parent_id,
                    text=part,
                    title=f"{c.std_no} {c.display_no}",
                    meta={
                        "clause_id": c.clause_id,
                        "doc_id": c.doc_id,
                        "std_no": c.std_no,
                        "std_name": c.std_name,
                        "category": c.category,
                        "chapter": c.chapter,
                        "clause_no": c.clause_no,
                        "level": c.level,
                        "page": c.page,
                        "part": i + 1,
                        "check_points": list(c.check_points),
                        "hazards": list(c.hazards),
                    },
                )
            )
    return chunks


def build_parent_text(corpus: Corpus, parent_id: str) -> str:
    p = corpus.parents.get(parent_id)
    return p["text"] if p else ""


# ---------------- 上传文档的分块 ----------------

def _split_long_paragraph(text: str, target: int = CHILD_TARGET) -> list[str]:
    out = []
    for para in re.split(r"\n{1,}", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target:
            out.append(para)
        else:
            out.extend(split_by_sentence(para, target=target, overlap=CHILD_OVERLAP))
    return out


def chunk_plain_document(text: str, doc_id: str, title: str, category: str = "未分类") -> tuple[list[Chunk], list[dict]]:
    """通用文档：父块 1200 字滚动，子块 300 字滑窗。"""
    paras = _split_long_paragraph(text)
    parents: list[dict] = []
    children: list[Chunk] = []

    # 父块
    buf, buf_len, pidx = [], 0, 0
    groups: list[list[str]] = []
    for p in paras:
        if buf_len + len(p) > PARENT_TARGET and buf:
            groups.append(buf)
            buf, buf_len = [], 0
        buf.append(p)
        buf_len += len(p)
    if buf:
        groups.append(buf)

    for pidx, group in enumerate(groups, start=1):
        pid = f"{doc_id}-p{pidx:03d}"
        body = "\n".join(group)
        parents.append({"parent_id": pid, "doc_id": doc_id, "title": title, "category": category, "text": body, "char_len": len(body)})
        # 子块：父块内按 300 字切
        for j, seg in enumerate(_split_long_paragraph(body), start=1):
            children.append(
                Chunk(
                    chunk_id=f"{pid}-c{j:02d}",
                    parent_id=pid,
                    text=seg,
                    title=title,
                    meta={"doc_id": doc_id, "category": category, "clause_no": "", "level": "一般", "page": pidx},
                )
            )
    return children, parents


def summarize_chunks(chunks: Iterable[Chunk]) -> dict:
    chunks = list(chunks)
    lens = [len(c.text) for c in chunks] or [0]
    return {
        "count": len(chunks),
        "avg_chars": round(sum(lens) / len(lens), 1),
        "max_chars": max(lens),
        "min_chars": min(lens),
    }
