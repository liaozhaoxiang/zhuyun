"""答案生成：有 LLM 走 Qwen（OpenAI 兼容流式），无 LLM 走严格抽取式模板。

抽取式的原则：只说条文里有的，每个结论挂引用编号，数字不推断。
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Iterator

import httpx

from app.config import settings

SYSTEM_PROMPT = """你是建筑施工安全规范问答助手，服务对象是现场安全员。
规则：
1. 只能使用【给定条文】中的内容回答，禁止编造规范编号、条文号、数值。
2. 每个结论句末尾必须标注引用编号，如 [1][2]。
3. 回答结构：先给结论（一句话），再给依据条文要点，最后给现场动作建议（如有）。
4. 如果给定条文不足以回答，直接说“现有知识库未收录对应条文”，并说明建议查阅哪类规范。
5. 语言简洁、术语准确，不要客套。"""


def _fmt_clause(h: dict) -> str:
    level = h.get("level") or ""
    page = h.get("page") or 0
    return f"[{h.get('std_no')}] {h.get('clause_no')} 条（{level}，P{page}）{h.get('text')}"


def build_context(parents: list[dict]) -> str:
    lines = []
    for i, p in enumerate(parents, start=1):
        lines.append(f"【条文{i}】{p.get('std_name')}（{p.get('std_no')}）{p.get('chapter')}")
        for h in p.get("hits", []):
            lines.append(f"- 第 {h.get('clause_no')} 条（{h.get('level')}）：{h.get('text')}")
        lines.append("")
    return "\n".join(lines)


def build_citations(parents: list[dict]) -> list[dict]:
    cites = []
    idx = 1
    for p in parents:
        for h in p.get("hits", []):
            cites.append(
                {
                    "index": idx,
                    "parent_id": p.get("parent_id"),
                    "clause_id": h.get("chunk_id"),
                    "std_no": h.get("std_no"),
                    "std_name": h.get("std_name"),
                    "clause_no": h.get("clause_no"),
                    "level": h.get("level"),
                    "chapter": h.get("chapter") or p.get("chapter"),
                    "page": h.get("page"),
                    "text": h.get("text"),
                    "check_points": h.get("check_points") or [],
                    "hazards": h.get("hazards") or [],
                    "score": h.get("rerank_score"),
                    "category": h.get("category"),
                }
            )
            idx += 1
            if idx > 8:
                return cites
    return cites


def _pick_key_sentences(text: str, n: int = 2) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[。；])", text) if len(s.strip()) > 6]
    with_num = [s for s in sents if re.search(r"\d", s)]
    return (with_num or sents)[:n]


def _extractive(intent: str, question: str, parents: list[dict], cites: list[dict], top_score: float) -> tuple[str, float]:
    if not parents or not cites:
        return (
            "现有知识库未收录与该问题直接对应的条文。\n\n建议补充关键词（如规范名称、构件名称、具体数值要求），或切换到对应分类后重试。",
            0.2,
        )

    first = cites[0]
    cite_no = f"[1]"
    head = ""
    body: list[str] = []
    tail: list[str] = []

    if intent == "规范条文查询":
        head = f"依据《{first['std_name']}》{first['std_no']} 第 {first['clause_no']} 条（{first['level']}）："
        body.append(f"> {first['text']} {cite_no}")
        if len(cites) > 1:
            body.append("\n**相关条文**")
            for c in cites[1:4]:
                body.append(f"- 《{c['std_name']}》第 {c['clause_no']} 条（{c['level']}）：{c['text'][:120]}… [{c['index']}]")
        cps = first.get("check_points") or []
        if cps:
            body.append("\n**现场检查要点**")
            body.extend([f"{i}. {x}" for i, x in enumerate(cps[:6], start=1)])

    elif intent == "隐患判定":
        level = first["level"]
        head = (
            f"**判定：不符合规范要求。** 依据《{first['std_name']}》第 {first['clause_no']} 条"
            f"（{level}）{cite_no}，现场做法与条文规定不一致。"
        )
        body.append(f"> {first['text']} {cite_no}")
        hz = first.get("hazards") or []
        if hz:
            body.append("\n**可能后果**")
            body.append("、".join(hz[:4]))
        body.append("\n**判定依据**")
        for c in cites[:3]:
            body.append(f"- [{c['index']}] 《{c['std_name']}》第 {c['clause_no']} 条 · {c['level']}")
        body.append(
            "\n**等级提示**：按企业《安全隐患分级判定与整改闭环管理办法》，涉及支护结构失稳、安全装置失效、"
            "临时用电系统性缺陷的，应判定为重大隐患并立即停工整改；其余按一般隐患 3 日内整改。"
        )

    elif intent == "整改要求":
        head = f"**整改依据**：《{first['std_name']}》第 {first['clause_no']} 条（{first['level']}）{cite_no}。"
        body.append(f"> {first['text']} {cite_no}")
        steps: list[str] = []
        for c in cites[:3]:
            for cp in (c.get("check_points") or [])[:2]:
                steps.append(cp)
        if not steps:
            steps = ["按条文规定恢复/调整现场设施", "整改过程留存影像资料", "整改完成后报项目安全总监复查"]
        body.append("\n**整改措施**")
        body.extend([f"{i}. {s}" for i, s in enumerate(dict.fromkeys(steps), start=1)])
        body.append("\n**时限与闭环**")
        body.append("一般隐患 3 日内整改完毕；重大隐患立即停工整改。整改完成后附整改前后对比照片，由安全总监复查合格后方可复工。")

    else:
        head = "我是筑安云建筑施工安全规范助手，可以回答规范条文、隐患判定与整改要求三类问题。"
        body.append("例如：脚手架连墙件间距要求是多少 / 基坑边堆土 2 米算不算重大隐患 / 这个隐患怎么整改。")

    # 置信度
    conf = max(0.0, min(0.99, 0.55 + 0.4 * top_score))
    if len(cites) >= 3:
        conf = min(0.99, conf + 0.03)
    md = head + "\n\n" + "\n".join(body)
    md += "\n\n**引用条文**\n" + "\n".join(
        f"[{c['index']}] 《{c['std_name']}》{c['std_no']} 第 {c['clause_no']} 条 · {c['level']} · P{c['page']}" for c in cites[:5]
    )
    return md, round(conf, 3)


def _stream_llm(question: str, parents: list[dict], history: list[dict] | None = None) -> Iterator[str]:
    ctx = build_context(parents)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-4:]:
        if h.get("role") in ("user", "assistant"):
            msgs.append({"role": h["role"], "content": h.get("content", "")[:800]})
    msgs.append({"role": "user", "content": f"【给定条文】\n{ctx}\n\n【问题】{question}"})
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {"model": settings.llm_model, "messages": msgs, "temperature": 0.1, "stream": True, "max_tokens": 900}
    headers = {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=settings.llm_timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0].get("delta", {}).get("content") or ""
                if delta:
                    yield delta
            except Exception:
                continue


def llm_available() -> bool:
    return bool(settings.llm_base_url and settings.llm_api_key)


def generate_stream(
    question: str,
    intent: str,
    parents: list[dict],
    cites: list[dict],
    top_score: float,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    """产出事件：token / citations / done。"""
    if llm_available() and parents:
        buf = []
        try:
            for delta in _stream_llm(question, parents, history):
                buf.append(delta)
                yield {"type": "token", "data": delta}
            text = "".join(buf)
            conf = max(0.0, min(0.99, 0.6 + 0.38 * top_score))
            yield {"type": "citations", "data": cites}
            yield {"type": "done", "data": {"answer": text, "confidence": round(conf, 3), "generator": f"llm:{settings.llm_model}"}}
            return
        except Exception as e:
            # LLM 失败退出，回退抽取式
            yield {"type": "token", "data": ""}
            yield {"type": "note", "data": f"LLM 调用失败，已回退抽取式生成（{type(e).__name__}）"}

    md, conf = _extractive(intent, question, parents, cites, top_score)
    # 分块流式，前端呈现打字效果
    for i in range(0, len(md), 18):
        yield {"type": "token", "data": md[i : i + 18]}
    yield {"type": "citations", "data": cites}
    yield {"type": "done", "data": {"answer": md, "confidence": conf, "generator": "extractive"}}
