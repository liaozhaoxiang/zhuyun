"""查询改写：子查询分解 + HyDE 假设文档 + 回溯扩展。"""
from __future__ import annotations

import re

from app.services.lexical import SYNONYMS, tokenize

_SPLIT = re.compile(r"(?:以及|并且|还有|同时|另外|，?和|，?与|；|;|\?|？)")


def decompose(query: str, max_parts: int = 3) -> list[str]:
    """把复合问句拆成子查询；拆不出有意义的子句时原样返回。"""
    q = query.strip()
    parts = [p.strip("，,。 ?？") for p in _SPLIT.split(q)]
    parts = [p for p in parts if len(p) >= 4]
    if len(parts) <= 1:
        return [q]
    # 过短的子句（如“要求是多少”）补上主语
    out = []
    for p in parts:
        if len(p) < 5 and out:
            p = out[0] + p
        out.append(p)
    # 去重保序
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:max_parts]


def expand_terms(query: str) -> list[str]:
    toks = tokenize(query)
    extra = []
    for t in toks:
        extra.extend(SYNONYMS.get(t, []))
    return list(dict.fromkeys(extra))


def hyde_text(query: str, intent: str = "规范条文查询") -> str:
    """无 LLM 时的模板化 HyDE：生成一段“规范体”假设文本，用于向量检索。"""
    terms = expand_terms(query)
    term_str = "、".join(terms[:6]) if terms else query
    if intent == "隐患判定":
        return (
            f"根据建筑施工安全规范，关于{query}的判定：现场存在不符合{term_str}相关规定的做法，"
            "应依据强制性条文判定隐患等级，属于重大隐患的应立即停工整改。"
            "检查要点包括防护设施完整性、构件规格、安全装置有效性及验收记录。"
        )
    if intent == "整改要求":
        return (
            f"针对{query}，整改措施应依据{term_str}相关条文执行：立即停止违规作业，"
            "按规范要求恢复防护设施或调整施工参数，整改完成后由项目安全总监组织复查验收，"
            "一般隐患 3 日内整改完毕，重大隐患立即停工整改。"
        )
    return (
        f"根据建筑施工安全规范，关于{query}的规定如下：{term_str}应符合规范要求，"
        "条文明确了构造要求、设置间距、材料规格、检查验收及安全防护措施，"
        "强制性条文必须严格执行，检查要点包括尺寸偏差、连接可靠性、验收记录。"
    )


def backtrack_queries(query: str, std_name: str = "", chapter: str = "", category: str = "") -> list[str]:
    """首轮召回不足时的回溯查询。"""
    out = []
    if std_name:
        out.append(f"{std_name} {query}")
    if chapter:
        out.append(f"{chapter} {query}")
    if category:
        out.append(f"{category} {query}")
    terms = expand_terms(query)
    if terms:
        out.append(" ".join(terms[:4]) + " " + query)
    return list(dict.fromkeys(out))[:3]
