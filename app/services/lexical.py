"""关键词检索：BM25 倒排索引 + 领域同义词扩展。

分词 = jieba（含领域词典）+ 中文单字兜底 + 英文/数字词。
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Iterable

_CJK = re.compile(r"[\u4e00-\u9fa5]")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-_/]*")

# 领域词表：保证“高支模/临电/塔吊/爬架”等口语词能被切出来
DOMAIN_WORDS = [
    "基坑", "支护", "锚杆", "土钉墙", "冠梁", "排桩", "地下连续墙", "降水", "回灌", "监测点", "报警值",
    "脚手架", "立杆", "横杆", "扫地杆", "剪刀撑", "连墙件", "扣件", "脚手板", "悬挑", "盘扣", "碗扣", "门式",
    "爬架", "附着式升降脚手架", "吊篮", "安全绳", "安全锁", "配重",
    "模板", "支架", "高支模", "顶托", "可调托撑", "拆模", "铝模", "爬模", "飞模",
    "临边", "洞口", "电梯井", "安全网", "密目网", "安全带", "安全帽", "攀登", "悬空", "操作平台", "交叉作业",
    "临时用电", "三级配电", "两级保护", "漏电保护器", "开关箱", "配电箱", "重复接地", "接零", "TN-S", "行灯",
    "塔吊", "塔式起重机", "施工升降机", "物料提升机", "汽车吊", "履带吊", "吊装", "吊索具", "力矩限制器", "防坠安全器",
    "动火", "消防", "灭火器", "消防通道", "噪声", "扬尘", "渣土", "沉淀池", "文明施工",
    "危大工程", "专项施工方案", "专家论证", "安全技术交底", "三级教育", "隐患", "整改单", "闭环",
]

# 口语 -> 规范用语（查询扩展）
SYNONYMS = {
    "塔吊": ["塔式起重机", "塔机"],
    "塔机": ["塔式起重机"],
    "临电": ["临时用电"],
    "临边": ["临边作业", "防护栏杆"],
    "洞口": ["洞口作业", "盖板"],
    "高支模": ["模板支架", "高大模板"],
    "爬架": ["附着式升降脚手架"],
    "吊篮": ["高处作业吊篮"],
    "触电": ["漏电保护", "接零", "临时用电"],
    "漏保": ["漏电保护器"],
    "安全带": ["坠落防护", "高挂低用"],
    "安全帽": ["头部防护"],
    "基坑": ["基坑支护", "基坑工程"],
    "塌方": ["坍塌", "边坡失稳", "基坑"],
    "脚手架": ["扣件式钢管脚手架", "架体"],
    "动火": ["动火作业", "动火证"],
    "消防": ["临时消防设施", "灭火器"],
    "扬尘": ["覆盖", "洒水", "环境"],
    "三级配电": ["三级配电两级保护", "一机一闸一漏一箱"],
    "一闸多机": ["一机一闸一漏一箱", "开关箱"],
    "危大": ["危险性较大的分部分项工程", "危大工程"],
}

_jieba_ready = False


def _init_jieba() -> None:
    global _jieba_ready
    if _jieba_ready:
        return
    try:
        import jieba

        for w in DOMAIN_WORDS:
            jieba.add_word(w, freq=20000)
        _jieba_ready = True
    except Exception:
        _jieba_ready = False


def tokenize(text: str, expand: bool = False) -> list[str]:
    _init_jieba()
    text = text.strip()
    toks: list[str] = []
    try:
        import jieba

        toks = [t for t in jieba.lcut(text) if t.strip() and t not in "，。；：、（）()【】《》“”\"' \t\n"]
    except Exception:
        pass
    toks += [t.lower() for t in _WORD.findall(text)]
    # 单字兜底（只补长词覆盖不到的短查询）
    if len(toks) < 3:
        toks += _CJK.findall(text)
    if expand:
        extra = []
        for t in toks:
            for syn in SYNONYMS.get(t, []):
                extra.extend(_simple_cut(syn))
        toks += extra
    return [t for t in toks if len(t) > 0]


def _simple_cut(s: str) -> list[str]:
    try:
        import jieba

        return [t for t in jieba.lcut(s) if t.strip()]
    except Exception:
        return _CJK.findall(s)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[str] = []
        self.ids: list[str] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0
        self.N = 0
        self._lock = threading.RLock()

    def build(self, items: Iterable[tuple[str, str]]) -> None:
        """items: (id, text)"""
        with self._lock:
            self.docs, self.ids, self.tf = [], [], []
            self.df = Counter()
            lens = []
            for cid, text in items:
                toks = tokenize(text)
                c = Counter(toks)
                self.ids.append(cid)
                self.docs.append(text)
                self.tf.append(c)
                lens.append(len(toks))
                for t in c:
                    self.df[t] += 1
            self.N = len(self.docs)
            self.avgdl = (sum(lens) / self.N) if self.N else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20, category: str | None = None, cat_map: dict[str, str] | None = None) -> list[tuple[str, float]]:
        with self._lock:
            q_toks = tokenize(query, expand=True)
            if not q_toks or not self.N:
                return []
            scores: dict[int, float] = defaultdict(float)
            for t in q_toks:
                idf = self._idf(t)
                if idf <= 0:
                    continue
                for i, c in enumerate(self.tf):
                    f = c.get(t)
                    if not f:
                        continue
                    dl = sum(c.values())
                    denom = f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                    scores[i] += idf * f * (self.k1 + 1) / denom
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            out = []
            for i, s in ranked:
                cid = self.ids[i]
                if category and cat_map and cat_map.get(cid) != category:
                    continue
                out.append((cid, float(s)))
                if len(out) >= top_k:
                    break
            return out

    def status(self) -> dict:
        return {"backend": "bm25", "docs": self.N, "vocab": len(self.df), "avgdl": round(self.avgdl, 1)}


class SparseIndex:
    """稀疏向量倒排（BGE-M3 lexical weights 或 TF 权重），按点积打分。"""

    def __init__(self) -> None:
        self.postings: dict[str, dict[int, float]] = defaultdict(dict)
        self.lengths: list[float] = []
        self.ids: list[str] = []
        self.cat: list[str] = []

    def build(self, items: Iterable[tuple[str, str, dict[str, float], str]]) -> None:
        self.postings = defaultdict(dict)
        self.ids, self.lengths, self.cat = [], [], []
        for i, (cid, _text, weights, category) in enumerate(items):
            self.ids.append(cid)
            self.cat.append(category)
            norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
            self.lengths.append(norm)
            for t, w in weights.items():
                self.postings[t][i] = w
        # idf 平滑
        n = max(1, len(self.ids))
        self.idf = {t: math.log(1 + n / (1 + len(p))) for t, p in self.postings.items()}

    def search(self, weights: dict[str, float], top_k: int = 20, category: str | None = None) -> list[tuple[str, float]]:
        if not self.postings:
            return []
        scores: dict[int, float] = defaultdict(float)
        for t, qw in weights.items():
            post = self.postings.get(t)
            if not post:
                continue
            idf = self.idf.get(t, 1.0)
            for i, dw in post.items():
                scores[i] += qw * dw * idf
        out = []
        for i, s in sorted(scores.items(), key=lambda x: -x[1]):
            if category and self.cat[i] != category:
                continue
            out.append((self.ids[i], float(s)))
            if len(out) >= top_k:
                break
        return out

    def status(self) -> dict:
        return {"backend": "sparse-inverted", "docs": len(self.ids), "terms": len(self.postings)}
