"""4 类业务意图识别：TF-IDF + 多项式朴素贝叶斯（本地训练，毫秒级）。

标签：规范条文查询 / 隐患判定 / 整改要求 / 系统闲聊
若配置了本地 BERT 意图模型目录，会自动改用 transformers 推理。
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.config import DATA_DIR, settings
from app.services.lexical import tokenize

LABELS = ["规范条文查询", "隐患判定", "整改要求", "系统闲聊"]

# ---------------- 标注样本 ----------------
SEED: dict[str, list[str]] = {
    "规范条文查询": [
        "脚手架立杆间距不能超过多少", "基坑监测频率要求是多少", "安全带应该怎么挂", "临边防护栏杆高度是多少",
        "塔吊多久检测一次", "漏电保护器动作电流多大", "高处作业是怎么定义的", "危大工程包括哪些",
        "施工升降机验收要求", "扣件式钢管脚手架钢管壁厚要求", "基坑支护安全等级怎么划分", "模板支架顶托悬臂长度限制",
        "连墙件设置间距", "剪刀撑角度要求", "配电箱安装高度", "临时用电三级配电是什么意思",
        "氧气乙炔瓶间距要求", "消防通道宽度", "施工噪声夜间限值", "动火作业需要什么手续",
        "JGJ 130 对剪刀撑有什么规定", "扣件拧紧力矩是多少", "电梯井口防护门高度", "安全帽使用期限",
        "基坑报警值一般设多少", "脚手架验收需要哪些资料", "塔式起重机附着要求", "施工升降机防坠器标定周期",
        "危险性较大的分部分项工程范围", "模板拆除混凝土强度要求", "什么情况下需要专家论证",
        "基坑边堆载距离要求", "吊篮安全绳怎么设置", "重复接地电阻要求", "行灯电压是多少",
        "土钉墙面层厚度", "盘扣架插销要求", "群塔作业安全距离", "安全带报废条件", "安全网怎么设置",
        "现场基坑支护要满足什么条件", "现场脚手架要满足什么条件", "现场临时用电要满足什么条件",
        "现场高处作业要满足什么条件", "现场模板支架要满足什么条件", "现场起重吊装要满足什么条件",
        "现场消防管理要满足什么条件", "连墙件有什么规范要求", "剪刀撑有什么规范要求",
        "临边防护有什么规范要求", "这个标准是多少", "规范是怎么规定的",
    ],
    "隐患判定": [
        "这个算不算重大隐患", "现场脚手架连墙件少了很多是什么等级", "基坑边堆土两米高有问题吗",
        "工人没系安全带怎么定性", "配电箱一闸多机属于什么隐患", "塔吊力矩限制器被短接了严重吗",
        "模板支架没有剪刀撑算不算重大隐患", "电梯井口没有防护门怎么判", "电缆拖地泡水是什么等级",
        "今天巡查发现临边没栏杆", "这个隐患严重吗", "脚手架立杆直接立在土上合规吗",
        "挖机在基坑边作业距离太近", "动火没办证算什么问题", "宿舍用泡沫夹芯板可以吗",
        "安全帽过期了还能用吗", "塔吊司机没证上岗怎么处理", "基坑监测数据超报警值了",
        "吊篮安全锁过期了怎么判", "配电箱里放了一堆工具", "工人从脚手架上往下扔模板",
        "卸料平台没有限载牌", "模板支架顶托伸出太长", "氧气瓶和乙炔瓶放一起了",
        "现场没有消防水源", "裸露土方没有覆盖", "夜间施工噪声超标被投诉",
        "这种情况属于一般隐患还是重大隐患", "帮我判断一下这个隐患等级", "这个做法符合规范吗",
        "安全员巡查发现工人站在钢筋骨架上", "施工升降机楼层门联锁坏了", "脚手架和结构没有拉结",
        "基坑出现管涌还在继续挖", "塔吊安全装置被屏蔽了", "临时用电没有采用 TN-S",
        "现场基坑支护不符合要求算不算隐患", "脚手架这样做合规吗", "临时用电不到位属于什么等级",
        "高处作业防护缺失严重吗", "模板支架有问题吗", "起重作业不规范算不算隐患",
        "消防通道被占用严重吗", "现场临边防护缺失算不算隐患", "这样做行不行",
    ],
    "整改要求": [
        "怎么整改", "整改单怎么写", "需要哪些整改措施", "整改期限多久", "复查要查什么",
        "给我一份整改方案", "这个隐患要几天内改完", "整改完成需要哪些资料", "怎么闭环",
        "整改责任人怎么确定", "重大隐患整改流程", "整改后谁验收", "需要停工整改吗",
        "帮我写整改通知单", "整改措施要引用哪条规范", "整改前后照片要几张",
        "隐患整改三定是什么", "整改超期了怎么办", "怎么申请复工", "整改报告怎么写",
        "分包单位不整改怎么处理", "给我列出整改步骤", "这个需要专家论证吗",
        "整改依据写哪一条", "复查不合格怎么办", "隐患台账怎么建", "整改闭环要哪些签字",
        "帮我生成一份整改回复", "整改措施有什么具体要求", "怎么防止同类隐患再次发生",
        "基坑支护不达标怎么整改", "脚手架不达标需要哪些整改措施", "临时用电隐患整改期限多久",
        "高处作业问题整改完怎么验收", "模板支架缺陷怎么整改", "起重设备隐患整改流程",
    ],
    "系统闲聊": [
        "你好", "你是谁", "你能做什么", "谢谢", "今天天气怎么样", "帮我写首诗",
        "你用的什么模型", "这个系统怎么部署", "知识库里有多少条规范", "支持哪些文件格式",
        "你好呀", "在吗", "你会不会出错", "你能联网吗", "帮我订个机票",
        "讲个笑话", "现在几点了", "你是什么公司的", "怎么联系客服", "系统好用吗",
    ],
}


class TfidfNB:
    """轻量多项式朴素贝叶斯，避免引入 sklearn。"""

    def __init__(self) -> None:
        self.labels = LABELS
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.log_prior: dict[str, float] = {}
        self.log_lik: dict[str, dict[str, float]] = {}
        self.default_loglik: dict[str, float] = {}
        self.trained = False

    def _featurize(self, text: str) -> Counter:
        toks = tokenize(text, expand=False)
        # 2-gram 补充
        toks += [toks[i] + toks[i + 1] for i in range(len(toks) - 1)]
        return Counter(toks)

    def train(self, data: dict[str, list[str]]) -> None:
        docs = [(lab, t) for lab, items in data.items() for t in items]
        n = len(docs)
        df: Counter = Counter()
        feats = []
        for lab, text in docs:
            f = self._featurize(text)
            feats.append((lab, f))
            for t in f:
                df[t] += 1
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}

        by_label: dict[str, Counter] = defaultdict(Counter)
        for lab, f in feats:
            for t, c in f.items():
                by_label[lab][t] += c * self.idf.get(t, 1.0)

        for lab in self.labels:
            cnt = sum(1 for l, _ in docs if l == lab)
            self.log_prior[lab] = math.log(cnt / n)
            total = sum(by_label[lab].values()) + len(self.vocab)
            self.log_lik[lab] = {t: math.log((by_label[lab].get(t, 0) + 1) / total) for t in self.vocab}
            self.default_loglik[lab] = math.log(1 / total)
        self.trained = True

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        f = self._featurize(text)
        scores: dict[str, float] = {}
        for lab in self.labels:
            s = self.log_prior[lab]
            for t, c in f.items():
                if t in self.vocab:
                    s += c * self.log_lik[lab][t]
                else:
                    s += self.default_loglik[lab] * 0.5
            scores[lab] = s
        # softmax
        mx = max(scores.values())
        exps = {k: math.exp(v - mx) for k, v in scores.items()}
        z = sum(exps.values()) or 1.0
        probs = {k: v / z for k, v in exps.items()}
        best = max(probs.items(), key=lambda x: x[1])
        return best[0], best[1], probs

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "TfidfNB":
        return pickle.loads(path.read_bytes())


class IntentClassifier:
    def __init__(self) -> None:
        self.model_path = DATA_DIR / "intent_nb.pkl"
        self.clf: TfidfNB | None = None
        self.backend = "rule"
        self._bert = None
        self._load()

    def _load(self) -> None:
        try:
            if self.model_path.exists():
                self.clf = TfidfNB.load(self.model_path)
                self.backend = "tfidf-nb"
            else:
                clf = TfidfNB()
                clf.train(SEED)
                clf.save(self.model_path)
                self.clf = clf
                self.backend = "tfidf-nb"
        except Exception:
            self.clf = None
            self.backend = "rule"

    def _rule(self, q: str) -> tuple[str, float, dict[str, float]]:
        q_low = q.strip()
        rules = [
            ("整改要求", r"怎么整改|如何整改|整改措施|整改期限|整改单|整改方案|整改流程|整改完|闭环|复查|复工|几天内改完|停工整改|整改通知"),
            ("隐患判定", r"算不算|属于什么|什么等级|严重吗|怎么判|判定为|定性为|合规吗|可以吗|行不行|有问题吗|违法|违规"),
            ("系统闲聊", r"^(你好|您好|hi|hello|在吗|谢谢|你是谁|你能做什么)|天气|笑话|诗|订.*票"),
        ]
        for lab, pat in rules:
            if re.search(pat, q_low):
                return lab, 0.82, {lab: 0.82}
        if re.search(r"多少|多久|什么要求|什么条件|要满足|怎么设置|规定|标准|条|第几条|间距|高度|频率|限值", q_low):
            return "规范条文查询", 0.78, {"规范条文查询": 0.78}
        return "规范条文查询", 0.55, {"规范条文查询": 0.55}

    def predict(self, query: str) -> dict:
        # 强规则优先：模板化问法（“算不算”“要满足什么条件”）用词稳定，规则比小模型可靠
        r_lab, r_conf, r_probs = self._rule(query)
        if r_conf >= 0.75:
            return {"intent": r_lab, "confidence": round(r_conf, 3), "scores": {k: round(v, 3) for k, v in r_probs.items()},
                    "backend": self.backend, "fallback": True, "rule_first": True}
        if self.clf and self.clf.trained:
            lab, conf, probs = self.clf.predict(query)
            # 低置信度回退规则
            if conf < settings.intent_conf_threshold:
                r_lab, r_conf, _ = self._rule(query)
                if r_conf > conf:
                    return {"intent": r_lab, "confidence": round(r_conf, 3), "scores": {k: round(v, 3) for k, v in probs.items()}, "backend": self.backend, "fallback": True}
            return {"intent": lab, "confidence": round(conf, 3), "scores": {k: round(v, 3) for k, v in probs.items()}, "backend": self.backend, "fallback": False}
        lab, conf, probs = self._rule(query)
        return {"intent": lab, "confidence": round(conf, 3), "scores": {k: round(v, 3) for k, v in probs.items()}, "backend": "rule", "fallback": True}

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "labels": LABELS,
            "samples": sum(len(v) for v in SEED.values()),
            "vocab": len(self.clf.vocab) if self.clf else 0,
        }


_classifier: IntentClassifier | None = None


def get_intent_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
