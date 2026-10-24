"""真实扫描的训练来源。

RealLines：run/build_reallines.py 从测试页以外的真实书页/词典页生成的伪标签行。
"""
import json
import pickle
import random

import cv2
import numpy as np

from . import config as C
from .preprocess import ink_map, to_input

DICT_HEADS_PKL = C.DATA / "real" / "dict_heads.pkl"


def build_dict_heads():
    hold = {json.loads(l)["page_id"] for l in open(C.require(C.SCAN_HOLDOUT, "词典扫描留出集"), encoding="utf-8")}
    root = C.require(C.SOURCE_ROOT, "原始资料根目录")
    out = []
    for l in open(C.require(C.HEAD_ANCHORS, "词典字头候选 heads.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        rm = r.get("reference_match") or {}
        chars = rm.get("character_candidates") or []
        if rm.get("status") != "reference_consistent_candidate" or len(chars) != 1 or r["page_id"] in hold:
            continue
        g = cv2.imdecode(np.fromfile(str(root / r["head_crop"]["path"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            out.append((g, chars[0], r["page_id"]))
    DICT_HEADS_PKL.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(DICT_HEADS_PKL, "wb"))
    return out


class DictHeads:
    def __init__(self, charset, seed=0):
        cs = set(charset)
        self.items = [(g, c) for g, c, _ in pickle.load(open(DICT_HEADS_PKL, "rb")) if c in cs]
        self.r = random.Random(seed)

    def sample(self):
        r = self.r
        k = 1 if r.random() < 0.3 else r.randint(2, 6)
        picks = [r.choice(self.items) for _ in range(k)]
        H = int(np.median([g.shape[0] for g, _ in picks]))
        parts, lab = [], ""
        for g, c in picks:
            s = H / g.shape[0]
            parts.append(cv2.resize(g, (max(4, round(g.shape[1] * s)), H), interpolation=cv2.INTER_AREA))
            if r.random() < 0.5:                                     # 字间距
                parts.append(np.full((H, max(1, round(H * r.uniform(0.0, 0.3)))), 255, np.uint8))
            lab += c
        g = np.concatenate(parts, 1)
        if r.random() < 0.5:                                         # 再降一点分辨率
            s = r.uniform(0.5, 1.0)
            small = cv2.resize(g, (max(8, round(g.shape[1] * s)), max(8, round(H * s))), interpolation=cv2.INTER_AREA)
            g = cv2.resize(small, (g.shape[1], H), interpolation=cv2.INTER_LINEAR)
        g = np.clip(g.astype(np.float32) + np.random.default_rng(r.getrandbits(32)).normal(0, r.uniform(0, 10), g.shape),
                    0, 255).astype(np.uint8)
        x, w = to_input(ink_map(g), r.uniform(56, 72))
        return x, w, lab, "dict_head"



class RealLines:
    """run/build_reallines.py 生成的真实扫描行（伪标签）。纯汉文行与含西夏字的行各抽一半。"""
    def __init__(self, charset, seed=0, path=C.DATA / "real" / "lines.jsonl"):
        cs = set(charset)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if all(c in cs for c in r["text"])]
        self.by_kind = {k: [r for r in rows if r["kind"] == k] for k in ("han", "mixed")}
        self.r = random.Random(seed)

    def sample(self):
        r = self.r
        while True:
            row = r.choice(self.by_kind["han" if r.random() < 0.5 else "mixed"])
            g = cv2.imdecode(np.fromfile(str(C.DATA / "real" / "lines" / row["path"]), np.uint8), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            if r.random() < 0.4:
                s = r.uniform(0.6, 1.0)
                h, w = g.shape
                g = cv2.resize(cv2.resize(g, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA),
                               (w, h), interpolation=cv2.INTER_LINEAR)
            x, w = to_input(ink_map(g), r.uniform(56, 72))
            if w // 8 < 2 * len(row["text"]) + 2:
                continue
            return x, w, row["text"], "real_" + row["kind"]


if __name__ == "__main__":
    out = build_dict_heads()
    print(f"{len(out)} 个字头，{len({c for _, c, _ in out})} 个不同字，{len({p for *_, p in out})} 页")
