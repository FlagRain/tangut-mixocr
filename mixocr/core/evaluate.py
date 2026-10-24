"""在 data/eval/ 的各评测集上跑模型并计分。"""
import pickle
from collections import Counter

import numpy as np
import torch

from . import config as C
from .model import greedy_decode
from .preprocess import batch
from .synth import is_tg, is_han

SETS = ("jlyx_tg", "jlyx_han", "lecture", "synth", "dict_head")


def script(ch):
    return "tg" if is_tg(ch) else ("han" if is_han(ch) else "other")


def align_counts(ref, hyp, mask=None):
    """编辑距离对齐。返回 Counter：按参考字的文字分 {s}_n/{s}_ok/{s}_sub/{s}_x（错成另一种文字）/{s}_del，以及 ins。"""
    mask = mask or "1" * len(ref)
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        r = ref[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (r != hyp[j - 1]))
    c = Counter()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if mask[i - 1] == "1":
                s, h = script(ref[i - 1]), hyp[j - 1]
                c[s + "_n"] += 1
                if h == ref[i - 1]:
                    c[s + "_ok"] += 1
                elif script(h) == s:
                    c[s + "_sub"] += 1
                else:
                    c[s + "_x"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            if mask[i - 1] == "1":
                c[script(ref[i - 1]) + "_n"] += 1
                c[script(ref[i - 1]) + "_del"] += 1
            i -= 1
        else:
            c["ins"] += 1
            j -= 1
    return c


def load_sets(names=SETS):
    paths = {n: C.DATA / "eval" / f"{n}.pkl" for n in names}
    return {n: pickle.load(open(p, "rb")) for n, p in paths.items() if p.exists()}


@torch.no_grad()
def predict(model, charset, items, device="cuda", cols=24000):
    model.eval()
    order = sorted(range(len(items)), key=lambda k: items[k][1])
    out = [None] * len(items)
    groups, cur = [], []
    for k in order:
        W = items[k][1] + (-items[k][1]) % 8
        if cur and (len(cur) + 1) * W > cols:
            groups.append(cur)
            cur = []
        cur.append(k)
    groups.append(cur)
    for idx in groups:
        t, ws = batch([items[k][0][:, : items[k][1] + (-items[k][1]) % 8] for k in idx])
        lg, ln = model(t.to(device), ws)
        for k, h in zip(idx, greedy_decode(lg, ln, charset)):
            out[k] = h
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def score_set(name, items, hyps):
    tot = Counter()
    for (x, w, ref, extra), hyp in zip(items, hyps):
        if name in ("jlyx_tg", "dict_head"):        # 与 mixbias 相同：只比西夏字和汉字，只在确认位置计分
            hyp = "".join(ch for ch in hyp if is_tg(ch) or is_han(ch))
            tot += align_counts(ref, hyp, extra)
        else:
            tot += align_counts("".join(ref.split()), "".join(hyp.split()))
    res = {}
    for s in ("tg", "han", "other"):
        n = tot[s + "_n"]
        if n:
            res[f"{s}_acc"] = tot[s + "_ok"] / n
            res[f"{s}_cross"] = tot[s + "_x"] / n
            res[f"{s}_n"] = n
    n_all = sum(tot[s + "_n"] for s in ("tg", "han", "other"))
    res["cer"] = (n_all - sum(tot[s + "_ok"] for s in ("tg", "han", "other")) + tot["ins"]) / max(1, n_all)
    return res


def evaluate(model, charset, sets, device="cuda"):
    out = {}
    for name, items in sets.items():
        out[name] = score_set(name, items, predict(model, charset, items, device))
    return out


def headline(r):
    """训练中挑模型用的单一指标：真实书页西夏字正确率与汉字一致率的平均；没有书页评测集时退回合成验证集。"""
    if "jlyx_tg" in r and "jlyx_han" in r:
        return 0.5 * r["jlyx_tg"]["tg_acc"] + 0.5 * r["jlyx_han"]["han_acc"]
    if "synth" in r:
        return 0.5 * r["synth"].get("tg_acc", 0) + 0.5 * r["synth"].get("han_acc", 0)
    return 0.0
