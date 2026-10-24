"""从真实扫描页取行，生成伪标签，供第二阶段训练用。输出 data/real/lines/<来源>/*.png 与 data/real/lines.jsonl。

来源（测试页一律排除）：
  jlyx  《经律异相》整理研究除第 50–115 页以外的页
  dxj   《西夏文德行集研究》的横排行（书中竖排的西夏文不取）
  dict  九册词典的扫描页（抽样；scan_holdout_v2 的 199 页排除）
行框和汉字参考都来自已有的 PP-OCRv5 结果（mixbias/book_ocr 与 ocr/dictionary_digitalization/chinese_raw）。

伪标签规则：
  - 模型读不出西夏字、PP-OCRv5 置信度 ≥0.97 的行：标签取 PP-OCRv5 的输出（汉字它比我们准）。
    另要求两者编辑距离不超过 30%，排除行框裁错的情况。
  - 模型读出西夏字的行：把模型输出和 PP-OCRv5 输出对齐，模型输出里非西夏字的部分至少 90% 与 PP-OCRv5 一致，
    且每个西夏字的置信度（发射帧的最大后验）≥0.6、平均 ≥0.9，才采用。标签以模型输出为准，
    其中与 PP-OCRv5 不同的非西夏字换成 PP-OCRv5 的字。
西夏字部分是模型自己的输出（自训练），会把模型已有的少量错误带进来；所以只取高置信度的行，比例也压低。

  python run/build_reallines.py --model runs/v1/best.pt --dict-pages 800
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from core import config as C
from core.model import load
from core.preprocess import batch, ink_map, to_input
from core.synth import is_han, is_tg

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dict-pages", type=int, default=800)
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-lines", type=int, default=0, help="只处理前若干行（试跑用）")
ap.add_argument("--out", default="lines.jsonl")
args = ap.parse_args()

OUT = C.DATA / "real" / "lines"
OUT.mkdir(parents=True, exist_ok=True)
model, charset = load(args.model)
model = model.to(args.device).eval()
cs_set = set(charset)


def crop(page, x0, y0, x1, y1):
    m = max(2, (y1 - y0) // 8)
    return page[max(0, y0 - m): y1 + m, max(0, x0 - m): x1 + m]


def book_lines():
    if not C.BOOK_OCR:
        return
    for tag in ("jlyx", "dxj"):
        for p in sorted((C.BOOK_OCR / tag).glob("p*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if tag == "jlyx" and C.JLYX_TEST_PAGES[0] <= d["page"] <= C.JLYX_TEST_PAGES[1]:
                continue
            img = C.BOOK_OCR / tag / "page_images" / f"p{d['page']:04d}.png"
            if not img.exists():
                continue
            page = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            for i, ln in enumerate(d["lines"]):
                b = np.asarray(ln["box"])
                x0, y0 = b.min(0).astype(int)
                x1, y1 = b.max(0).astype(int)
                if x1 - x0 < 2.5 * (y1 - y0):                     # 竖排或过短的块不取
                    continue
                yield f"{tag}_p{d['page']:04d}_l{i:03d}", tag, crop(page, x0, y0, x1, y1), ln["text"], ln["score"]


def dict_lines():
    if not (C.DICT_RAW_PAGES and C.SCAN_HOLDOUT):
        return
    hold = {json.loads(l)["page_id"].replace(":", "-") for l in open(C.SCAN_HOLDOUT, encoding="utf-8")}
    files = sorted(C.DICT_RAW_PAGES.glob("*/*.json"))
    random.Random(0).shuffle(files)
    seen, n = set(), 0
    for p in files:
        if p.stem in seen or p.stem in hold:
            continue
        seen.add(p.stem)
        d = json.loads(p.read_text(encoding="utf-8"))
        img = C.require(C.SOURCE_ROOT, "原始资料根目录") / d["native_image_path"]
        if not img.exists():
            continue
        page = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        for i, ln in enumerate(d.get("lines", [])):
            b = np.asarray(ln["polygon_image_px"])
            x0, y0 = b.min(0).astype(int)
            x1, y1 = b.max(0).astype(int)
            if x1 - x0 < 2.0 * (y1 - y0):
                continue
            yield f"dict_{p.stem}_l{i:03d}", "dict", crop(page, x0, y0, x1, y1), ln.get("raw_ocr_candidate") or "", \
                ln.get("recognition_confidence") or 0.0
        n += 1
        if n >= args.dict_pages:
            break


@torch.no_grad()
def read(grays):
    """返回 [(文字, 每个字的置信度)]。置信度 = 该字发射的那段帧里，该字后验的最大值。"""
    X = [to_input(ink_map(g)) for g in grays]
    t, ws = batch([x for x, _ in X])
    lg, ln = model(t.to(args.device), ws)
    prob = lg.float().softmax(-1).cpu().numpy()
    out = []
    for b, n in enumerate(ln.tolist()):
        best = prob[:n, b].argmax(-1)
        s, conf, prev = [], [], 0
        for f, k in enumerate(best):
            if k and k != prev:
                s.append(charset[k])
                conf.append(prob[f, b, k])
            elif k and k == prev:
                conf[-1] = max(conf[-1], prob[f, b, k])
            prev = k
        out.append(("".join(s), conf))
    return out


def align(a, b):
    """编辑距离对齐。返回完整路径 [(i, j)]：i 或 j 为 None 表示只在一边出现；以及距离。"""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    path, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    return path[::-1], int(D[n, m])


def label(hyp, conf, pp, score):
    hyp_ns = "".join(hyp.split())
    pp = "".join(ch for ch in "".join(pp.split()) if ch in cs_set)
    n_tg = sum(map(is_tg, hyp_ns))
    if not hyp_ns or not pp:
        return None, ""
    path, dist = align(hyp_ns, pp)
    if not n_tg:
        if score >= 0.97 and dist <= 0.3 * max(len(pp), len(hyp_ns)) and sum(map(is_han, pp)) >= 0.5 * len(pp):
            return pp, "han"
        return None, ""
    c = [float(x) for x, ch in zip(conf, hyp) if is_tg(ch)]          # conf 与含空格的 hyp 逐字对应
    if not c or min(c) < 0.6 or np.mean(c) < 0.9:
        return None, ""
    # 沿对齐路径拼标签：西夏字取模型的；非西夏字两边不同时取 PP-OCRv5 的；PP-OCRv5 多出的字，
    # 只有前后都不挨着西夏字时才补进来（挨着西夏字的多半是它把西夏字读成的形近汉字）
    tg_at = [is_tg(hyp_ns[i]) if i is not None else None for i, _ in path]
    lab, disagree = [], 0
    for k, (i, j) in enumerate(path):
        if i is not None and is_tg(hyp_ns[i]):
            lab.append(hyp_ns[i])
        elif i is not None:
            if j is not None and pp[j] != hyp_ns[i]:
                disagree += 1
                lab.append(pp[j] if not is_tg(pp[j]) else hyp_ns[i])
            else:
                disagree += j is None
                lab.append(hyp_ns[i])
        else:
            near_tg = any(tg_at[q] for q in (k - 1, k + 1) if 0 <= q < len(path) and tg_at[q] is not None)
            if not near_tg:
                disagree += 1
                lab.append(pp[j])
    n_other = len(hyp_ns) - n_tg
    if n_other and disagree > 0.1 * n_other + 0.5:
        return None, ""
    return "".join(lab), "mixed"


def main():
    rows, buf = [], []
    stats = {}

    def flush():
        res = read([g for _, _, g, _, _ in buf])
        for (lid, src, g, pp, score), (hyp, conf) in zip(buf, res):
            lab, kind = label(hyp, conf, pp, score)
            st = stats.setdefault(src, {"lines": 0, "han": 0, "mixed": 0})
            st["lines"] += 1
            if lab:
                st[kind] += 1
                d = OUT / src
                d.mkdir(exist_ok=True)
                cv2.imencode(".png", g)[1].tofile(str(d / f"{lid}.png"))
                rows.append({"id": lid, "source": src, "kind": kind, "path": f"{src}/{lid}.png", "text": lab,
                             "model": hyp, "ppocr": pp, "ppocr_score": score})
        buf.clear()

    for gen in (book_lines(), dict_lines()):
        for k, item in enumerate(gen):
            if args.max_lines and k >= args.max_lines:
                break
            if item[2].size == 0 or min(item[2].shape) < 8:
                continue
            buf.append(item)
            if len(buf) >= 16:
                flush()
    if buf:
        flush()
    with open(C.DATA / "real" / args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
