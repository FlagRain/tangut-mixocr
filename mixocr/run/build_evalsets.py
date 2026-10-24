"""建评测集，预处理成模型输入后存到 data/eval/<name>.pkl。每条：(96×W 数组, 宽度, 参考字串, 计分掩码或 None)。

jlyx_tg   《经律异相》整理研究真实扫描页的西夏文行（mixbias 的 realtest：42 行，只在模板匹配确认的 874 个西夏字上计分）
lecture   讲义 dev 67 行（电子排版，全文真值，含拉丁、拟音、标点）
jlyx_han  同一本书测试页（第 50–115 页）的纯汉文行
synth     合成验证行 1,500 条，汉字用训练不用的华文中宋，西夏三套字体轮换，带扫描退化
dict_head 词典扫描留出集里的 208 个西夏字头（1–8 册），单字裁图；真值由印刷的 K/L/S 编号对到拟音表，

  python run/build_evalsets.py                 # 全部
  python run/build_evalsets.py --only dict_head
"""
import argparse
import json
import pickle
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from core import config as C
from core import charset as CS
from core.model import greedy_decode, load
from core.preprocess import ink_map, to_input, batch
from core.synth import Synth, is_tg

OUT = C.DATA / "eval"


def prep(gray):
    return to_input(ink_map(gray))


def norm_ref(s):
    """讲义 PDF 文字层里有康熙部首等兼容码位（⽂ U+2F42 之类），映射到统一汉字；其余不动。"""
    return "".join(unicodedata.normalize("NFKC", ch) if 0x2E80 <= ord(ch) <= 0x2FDF else ch for ch in s)


def jlyx_tg():
    C.require(C.JLYX_REALTEST, "真实书页测试集（MIXOCR_JLYX_REALTEST）")
    items = []
    for l in open(C.JLYX_REALTEST / "lines.jsonl", encoding="utf-8"):
        l = json.loads(l)
        if l["verified"] and l["gt"]:
            x, w = prep(cv2.imread(str(C.JLYX_REALTEST / "crops" / l["crop"]), cv2.IMREAD_GRAYSCALE))
            items.append((x, w, l["gt"], l["mask"]))
    print("jlyx_tg", len(items), sum(m.count("1") for *_, m in items))
    return items


def lecture():
    C.require(C.LECTURE_DEV, "讲义 dev 行图（MIXOCR_LECTURE_DEV）")
    items = []
    for l in open(C.LECTURE_DEV / "predictions_cuda.jsonl", encoding="utf-8"):     # 这里有 行号 → 行图 的对应
        r = json.loads(l)
        p = C.LECTURE_DEV / "images" / Path(r["image_path"].replace("\\", "/")).name
        x, w = prep(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        items.append((x, w, norm_ref(r["expected"]), None))
    print("lecture", len(items))
    return items


def jlyx_han():
    C.require(C.BOOK_OCR, "扫描书逐页 OCR 结果（MIXOCR_BOOK_OCR）")
    m, cs8 = load(C.CODEX_V8)
    m = m.cuda().eval()
    cand = []
    for pno in range(C.JLYX_TEST_PAGES[0], C.JLYX_TEST_PAGES[1] + 1):
        p = C.BOOK_OCR / "jlyx" / f"p{pno:04d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        page = cv2.imread(str(C.BOOK_OCR / "jlyx" / "page_images" / f"p{pno:04d}.png"), cv2.IMREAD_GRAYSCALE)
        for ln in d["lines"]:
            s = "".join(ln["text"].split())
            n_han = sum(0x3400 <= ord(c) <= 0x9FFF for c in s)
            if ln["score"] < 0.98 or len(s) < 6 or n_han < 0.6 * len(s):
                continue
            b = np.asarray(ln["box"])
            x0, y0 = b.min(0).astype(int)
            x1, y1 = b.max(0).astype(int)
            mg = max(2, (y1 - y0) // 8)
            cand.append((page[max(0, y0 - mg): y1 + mg, max(0, x0 - mg): x1 + mg], s))
    X = [prep(c) for c, _ in cand]
    keep = []
    for i in range(0, len(X), 32):
        t, ws = batch([x for x, _ in X[i: i + 32]])
        with torch.no_grad():
            lg, ln = m(t.cuda(), ws)
        for j, h in enumerate(greedy_decode(lg, ln, cs8)):
            if not any(map(is_tg, h)):
                keep.append(i + j)
    random.Random(0).shuffle(keep)
    items = [(X[k][0], X[k][1], cand[k][1], None) for k in sorted(keep[:400])]
    print("jlyx_han", len(items), "of", len(cand), "candidates;", sum(len(it[2]) for it in items), "chars")
    return items


def synth():
    syn = Synth(CS.load()[1:], seed=12345)
    fonts = [{"tg": tg, "han": C.HAN_VAL_FONT, "lat": "lat_times"} for tg in C.TANGUT_FONTS]
    items = []
    for i in range(1500):
        syn.fonts = fonts[i % 3]
        x, w, lab, t = syn.sample()
        items.append((x, w, lab, t))                 # 第 4 项存版式名
    print("synth", len(items))
    return items


def dict_head():
    C.require(C.SCAN_HOLDOUT, "词典扫描留出集（MIXOCR_SCAN_HOLDOUT）")
    items, pages = [], {}
    for l in open(C.SCAN_HOLDOUT, encoding="utf-8"):
        r = json.loads(l)
        if r.get("script_label_candidate") != "txg" or not r.get("unicode_transcription"):
            continue
        p = r["source_image_path"]
        if p not in pages:
            pages[p] = cv2.imread(str(C.require(C.SOURCE_ROOT, "原始资料根目录") / p), cv2.IMREAD_GRAYSCALE)
        x0, y0, x1, y1 = r["bbox_image_px"]
        mg = max(3, (y1 - y0) // 6)
        crop = pages[p][max(0, y0 - mg): y1 + mg, max(0, x0 - mg): x1 + mg]
        x, w = prep(crop)
        items.append((x, w, r["unicode_transcription"], None))
    print("dict_head", len(items))
    return items


BUILDERS = {"jlyx_tg": jlyx_tg, "lecture": lecture, "jlyx_han": jlyx_han, "synth": synth, "dict_head": dict_head}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        if not args.only or name in args.only.split(","):
            try:
                pickle.dump(fn(), open(OUT / f"{name}.pkl", "wb"))
            except SystemExit as e:                      # 缺原始资料的评测集跳过
                print(f"跳过 {name}：{e}")
