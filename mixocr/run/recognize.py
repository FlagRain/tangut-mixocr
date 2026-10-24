"""用 PyTorch 检查点识别夏汉混排文字（看训练中的 runs/<name>/best.pt 效果用；日常使用请用仓库根目录的 ocr.py）。

  # 已经切好的行图（一张图一行）
  python run/recognize.py line1.png line2.png
  # 整页：先用 PP-OCR 检测模型（weights/det/，ONNX）找行框，再逐行识别
  python run/recognize.py --page page.png --out page.json
  # PDF 的某几页（扫描 PDF 取页内嵌图像，电子 PDF 按 300 dpi 渲染）
  python run/recognize.py --pdf book.pdf --pages 120-122 --out book.jsonl

默认模型 weights/mixocr-v1.pt；--model 可换成 runs/<name>/best.pt。
输出每行的框（整页时）和文字；不做语言模型校正。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from core import config as C
from core.model import greedy_decode, load
from core.preprocess import batch, ink_map, to_input

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="*")
ap.add_argument("--page", nargs="*", default=[])
ap.add_argument("--pdf")
ap.add_argument("--pages", default="")
ap.add_argument("--model", default=str(C.WEIGHTS / "mixocr-v1.pt"))
ap.add_argument("--out")
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()

dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
model, charset = load(args.model)
model = model.to(dev).eval()


@torch.no_grad()
def read_lines(grays, bs=24):
    X = [to_input(ink_map(g)) for g in grays]
    order = sorted(range(len(X)), key=lambda k: X[k][1])
    out = [""] * len(X)
    for a in range(0, len(order), bs):
        idx = order[a: a + bs]
        t, ws = batch([X[k][0] for k in idx])
        lg, ln = model(t.to(dev), ws)
        for k, h in zip(idx, greedy_decode(lg.float(), ln, charset)):
            out[k] = h
    return out


_det = None


def detect(gray):
    """行框，按阅读顺序；与网站和 ocr.py 用同一个检测器（web/app/engine.py）。"""
    global _det
    if _det is None:
        sys.path.insert(0, str(C.REPO / "web"))
        from app.engine import Detector
        _det = Detector(C.WEIGHTS / "det" / "PP-OCRv6_det_small.onnx")
    return _det(gray)


def read_page(gray):
    boxes = detect(gray)
    crops = []
    for x0, y0, x1, y1 in boxes:
        m = max(2, (y1 - y0) // 8)
        crops.append(gray[max(0, y0 - m): y1 + m, max(0, x0 - m): x1 + m])
    texts = read_lines(crops) if crops else []
    return [{"box": b, "text": t} for b, t in zip(boxes, texts)]


def pdf_pages(path, spec):
    import pymupdf
    doc = pymupdf.open(path)
    a, b = (map(int, spec.split("-")) if "-" in spec else (int(spec), int(spec))) if spec else (1, len(doc))
    for pno in range(a, b + 1):
        page = doc[pno - 1]
        imgs = page.get_images()
        if len(imgs) == 1 and not page.get_text().strip():          # 扫描页：直接取原图
            pix = pymupdf.Pixmap(doc, imgs[0][0])
        else:
            pix = page.get_pixmap(dpi=300)
        if pix.n > 1:
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        yield pno, np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


results = []
if args.images:
    texts = read_lines([cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE) for p in args.images])
    for p, t in zip(args.images, texts):
        results.append({"image": p, "text": t})
        print(t)
for p in args.page:
    lines = read_page(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE))
    results.append({"image": p, "lines": lines})
    print("\n".join(l["text"] for l in lines))
if args.pdf:
    for pno, gray in pdf_pages(args.pdf, args.pages):
        lines = read_page(gray)
        results.append({"pdf": args.pdf, "page": pno, "lines": lines})
        print(f"--- 第 {pno} 页")
        print("\n".join(l["text"] for l in lines))
if args.out:
    with open(args.out, "w", encoding="utf-8") as fh:
        if args.out.endswith(".jsonl"):
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            json.dump(results, fh, ensure_ascii=False, indent=1)
