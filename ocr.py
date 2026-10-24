"""命令行识别夏汉混排文字。只用 ONNX Runtime，不需要 PyTorch，也不需要显卡。

  python ocr.py 行图1.png 行图2.png               # 已经切好的行图，一张图一行
  python ocr.py --page 整页.png --out 结果.json    # 整页：先找行框，再逐行识别
  python ocr.py --pdf 书.pdf --pages 120-121 --out 结果.jsonl

依赖见 requirements.txt。权重在 weights/：识别模型 mixocr-v1.onnx 与字表 charset.json，
行检测用 PaddleOCR 的 PP-OCRv6 small（weights/det/）。输出没有语言模型校正。
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "web"))
from app.engine import Engine          # noqa: E402  网站和命令行共用同一个识别引擎

W = REPO / "weights"


def pdf_pages(path, spec):
    import pymupdf
    with pymupdf.open(path) as doc:
        a, b = (map(int, spec.split("-")) if "-" in spec else (int(spec), int(spec))) if spec else (1, doc.page_count)
        for pno in range(a, b + 1):
            page = doc[pno - 1]
            imgs = page.get_images()
            if len(imgs) == 1 and page.rotation == 0 and not page.get_text().strip():
                pix = pymupdf.Pixmap(doc, imgs[0][0])          # 扫描页：直接取内嵌原图
            else:
                pix = page.get_pixmap(dpi=300)
            if pix.n - pix.alpha != 1:
                pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            yield pno, np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def imread(p):
    g = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        sys.exit(f"读不了图片：{p}")
    return g


def main():
    ap = argparse.ArgumentParser(description="夏汉混排 OCR（mixocr-v1）")
    ap.add_argument("images", nargs="*", help="行图（一张图一行）")
    ap.add_argument("--page", nargs="*", default=[], help="整页图片")
    ap.add_argument("--pdf")
    ap.add_argument("--pages", default="", help="PDF 页码范围，如 3-5；默认全部")
    ap.add_argument("--out", help="输出 .json 或 .jsonl（含行框、每个字的大致位置和候选）")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if not (args.images or args.page or args.pdf):
        ap.print_help()
        return
    eng = Engine(det=W / "det" / "PP-OCRv6_det_small.onnx", rec=W / "mixocr-v1.onnx", charset=W / "charset.json",
                 threads=args.threads)
    results = []
    for p in args.images:
        text, chars = eng.read_line(imread(p))
        results.append({"image": p, "text": text})
        print(text)
    for p in args.page:
        res = eng.page(imread(p))
        results.append({"image": p, **res})
        print("\n".join(l["text"] for l in res["lines"]))
    if args.pdf:
        for pno, gray in pdf_pages(args.pdf, args.pages):
            res = eng.page(gray)
            results.append({"pdf": args.pdf, "page": pno, **res})
            print(f"--- 第 {pno} 页")
            print("\n".join(l["text"] for l in res["lines"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            if args.out.endswith(".jsonl"):
                fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
            else:
                json.dump(results, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
