"""整页识别：PP-OCR 检测模型找行框 → mixocr 模型逐行识别。只用 onnxruntime，不需要 PyTorch。

models/ 目录里应有：
  det.onnx          PP-OCRv6 small 文本检测模型（PaddleOCR，Apache-2.0；取自 RapidOCR）
  rec.onnx          mixocr-v1（mixocr/run/export_onnx.py 导出；输出每帧前 5 名类别与概率）
  charset.json      rec 的字表，下标 0 是 CTC 空白
  model_info.json   版本、评测数字（显示在网页“说明”里）
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

from .preprocess import ink_map, to_input

STRIDE = 8          # 识别模型每一帧对应输入图的 8 个像素宽
DET_MIN_SIDE = 736  # 检测输入短边至少这么大（与 RapidOCR 默认一致）
DET_MAX_SIDE = 2048  # 检测输入长边上限；更大的页面先缩小检测，再到原图上裁行
LOW_CONF = 0.6      # 低于这个概率的字在界面上标黄


def _session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False          # 行宽每次不同，内存池会越涨越大
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class Detector:
    """PP-OCR 文本检测（DB）：灰度页 → 行框 [x0, y0, x1, y1]（原图坐标），按阅读顺序排列。"""
    def __init__(self, path, threads=2):
        self.sess = _session(path, threads)

    def __call__(self, gray):
        h, w = gray.shape
        r = DET_MIN_SIDE / min(h, w) if min(h, w) < DET_MIN_SIDE else 1.0
        r = min(r, DET_MAX_SIDE / max(h, w))
        rh, rw = max(32, int(round(h * r / 32)) * 32), max(32, int(round(w * r / 32)) * 32)
        img = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        x = np.repeat(x[None, None], 3, axis=1)
        prob = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0, 0]
        boxes = _db_boxes(prob, w / rw, h / rh)
        boxes = [b for b in boxes if b[3] - b[1] >= 8 and b[2] - b[0] >= 8]
        return _reading_order(boxes, w)


class Engine:
    """model_dir 里放 det.onnx / rec.onnx / charset.json / model_info.json；也可以逐个文件指定。"""
    def __init__(self, model_dir=None, threads=2, det=None, rec=None, charset=None, info=None):
        d = Path(model_dir) if model_dir else None
        self.detect = Detector(det or d / "det.onnx", threads)
        self.rec = _session(rec or d / "rec.onnx", threads)
        self.charset = json.loads(Path(charset or d / "charset.json").read_text(encoding="utf-8"))
        info = Path(info) if info else (d / "model_info.json" if d else None)
        self.info = json.loads(info.read_text(encoding="utf-8")) if info and info.exists() else {}

    # ------------------------------------------------------------ 识别一行
    def read_line(self, crop):
        """crop: 行图（灰度）。返回 (文字, 每个字的 [字, 起点 x, 终点 x, 概率, 候选])，x 为行图坐标。"""
        x, w = to_input(ink_map(crop))
        idx, prob = self.rec.run(None, {"image": x[None, None]})
        scale = w / crop.shape[1]
        cs = self.charset
        runs = []                                     # 每个发射的字：[类别, 首帧, 末帧, 峰值帧]
        prev = 0
        for t, k in enumerate(idx[:, 0]):
            k = int(k)
            if k and k == prev:
                runs[-1][2] = t
                if prob[t, 0] > prob[runs[-1][3], 0]:
                    runs[-1][3] = t
            elif k:
                runs.append([k, t, t, t])
            prev = k
        chars = []
        for i, (k, a, b, peak) in enumerate(runs):
            # 字的左右边界取相邻两字中心的中点（CTC 只给出大致位置）
            c = (a + b + 1) / 2 * STRIDE / scale
            left = (runs[i - 1][1] + runs[i - 1][2] + 1) / 2 * STRIDE / scale if i else None
            right = (runs[i + 1][1] + runs[i + 1][2] + 1) / 2 * STRIDE / scale if i + 1 < len(runs) else None
            half = ((right - c) if right else (c - left) if left else 32 / scale) / 2
            x0 = (left + c) / 2 if left is not None else c - half
            x1 = (c + right) / 2 if right is not None else c + half
            alts = [[cs[int(j)], round(float(p), 3)] for j, p in zip(idx[peak], prob[peak])
                    if int(j) and int(j) != k][:4]
            chars.append([cs[k], max(0.0, x0), min(float(crop.shape[1]), x1), round(float(prob[peak, 0]), 3), alts])
        return "".join(c[0] for c in chars), chars

    # ------------------------------------------------------------ 整页
    def page(self, gray):
        t0 = time.time()
        boxes = self.detect(gray)
        H, W = gray.shape
        lines = []
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            if y1 - y0 > 1.6 * (x1 - x0) and y1 - y0 > 60:
                lines.append({"id": i, "box": [x0, y0, x1, y1], "text": "", "chars": [], "vertical": True})
                continue
            m = max(2, (y1 - y0) // 8)
            cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
            crop = gray[cy0: min(H, y1 + m), cx0: min(W, x1 + m)]
            text, chars = self.read_line(crop)
            lines.append({"id": i, "box": [x0, y0, x1, y1], "text": text, "vertical": False,
                          "chars": [{"c": c, "x0": round(cx0 + a), "x1": round(cx0 + b), "p": p, "alt": alts}
                                    for c, a, b, p, alts in chars]})
        return {"width": W, "height": H, "lines": lines, "model": self.info.get("name", "mixocr"),
                "seconds": round(time.time() - t0, 2)}


def _db_boxes(prob, sx, sy, thresh=0.3, box_thresh=0.5, unclip_ratio=1.6):
    """DB 文本检测的后处理（与 PaddleOCR / RapidOCR 相同的参数），返回原图坐标的外接矩形。"""
    mask = cv2.dilate((prob > thresh).astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours[:1000]:
        rect = cv2.minAreaRect(c)
        if min(rect[1]) < 3:
            continue
        pts = cv2.boxPoints(rect)
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(prob.shape[1] - 1, x1), min(prob.shape[0] - 1, y1)
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(m, [(pts - [x0, y0]).astype(np.int32)], 1)
        if cv2.mean(prob[y0: y1 + 1, x0: x1 + 1], m)[0] < box_thresh:
            continue
        area, length = cv2.contourArea(pts), cv2.arcLength(pts, True)
        if length == 0:
            continue
        off = pyclipper.PyclipperOffset()
        off.AddPath(pts.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        exp = off.Execute(area * unclip_ratio / length)
        if not exp:
            continue
        exp = np.array(exp[0], np.float32)
        r2 = cv2.minAreaRect(exp)
        if min(r2[1]) < 5:
            continue
        p2 = cv2.boxPoints(r2)
        bx0, by0 = p2.min(0)
        bx1, by1 = p2.max(0)
        out.append([int(max(0, bx0 * sx)), int(max(0, by0 * sy)), int(bx1 * sx), int(by1 * sy)])
    return out


def _reading_order(boxes, W):
    """阅读顺序。若页面中间有一条几乎没有行框跨过的竖向空白（最多容许少数标题之类跨栏）、
    两侧各有 3 行以上，按双栏处理：跨栏的行把页面分成几段，每段先左栏后右栏；否则按单栏排。"""
    cover = np.zeros(W + 2, np.int32)
    for b in boxes:
        cover[b[0]: b[2] + 1] += 1
    lo, hi = int(0.3 * W), int(0.7 * W)
    if hi > lo:
        m = cover[lo:hi].min()
        xs = np.flatnonzero(cover[lo:hi] == m) + lo
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        best = max(runs, key=len)
    if hi > lo and m <= max(1, 0.05 * len(boxes)):
        g = (best[0] + best[-1]) / 2
        left = [b for b in boxes if b[2] <= g]
        right = [b for b in boxes if b[0] >= g]
        if len(left) >= 3 and len(right) >= 3:
            span = sorted((b for b in boxes if b[0] < g < b[2]), key=lambda b: b[1])
            out, top = [], -1
            for s in span + [None]:
                bottom = s[1] if s else float("inf")
                for col in (left, right):
                    out += _rows([b for b in col if top <= (b[1] + b[3]) / 2 < bottom])
                if s:
                    out.append(s)
                    top = (s[1] + s[3]) / 2
            return out
    return _rows(boxes)


def _rows(boxes):
    """先上后下；纵向重叠过半的框算同一行，行内从左到右。"""
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        r = rows[-1] if rows else None
        if r and min(b[3], r[1]) - max(b[1], r[0]) > 0.5 * min(b[3] - b[1], r[1] - r[0]):
            r[2].append(b)
            r[0], r[1] = min(r[0], b[1]), max(r[1], b[3])
        else:
            rows.append([b[1], b[3], [b]])
    return [b for r in rows for b in sorted(r[2], key=lambda b: b[0])]
