"""行图 → 模型输入：暗底亮字，高 96，字身高约 TEXT_H 像素并垂直居中。

训练和推理走同一个函数，所以合成行、真实书页、讲义 PDF 行的字大小与位置一致。
"""
import cv2
import numpy as np

INPUT_HEIGHT = 96    # 模型输入高度（core/model.py 从这里取）。本文件不依赖 torch，部署的网站直接复制使用

TEXT_H = 64          # 字身（行中墨迹的上下范围）缩放到的高度；v8 在真实书页上以 64 最好
MAX_W = 4096


def ink_map(gray):
    """灰度图（白底黑字或黑底白字均可）→ 0–1 的墨迹强度，背景为 0。"""
    g = gray.astype(np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if np.median(border) < 128:                     # 黑底白字，先反相
        g = 255.0 - g
    bg, ink = np.percentile(g, 75), np.percentile(g, 1)
    x = (bg - g) / max(8.0, bg - ink)
    return np.clip((x - 0.15) / 0.85, 0, 1)          # 纸面噪声和底色不算墨迹


def text_band(x):
    """行的主体墨迹带 (top, bottom)。按行投影找墨迹最多的连续带，忽略上下相邻行露进来的残笔。"""
    prof = (x > 0.5).sum(1).astype(np.float32)
    if prof.max() <= 0:
        return 0, x.shape[0]
    on = np.flatnonzero(prof > 0.06 * prof.max())
    gap = max(2, round(0.12 * x.shape[0]))
    runs, a = [], on[0]
    for p, q in zip(on[:-1], on[1:]):
        if q - p > gap:
            runs.append((a, p + 1))
            a = q
    runs.append((a, on[-1] + 1))
    return max(runs, key=lambda ab: prof[ab[0]: ab[1]].sum())


def to_input(x, text_h=TEXT_H, height=INPUT_HEIGHT):
    """x: 0–1 墨迹图。按主体墨迹带缩放到 text_h，垂直居中，宽度补到 8 的倍数。"""
    top, bot = text_band(x)
    s = text_h / max(4, bot - top)
    w = max(16, min(MAX_W, round(x.shape[1] * s)))
    h = max(1, round(x.shape[0] * s))
    y = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y = np.clip(y, 0, 1)
    out = np.zeros((height, w + (-w) % 8), np.float32)
    cy = round((top + bot) / 2 * s)
    src0 = cy - height // 2
    a, b = max(0, src0), min(h, src0 + height)
    out[a - src0: b - src0, :w] = y[a:b]
    return out, w


def batch(images):
    """一组 96×W 的数组 → 右补零的张量和真实宽度。"""
    import torch
    W = max(im.shape[1] for im in images)
    t = torch.zeros(len(images), 1, INPUT_HEIGHT, W)
    for i, im in enumerate(images):
        t[i, 0, :, : im.shape[1]] = torch.from_numpy(im)
    return t, [im.shape[1] for im in images]
