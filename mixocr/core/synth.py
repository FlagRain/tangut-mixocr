"""在线合成夏汉混排行：先按版式生成文字，再逐字排版，最后加扫描退化。

版式（权重见 TEMPLATES）：
  zh_tg     汉文正文中夹西夏字（可带引号、书名号、括号）
  tg_punct  西夏文录文行：西夏字 + 汉文标点 + 注码，偶尔带汉文标签或括注
  mixed     讲义 PDF 文字层里的真实混排行
  zh        纯汉文正文，带数字、页码、注码
  gloss     字典式释义：西夏字 + 拟音 + 汉字释义
  tg_rand   均匀随机的西夏字，保证 5,878 个字都有曝光
  short     只有 1–3 个字的短行（字头、页眉、孤立字）
每行固定一套西夏字体、一套汉字字体、一套拉丁字体，与书籍排版一致。
"""
import io
import json
import pickle
import random
import unicodedata
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C
from .preprocess import ink_map, to_input

BASE = 48            # 排版时的字号（像素）

TEMPLATES = {"zh_tg": 30, "tg_punct": 22, "mixed": 8, "zh": 15, "gloss": 10, "tg_rand": 15, "short": 5}
CJK_PUNCT_TG = "，。：；、，，。"
FOOTNOTE_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
SUP_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def is_tg(ch):
    return 0x17000 <= ord(ch) <= 0x18DFF


def is_han(ch):
    cp = ord(ch)
    return 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x323AF


def is_fullwidth(ch):
    return is_tg(ch) or is_han(ch) or 0x3000 <= ord(ch) <= 0x303F or 0xFF00 <= ord(ch) <= 0xFFEF \
        or ch in "“”‘’…—①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


# ---------------------------------------------------------------- 字体
def _cmaps():
    path = C.DATA / "font_cmaps.pkl"
    key = sorted((n, str(p), i) for n, (p, i, _) in C.FONTS.items())
    if path.exists():
        cached = pickle.loads(path.read_bytes())
        if isinstance(cached, dict) and cached.get("key") == key:
            return cached["cmaps"]
    from fontTools.ttLib import TTFont
    out = {}
    for name, (p, idx, _) in C.FONTS.items():
        f = TTFont(str(p), fontNumber=idx, lazy=True)
        out[name] = frozenset(chr(c) for c in f.getBestCmap())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"key": key, "cmaps": out}))
    return out


CMAPS = _cmaps()


@lru_cache(maxsize=64)
def font(name, px=BASE):
    p, idx, inst = C.FONTS[name]
    f = ImageFont.truetype(str(p), px, index=idx)
    if inst is not None:
        f.set_variation_by_name(inst)
    return f


@lru_cache(maxsize=60000)
def glyph(ch, name):
    """单字在 BASE 字号下的墨迹图（0–255，墨迹为高值）、相对基线的左上角偏移、步进宽度。"""
    f = font(name)
    adv = f.getlength(ch)
    w = BASE * 3
    img = Image.new("L", (w, w), 0)
    ImageDraw.Draw(img).text((BASE, 2 * BASE), ch, font=f, fill=255, anchor="ls")
    a = np.asarray(img)
    ys, xs = np.nonzero(a > 20)
    if len(ys) == 0:
        return None, 0, 0, adv
    a = a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return a, xs.min() - BASE, ys.min() - 2 * BASE, adv


def pick_font(ch, preferred, pool):
    if preferred in CMAPS and ch in CMAPS[preferred]:
        return preferred
    for n in pool:
        if n in CMAPS and ch in CMAPS[n]:
            return n
    return None


# ---------------------------------------------------------------- 文本
class TextSource:
    def __init__(self, charset):
        """文本库、字表 TSV、释义表都是可选的：缺了就用字表里的随机字代替，相应版式自动关闭。"""
        def lines(name):
            p = C.TEXTBANK / name
            return [l for l in p.read_text(encoding="utf-8").splitlines() if l] if p.exists() else []
        self.zh, self.tg, self.mixed = lines("zh.txt"), lines("tg.txt"), lines("mixed.txt")
        cs = set(charset)
        self.charset = cs
        self.tg_chars = [c for c in charset if is_tg(c)]
        self.han_chars = [c for c in charset if is_han(c)]
        self.tg_natural = "".join(ch for s in self.tg + self.mixed for ch in s if is_tg(ch))
        self.gloss = []
        if C.TANGUT_TSV and C.TANGUT_TSV.exists():
            for line in open(C.TANGUT_TSV, encoding="utf-8"):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 6 and len(p[1]) == 1 and is_tg(p[1]):
                    self.gloss.append((p[1], [x for x in (p[2], p[5]) if x]))
        self.lexicon = []
        if C.LEXICON and C.LEXICON.exists():
            for line in open(C.LEXICON, encoding="utf-8"):
                r = json.loads(line)
                if r.get("source_text") and r.get("target_text") and r.get("category", "lexicon") == "lexicon":
                    s, t = r["source_text"], r["target_text"]
                    if all(is_tg(c) for c in s) and len(t) <= 12:
                        self.lexicon.append((s, t))
        self.templates = {k: w for k, w in TEMPLATES.items()
                          if not (k == "mixed" and not self.mixed) and not (k == "gloss" and not self.gloss)}

    def zh_chunk(self, r, lo, hi):
        n = r.randint(lo, hi)
        if not self.zh:                                              # 没有文本库：随机常用字
            return "".join(r.choice(self.han_chars[:3500]) for _ in range(n))
        s = r.choice(self.zh)
        if len(s) > n:
            a = r.randint(0, len(s) - n)
            s = s[a: a + n]
        if r.random() < 0.15:                                        # 混入罕用字
            s = "".join(r.choice(self.han_chars) if is_han(c) and r.random() < 0.3 else c for c in s)
        return s

    def tg_run(self, r, lo, hi):
        n = r.randint(lo, hi)
        if r.random() < 0.5 and len(self.tg_natural) > n:
            a = r.randint(0, len(self.tg_natural) - n)
            return self.tg_natural[a: a + n]
        return "".join(r.choice(self.tg_chars) for _ in range(n))

    def footnote(self, r):
        k = r.random()
        n = r.randint(1, 40)
        if k < 0.45:
            return f"[{n}]", False
        if k < 0.7:
            return FOOTNOTE_CIRCLED[n % 20], r.random() < 0.6
        if k < 0.85:
            return "".join(SUP_DIGITS[int(d)] for d in str(n)), True
        return str(n), True                                             # 上标形式的普通数字

    # 返回 [(text, kind)]，kind ∈ tg / han / lat / sup（上标小字）
    def make(self, r):
        t = r.choices(list(self.templates), weights=list(self.templates.values()))[0]
        segs = getattr(self, "_" + t)(r)
        out = []
        for s, k in segs:
            s = "".join(ch for ch in s if ch in self.charset or ch == " ")
            if s:
                out.append((s, k))
        return out, t

    def _wrap(self, r, s):
        k = r.random()
        if k < 0.6:
            return [(s, "tg")]
        a, b = r.choice(["“”", "《》", "（）", "「」", "[]", "()"])
        kind = "lat" if a in "[(" else "han"
        return [(a, kind), (s, "tg"), (b, kind)]

    def _zh_tg(self, r):
        base = self.zh_chunk(r, 8, 32)
        cuts = sorted(r.sample(range(len(base) + 1), min(len(base) + 1, r.randint(1, 3))))
        segs, prev = [], 0
        for c in cuts:
            segs.append((base[prev:c], "han"))
            segs += self._wrap(r, self.tg_run(r, 1, 6))
            if r.random() < 0.3:
                segs.append((r.choice("，、：/"), "han"))
                if r.random() < 0.5:
                    segs.append((self.zh_chunk(r, 1, 4), "han"))
            prev = c
        segs.append((base[prev:], "han"))
        if r.random() < 0.25:
            fn, sup = self.footnote(r)
            segs.insert(r.randint(1, len(segs)), (fn, "sup" if sup else "lat"))
        return segs

    def _tg_punct(self, r):
        segs = []
        if r.random() < 0.15:
            segs.append((r.choice(["西夏文本：", "夏译：", "原文：", "汉文本：", "译文："]), "han"))
        if r.random() < 0.2:
            segs.append(("“", "han"))
        for _ in range(r.randint(1, 4)):
            segs.append((self.tg_run(r, 2, 10), "tg"))
            if r.random() < 0.3:
                fn, sup = self.footnote(r)
                segs.append((fn, "sup" if sup else "lat"))
            segs.append((r.choice(CJK_PUNCT_TG) if r.random() < 0.85 else r.choice("“”：？！"), "han"))
            if r.random() < 0.25:
                segs.append((" " * r.randint(1, 2), "lat"))
        if r.random() < 0.15:
            segs.append(("（" + self.zh_chunk(r, 3, 12), "han"))
        return segs

    def _mixed(self, r):
        s = r.choice(self.mixed)
        n = r.randint(6, 40)
        if len(s) > n:
            a = r.randint(0, len(s) - n)
            s = s[a: a + n]
        return [(s, "auto")]

    def _zh(self, r):
        segs = [(self.zh_chunk(r, 6, 34), "han")]
        k = r.random()
        if k < 0.15:
            a = r.randint(1, 600)
            segs.insert(0, (f"第{a}页（{r.randint(1000, 5999)}~{r.randint(1000, 5999)}）", "han"))
        elif k < 0.35:
            fn, sup = self.footnote(r)
            segs.insert(r.randint(0, 1), (fn, "sup" if sup else "lat"))
        return segs

    def _gloss(self, r):
        segs = []
        for _ in range(r.randint(1, 3)):
            k = r.random()
            if k < 0.45 and self.lexicon:
                s, t = r.choice(self.lexicon)
                segs += [(s, "tg"), (r.choice(["，", " ", "/", "：", "　"]), "han"), (t, "han")]
            else:
                ch, roms = r.choice(self.gloss)
                rom = unicodedata.normalize("NFC", r.choice(roms))
                segs += [(ch, "tg"), (" ", "lat"), (rom, "lat")]
                if r.random() < 0.5 and self.lexicon:
                    segs += [(" ", "lat"), (r.choice(self.lexicon)[1], "han")]
            segs.append((r.choice(["；", "。", "  ", "，"]), "han"))
        return segs

    def _short(self, r):
        if r.random() < 0.7:
            return [(self.tg_run(r, 1, 3), "tg")]
        return [(self.zh_chunk(r, 1, 3), "han")]

    def _tg_rand(self, r):
        n = r.randint(4, 24)
        s = "".join(r.choice(self.tg_chars) for _ in range(n))
        if r.random() < 0.3:
            s = "".join(c + (r.choice("，。、") if r.random() < 0.12 else "") for c in s)
        return [(s, "tg")]


# ---------------------------------------------------------------- 排版
def _weighted(r, d):
    return r.choices(list(d), weights=list(d.values()))[0]


def layout(segs, r, fonts=None):
    """segs → (白底黑字的 uint8 图, 标签字串)。逐字排版，不依赖字体的字距表。"""
    tg_f = fonts["tg"] if fonts else r.choice(C.TANGUT_FONTS)
    han_f = fonts["han"] if fonts else _weighted(r, C.HAN_FONTS)
    lat_f = fonts["lat"] if fonts else _weighted(r, C.LATIN_FONTS)
    han_pool = [han_f] + C.HAN_FALLBACK
    tg_scale = r.uniform(0.9, 1.2)
    lat_scale = r.uniform(0.85, 1.05)
    track = r.uniform(-0.08, 0.1)                                    # 字距
    items, label = [], []
    x = 0.0
    for text, kind in segs:
        for ch in text:
            k = kind
            if k == "auto":
                k = "tg" if is_tg(ch) else ("han" if is_fullwidth(ch) else "lat")
            if ch == " ":
                x += BASE * r.uniform(0.25, 0.5)
                label.append(" ")
                continue
            if k == "tg":
                fn, sc = pick_font(ch, tg_f, C.TANGUT_FONTS), tg_scale
            elif k in ("han",) or is_fullwidth(ch):
                fn, sc = pick_font(ch, han_f, han_pool), 1.0
            else:
                fn, sc = pick_font(ch, lat_f, [*C.LATIN_FONTS, han_f]), lat_scale
            if fn is None:
                continue
            if k == "sup":
                sc *= 0.6
            g, dx, dy, adv = glyph(ch, fn)
            if g is None:
                continue
            if is_fullwidth(ch):
                adv = BASE * (1 + track)
            rise = -BASE * 0.35 if k == "sup" else 0.0
            jitter = r.gauss(0, 0.6)
            items.append((g, x + dx * sc, dy * sc + rise + jitter, sc))
            x += adv * sc
            label.append(ch)
    if not items:
        return None, ""
    lab = " ".join("".join(label).split())
    pad = BASE * r.uniform(0.1, 0.8)
    W = int(x + 2 * pad) + 4
    H = int(BASE * 2.2)
    base_y = int(BASE * 1.55)
    canvas = np.zeros((H, W), np.float32)
    for g, gx, gy, sc in items:
        if sc != 1.0:
            g = cv2.resize(g, (max(1, round(g.shape[1] * sc)), max(1, round(g.shape[0] * sc))),
                           interpolation=cv2.INTER_AREA)
        x0, y0 = int(round(pad + gx)), int(round(base_y + gy))
        y1, x1 = min(H, y0 + g.shape[0]), min(W, x0 + g.shape[1])
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            continue
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], g[: y1 - y0, : x1 - x0] / 255.0)
    return canvas, lab


# ---------------------------------------------------------------- 退化
def degrade(ink, r):
    """ink: 0–1 墨迹图（BASE 字号）。返回模拟扫描的白底黑字 uint8 灰度图。"""
    k = r.random()
    if k < 0.15:
        ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
    elif k < 0.25:
        ink = cv2.erode(ink, np.ones((2, 2), np.uint8))
    if r.random() < 0.4:                                             # 轻微旋转/错切
        h, w = ink.shape
        a = r.uniform(-0.8, 0.8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        M[0, 1] += r.uniform(-0.04, 0.04)
        ink = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if r.random() < 0.2:                                             # 上下相邻行的残笔
        h, w = ink.shape
        band = int(h * r.uniform(0.03, 0.1))
        noise = (np.random.default_rng(r.getrandbits(32)).random((band, w)) < r.uniform(0.002, 0.02)).astype(np.float32)
        noise = cv2.dilate(noise, np.ones((3, 5), np.uint8))
        if r.random() < 0.5:
            ink[:band] = np.maximum(ink[:band], noise)
        else:
            ink[-band:] = np.maximum(ink[-band:], noise)
    # 降到扫描分辨率：字高约 14–44 像素
    th = r.uniform(16, 44)
    s = th / BASE
    h, w = ink.shape
    x = cv2.resize(ink, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(r.getrandbits(32))
    if r.random() < 0.6:
        x = cv2.GaussianBlur(x, (0, 0), r.uniform(0.3, 0.9))
    x = np.clip(x, 0, 1) ** np.exp(r.uniform(-0.4, 0.4)) * r.uniform(0.65, 1.0)
    paper = r.uniform(0.0, 0.25)
    if r.random() < 0.4:                                             # 纸面明暗不均
        lo = cv2.resize(rng.random((3, 6)).astype(np.float32), (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
        paper = paper + lo * r.uniform(0, 0.2)
    g = 1.0 - np.clip(x + paper * (1 - x), 0, 1)                    # 白底黑字，0–1
    if r.random() < 0.3:
        g = np.where(rng.random(g.shape) < r.uniform(0.001, 0.01), rng.uniform(0, 1, g.shape), g)
    g = g + rng.normal(0, r.uniform(0, 0.07), g.shape)
    if r.random() < 0.25:                                            # 二值化扫描
        t = r.uniform(0.35, 0.65)
        g = np.where(g > t, 1.0, 0.0)
        if r.random() < 0.5:
            g = cv2.GaussianBlur(g, (0, 0), 0.5)
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    if r.random() < 0.3:
        q = r.randint(30, 90)
        ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return g


class Synth:
    def __init__(self, charset, seed=0, clean=False, fonts=None):
        self.src = TextSource(charset)
        self.r = random.Random(seed)
        self.clean = clean
        self.fonts = fonts

    def sample(self):
        """返回 (模型输入 96×W float32, 真实宽度, 标签, 版式)。"""
        while True:
            segs, t = self.src.make(self.r)
            ink, lab = layout(segs, self.r, self.fonts)
            if ink is None or not lab.strip():
                continue
            g = (255 - ink * 255).astype(np.uint8) if self.clean else degrade(ink, self.r)
            x, w = to_input(ink_map(g), self.r.uniform(56, 72) if not self.clean else 64)
            if w // 8 < 2 * len(lab) + 2:                               # CTC 帧数不够（极端窄的行）
                continue
            return x, w, lab, t
