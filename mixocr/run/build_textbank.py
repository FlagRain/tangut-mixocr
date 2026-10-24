"""收集合成训练行用的文本。输出 data/textbank/{zh,tg,mixed}.txt，每行一段。

- tg：自然西夏文。来自课程标注（\\ex 行与正文行）和讲义 PDF 文字层。
- zh：汉文句子。来自课程标注的译文（\\ft）、讲义 PDF、两本扫描书的 PP-OCR 输出（只取高置信度的纯汉文行）、
      词典扫描页的 PP-OCR 输出（同样只取高置信度行）。
- mixed：讲义 PDF 里同一行同时含西夏字和汉字的原文，排版合成时原样使用。

留出：《经律异相》转写（真实书页测试集的真值）整个目录不用；《经律异相》整理研究第 50–115 页
（测试页）不用；讲义 dev 的三份 PDF不用。
"""
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf

from core import config as C

OUT = C.TEXTBANK
OUT.mkdir(parents=True, exist_ok=True)

is_tg = lambda ch: 0x17000 <= ord(ch) <= 0x18DFF
is_han = lambda ch: 0x3400 <= ord(ch) <= 0x9FFF or 0xF900 <= ord(ch) <= 0xFAFF or 0x20000 <= ord(ch) <= 0x323AF
GARBAGE = re.compile(r"[-�]")


def frac(s, f):
    s = s.replace(" ", "")
    return sum(map(f, s)) / max(1, len(s))


tg, zh, mixed = [], [], []

# ---- 课程标注
for p in (glob.glob(str(C.ANNOTATION_DIR / "**" / "*.md"), recursive=True) if C.ANNOTATION_DIR else []):
    if "经律异相" in p:
        continue
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line.startswith("\\ex "):
            s = line[4:].replace("(", "").replace(")", "").strip()
            if frac(s, is_tg) > 0.5:
                tg.append(s)
        elif line.startswith("\\ft "):
            s = line[4:].strip()
            if frac(s, is_han) > 0.5:
                zh.append(s)
        elif line and not line.startswith(("\\", "```", "#")):
            if frac(line, is_tg) > 0.6:
                tg.append(line)
            elif frac(line, is_han) > 0.6:
                zh.append(line)

# ---- 讲义 PDF 文字层
held = set(C.LECTURE_DEV_DOCS)
for p in (sorted(C.LECTURE_DIR.glob("*.pdf")) if C.LECTURE_DIR else []):
    doc_id = "gr_" + hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    if doc_id in held:
        continue
    for page in pymupdf.open(p):
        for block in page.get_text("dict")["blocks"]:
            for ln in block.get("lines", []):
                s = "".join(sp["text"] for sp in ln["spans"]).strip()
                if len(s) < 2 or GARBAGE.search(s):
                    continue
                n_tg, n_han = sum(map(is_tg, s)), sum(map(is_han, s))
                if n_tg and n_han:
                    mixed.append(s)
                elif n_tg >= 2:
                    tg.append(s)
                elif n_han >= 4:
                    zh.append(s)

# ---- 两本扫描书的 PP-OCR 输出（mixbias/scan_book_pages.py 生成）
for tag in ("dxj", "jlyx"):
    for p in (sorted((C.BOOK_OCR / tag).glob("p*.json")) if C.BOOK_OCR else []):
        d = json.loads(p.read_text(encoding="utf-8"))
        if tag == "jlyx" and C.JLYX_TEST_PAGES[0] <= d["page"] <= C.JLYX_TEST_PAGES[1]:
            continue
        for ln in d["lines"]:
            s = ln["text"].strip()
            if ln["score"] >= 0.97 and len(s) >= 4 and frac(s, is_han) > 0.6:
                zh.append(s)

# ---- 词典扫描页的 PP-OCR 输出
dict_zh = []
for p in (glob.glob(str(C.DICT_RAW_PAGES / "*" / "*.json")) if C.DICT_RAW_PAGES else []):
    d = json.loads(open(p, encoding="utf-8").read())
    for ln in d.get("lines", []):
        s = (ln.get("raw_ocr_candidate") or "").strip()
        if (ln.get("recognition_confidence") or 0) >= 0.97 and len(s) >= 4 and frac(s, is_han) > 0.7:
            dict_zh.append(s)
random.Random(0).shuffle(dict_zh)
zh += dict_zh[:150_000]


def dedup(xs):
    return list(dict.fromkeys(x for x in xs if not GARBAGE.search(x)))


for name, xs in (("tg", tg), ("zh", zh), ("mixed", mixed)):
    xs = dedup(xs)
    (OUT / f"{name}.txt").write_text("\n".join(xs) + "\n", encoding="utf-8")
    chars = "".join(xs)
    print(f"{name}: {len(xs)} 段, {len(chars)} 字, 西夏字 {sum(map(is_tg, chars))}, 汉字 {sum(map(is_han, chars))}")
