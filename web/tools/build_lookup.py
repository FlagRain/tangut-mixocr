"""（可选）生成网页的“查西夏字”表 web/static/lookup.json：按拟音、汉义、英义查字。

仓库不附带这个表，因为拟音和释义来自整理者自己的资料。有资料的话：
  python web/tools/build_lookup.py --tsv 西夏文拟音.tsv --dict 字典.json

--tsv  制表符分隔，第 2 列为西夏字，第 3、6、8 列为拟音（与 mixocr 训练用的字表格式相同）
--dict JSON 数组，每项 {"character", "GHC", "GX", "explanationCN", "explanationEN"}
两者都可以只给一个。没有这个文件时，网页自动隐藏查字面板。
"""
import argparse
import json
import unicodedata
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tsv")
ap.add_argument("--dict")
args = ap.parse_args()
is_tg = lambda c: len(c) == 1 and 0x17000 <= ord(c) <= 0x18DFF

ent = {}
if args.tsv:
    for line in open(args.tsv, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 8 and is_tg(p[1]):
            ent.setdefault(p[1], {"r": [], "zh": "", "en": ""})["r"] += [x for x in (p[2], p[5], p[7]) if x]
if args.dict:
    for d in json.load(open(args.dict, encoding="utf-8")):
        c = d.get("character", "")
        if is_tg(c):
            e = ent.setdefault(c, {"r": [], "zh": "", "en": ""})
            e["r"] += [x for x in (d.get("GHC"), d.get("GX")) if x]
            e["zh"] = e["zh"] or (d.get("explanationCN") or "").strip()
            e["en"] = e["en"] or (d.get("explanationEN") or "").strip()
rows = [[c, " / ".join(dict.fromkeys(unicodedata.normalize("NFC", x) for x in e["r"])), e["zh"][:40], e["en"][:60]]
        for c, e in sorted(ent.items())]
out = Path(__file__).resolve().parents[1] / "static" / "lookup.json"
out.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"{out}：{len(rows)} 字")
