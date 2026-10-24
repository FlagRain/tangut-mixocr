"""在全部评测集上比较模型，写 results/eval.md 与 results/eval.csv。

"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from core import config as C
from core.evaluate import SETS, evaluate, load_sets
from core.model import load

COLS = [("jlyx_tg", "tg_acc", "书页西夏字"), ("jlyx_tg", "tg_cross", "书页西夏→汉"),
        ("jlyx_han", "han_acc", "书页汉字*"), ("dict_head", "tg_acc", "词典字头"),
        ("lecture", "tg_acc", "讲义西夏字"), ("lecture", "han_acc", "讲义汉字"), ("lecture", "other_acc", "讲义其他"),
        ("synth", "tg_acc", "合成西夏字"), ("synth", "han_acc", "合成汉字"), ("synth", "cer", "合成 CER")]

sets = load_sets(SETS)
rows = []
for spec in sys.argv[1:]:
    m, cs = load(spec)
    r = evaluate(m.cuda(), cs, sets)
    rows.append({"模型": spec, **{name: r[s][k] for s, k, name in COLS}})
    print(spec, {name: round(r[s][k], 4) for s, k, name in COLS}, flush=True)

df = pd.DataFrame(rows)
C.RESULTS.mkdir(parents=True, exist_ok=True)
df.to_csv(C.RESULTS / "eval.csv", index=False, encoding="utf-8-sig")
fmt = df.copy()
for c in fmt.columns[1:]:
    fmt[c] = fmt[c].map(lambda v: f"{v:.2%}")
n = {k: sets[k] for k in sets}
lines = ["# 评测结果\n",
         f"书页西夏字：{sum(m.count('1') for *_, m in n['jlyx_tg'])} 个确认的西夏字（{len(n['jlyx_tg'])} 行）；"
         f"书页汉字：{len(n['jlyx_han'])} 行；词典字头：{len(n['dict_head'])} 个；讲义：{len(n['lecture'])} 行；"
         f"合成：{len(n['synth'])} 行。\n",
         "*书页汉字的参考是 PP-OCRv5 的高置信度输出\n",
         "| " + " | ".join(fmt.columns) + " |", "|" + " --- |" * len(fmt.columns)]
lines += ["| " + " | ".join(map(str, r)) + " |" for r in fmt.values]
(C.RESULTS / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
