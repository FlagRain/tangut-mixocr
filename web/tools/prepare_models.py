"""把仓库 weights/ 里的模型放到网站用的 web/models/，并写 model_info.json（网页“说明”里的评测数字）。

  python web/tools/prepare_models.py                 # 默认 mixocr-v1
  python web/tools/prepare_models.py --model mixocr-v2

换模型：先用 mixocr/run/export_onnx.py 导出 weights/<名>.onnx 与 <名>.charset.json，再运行本脚本、重新打包。
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
W = REPO / "weights"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mixocr-v1")
args = ap.parse_args()

out = WEB / "models"
out.mkdir(exist_ok=True)
charset = W / f"{args.model}.charset.json"
if not charset.exists():
    charset = W / "charset.json"
shutil.copyfile(W / f"{args.model}.onnx", out / "rec.onnx")
shutil.copyfile(charset, out / "charset.json")
shutil.copyfile(W / "det" / "PP-OCRv6_det_small.onnx", out / "det.onnx")

info = {"name": args.model}
ev = REPO / "training_logs" / "eval_results.csv"
if ev.exists():
    rows = list(csv.DictReader(open(ev, encoding="utf-8-sig")))
    row = next((r for r in rows if args.model in r["模型"]), None)
    base = next((r for r in rows if r["模型"] == "codex_v8"), None)
    keys = ("书页西夏字", "书页汉字*", "词典字头", "讲义西夏字", "讲义汉字")
    if row:
        info["metrics"] = {k: round(float(row[k]), 4) for k in keys}
    if base:
        info["baseline_v8"] = {k: round(float(base[k]), 4) for k in keys[:2]}
(out / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
print("web/models 已就绪：", info)
