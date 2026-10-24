"""把模型导出成 ONNX。输出与输入同目录的 <名>.onnx 与 <名>.charset.json。

  python mixocr/run/export_onnx.py weights/mixocr-v1.pt

导出的是单行推理（batch=1，宽度可变）：输入 1×1×96×W（core.preprocess.to_input 的结果），
输出每帧后验最高的 5 个类别下标（T×5）和它们的概率（T×5）；第 1 列就是贪心解码用的类别，
其余几列给网站的纠错界面当候选字。导出后在讲义与书页评测集上核对（没有评测集时用随机输入核对每帧第 1 名），确认与 PyTorch 结果相同。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from core import config as C
from core.evaluate import load_sets, predict
from core.model import load


class SingleLine(nn.Module):
    """batch=1 时不需要 pack_padded_sequence，结果与带 pack 的前向完全相同。"""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out, _ = self.m.sequence(self.m.encoder(x))
        p = self.m.head(out)[0].softmax(-1)
        conf, idx = p.topk(5, dim=-1)
        return idx, conf


def decode(idx, charset):
    s, prev = [], 0
    for k in idx:
        if k and k != prev:
            s.append(charset[k])
        prev = k
    return "".join(s)


def main():
    src = Path(sys.argv[1]).resolve()
    model, charset = load(src)
    model.eval()
    out = src.with_suffix(".onnx")
    wrapper = SingleLine(model).eval()          # 导出后会恢复包装模块原来的模式；新建的模块默认是训练模式
    torch.onnx.export(wrapper, torch.zeros(1, 1, 96, 640), str(out), input_names=["image"],
                      output_names=["index", "prob"], dynamic_axes={"image": {3: "width"}, "index": {0: "t"},
                                                                    "prob": {0: "t"}},
                      opset_version=17, dynamo=False)
    model.eval()
    out.with_suffix(".charset.json").write_text(json.dumps(charset, ensure_ascii=False), encoding="utf-8")
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    same = total = 0
    sets = load_sets(["lecture", "jlyx_tg"])
    for items in sets.values():
        ref = predict(model, charset, items, "cpu", cols=1)              # 一次一行，与 ONNX 一样
        for (x, w, *_), r in zip(items, ref):
            idx, _ = sess.run(None, {"image": x[None, None, :, : w + (-w) % 8]})
            same += decode(idx[:, 0], charset) == r
            total += 1
    if not sets:                                                     # 没有评测集：随机输入比每帧第 1 名
        rng = np.random.default_rng(0)
        for _ in range(8):
            x = (rng.random((1, 1, 96, 8 * int(rng.integers(20, 120)))) > 0.8).astype(np.float32)
            with torch.no_grad():
                ref = wrapper(torch.from_numpy(x))[0].numpy()
            idx, _ = sess.run(None, {"image": x})
            same += int((idx[:, 0] == ref[:, 0]).mean() > 0.99)     # 解码只用第 1 名；后几名在接近并列时顺序可能互换
            total += 1
    print(f"{out.name}: {out.stat().st_size / 2**20:.1f} MiB；与 PyTorch 逐行相同 {same}/{total}")


if __name__ == "__main__":
    main()
