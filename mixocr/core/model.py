"""整行 CTC 识别模型：卷积编码器 + 双层 BiLSTM + 线性输出层。

输入：B×1×96×W，暗底亮字，像素 0–1。横向步长 8，每 8 像素输出一帧。
"""
import torch
from torch import nn

CONV_CHANNELS = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256))
from .preprocess import INPUT_HEIGHT
STRIDE = 8


def out_len(width):
    """三次 2×2、步长 2、不补边的最大池化之后的宽度。"""
    for _ in range(3):
        width = (width - 2) // 2 + 1
    return width


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for i, (a, b) in enumerate(CONV_CHANNELS):
            blocks.append(ConvBlock(a, b))
            if i in (1, 3, 5):
                blocks.append(nn.MaxPool2d(2, 2))
        self.stem = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stem(x).mean(dim=2).transpose(1, 2)      # B×T×256


class LineCTC(nn.Module):
    def __init__(self, n_classes, hidden=128, layers=2):
        super().__init__()
        self.encoder = Encoder()
        self.sequence = nn.LSTM(256, hidden, num_layers=layers, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, n_classes)

    def freeze_bn(self):
        """BatchNorm 统计量保持冻结，仿射参数仍可训练。"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, widths):
        """x: B×1×96×W；widths: 每行真实像素宽。返回 T×B×C 的 logits 和每行帧数。"""
        feats = self.encoder(x)
        lengths = torch.tensor([min(out_len(int(w)), feats.shape[1]) for w in widths], dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.sequence(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=feats.shape[1])
        return self.head(out).transpose(0, 1), lengths


def greedy_decode(logits, lengths, charset):
    """logits: T×B×C。charset[0] 是空白。"""
    best = logits.argmax(-1).cpu().numpy()
    texts = []
    for b, n in enumerate(lengths.tolist()):
        prev, s = 0, []
        for k in best[:n, b]:
            if k and k != prev:
                s.append(charset[k])
            prev = k
        texts.append("".join(s))
    return texts


def load_codex_v8(path):
    """读检查点，转成本模块的结构。返回 (model, charset)，charset[0] 为空白。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    src = ck["model"]
    charset = ["<b>"] + list(ck["characters"])
    m = LineCTC(len(charset))
    state = {}
    for k, v in src.items():
        if k.startswith("student_glyph.encoder.stem."):
            state[k.replace("student_glyph.encoder.", "encoder.")] = v
        elif k.startswith("sequence."):
            state[k] = v
    for p in ("weight", "bias"):
        state[f"head.{p}"] = torch.cat([src[f"head.original.{p}"], src[f"head.added.{p}"]], 0)
    m.load_state_dict(state, strict=True)
    return m, charset


def save(model, charset, path, **meta):
    torch.save({"state_dict": model.state_dict(), "charset": charset, "hidden": model.sequence.hidden_size,
                "layers": model.sequence.num_layers, **meta}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = LineCTC(len(ck["charset"]), ck.get("hidden", 128), ck.get("layers", 2))
    m.load_state_dict(ck["state_dict"])
    return m, ck["charset"]


def expand_charset(model, charset, new_chars):
    """在输出层末尾追加新字符。旧行原样复制；新行取旧字符行的平均（logit 约等于各字的平均水平）再把偏置压低，
    避免一开始就抢输出。旧输出层的 logit 量级很大，只给新行一个常数偏置是不够的。"""
    add = [c for c in dict.fromkeys(new_chars) if c not in set(charset)]
    if not add:
        return model, charset
    old = model.head
    head = nn.Linear(old.in_features, old.out_features + len(add))
    with torch.no_grad():
        head.weight[: old.out_features] = old.weight
        head.bias[: old.out_features] = old.bias
        head.weight[old.out_features:] = old.weight[1:].mean(0) + 0.01 * torch.randn(len(add), old.in_features)
        head.bias[old.out_features:] = old.bias[1:].mean() - 4.0
    model.head = head
    return model, charset + add
