"""继续训练

  python run/train.py --name v1 --steps 60000
中断后加 --resume 从 runs/<name>/last.pt 接着训。每 --eval-every 步在四个评测集上评估，
按真实书页指标（core.evaluate.headline）保存 best.pt。日志在 runs/<name>/log.jsonl。
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from core import config as C
from core import charset as CS
from core.evaluate import evaluate, headline, load_sets
from core.model import expand_charset, load, save
from core.real import DICT_HEADS_PKL, DictHeads, RealLines
from core.synth import Synth

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--init", default=str(C.CODEX_V8),
                help="初始化权重：默认weights/init/，也可以是 weights/mixocr-v1.pt 或某个 runs/<name>/best.pt")
ap.add_argument("--steps", type=int, default=60000)
ap.add_argument("--cols", type=int, default=40000, help="每批的总像素宽度上限（显存预算）")
ap.add_argument("--max-batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--enc-lr-mult", type=float, default=0.5)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--p-dict", type=float, default=0.08, help="词典真实字头拼行的比例")
ap.add_argument("--p-real", type=float, default=0.0, help="真实扫描行（伪标签，run/build_reallines.py）的比例")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

RUN = C.RUNS / args.name
P_DICT = args.p_dict if DICT_HEADS_PKL.exists() else 0.0   # 词典字头（data/real/dict_heads.pkl）不在就不用
RUN.mkdir(parents=True, exist_ok=True)
charset = CS.load()


class Lines(IterableDataset):
    """每个 worker 自己合成；一次合成 8 批的量，按宽度排序后切批，减少右侧补零。"""
    def __init__(self, seed):
        self.seed = seed

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        syn = Synth(charset[1:], seed=self.seed * 1000 + wid)
        heads = DictHeads(charset[1:], seed=self.seed * 1000 + wid) if P_DICT else None
        real = RealLines(charset[1:], seed=self.seed * 1000 + wid) if args.p_real else None

        def one():
            u = syn.r.random()
            if u < P_DICT:
                return heads.sample()
            if u < P_DICT + args.p_real:
                return real.sample()
            return syn.sample()

        idx = {c: i for i, c in enumerate(charset)}
        while True:
            pool = [one() for _ in range(8 * 24)]
            pool.sort(key=lambda s: s[1])
            batches, cur = [], []
            for s in pool:
                W = s[0].shape[1]
                if cur and (len(cur) + 1) * W > args.cols or len(cur) >= args.max_batch:
                    batches.append(cur)
                    cur = []
                cur.append(s)
            batches.append(cur)
            random.Random(syn.r.random()).shuffle(batches)
            for b in batches:
                W = b[-1][0].shape[1]
                x = np.zeros((len(b), 1, 96, W), np.float32)
                for i, s in enumerate(b):
                    x[i, 0, :, : s[0].shape[1]] = s[0]
                tgt = [idx[c] for s in b for c in s[2] if c in idx]
                lens = [sum(c in idx for c in s[2]) for s in b]
                yield (torch.from_numpy(x), torch.tensor([s[1] for s in b]), torch.tensor(tgt),
                       torch.tensor(lens), [s[3] for s in b])


def lr_at(step):
    if step < args.warmup:
        return (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    torch.manual_seed(args.seed)
    dev = "cuda"
    # 超过显存就直接报错，而不是让 Windows 溢出到共享内存（那样会慢一个数量级）
    torch.cuda.set_per_process_memory_fraction(0.92)
    model, cs0 = load(args.init)
    assert charset[: len(cs0)] == cs0, "新字表必须以初始化模型的字表为前缀"
    model, cs1 = expand_charset(model, cs0, charset[len(cs0):])
    assert cs1 == charset
    model = model.to(dev)

    enc = list(model.encoder.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr * args.enc_lr_mult},
                             {"params": rest, "lr": args.lr}], weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    step, best = 0, -1.0
    if args.resume and (RUN / "last.pt").exists():
        ck = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        step, best = ck["step"], ck["best"]
        print(f"从第 {step} 步继续，best={best:.4f}", flush=True)
    (RUN / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=1), encoding="utf-8")

    sets = load_sets()                     # 只加载 data/eval/ 里存在的评测集；一个都没有就只训练不评估
    if not sets:
        print("没有评测集（data/eval/*.pkl），只训练、不挑模型；最后一步存为 best.pt", flush=True)
    log = open(RUN / "log.jsonl", "a", encoding="utf-8")

    def run_eval():
        if not sets:
            return float(step)             # 没有评测集时“越新越好”
        te = time.time()
        r = evaluate(model, charset, sets, dev)
        h = headline(r)
        parts = [f"{n}.{k}={r[n][k]:.4f}" for n in r for k in ("tg_acc", "han_acc") if k in r[n]]
        print(f"[eval {step}] headline={h:.4f} " + " ".join(parts) + f" ({time.time() - te:.0f}s)", flush=True)
        log.write(json.dumps({"step": step, "eval": r, "headline": h}) + "\n")
        log.flush()
        return h

    if step == 0:
        run_eval()
    dl = DataLoader(Lines(args.seed + step), batch_size=None, num_workers=args.workers,
                    persistent_workers=True, prefetch_factor=4)
    model.train().freeze_bn()
    scaler = torch.amp.GradScaler()
    t0, seen, run_loss = time.time(), 0, []
    for x, widths, tgt, lens, kinds in dl:
        if step >= args.steps:
            break
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_at(step)
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):      # 这块显卡上 fp16 比 bf16 快约 3 倍
            logits, olens = model(x, widths.tolist())
        logp = logits.float().log_softmax(-1)
        loss = ctc(logp, tgt, olens, lens)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += len(widths)
        run_loss.append(loss.item())
        if step % 100 == 0:
            el = time.time() - t0
            msg = {"step": step, "loss": float(np.mean(run_loss)), "lr": opt.param_groups[1]["lr"],
                   "lines_per_s": seen / el, "batch": len(widths), "width": x.shape[-1]}
            print(json.dumps(msg), flush=True)
            log.write(json.dumps(msg) + "\n")
            run_loss, seen, t0 = [], 0, time.time()
        if step % args.eval_every == 0 or step == args.steps:
            h = run_eval()
            model.train().freeze_bn()
            t0, seen = time.time(), 0
            if h > best:
                best = h
                save(model, charset, RUN / "best.pt", step=step, headline=h)
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                        "best": best, "charset": charset}, RUN / "last.pt")
    save(model, charset, RUN / "final.pt", step=step)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
