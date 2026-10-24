"""打部署包：dist/tangut-ocr-web-<日期>.tar.gz（根目录 tangut-ocr/）与同名 .sha256。

  python web/tools/pack.py

先跑过 web/tools/prepare_models.py。文本文件统一换成 LF 换行（Linux 上的 shell 脚本和 systemd 单元必须是 LF），
install.sh 与 smoke_test.py 设为可执行。
"""
import hashlib
import io
import sys
import tarfile
import time
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
PREFIX = "tangut-ocr"
TEXT = {".py", ".sh", ".service", ".example", ".md", ".txt", ".html", ".js", ".css", ".json"}
EXEC = {"deploy/install.sh", "tools/smoke_test.py"}


def files():
    for d in ("app", "static", "models"):
        for p in sorted((WEB / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                yield p
    for rel in ("deploy/install.sh", "deploy/tangut-ocr.service", "deploy/tangut-ocr.env.example",
                "tools/smoke_test.py", "requirements.txt", "README.md", "部署说明.md"):
        yield WEB / rel


def main():
    missing = [p for p in ("models/rec.onnx", "models/det.onnx", "models/charset.json",
                           "static/fonts/NotoSerifTangut.woff2", "static/sample.png", "部署说明.md")
               if not (WEB / p).exists()]
    if missing:
        sys.exit("缺少文件（先跑 web/tools/prepare_models.py）：" + ", ".join(missing))
    if not (WEB / "static" / "lookup.json").exists():
        print("提示：没有 static/lookup.json，网页上不会显示查字面板（见 tools/build_lookup.py）")
    mix = WEB.parent / "mixocr" / "core" / "preprocess.py"
    if mix.exists() and mix.read_bytes() != (WEB / "app" / "preprocess.py").read_bytes():
        sys.exit("web/app/preprocess.py 与 mixocr/core/preprocess.py 不一致：训练和线上必须用同一份预处理，请复制过来")
    out = WEB / "dist" / f"tangut-ocr-web-{time.strftime('%Y%m%d')}.tar.gz"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for p in files():
            rel = p.relative_to(WEB).as_posix()
            data = p.read_bytes()
            if p.suffix in TEXT:
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(f"{PREFIX}/{rel}")
            info.size = len(data)
            info.mtime = int(p.stat().st_mtime)
            info.mode = 0o755 if rel in EXEC else 0o644
            tar.addfile(info, io.BytesIO(data))
            n += 1
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="ascii")
    print(f"{out}  {n} 个文件，{out.stat().st_size / 2**20:.1f} MiB\nsha256 {digest}")


if __name__ == "__main__":
    main()
