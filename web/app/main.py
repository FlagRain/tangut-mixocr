"""网站后端：上传 → 排队识别 → 返回结果；收集纠错反馈；管理员审核与导出。

识别在一个后台线程里逐页进行（小服务器一次只跑一页），网页轮询进度。
"""
import base64
import collections
import hmac
import io
import json
import queue
import secrets
import threading
import time
import zipfile

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import config as C
from .engine import Engine
from .store import Store

Image.MAX_IMAGE_PIXELS = 120_000_000
DISPLAY_MAX = 2200            # 网页显示用 JPEG 的长边

store = Store()
engine = Engine(C.MODEL_DIR, C.THREADS)
app = FastAPI(title="西夏文 OCR", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def headers(request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    path = request.url.path
    if path.startswith("/api/") and "cache-control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-store"
    elif path.startswith("/static/fonts/"):
        resp.headers["Cache-Control"] = "public, max-age=604800"
    elif path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"          # 每次向服务器确认，升级后不会用到旧脚本
    return resp


# ------------------------------------------------------------------ 限速
class RateLimit:
    def __init__(self, per_hour):
        self.per_hour = per_hour
        self.hits = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def check(self, key):
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and q[0] < now - 3600:
                q.popleft()
            if len(q) >= self.per_hour:
                raise HTTPException(429, "操作太频繁，请稍后再试")
            q.append(now)


upload_limit = RateLimit(C.UPLOADS_PER_HOUR)
feedback_limit = RateLimit(C.FEEDBACK_PER_HOUR)


def client_ip(request):
    return request.client.host if request.client else ""


# ------------------------------------------------------------------ 页面图像
def _fit(gray, max_side):
    h, w = gray.shape
    s = max_side / max(h, w)
    return cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray


def _save_page(d, n, gray):
    gray = _fit(gray, C.MAX_SIDE)
    cv2.imwrite(str(d / f"p{n}.png"), gray)
    ok, buf = cv2.imencode(".jpg", _fit(gray, DISPLAY_MAX), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (d / f"p{n}.jpg").write_bytes(buf.tobytes())
    return gray


def _render_pdf_page(path, n):
    import pymupdf
    with pymupdf.open(path) as doc:
        page = doc[n - 1]
        imgs = page.get_images()
        if len(imgs) == 1 and page.rotation == 0 and not page.get_text().strip():
            pix = pymupdf.Pixmap(doc, imgs[0][0])           # 扫描页：直接取原图，不重采样
        else:
            r = page.rect
            dpi = min(300, int(C.MAX_SIDE / max(r.width, r.height) * 72))
            pix = page.get_pixmap(dpi=dpi)
        if pix.n - pix.alpha != 1:
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def page_gray(doc, n):
    """识别用的灰度页；PDF 的页在第一次用到时才渲染。"""
    d = store.doc_dir(doc)
    p = d / f"p{n}.png"
    if p.exists():
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    meta = store.meta(doc)
    if meta["kind"] != "pdf":
        raise RuntimeError("页面文件不存在")
    return _save_page(d, n, _render_pdf_page(d / "source.pdf", n))


# ------------------------------------------------------------------ 识别队列
jobs = {}                     # "doc:page" -> {"status", "error", "ip", "t"}
order = collections.deque()   # 排队中的 key
jobs_lock = threading.Lock()
work = queue.Queue()


def worker():
    while True:
        key = work.get()
        doc, n = key.rsplit(":", 1)
        with jobs_lock:
            if key in order:
                order.remove(key)
            jobs[key]["status"] = "running"
        try:
            gray = page_gray(doc, int(n))
            res = engine.page(gray)
            d = store.doc_dir(doc)
            if d:
                (d / f"p{n}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            status, err = "done", None
        except Exception as e:                      # noqa: BLE001  识别失败不能拖垮工作线程
            status, err = "error", f"识别失败：{type(e).__name__}"
            print("识别失败", key, repr(e), flush=True)
        with jobs_lock:
            jobs[key].update(status=status, error=err, t=time.time())


def janitor():
    while True:
        try:
            n = store.cleanup()
            if n:
                print(f"已清理 {n} 个过期上传", flush=True)
            with jobs_lock:
                for k in [k for k, j in jobs.items() if j["status"] in ("done", "error") and time.time() - j["t"] > 3600]:
                    del jobs[k]
        except Exception as e:                      # noqa: BLE001
            print("清理出错", repr(e), flush=True)
        time.sleep(3600)


threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=janitor, daemon=True).start()


# ------------------------------------------------------------------ 公开接口
@app.get("/api/info")
def info(client: str = ""):
    return {"model": engine.info, "limits": {"max_upload_mb": C.MAX_UPLOAD_MB, "retention_hours": C.RETENTION_HOURS},
            "features": {"lookup": (C.STATIC_DIR / "lookup.json").exists()},
            "stats": store.stats(client or None)}


def _new_pdf_doc(path, name, ip):
    """path 是已经落盘的上传文件；成功时移进文档目录，失败时删掉。"""
    import pymupdf
    try:
        with pymupdf.open(path) as pdf:
            pages = pdf.page_count
    except Exception:                               # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(400, "无法打开这个 PDF")
    if not 1 <= pages <= C.MAX_PDF_PAGES:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"PDF 页数须在 1–{C.MAX_PDF_PAGES} 之间")
    doc, d = store.new_doc({"kind": "pdf", "name": name, "pages": pages, "ip": store.ip_hash(ip)})
    path.replace(d / "source.pdf")
    return {"doc": doc, "kind": "pdf", "pages": pages, "name": name}


def _new_image_doc(data, name, ip):
    try:
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > 60_000_000:
            raise HTTPException(400, "图片太大（超过 6000 万像素）")
        im = ImageOps.exif_transpose(im)
        gray = np.asarray(im.convert("L"))
    except HTTPException:
        raise
    except Exception:                               # noqa: BLE001
        raise HTTPException(400, "无法识别的文件。请上传 PNG、JPG、WebP、TIFF 图片或 PDF")
    doc, d = store.new_doc({"kind": "image", "name": name, "pages": 1, "ip": store.ip_hash(ip)})
    _save_page(d, 1, gray)
    return {"doc": doc, "kind": "image", "pages": 1, "name": name}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """上传按 1 MB 一块写到磁盘，不把整个文件放进内存（扫描书 PDF 动辄近百 MB）。"""
    ip = client_ip(request)
    upload_limit.check(ip)
    cap = C.MAX_UPLOAD_MB * 2**20
    tmp = store.root / "tmp"
    tmp.mkdir(exist_ok=True)
    path = tmp / f"up-{secrets.token_hex(8)}"
    size = 0
    try:
        with open(path, "wb") as fh:
            while chunk := await file.read(2**20):
                size += len(chunk)
                if size > cap:
                    raise HTTPException(413, f"文件超过 {C.MAX_UPLOAD_MB} MB；可以先把需要的几页另存为一个小 PDF")
                fh.write(chunk)
        if not size:
            raise HTTPException(400, "空文件")
        with open(path, "rb") as fh:
            head = fh.read(5)
        name = (file.filename or "未命名")[:120]
        if head == b"%PDF-":
            return _new_pdf_doc(path, name, ip)
        return _new_image_doc(path.read_bytes(), name, ip)
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/sample")
def sample(request: Request):
    return _new_image_doc((C.STATIC_DIR / "sample.png").read_bytes(), "示例页.png", client_ip(request))


def _doc_page(doc, n):
    meta = store.meta(doc)
    if not meta:
        raise HTTPException(404, "文件已过期或不存在，请重新上传")
    if not 1 <= n <= meta["pages"]:
        raise HTTPException(404, "没有这一页")
    return store.doc_dir(doc), meta


def _status(key, d, n):
    if (d / f"p{n}.json").exists():
        return {"status": "done"}
    with jobs_lock:
        j = jobs.get(key)
        if not j:
            return {"status": "none"}
        pos = list(order).index(key) if key in order else 0
        return {"status": j["status"], "ahead": pos, "error": j.get("error")}


@app.post("/api/docs/{doc}/pages/{n}/ocr")
def start_ocr(doc: str, n: int, request: Request):
    d, _ = _doc_page(doc, n)
    key = f"{doc}:{n}"
    ip = store.ip_hash(client_ip(request))
    with jobs_lock:
        if not (d / f"p{n}.json").exists() and (key not in jobs or jobs[key]["status"] == "error"):
            if len(order) >= C.MAX_QUEUE:
                raise HTTPException(503, "排队的人太多，请过几分钟再试")
            if sum(1 for k in order if jobs[k]["ip"] == ip) >= C.MAX_JOBS_PER_IP:
                raise HTTPException(429, f"你已有 {C.MAX_JOBS_PER_IP} 页在排队，请等它们完成")
            jobs[key] = {"status": "queued", "error": None, "ip": ip, "t": time.time()}
            order.append(key)
            work.put(key)
    return _status(key, d, n)


@app.get("/api/docs/{doc}/pages/{n}/status")
def page_status(doc: str, n: int):
    d, _ = _doc_page(doc, n)
    return _status(f"{doc}:{n}", d, n)


@app.get("/api/docs/{doc}/pages/{n}")
def page_result(doc: str, n: int):
    d, _ = _doc_page(doc, n)
    p = d / f"p{n}.json"
    if not p.exists():
        raise HTTPException(404, "这一页还没有识别")
    return Response(p.read_bytes(), media_type="application/json")


@app.get("/api/docs/{doc}/pages/{n}/image")
def page_image(doc: str, n: int):
    d, _ = _doc_page(doc, n)
    p = d / f"p{n}.jpg"
    if not p.exists():
        raise HTTPException(404, "页面图像还没有生成")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


class Feedback(BaseModel):
    doc: str = Field(max_length=40)
    page: int
    line: int
    action: str = Field(pattern="^(confirm|correct|bad)$")
    text: str = Field("", max_length=600)
    client: str = Field(min_length=8, max_length=64)
    contributor: str = Field("", max_length=40)
    consent: bool


@app.post("/api/feedback")
def feedback(fb: Feedback, request: Request):
    ip = client_ip(request)
    feedback_limit.check(ip)
    if not fb.consent:
        raise HTTPException(400, "需要先同意把提交的行图和文字用于模型训练")
    d, _ = _doc_page(fb.doc, fb.page)
    p = d / f"p{fb.page}.json"
    if not p.exists():
        raise HTTPException(404, "这一页还没有识别")
    res = json.loads(p.read_text(encoding="utf-8"))
    if not 0 <= fb.line < len(res["lines"]):
        raise HTTPException(400, "没有这一行")
    line = res["lines"][fb.line]
    text = " ".join(fb.text.split())
    if fb.action == "confirm":
        text = line["text"]
    elif fb.action == "bad":
        text = ""
    elif not text:
        raise HTTPException(400, "改正后的文字不能为空；如果这里不是一行文字，请用“框有问题”")
    x0, y0, x1, y1 = line["box"]
    gray = page_gray(fb.doc, fb.page)
    m = max(2, (y1 - y0) // 8)
    crop = gray[max(0, y0 - m): y1 + m, max(0, x0 - m): x1 + m]
    ok, png = cv2.imencode(".png", crop)
    fid = store.save_feedback({"client": fb.client, "contributor": fb.contributor.strip() or None,
                               "ip_hash": store.ip_hash(ip), "doc": fb.doc, "page": fb.page, "line": fb.line,
                               "action": fb.action, "model_text": line["text"], "text": text, "box": line["box"],
                               "model": res.get("model")}, png.tobytes())
    return {"id": fid, "stats": store.stats(fb.client)}


@app.get("/api/stats")
def stats(client: str = ""):
    return store.stats(client or None)


# ------------------------------------------------------------------ 管理后台
def admin(request: Request):
    tok = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not C.ADMIN_TOKEN or not hmac.compare_digest(tok.encode(), C.ADMIN_TOKEN.encode()):
        raise HTTPException(401, "口令不对")


@app.get("/api/admin/items")
def admin_items(request: Request, status: str = "pending", action: str = "all", offset: int = 0, limit: int = 40):
    admin(request)
    if status not in ("pending", "approved", "rejected", "all") or action not in ("confirm", "correct", "bad", "all"):
        raise HTTPException(400, "参数不对")
    total, rows = store.list_feedback(status, action, max(0, offset), min(100, max(1, limit)))
    for r in rows:
        p = store.fb / r["crop"]
        r["image"] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else None
        r["box"] = json.loads(r["box"])
        r.pop("ip_hash", None)
    return {"total": total, "items": rows, "stats": store.stats()}


class Review(BaseModel):
    ids: list[int]
    status: str = Field(pattern="^(pending|approved|rejected)$")
    text: str | None = Field(None, max_length=600)


@app.post("/api/admin/review")
def admin_review(r: Review, request: Request):
    admin(request)
    text = " ".join(r.text.split()) if r.text is not None else None
    store.review(r.ids[:500], r.status, text)
    return {"ok": True, "stats": store.stats()}


@app.get("/api/admin/export")
def admin_export(request: Request, status: str = "approved"):
    """导出成 mixocr 训练能直接读的格式：lines/web/*.png + lines.jsonl（字段与 run/build_reallines.py 相同）。"""
    admin(request)
    if status not in ("approved", "pending", "all"):
        raise HTTPException(400, "参数不对")
    rows = store.export_rows(status)
    tmp = store.root / "tmp"
    tmp.mkdir(exist_ok=True)
    path = tmp / f"export-{secrets.token_hex(6)}.zip"           # 写到磁盘再发，不在内存里攒整个压缩包
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        out = []
        for r in rows:
            p = store.fb / r["crop"]
            if not p.exists():
                continue
            name = f"web/web_{r['id']:07d}.png"
            z.write(p, "lines/" + name)
            text = r["reviewed_text"] or r["text"]
            kind = "mixed" if any(0x17000 <= ord(c) <= 0x18DFF for c in text) else "han"
            out.append({"id": f"web_{r['id']:07d}", "source": "web", "kind": kind, "path": name, "text": text,
                        "model": r["model_text"], "action": r["action"], "status": r["status"],
                        "contributor": r["contributor"], "created": r["created"], "model_version": r["model"]})
        z.writestr("lines.jsonl", "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
        z.writestr("README.txt", "lines/web/*.png 是行图，lines.jsonl 每行一条标签。\n"
                   "解压到 mixocr/data/real/ 下（lines.jsonl 可与 build_reallines.py 的输出合并），"
                   "训练时加 --p-real 即可使用。\n")
    stamp = time.strftime("%Y%m%d-%H%M")
    return FileResponse(path, media_type="application/zip", filename=f"tangut-ocr-feedback-{status}-{stamp}.zip",
                        background=BackgroundTask(path.unlink, missing_ok=True))


# ------------------------------------------------------------------ 网页
@app.get("/")
def index():
    return FileResponse(C.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/admin")
def admin_page():
    return FileResponse(C.STATIC_DIR / "admin.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=C.STATIC_DIR), name="static")


@app.exception_handler(HTTPException)
def http_error(request, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
def bad_request(request, exc):
    return JSONResponse({"error": "请求内容不完整或格式不对"}, status_code=400)
