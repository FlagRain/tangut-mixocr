"""对一个正在运行的网站做端到端检查（只用标准库，可以在服务器上直接跑）。

  python3 tools/smoke_test.py http://127.0.0.1:7777 [管理口令] [PDF 文件]

依次检查：首页、示例页上传与识别、坏文件被拒、反馈提交、（给了口令时）后台列表与导出、（给了 PDF 时）PDF 翻页识别。
会在服务器上留下一条测试反馈，后台里可以拒绝掉。
"""
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:7777"
TOKEN = sys.argv[2] if len(sys.argv) > 2 else ""
PDF = sys.argv[3] if len(sys.argv) > 3 else ""


def req(method, path, data=None, headers=None, raw=False):
    h = dict(headers or {})
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            body = resp.read()
            return resp.status, (body if raw else json.loads(body or b"null"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def upload(name, content, ctype):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + content + f"\r\n--{b}--\r\n".encode()
    return req("POST", "/api/upload", body, {"Content-Type": f"multipart/form-data; boundary={b}"})


def ocr(doc, page):
    t0 = time.time()
    st, s = req("POST", f"/api/docs/{doc}/pages/{page}/ocr")
    assert st == 200, s
    while s["status"] not in ("done", "error"):
        time.sleep(0.5)
        st, s = req("GET", f"/api/docs/{doc}/pages/{page}/status")
    assert s["status"] == "done", s
    st, res = req("GET", f"/api/docs/{doc}/pages/{page}")
    assert st == 200 and res["lines"], res
    st, img = req("GET", f"/api/docs/{doc}/pages/{page}/image", raw=True)
    assert st == 200 and img[:2] == b"\xff\xd8", "页面 JPEG 取不到"
    return res, time.time() - t0


def ok(msg):
    print("  ✓", msg, flush=True)


print("检查", BASE)
st, html = req("GET", "/", raw=True)
assert st == 200 and "西夏文 OCR".encode() in html
ok("首页")
st, info = req("GET", "/api/info")
assert st == 200 and info["model"].get("name"), info
ok(f"模型 {info['model'].get('name')}")

st, d = req("POST", "/api/sample")
assert st == 200, d
res, sec = ocr(d["doc"], 1)
ok(f"示例页识别 {len(res['lines'])} 行，{sec:.1f} 秒；第 2 行：{res['lines'][1]['text'][:20]}")

st, e = upload("bad.png", b"not an image", "image/png")
assert st == 400, (st, e)
ok(f"坏文件被拒：{e['error']}")

client = "smoke-" + uuid.uuid4().hex[:12]
fb = {"doc": d["doc"], "page": 1, "line": 1, "action": "confirm", "text": "", "client": client,
      "contributor": "自动检查", "consent": True}
st, out = req("POST", "/api/feedback", {**fb, "consent": False})
assert st == 400, out
st, out = req("POST", "/api/feedback", fb)
assert st == 200 and out["stats"]["mine"] == 1, out
ok(f"反馈提交（id {out['id']}）")

if TOKEN:
    H = {"Authorization": "Bearer " + TOKEN}
    st, _ = req("GET", "/api/admin/items")
    assert st == 401
    st, items = req("GET", "/api/admin/items?status=all&limit=5", headers=H)
    assert st == 200 and items["items"], items
    st, z = req("GET", "/api/admin/export?status=all", headers=H, raw=True)
    assert st == 200
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert "lines.jsonl" in names
    ok(f"后台：共 {items['total']} 条；导出包 {len(names)} 个文件")

if PDF:
    st, d = upload(PDF.replace("\\", "/").split("/")[-1], open(PDF, "rb").read(), "application/pdf")
    assert st == 200, d
    for p in sorted({1, min(2, d["pages"])}):
        res, sec = ocr(d["doc"], p)
        ok(f"PDF 第 {p}/{d['pages']} 页：{len(res['lines'])} 行，{sec:.1f} 秒")

print("全部通过")
