"""数据存放。

data/
  app.sqlite3            纠错反馈
  secret                 IP 散列用的随机盐（首次启动生成）
  docs/<doc>/            上传的文件与逐页结果，保留 RETENTION_HOURS 小时后自动删除
    meta.json  source.*  p<n>.png（识别用灰度图） p<n>.jpg（网页显示用） p<n>.json（识别结果）
  feedback/<年月>/<id>.png  反馈对应的行图（永久保留）
"""
import hashlib
import json
import secrets
import shutil
import sqlite3
import threading
import time

from . import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  client TEXT NOT NULL,          -- 浏览器本地生成的随机 id，同一人对同一行重复提交时覆盖
  contributor TEXT,              -- 自愿填写的署名
  ip_hash TEXT,
  doc TEXT NOT NULL, page INTEGER NOT NULL, line INTEGER NOT NULL,
  action TEXT NOT NULL,          -- confirm：模型读对了；correct：改正后的文字；bad：框有问题，不是一行文字
  model_text TEXT NOT NULL,
  text TEXT NOT NULL,
  box TEXT NOT NULL,             -- 行框 [x0,y0,x1,y1]，页面原图坐标
  crop TEXT NOT NULL,            -- 行图相对 feedback/ 的路径
  model TEXT,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending / approved / rejected
  reviewed_text TEXT,            -- 管理员修订过的文字（为空表示采用 text）
  reviewed_at REAL,
  UNIQUE (client, doc, page, line)
);
CREATE INDEX IF NOT EXISTS feedback_status ON feedback(status, id);
"""


class Store:
    def __init__(self, root=C.DATA_DIR):
        self.root = root
        self.docs = root / "docs"
        self.fb = root / "feedback"
        for d in (self.docs, self.fb):
            d.mkdir(parents=True, exist_ok=True)
        sec = root / "secret"
        if not sec.exists():
            sec.write_text(secrets.token_hex(16))
        self.salt = sec.read_text().strip()
        self.lock = threading.Lock()
        self.db = sqlite3.connect(root / "app.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def ip_hash(self, ip):
        return hashlib.sha256((self.salt + (ip or "")).encode()).hexdigest()[:16]

    # ---------------------------------------------------------------- 文档
    def new_doc(self, meta):
        doc = secrets.token_urlsafe(12)
        d = self.docs / doc
        d.mkdir()
        meta = {**meta, "created": time.time()}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return doc, d

    def doc_dir(self, doc):
        if not doc or not all(c.isalnum() or c in "-_" for c in doc):
            return None
        d = self.docs / doc
        return d if (d / "meta.json").exists() else None

    def meta(self, doc):
        d = self.doc_dir(doc)
        return json.loads((d / "meta.json").read_text(encoding="utf-8")) if d else None

    def cleanup(self):
        cutoff = time.time() - C.RETENTION_HOURS * 3600
        n = 0
        for d in self.docs.iterdir():
            m = d / "meta.json"
            try:
                created = json.loads(m.read_text(encoding="utf-8"))["created"] if m.exists() else d.stat().st_mtime
            except (OSError, ValueError, KeyError):
                created = d.stat().st_mtime
            if created < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        return n

    # ---------------------------------------------------------------- 反馈
    def save_feedback(self, row, crop_png):
        now = time.time()
        with self.lock:
            old = self.db.execute("SELECT id, crop FROM feedback WHERE client=? AND doc=? AND page=? AND line=?",
                                  (row["client"], row["doc"], row["page"], row["line"])).fetchone()
            if old:
                rel = old["crop"]
                self.db.execute("""UPDATE feedback SET updated=?, contributor=?, ip_hash=?, action=?, text=?,
                                   status='pending', reviewed_text=NULL, reviewed_at=NULL WHERE id=?""",
                                (now, row["contributor"], row["ip_hash"], row["action"], row["text"], old["id"]))
                fid = old["id"]
            else:
                month = time.strftime("%Y%m")
                (self.fb / month).mkdir(exist_ok=True)
                rel = f"{month}/{secrets.token_hex(8)}.png"
                cur = self.db.execute(
                    """INSERT INTO feedback (created, updated, client, contributor, ip_hash, doc, page, line, action,
                       model_text, text, box, crop, model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (now, now, row["client"], row["contributor"], row["ip_hash"], row["doc"], row["page"], row["line"],
                     row["action"], row["model_text"], row["text"], json.dumps(row["box"]), rel, row["model"]))
                fid = cur.lastrowid
            (self.fb / rel).write_bytes(crop_png)
            self.db.commit()
        return fid

    def stats(self, client=None):
        q = self.db.execute("""SELECT COUNT(*) n, COUNT(DISTINCT client) people,
                               SUM(status='approved') approved, SUM(status='pending') pending FROM feedback""").fetchone()
        out = {"lines": q["n"], "contributors": q["people"], "approved": q["approved"] or 0, "pending": q["pending"] or 0}
        if client:
            out["mine"] = self.db.execute("SELECT COUNT(*) FROM feedback WHERE client=?", (client,)).fetchone()[0]
        return out

    def list_feedback(self, status, action, offset, limit):
        where, args = [], []
        if status != "all":
            where.append("status=?")
            args.append(status)
        if action != "all":
            where.append("action=?")
            args.append(action)
        w = ("WHERE " + " AND ".join(where)) if where else ""
        total = self.db.execute(f"SELECT COUNT(*) FROM feedback {w}", args).fetchone()[0]
        rows = self.db.execute(f"SELECT * FROM feedback {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                               args + [limit, offset]).fetchall()
        return total, [dict(r) for r in rows]

    def review(self, ids, status, text=None):
        now = time.time()
        with self.lock:
            for i in ids:
                if text is None:
                    self.db.execute("UPDATE feedback SET status=?, reviewed_at=? WHERE id=?", (status, now, i))
                else:
                    self.db.execute("UPDATE feedback SET status=?, reviewed_text=?, reviewed_at=? WHERE id=?",
                                    (status, text, now, i))
            self.db.commit()

    def export_rows(self, status):
        q = "SELECT * FROM feedback WHERE action != 'bad'" + ("" if status == "all" else " AND status=?")
        return [dict(r) for r in self.db.execute(q + " ORDER BY id", () if status == "all" else (status,))]
