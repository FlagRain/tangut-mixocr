/* 审核页：逐条采用/拒绝/修改用户提交的校对，导出训练数据。 */
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const chars = (s) => Array.from(s || "");
const A = { token: sessionStorage.getItem("tocr.admin") || "", status: "pending", action: "all", offset: 0, items: [], cur: -1 };
const PAGE = 30;

let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (err ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), 2400);
}
async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), Authorization: "Bearer " + A.token } });
  if (r.status === 401) { logout(); throw new Error("口令不对或已失效"); }
  const data = r.headers.get("content-type")?.includes("json") ? await r.json() : null;
  if (!r.ok) throw new Error((data && data.error) || `服务器出错（${r.status}）`);
  return data ?? r;
}

/* 逐字差异：最长公共子序列 */
function diff(a, b) {
  a = chars(a); b = chars(b);
  const n = a.length, m = b.length, D = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    D[i][j] = a[i] === b[j] ? D[i + 1][j + 1] + 1 : Math.max(D[i + 1][j], D[i][j + 1]);
  let i = 0, j = 0, x = "", y = "";
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { x += esc(a[i]); y += esc(b[j]); i++; j++; }
    else if (j < m && (i === n || D[i][j + 1] >= D[i + 1][j])) { y += `<ins>${esc(b[j])}</ins>`; j++; }
    else { x += `<del>${esc(a[i])}</del>`; i++; }
  }
  return [x, y];
}

function showStats(st) {
  const el = $("#statLine");
  el.hidden = false;
  el.innerHTML = `待审 <b>${st.pending}</b> · 已采用 <b>${st.approved}</b> · 共 ${st.lines} 条 · ${st.contributors} 位提交者`;
}

async function load(append = false) {
  if (!append) { A.offset = 0; A.items = []; A.cur = -1; }
  try {
    const d = await api(`/api/admin/items?status=${A.status}&action=${A.action}&offset=${A.offset}&limit=${PAGE}`);
    A.items = A.items.concat(d.items);
    A.offset += d.items.length;
    showStats(d.stats);
    render();
    $("#moreBtn").hidden = A.offset >= d.total;
    $("#countNote").textContent = d.total ? `已显示 ${A.items.length} / ${d.total} 条` : "";
    if (!append && A.items.length) setCur(0, false);
  } catch (e) { toast(e.message, true); }
}

function card(it, k) {
  const final = it.reviewed_text ?? it.text;
  const [m, t] = it.action === "bad" ? [esc(it.model_text), ""] : diff(it.model_text, final);
  const when = new Date(it.updated * 1000).toLocaleString("zh-CN", { hour12: false });
  const label = { correct: "改正", confirm: "确认无误", bad: "框有问题" }[it.action];
  return `<div class="card st-${it.status}" data-k="${k}">
    ${it.image ? `<img src="${it.image}" alt="">` : ""}
    <div class="row"><span class="k">模型</span><span class="v ${it.model_text ? "" : "empty"}">${m || "（空）"}</span></div>
    ${it.action === "bad" ? `<div class="row"><span class="k">提交</span><span class="v empty">用户认为这里不是一行文字（不会导出）</span></div>`
      : `<div class="row"><span class="k">${it.reviewed_text != null ? "审定" : "提交"}</span><span class="v">${t}</span></div>`}
    <div class="edit" hidden><input value="${esc(final)}"><button class="primary" data-op="save">保存并采用</button></div>
    <div class="foot">
      <span class="meta"><span class="tag ${it.action}">${label}</span>#${it.id} · ${esc(it.contributor || "匿名")} · ${when} · 第 ${it.page} 页第 ${it.line + 1} 行</span>
      ${it.action !== "bad" ? `<button class="ghost" data-op="edit">修改</button>` : ""}
      <button class="ghost" data-op="rejected">拒绝</button>
      <button class="secondary" data-op="approved">采用</button>
    </div>
  </div>`;
}
function render() {
  $("#cards").innerHTML = A.items.length ? A.items.map(card).join("")
    : `<p style="text-align:center;color:var(--muted);padding:40px 0">这里没有条目。</p>`;
  if (A.cur >= 0) setCur(A.cur, false);
}
function setCur(k, scroll = true) {
  A.cur = Math.max(0, Math.min(A.items.length - 1, k));
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("cur", +c.dataset.k === A.cur));
  const el = document.querySelector(`.card[data-k="${A.cur}"]`);
  if (el && scroll) el.scrollIntoView({ block: "center", behavior: "smooth" });
}

async function review(k, status, text) {
  const it = A.items[k];
  try {
    const d = await api("/api/admin/review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [it.id], status, ...(text !== undefined ? { text } : {}) }),
    });
    it.status = status;
    if (text !== undefined) it.reviewed_text = text;
    showStats(d.stats);
    const el = document.querySelector(`.card[data-k="${k}"]`);
    el.outerHTML = card(it, k);
    setCur(k + 1);
  } catch (e) { toast(e.message, true); }
}

$("#cards").addEventListener("click", (e) => {
  const c = e.target.closest(".card");
  if (!c) return;
  const k = +c.dataset.k, b = e.target.closest("button");
  setCur(k, false);
  if (!b) return;
  const op = b.dataset.op;
  if (op === "edit") { c.querySelector(".edit").hidden = false; c.querySelector(".edit input").focus(); }
  else if (op === "save") review(k, "approved", c.querySelector(".edit input").value.trim());
  else review(k, op);
});
$("#cards").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.matches(".edit input")) {
    review(+e.target.closest(".card").dataset.k, "approved", e.target.value.trim());
  }
});
document.addEventListener("keydown", (e) => {
  if ($("#adm").hidden || e.target.matches("input, select") || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "j") setCur(A.cur + 1);
  else if (k === "k") setCur(A.cur - 1);
  else if (A.cur >= 0 && k === "a") review(A.cur, "approved");
  else if (A.cur >= 0 && k === "r") review(A.cur, "rejected");
  else if (A.cur >= 0 && k === "e") {
    const c = document.querySelector(`.card[data-k="${A.cur}"] .edit`);
    if (c) { e.preventDefault(); c.hidden = false; c.querySelector("input").focus(); }
  }
});

$("#tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  A.status = b.dataset.s;
  load();
});
$("#actionSel").addEventListener("change", (e) => { A.action = e.target.value; load(); });
$("#moreBtn").addEventListener("click", () => load(true));
$("#bulkBtn").addEventListener("click", async () => {
  const ids = A.items.filter((it) => it.status !== "approved" && it.action !== "bad").map((it) => it.id);
  if (!ids.length) return toast("本页没有可采用的条目");
  if (!confirm(`把本页 ${ids.length} 条（不含“框有问题”）都标为采用？`)) return;
  try {
    await api("/api/admin/review", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids, status: "approved" }) });
    toast(`已采用 ${ids.length} 条`);
    load();
  } catch (e) { toast(e.message, true); }
});
$("#exportBtn").addEventListener("click", async () => {
  try {
    toast("正在打包…");
    const r = await api("/api/admin/export?status=approved");
    const blob = await r.blob();
    const name = (r.headers.get("content-disposition") || "").match(/filename="?([^";]+)/)?.[1] || "feedback.zip";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = decodeURIComponent(name);
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  } catch (e) { toast(e.message, true); }
});

function logout() {
  sessionStorage.removeItem("tocr.admin");
  A.token = "";
  $("#adm").hidden = true;
  $("#logoutBtn").hidden = true;
  $("#statLine").hidden = true;
  $("#login").hidden = false;
}
$("#logoutBtn").addEventListener("click", logout);
$("#loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  A.token = $("#tokenInput").value.trim();
  try {
    await api("/api/admin/items?limit=1");
    sessionStorage.setItem("tocr.admin", A.token);
    enter();
  } catch (err) { toast(err.message, true); }
});
function enter() {
  $("#login").hidden = true;
  $("#adm").hidden = false;
  $("#logoutBtn").hidden = false;
  load();
}
if (A.token) enter();
