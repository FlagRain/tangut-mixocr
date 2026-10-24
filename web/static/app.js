/* 西夏文 OCR —— 识别与校对页面
 * 西夏字在 Unicode 辅助平面，JS 字符串里是两个 UTF-16 单元：凡是按“字”处理的地方一律用 Array.from。
 */
"use strict";

const $ = (s) => document.querySelector(s);
const LOW = 0.6;                         // 低于这个概率的字标黄
const store = {
  get(k, d) { try { const v = localStorage.getItem("tocr." + k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("tocr." + k, v); } catch { /* 隐私模式 */ } },
};

const S = {
  client: store.get("client", null),
  doc: null, kind: null, pages: 1, name: "", page: 1,
  results: {},          // 页码 → 识别结果
  marks: {},            // "页:行" → {action, text}
  sel: -1,              // 当前校对的行
  chip: -1,             // 选中的字（查字结果会替换它）
  caret: null,          // 输入框里最后的光标位置
  zoom: "fit",
  lookup: null,
  pollTimer: null,
};
if (!S.client) {
  S.client = (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36));
  store.set("client", S.client);
}

/* ---------------------------------------------------------------- 工具 */
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch { /* 非 JSON */ }
  if (!r.ok) throw new Error((data && data.error) || `服务器出错（${r.status}）`);
  return data;
}
let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (err ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast" + (err ? " err" : "")), err ? 4200 : 2200);
}
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const chars = (s) => Array.from(s || "");
const isTangut = (ch) => { const c = ch.codePointAt(0); return c >= 0x17000 && c <= 0x18dff; };
const pct = (p) => (p >= 0.995 ? ">99" : p < 0.01 ? "<1" : Math.round(p * 100)) + "%";

/* ---------------------------------------------------------------- 统计 */
function showStats(st) {
  if (!st) return;
  if (st.lines > 0) {
    $("#community").hidden = false;
    $("#commLines").textContent = st.lines.toLocaleString();
    $("#commPeople").textContent = st.contributors.toLocaleString();
  }
  if (st.mine) {
    $("#mine").hidden = false;
    $("#mine b").textContent = st.mine;
  }
}
async function loadInfo() {
  try {
    const info = await api("/api/info?client=" + encodeURIComponent(S.client));
    showStats(info.stats);
    $("#retention").textContent = info.limits.retention_hours;
    $("#lookup").hidden = !(info.features && info.features.lookup);       // 没装查字表就不显示查字面板
    const m = (info.model && info.model.metrics) || {};
    const b = (info.model && info.model.baseline_v8) || {};
    const cell = (v, label, base) => v == null ? "" :
      `<div class="metric"><b>${(v * 100).toFixed(1)}%</b><span>${label}${base != null ? `（旧模型 ${(base * 100).toFixed(1)}%）` : ""}</span></div>`;
    $("#metrics").innerHTML = cell(m["书页西夏字"], "扫描书页 · 西夏字", b["书页西夏字"]) +
      cell(m["书页汉字*"], "扫描书页 · 汉字", b["书页汉字*"]) + cell(m["词典字头"], "词典扫描 · 字头");
  } catch { /* 首页统计不是必须的 */ }
}

/* ---------------------------------------------------------------- 上传 */
function upload(file) {
  if (!file) return;
  const bar = $("#upBar"), fill = bar.querySelector("i");
  bar.hidden = false;
  fill.style.width = "0";
  $("#dropTitle").textContent = `正在上传 ${file.name || "图片"}…`;
  const fd = new FormData();
  fd.append("file", file, file.name || "paste.png");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) fill.style.width = (e.loaded / e.total * 100) + "%"; };
  xhr.onload = () => {
    resetDrop();
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch { /* */ }
    if (xhr.status !== 200) return toast((data && data.error) || `上传失败（${xhr.status}）`, true);
    openDoc(data);
  };
  xhr.onerror = () => { resetDrop(); toast("上传失败，请检查网络", true); };
  xhr.send(fd);
}
function resetDrop() {
  $("#upBar").hidden = true;
  $("#dropTitle").textContent = "把图片或 PDF 拖到这里，或点击选择";
}

$("#file").addEventListener("change", (e) => { upload(e.target.files[0]); e.target.value = ""; });
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); upload(e.dataTransfer.files[0]); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  if (!$("#home").hidden && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
});
document.addEventListener("paste", (e) => {
  if ($("#home").hidden) return;
  const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (f) upload(f);
});
$("#sampleBtn").addEventListener("click", async () => {
  try { openDoc(await api("/api/sample", { method: "POST" })); } catch (e) { toast(e.message, true); }
});

/* ---------------------------------------------------------------- 文档与翻页 */
function openDoc(d) {
  Object.assign(S, { doc: d.doc, kind: d.kind, pages: d.pages, name: d.name, results: {}, marks: {} });
  $("#home").hidden = true;
  $("#work").hidden = false;
  $("#docName").textContent = d.name;
  $("#docName").title = d.name;
  $("#pager").hidden = d.pages <= 1;
  $("#pageCount").textContent = d.pages;
  $("#pageInput").max = d.pages;
  history.pushState({ work: true }, "");
  goPage(1);
}
function goHome() {
  closeEditor();
  clearTimeout(S.pollTimer);
  $("#work").hidden = true;
  $("#home").hidden = false;
  S.doc = null;
  loadInfo();
}
$("#newBtn").addEventListener("click", () => history.back());
window.addEventListener("popstate", () => { if (!$("#work").hidden) goHome(); });

async function goPage(n) {
  n = Math.max(1, Math.min(S.pages, n | 0));
  S.page = n;
  $("#pageInput").value = n;
  $("#prevPage").disabled = n <= 1;
  $("#nextPage").disabled = n >= S.pages;
  closeEditor();
  clearTimeout(S.pollTimer);
  $("#lines").innerHTML = "";
  $("#overlay").innerHTML = "";
  $("#lineCount").textContent = "";
  if (S.results[n]) return render();
  $("#pageImg").removeAttribute("src");
  busy("正在准备这一页…");
  try {
    const st = await api(`/api/docs/${S.doc}/pages/${n}/ocr`, { method: "POST" });
    poll(n, st);
  } catch (e) { busy(null); showLinesNote(e.message); toast(e.message, true); }
}
function poll(n, st) {
  if (n !== S.page || !S.doc) return;
  if (st.status === "done") return fetchResult(n);
  if (st.status === "error") { busy(null); showLinesNote(st.error || "识别失败", true); return; }
  busy(st.status === "queued" ? (st.ahead ? `排队中，前面还有 ${st.ahead} 页…` : "马上开始识别…")
    : "正在识别这一页，通常需要 5–20 秒…");
  S.pollTimer = setTimeout(async () => {
    try { poll(n, await api(`/api/docs/${S.doc}/pages/${n}/status`)); }
    catch (e) { busy(null); showLinesNote(e.message); }
  }, 900);
}
async function fetchResult(n) {
  try {
    S.results[n] = await api(`/api/docs/${S.doc}/pages/${n}`);
    if (n === S.page) render();
  } catch (e) { busy(null); showLinesNote(e.message); }
}
function busy(msg) {
  $("#busy").hidden = !msg;
  if (msg) $("#busyText").textContent = msg;
}
function showLinesNote(msg, retry = false) {
  $("#lines").innerHTML = `<li class="empty-note" style="display:block">${esc(msg)}${retry ? '<br><button class="ghost" id="retryBtn" style="margin-top:10px">重新识别</button>' : ""}</li>`;
  if (retry) $("#retryBtn").onclick = () => goPage(S.page);
}
$("#prevPage").addEventListener("click", () => goPage(S.page - 1));
$("#nextPage").addEventListener("click", () => goPage(S.page + 1));
$("#pageInput").addEventListener("change", (e) => goPage(+e.target.value));

/* ---------------------------------------------------------------- 显示结果 */
const res = () => S.results[S.page];
const mark = (i) => S.marks[`${S.page}:${i}`];
const lineText = (i) => { const m = mark(i); return m && m.action === "correct" ? m.text : res().lines[i].text; };

function render() {
  const r = res();
  const img = $("#pageImg");
  img.onload = () => { busy(null); applyZoom(); };
  img.onerror = () => { busy(null); toast("页面图像加载失败", true); };
  img.src = `/api/docs/${S.doc}/pages/${S.page}/image`;
  if (img.complete && img.naturalWidth) img.onload();
  const svg = $("#overlay");
  svg.setAttribute("viewBox", `0 0 ${r.width} ${r.height}`);
  svg.innerHTML = r.lines.map((l, i) => {
    const [x0, y0, x1, y1] = l.box;
    return `<rect data-i="${i}" class="line${l.vertical ? " vertical" : ""}" x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" rx="3"><title>第 ${i + 1} 行</title></rect>`;
  }).join("");
  const n = r.lines.filter((l) => !l.vertical).length;
  $("#lineCount").textContent = r.lines.length ? `${n} 行 · 用时 ${r.seconds} 秒` : "";
  if (!r.lines.length) return showLinesNote("这一页没有找到文字。");
  $("#lines").innerHTML = r.lines.map((l, i) => `<li data-i="${i}">${lineRow(i)}</li>`).join("");
  refreshMarks();
}
function lineRow(i) {
  const l = res().lines[i];
  const m = mark(i);
  let body;
  if (l.vertical) body = `<span class="t muted">竖排文字，暂不支持识别</span>`;
  else if (m && m.action === "correct") body = `<span class="t">${esc(m.text)}</span>`;
  else body = `<span class="t">${l.chars.map((c) => c.p < LOW ? `<span class="low">${esc(c.c)}</span>` : esc(c.c)).join("") || "&nbsp;"}</span>`;
  const badge = m ? `<span class="badge ${m.action}">${{ confirm: "✓ 已确认", correct: "已改正", bad: "框有问题" }[m.action]}</span>` : "<span></span>";
  return `<span class="no">${i + 1}</span>${body}${badge}`;
}
function refreshMarks() {
  document.querySelectorAll("#overlay rect").forEach((r) => {
    const i = +r.dataset.i;
    r.classList.toggle("done", !!mark(i));
    r.classList.toggle("sel", i === S.sel);
  });
  document.querySelectorAll("#lines li[data-i]").forEach((li) => li.classList.toggle("sel", +li.dataset.i === S.sel));
}
$("#overlay").addEventListener("click", (e) => { const r = e.target.closest("rect"); if (r) selectLine(+r.dataset.i, "list"); });
$("#lines").addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) selectLine(+li.dataset.i, "image"); });

/* 缩放 */
function applyZoom() {
  const viewer = $("#viewer"), img = $("#pageImg"), stage = $("#stage");
  if (!img.naturalWidth) return;
  const fit = Math.max(200, viewer.clientWidth - 32);
  const w = S.zoom === "fit" ? fit : Math.round(fit * S.zoom);
  stage.style.width = w + "px";
}
document.querySelectorAll(".zoom button").forEach((b) => b.addEventListener("click", () => {
  const z = b.dataset.z, cur = S.zoom === "fit" ? 1 : S.zoom;
  S.zoom = z === "fit" ? "fit" : Math.max(0.5, Math.min(5, cur * (z === "in" ? 1.3 : 1 / 1.3)));
  if (S.zoom !== "fit" && Math.abs(S.zoom - 1) < 0.01) S.zoom = "fit";
  applyZoom();
}));
window.addEventListener("resize", () => { applyZoom(); if (S.sel >= 0) drawCrop(); });

/* ---------------------------------------------------------------- 校对面板 */
function selectLine(i, scrollTarget) {
  const r = res();
  if (!r || i < 0 || i >= r.lines.length) return;
  if (r.lines[i].vertical) return toast("竖排文字暂不支持识别");
  S.sel = i;
  S.chip = -1;
  hidePop();
  refreshMarks();
  $("#editor").hidden = false;
  document.body.classList.add("editing");
  $("#edTitle").textContent = `第 ${i + 1} 行（共 ${r.lines.length} 行）`;
  const m = mark(i), badge = $("#edBadge");
  badge.hidden = !m;
  if (m) { badge.className = "badge " + m.action; badge.textContent = { confirm: "已确认", correct: "已改正", bad: "已标为框有问题" }[m.action]; }
  const input = $("#lineInput");
  input.value = lineText(i);
  S.caret = null;
  drawCrop();
  renderChips();
  if (matchMedia("(pointer: fine)").matches) input.focus({ preventScroll: true });
  // 两边都滚到这一行
  const li = document.querySelector(`#lines li[data-i="${i}"]`);
  if (li) li.scrollIntoView({ block: "nearest", behavior: scrollTarget === "list" ? "smooth" : "auto" });
  if (scrollTarget !== "none") {
    const viewer = $("#viewer"), stage = $("#stage");
    const s = stage.clientWidth / r.width, [x0, y0] = r.lines[i].box;
    viewer.scrollTo({ top: stage.offsetTop + y0 * s - viewer.clientHeight * 0.3, left: Math.max(0, stage.offsetLeft + x0 * s - 40), behavior: "smooth" });
  }
  requestAnimationFrame(syncEditorHeight);
}
function closeEditor() {
  S.sel = -1;
  hidePop();
  $("#editor").hidden = true;
  document.body.classList.remove("editing");
  syncEditorHeight();
  if (res()) refreshMarks();
}
function syncEditorHeight() {
  const h = $("#editor").hidden ? 0 : $("#editor").offsetHeight;
  if (matchMedia("(max-width: 860px)").matches) { $("#work").style.paddingBottom = h + "px"; $("#work").style.height = ""; }
  else { $("#work").style.height = `calc(100vh - 56px - ${h}px)`; $("#work").style.paddingBottom = ""; }
}
new ResizeObserver(syncEditorHeight).observe($("#editor"));

function cropGeom() {
  const r = res(), l = r.lines[S.sel], [x0, y0, x1, y1] = l.box;
  const m = Math.max(2, Math.floor((y1 - y0) / 8));
  return { r, l, cx0: Math.max(0, x0 - m), cy0: Math.max(0, y0 - m), cx1: Math.min(r.width, x1 + m), cy1: Math.min(r.height, y1 + m) };
}
function drawCrop() {
  if (S.sel < 0) return;
  const img = $("#pageImg"), cv = $("#cropCanvas");
  if (!img.naturalWidth) return;
  const { r, cx0, cy0, cx1, cy1 } = cropGeom();
  const k = img.naturalWidth / r.width;                     // 显示图与原图的比例
  const box = $("#crop"), W = box.clientWidth;
  const natW = (cx1 - cx0) * k, natH = (cy1 - cy0) * k;
  const maxH = Math.min(150, window.innerHeight * 0.16);
  const scale = Math.min(W / natW, maxH / natH, 3);
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(natW * scale * dpr);
  cv.height = Math.round(natH * scale * dpr);
  cv.style.width = Math.round(natW * scale) + "px";
  cv.style.height = Math.round(natH * scale) + "px";
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, cx0 * k, cy0 * k, natW, natH, 0, 0, cv.width, cv.height);
  box.querySelectorAll(".mark").forEach((e) => e.remove());
}
function markChar(i) {
  const box = $("#crop");
  box.querySelectorAll(".mark").forEach((e) => e.remove());
  const { l, cx0, cx1 } = cropGeom();
  if (i < 0 || !aligned() || !l.chars[i]) return;
  const c = l.chars[i], w = $("#cropCanvas").clientWidth / (cx1 - cx0);
  const d = document.createElement("div");
  d.className = "mark";
  d.style.left = (c.x0 - cx0) * w + "px";
  d.style.width = Math.max(4, (c.x1 - c.x0) * w) + "px";
  box.appendChild(d);
}
/* 输入框的字数和模型输出一样时，第 i 个字仍对应模型的第 i 个字，可以用它的候选 */
function aligned() {
  const l = res().lines[S.sel];
  return chars($("#lineInput").value).length === l.chars.length;
}
function renderChips() {
  const l = res().lines[S.sel], cs = chars($("#lineInput").value), al = aligned();
  $("#chips").innerHTML = cs.map((ch, i) => {
    const low = al && l.chars[i] && l.chars[i].p < LOW && ch === l.chars[i].c;
    const cls = ["chip", ch === " " ? "space" : "", low ? "low" : "", i === S.chip ? "sel" : ""].join(" ");
    return `<button class="${cls}" data-i="${i}" title="${al && l.chars[i] ? "把握 " + pct(l.chars[i].p) : ""}">${ch === " " ? "空格" : esc(ch)}</button>`;
  }).join("");
}
$("#chips").addEventListener("mouseover", (e) => { const b = e.target.closest(".chip"); if (b) markChar(+b.dataset.i); });
$("#chips").addEventListener("mouseleave", () => markChar(S.chip));
$("#chips").addEventListener("click", (e) => {
  const b = e.target.closest(".chip");
  if (!b) return;
  const i = +b.dataset.i;
  S.chip = S.chip === i ? -1 : i;
  renderChips();                                     // 重画后原来的按钮已不在页面上，要重新取
  markChar(S.chip);
  if (S.chip >= 0) showPop(document.querySelector(`.chip[data-i="${i}"]`), i); else hidePop();
});
$("#lineInput").addEventListener("input", () => { S.chip = -1; hidePop(); renderChips(); });
["keyup", "click", "blur"].forEach((t) => $("#lineInput").addEventListener(t, (e) => { S.caret = e.target.selectionStart; }));

/* 候选字浮层 */
function showPop(anchor, i) {
  const l = res().lines[S.sel], pop = $("#pop");
  const cur = chars($("#lineInput").value)[i];
  let alts = [];
  if (aligned() && l.chars[i]) {
    const mc = l.chars[i];
    alts = [[mc.c, mc.p], ...mc.alt].filter(([c]) => c !== cur && c.trim());
  }
  pop.innerHTML = `<div class="ttl">${alts.length ? "模型认为可能是：" : "这个位置没有候选字"}</div>
    <div class="alts">${alts.map(([c, p]) => `<button class="alt" data-c="${esc(c)}"><span class="g">${esc(c)}</span><span class="p">${pct(p)}</span></button>`).join("")}</div>
    <div class="ops"><button class="ghost" data-op="del">删除这个字</button><button class="ghost" data-op="look">查字替换…</button></div>`;
  pop.hidden = false;
  const a = anchor.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
  pop.style.left = Math.max(8, Math.min(window.innerWidth - pw - 8, a.left)) + "px";
  pop.style.top = (a.top - ph - 6 > 8 ? a.top - ph - 6 : a.bottom + 6) + "px";
}
function hidePop() { $("#pop").hidden = true; }
$("#pop").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || S.chip < 0) return;
  const cs = chars($("#lineInput").value);
  if (b.dataset.c) cs[S.chip] = b.dataset.c;
  else if (b.dataset.op === "del") { cs.splice(S.chip, 1); S.chip = -1; }
  else if (b.dataset.op === "look") {
    hidePop();
    $("#lookup").open = true;
    $("#lookupInput").focus();
    loadLookup();
    return;
  }
  $("#lineInput").value = cs.join("");
  hidePop();
  renderChips();
  markChar(S.chip);
});
document.addEventListener("mousedown", (e) => {
  if (!$("#pop").hidden && !e.target.closest("#pop") && !e.target.closest(".chip")) hidePop();
});

/* ---------------------------------------------------------------- 查西夏字 */
const fold = (s) => s.normalize("NFD").replace(/[̀-ͯ·ˑˑ]/g, "")
  .replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹0-9]/g, "").toLowerCase();
async function loadLookup() {
  if (S.lookup) return S.lookup;
  try {
    const rows = await (await fetch("/static/lookup.json")).json();
    const toneless = (s) => s.normalize("NFC").replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹0-9]/g, "").toLowerCase();
    S.lookup = rows.map(([c, r, zh, en]) => {
      const rs = r.split(/[\s/]+/).filter(Boolean);
      return { c, r, zh, en, exact: rs.map(toneless), keys: rs.map(fold),
               words: zh.split(/[\s、，,；;.。（）()【】0-9]+/).filter(Boolean) };
    });
  } catch { toast("查字表加载失败", true); }
  return S.lookup;
}
$("#lookup").addEventListener("toggle", () => { if ($("#lookup").open) loadLookup(); });
let lookTimer;
$("#lookupInput").addEventListener("input", () => { clearTimeout(lookTimer); lookTimer = setTimeout(doLookup, 120); });
async function doLookup() {
  const q = $("#lookupInput").value.trim(), out = $("#lookupResults");
  if (!q) { out.innerHTML = ""; return; }
  const L = await loadLookup();
  if (!L) return;
  let hits = [];
  const u = q.match(/^(?:u\+)?([0-9a-f]{5})$/i);
  if (u) hits = L.filter((e) => e.c.codePointAt(0) === parseInt(u[1], 16));
  else if (chars(q).some(isTangut)) hits = L.filter((e) => q.includes(e.c));
  else if (/[㐀-鿿]/.test(q)) {
    // 释义里单独成项的字排最前（如“1.身2.形”里的“身”），其次是以它开头的，再次是包含它的
    const rank = (e) => e.words.includes(q) ? 0 : e.words.some((w) => w.startsWith(q)) ? 1 : e.zh.includes(q) ? 2 : 9;
    hits = L.map((e) => [rank(e), e]).filter(([r]) => r < 9).sort((a, b) => a[0] - b[0]).map(([, e]) => e);
  } else {
    // 带附加符号完全一致 > 去掉附加符号一致 > 前缀 > 英文释义
    const f = fold(q), lq = q.normalize("NFC").replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹0-9]/g, "").toLowerCase();
    const enRe = new RegExp("\\b" + lq.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const rank = (e) => e.exact.includes(lq) ? 0 : e.keys.includes(f) ? 1 : e.keys.some((k) => k.startsWith(f)) ? 2
      : enRe.test(e.en.toLowerCase()) ? 3 : 9;
    hits = L.map((e) => [rank(e), e]).filter(([r]) => r < 9).sort((a, b) => a[0] - b[0]).map(([, e]) => e);
  }
  out.innerHTML = hits.slice(0, 80).map((e) =>
    `<button class="lk" data-c="${e.c}" title="${esc(e.r)}\n${esc(e.zh)} ${esc(e.en)}"><span class="g">${e.c}</span>` +
    `<span class="m"><b>${esc(e.r.split(" / ")[0] || "")}</b><br>${esc(e.zh || e.en)}</span></button>`).join("")
    || `<span style="color:var(--muted);font-size:13px">没有找到</span>`;
}
$("#lookupResults").addEventListener("click", (e) => {
  const b = e.target.closest(".lk");
  if (!b) return;
  const input = $("#lineInput"), cs = chars(input.value);
  hidePop();
  if (S.chip >= 0 && S.chip < cs.length) {           // 替换选中的字
    cs[S.chip] = b.dataset.c;
    input.value = cs.join("");
  } else {                                            // 插到光标处
    const at = S.caret ?? input.value.length;
    input.setRangeText(b.dataset.c, at, at, "end");
    S.caret = at + b.dataset.c.length;
  }
  renderChips();
  markChar(S.chip);
  toast(`已填入 ${b.dataset.c}`);
});

/* ---------------------------------------------------------------- 提交 */
function needConsent() {
  return new Promise((resolve) => {
    if (store.get("consent", "") === "1") return resolve(true);
    const dlg = $("#consentDlg");
    $("#nameInput").value = store.get("name", "");
    $("#consentChk").checked = false;
    $("#consentOk").disabled = true;
    dlg.onclose = () => {
      if (dlg.returnValue === "ok" && $("#consentChk").checked) {
        store.set("consent", "1");
        store.set("name", $("#nameInput").value.trim());
        resolve(true);
      } else resolve(false);
    };
    dlg.showModal();
  });
}
$("#consentChk").addEventListener("change", (e) => { $("#consentOk").disabled = !e.target.checked; });

async function submit(action) {
  if (S.sel < 0) return;
  const i = S.sel, l = res().lines[i];
  let text = $("#lineInput").value.replace(/\s+/g, " ").trim();
  if (action === "correct" && text === l.text) action = "confirm";
  if (action === "correct" && !text) return toast("文字不能为空；如果这里不是一行文字，请点“这里不是一行文字”", true);
  if (!(await needConsent())) return;
  const btns = ["#submitBtn", "#confirmBtn", "#badBtn"].map((s) => $(s));
  btns.forEach((b) => (b.disabled = true));
  try {
    const out = await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc: S.doc, page: S.page, line: i, action, text, client: S.client,
        contributor: store.get("name", ""), consent: true }),
    });
    S.marks[`${S.page}:${i}`] = { action, text: action === "confirm" ? l.text : text };
    const li = document.querySelector(`#lines li[data-i="${i}"]`);
    if (li) li.innerHTML = lineRow(i);
    showStats(out.stats);
    toast({ confirm: "已确认，谢谢！", correct: "已提交改正，谢谢！", bad: "已标记，谢谢！" }[action]);
    nextLine(1);
  } catch (e) { toast(e.message, true); }
  finally { btns.forEach((b) => (b.disabled = false)); }
}
function nextLine(d) {
  const r = res();
  for (let j = S.sel + d; j >= 0 && j < r.lines.length; j += d) {
    if (!r.lines[j].vertical) return selectLine(j);
  }
  if (d > 0) {
    closeEditor();
    toast(S.page < S.pages ? "这一页校对完了，可以翻到下一页" : "这一页校对完了");
  }
}
$("#submitBtn").addEventListener("click", () => submit("correct"));
$("#confirmBtn").addEventListener("click", () => submit("confirm"));
$("#badBtn").addEventListener("click", () => submit("bad"));
$("#edPrev").addEventListener("click", () => nextLine(-1));
$("#edNext").addEventListener("click", () => nextLine(1));
$("#edClose").addEventListener("click", closeEditor);
$("#lineInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); submit("correct"); }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!$("#pop").hidden) return hidePop();
    if (S.sel >= 0 && !document.querySelector("dialog[open]")) closeEditor();
  } else if (e.altKey && (e.key === "ArrowDown" || e.key === "ArrowUp") && S.sel >= 0) {
    e.preventDefault();
    nextLine(e.key === "ArrowDown" ? 1 : -1);
  }
});

/* ---------------------------------------------------------------- 导出文字 */
function allText() {
  const pages = Object.keys(S.results).map(Number).sort((a, b) => a - b);
  return pages.map((p) => {
    const r = S.results[p];
    const body = r.lines.map((l, i) => {
      const m = S.marks[`${p}:${i}`];
      if (l.vertical || (m && m.action === "bad")) return null;
      return m && m.action === "correct" ? m.text : l.text;
    }).filter((t) => t !== null).join("\n");
    return pages.length > 1 ? `—— 第 ${p} 页 ——\n${body}` : body;
  }).join("\n\n");
}
$("#copyBtn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(allText()); toast("已复制到剪贴板"); }
  catch { toast("复制失败，请用“下载 TXT”", true); }
});
$("#txtBtn").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + allText()], { type: "text/plain;charset=utf-8" }));
  a.download = (S.name || "ocr").replace(/\.[^.]+$/, "") + ".txt";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

/* ---------------------------------------------------------------- 说明 */
$("#aboutBtn").addEventListener("click", () => $("#aboutDlg").showModal());

loadInfo();
