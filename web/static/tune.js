/* 调样式面板 —— 只在本机打开网站时加载（index.html 末尾按网址判断），线上访客看不到。
 * 从 style.css 开头的 :root 读出全部变量，改动立刻作用在页面上，并记在浏览器里，刷新不丢。
 * 用 tools/preview.py 预览时可以一键写回 style.css；否则“复制改动”后把值抄回 style.css。
 */
"use strict";
(() => {
  if (!/^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)) return;

  const KEY = "tune.v1";
  const st = { open: false, left: false, scheme: "auto", light: {}, dark: {} };
  try { Object.assign(st, JSON.parse(localStorage.getItem(KEY)) || {}); } catch { /* 隐私模式 */ }
  const remember = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch { /* 隐私模式 */ } };

  // 已知变量的中文名和分组；style.css 里以后新加的变量会自动出现在“其他”里
  const GROUPS = [
    ["底色与文字", { "--bg": "页面底色", "--surface": "面板、卡片", "--surface-2": "次一级底色", "--text": "正文",
      "--muted": "次要文字", "--border": "边框", "--border-strong": "深一点的边框" }],
    ["主色", { "--accent": "主色（按钮、链接）", "--accent-hover": "主色 · 鼠标悬停", "--accent-soft": "主色 · 浅底",
      "--on-accent": "主色按钮上的字" }],
    ["提示色", { "--ok": "已确认", "--ok-soft": "已确认 · 浅底", "--warn": "没把握的字", "--warn-soft": "没把握 · 浅底" }],
    ["扫描页", { "--paper": "页面和行图底色", "--mark": "行框", "--mark-done": "已校对的行框", "--mark-sel": "选中的行框" }],
    ["圆角", { "--radius-sm": "小（键帽、字块、行图）", "--radius": "中（按钮、输入框）", "--radius-lg": "大（卡片、对话框）",
      "--radius-pill": "胶囊标签" }],
    ["阴影", { "--shadow": "卡片", "--shadow-lg": "弹层、对话框", "--shadow-up": "校对面板", "--backdrop": "对话框后面的遮罩" }],
    ["字体", { "--ui": "界面文字", "--tx": "西夏文和识别结果", "--mono": "数字、格式（等宽）" }],
    ["动效", { "--ease": "缓动曲线" }],
  ];
  const FONTS = {
    "--ui": {
      "黑体": 'system-ui, -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif',
      "宋体": '"Songti SC", "STSong", "Noto Serif SC", "Source Han Serif SC", "SimSun", serif',
      "楷体": '"Kaiti SC", "STKaiti", "KaiTi", serif',
      "仿宋": '"STFangsong", "FangSong", serif',
    },
    "--tx": {   // 第一个必须是 Tangut，否则西夏字没有字形
      "宋体": '"Tangut", "Songti SC", "Noto Serif SC", "Source Han Serif SC", "SimSun", serif',
      "楷体": '"Tangut", "Kaiti SC", "STKaiti", "KaiTi", serif',
      "黑体": '"Tangut", "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif',
    },
  };
  const LABEL = Object.assign({}, ...GROUPS.map(([, g]) => g));
  const squash = (s) => s.replace(/["'\s]/g, "").toLowerCase();
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* ---------------------------------------------------------------- 读 style.css 里的原值 */
  let base = { light: {}, dark: {} };
  function readBase() {
    base = { light: {}, dark: {} };
    const sheet = [...document.styleSheets].find((s) => /\/static\/style\.css(\?|$)/.test(s.href || ""));
    if (!sheet) return;
    const grab = (rule, into) => {
      for (const n of Array.from(rule.style)) if (n.startsWith("--")) into[n] = rule.style.getPropertyValue(n).trim();
    };
    for (const r of sheet.cssRules) {
      if (r.selectorText === ":root") grab(r, base.light);
      else if (r.media && /prefers-color-scheme:\s*dark/.test(r.media.mediaText)) {
        for (const rr of r.cssRules) if (rr.selectorText === ":root") grab(rr, base.dark);
      }
    }
    // 已经写回文件的改动不再算改动
    for (const k of ["light", "dark"]) for (const n in st[k]) if (st[k][n] === base[k][n] || !(n in base[k])) delete st[k][n];
  }

  /* 只有深色块里也写了的变量（主要是颜色）分浅色、深色两套；圆角、字体等两套共用，记在 light 里 */
  const mq = matchMedia("(prefers-color-scheme: dark)");
  const scheme = () => (st.scheme === "auto" ? (mq.matches ? "dark" : "light") : st.scheme);
  const slot = (s, n) => (s === "dark" && n in base.dark ? "dark" : "light");
  const orig = (s, n) => base[slot(s, n)][n] ?? base.light[n];
  const val = (s, n) => st[slot(s, n)][n] ?? orig(s, n);
  const names = () => Object.keys({ ...base.light, ...base.dark });
  const changed = () => Object.keys(st.light).length + Object.keys(st.dark).length;

  /* 用 html:root 压过 style.css 的 :root；强制浅色/深色时只写一套 */
  const css = document.createElement("style");
  function apply() {
    const block = (s) => names().map((n) => `${n}: ${val(s, n)};`).join(" ") + ` color-scheme: ${s};`;
    css.textContent = st.scheme === "auto"
      ? `html:root { ${block("light")} } @media (prefers-color-scheme: dark) { html:root { ${block("dark")} } }`
      : `html:root { ${block(st.scheme)} }`;
    document.head.append(css);
    remember();
  }
  function set(n, v) {
    const s = scheme(), k = slot(s, n);
    v = v.trim();
    if (!v || v === orig(s, n)) delete st[k][n]; else st[k][n] = v;
    apply();
    sync();
  }

  /* ---------------------------------------------------------------- 面板 */
  const host = document.createElement("div");
  host.id = "tune";
  const root = host.attachShadow({ mode: "open" });     // 页面的样式进不来，面板也不会被自己调的颜色影响
  root.innerHTML = `<style>
    :host { all: initial; }
    * { box-sizing: border-box; }
    .wrap { --bg: #ffffff; --bg2: #f3f3f3; --fg: #1a1a1a; --mute: #6b6b6b; --line: #dedede; --hi: #2563eb;
      font: 13px/1.45 system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; color: var(--fg); }
    @media (prefers-color-scheme: dark) {
      .wrap { --bg: #1d1d1f; --bg2: #2a2a2d; --fg: #ececec; --mute: #9a9a9a; --line: #3a3a3d; --hi: #6ea0ff; }
    }
    button, input, select { font: inherit; color: inherit; }
    button { cursor: pointer; background: var(--bg2); border: 1px solid var(--line); border-radius: 6px; padding: 4px 9px; }
    button:hover { border-color: var(--mute); }
    .tab { position: fixed; top: 28%; right: 0; transform: translateY(-50%); z-index: 2147483000; opacity: .6;
      writing-mode: vertical-rl; padding: 10px 5px; border-radius: 6px 0 0 6px; border-right: none;
      background: var(--fg); color: var(--bg); box-shadow: 0 2px 10px rgba(0, 0, 0, .2); }
    .tab:hover { opacity: 1; }
    .left .tab { right: auto; left: 0; border-radius: 0 6px 6px 0; border: 1px solid var(--line); border-left: none; }
    .tab b { display: inline-block; margin-top: 6px; min-width: 16px; padding: 2px 0; border-radius: 8px; background: var(--hi); color: #fff;
      font-weight: 600; text-align: center; }
    .panel { position: fixed; top: 0; bottom: 0; right: 0; z-index: 2147483001; width: 330px; max-width: 100vw;
      display: flex; flex-direction: column; background: var(--bg); border-left: 1px solid var(--line);
      box-shadow: -6px 0 24px rgba(0, 0, 0, .12); }
    .left .panel { right: auto; left: 0; border-left: none; border-right: 1px solid var(--line); box-shadow: 6px 0 24px rgba(0, 0, 0, .12); }
    [hidden] { display: none !important; }
    header { display: flex; align-items: center; gap: 6px; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    header b { font-size: 14px; }
    header .n { color: var(--mute); }
    header .sp { flex: 1; }
    header button { padding: 2px 8px; }
    .scheme { padding: 8px 12px; border-bottom: 1px solid var(--line); }
    .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; margin-left: 6px; }
    .seg button { border: none; border-radius: 0; background: transparent; padding: 3px 10px; }
    .seg button.on { background: var(--fg); color: var(--bg); }
    .scheme p { margin: 6px 0 0; color: var(--mute); font-size: 12px; }
    .body { flex: 1; overflow: auto; padding: 4px 0 12px; }
    details { border-bottom: 1px solid var(--line); }
    summary { cursor: pointer; padding: 8px 12px; font-weight: 600; user-select: none; }
    .row { display: grid; grid-template-columns: 1fr auto 24px; gap: 4px 8px; align-items: center; padding: 4px 12px; }
    .row.on { background: color-mix(in srgb, var(--hi) 10%, transparent); }
    .row .lb { min-width: 0; }
    .row .lb small { display: block; color: var(--mute); font: 11px ui-monospace, Menlo, monospace; }
    .ctl { display: flex; align-items: center; gap: 6px; }
    .row.wide { grid-template-columns: 1fr 24px; }
    .row.wide .ctl { grid-column: 1; flex-wrap: wrap; }
    .row.wide .undo { grid-row: 1; grid-column: 2; }
    input[type=color] { width: 30px; height: 26px; padding: 0; border: 1px solid var(--line); border-radius: 5px; background: none; cursor: pointer; }
    input.txt { width: 92px; padding: 3px 6px; border: 1px solid var(--line); border-radius: 5px; background: var(--bg); font: 12px ui-monospace, Menlo, monospace; }
    .wide input.txt { flex: 1; min-width: 0; }
    .row[data-k=font] input.txt { flex-basis: 100%; }
    input[type=range] { width: 96px; accent-color: var(--hi); }
    .px { width: 46px; text-align: right; color: var(--mute); font-variant-numeric: tabular-nums; }
    select { padding: 3px 4px; border: 1px solid var(--line); border-radius: 5px; background: var(--bg); }
    .undo { width: 24px; height: 24px; padding: 0; border: none; background: none; color: var(--mute); visibility: hidden; }
    .row.on .undo { visibility: visible; }
    footer { padding: 10px 12px 12px; border-top: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 6px; }
    footer .main { background: var(--hi); border-color: var(--hi); color: #fff; }
    footer .sp { flex: 1; }
    footer p { width: 100%; margin: 4px 0 0; color: var(--mute); font-size: 12px; min-height: 1em; }
    footer textarea { width: 100%; height: 120px; font: 11px ui-monospace, Menlo, monospace; }
  </style>
  <div class="wrap">
    <button class="tab" data-act="open" title="调样式（只在本机出现）">调样式<b hidden></b></button>
    <aside class="panel" hidden>
      <header>
        <b>调样式</b><span class="n"></span><span class="sp"></span>
        <button data-act="side" title="换到另一边">⇆</button>
        <button data-act="close" title="收起">✕</button>
      </header>
      <div class="scheme">预览<span class="seg">
        <button data-scheme="auto">跟随系统</button><button data-scheme="light">浅色</button><button data-scheme="dark">深色</button>
      </span><p></p></div>
      <div class="body"></div>
      <footer>
        <button class="main" data-act="save" hidden>保存到 style.css</button>
        <button data-act="copy">复制改动</button>
        <span class="sp"></span>
        <button data-act="reset">全部还原</button>
        <p class="note"></p>
      </footer>
    </aside>
  </div>`;
  const $ = (s) => root.querySelector(s);
  const note = (msg) => { $(".note").textContent = msg; };

  const kind = (n, v) => (FONTS[n] ? "font" : /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v) ? "color"
    : /^\d+(\.\d+)?px$/.test(v) ? "px" : "text");
  const hex6 = (v) => (/^#[0-9a-f]{3}$/i.test(v) ? "#" + [...v.slice(1)].map((c) => c + c).join("") : v);

  function render() {
    const all = names(), known = new Set(Object.keys(LABEL));
    const groups = [...GROUPS.map(([t, g]) => [t, Object.keys(g).filter((n) => all.includes(n))]),
      ["其他", all.filter((n) => !known.has(n))]].filter(([, ns]) => ns.length);
    $(".body").innerHTML = groups.map(([t, ns]) => `<details open><summary>${t}</summary>${ns.map(row).join("")}</details>`).join("");
    sync();
  }
  function row(n) {
    const k = kind(n, base.light[n] ?? base.dark[n] ?? "");
    let ctl;
    if (k === "color") ctl = `<input type="color"><input class="txt" spellcheck="false">`;
    else if (k === "px") ctl = `<input type="range" min="0" max="32" step="1"><span class="px"></span>`;
    else if (k === "font") ctl = `<select>${Object.keys(FONTS[n]).map((f) => `<option>${f}</option>`).join("")}<option value="">自己填</option></select>` +
      `<input class="txt" spellcheck="false">`;
    else ctl = `<input class="txt" spellcheck="false">${/shadow/.test(n) ? `<button data-none>无</button>` : ""}`;
    const wide = k === "font" || k === "text";
    return `<div class="row${wide ? " wide" : ""}" data-n="${n}" data-k="${k}">
      <div class="lb">${esc(LABEL[n] || n)}<small>${n}</small></div><div class="ctl">${ctl}</div>
      <button class="undo" title="还原成 style.css 里的值">↺</button></div>`;
  }
  function sync() {
    const s = scheme(), focus = root.activeElement;
    root.querySelectorAll(".row").forEach((r) => {
      const n = r.dataset.n, v = val(s, n);
      r.classList.toggle("on", n in st[slot(s, n)]);
      r.querySelector(".lb").title = `style.css 原值：${orig(s, n)}`;
      const txt = r.querySelector(".txt"), color = r.querySelector("[type=color]"), range = r.querySelector("[type=range]");
      if (txt && txt !== focus) txt.value = v;
      if (color && /^#[0-9a-f]{3,6}$/i.test(v)) color.value = hex6(v);
      if (range) { if (range !== focus) range.value = parseFloat(v); r.querySelector(".px").textContent = v; }
      const sel = r.querySelector("select");
      if (sel) sel.value = Object.keys(FONTS[n]).find((f) => squash(FONTS[n][f]) === squash(v)) ?? "";
    });
    root.querySelectorAll("[data-scheme]").forEach((b) => b.classList.toggle("on", b.dataset.scheme === st.scheme));
    $(".scheme p").textContent = `颜色改的是${s === "dark" ? "深色" : "浅色"}那一套；圆角、阴影、字体两套共用。`;
    const c = changed();
    $(".n").textContent = c ? `改了 ${c} 项` : "";
    $(".tab b").hidden = !c;
    $(".tab b").textContent = c;
  }
  function layout() {
    $(".wrap").classList.toggle("left", st.left);
    $(".panel").hidden = !st.open;
    $(".tab").hidden = st.open;
    remember();
  }

  root.addEventListener("input", (e) => {
    const r = e.target.closest(".row");
    if (!r) return;
    const t = e.target;
    if (t.type === "range") set(r.dataset.n, t.value + "px");
    else if (t.matches(".txt, [type=color]")) set(r.dataset.n, t.value);
  });
  root.addEventListener("change", (e) => {
    const r = e.target.closest(".row");
    if (r && e.target.tagName === "SELECT" && e.target.value) set(r.dataset.n, FONTS[r.dataset.n][e.target.value]);
  });
  root.addEventListener("click", async (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    const r = b.closest(".row");
    if (r && b.matches(".undo")) set(r.dataset.n, "");
    else if (r && b.hasAttribute("data-none")) set(r.dataset.n, "none");
    else if (b.dataset.scheme) { st.scheme = b.dataset.scheme; apply(); sync(); }
    else if (b.dataset.act === "open") { st.open = true; layout(); }
    else if (b.dataset.act === "close") { st.open = false; layout(); }
    else if (b.dataset.act === "side") { st.left = !st.left; layout(); }
    else if (b.dataset.act === "reset") {
      if (changed() && confirm("把所有改动还原成 style.css 里的值？")) { st.light = {}; st.dark = {}; apply(); sync(); note("已还原"); }
    } else if (b.dataset.act === "copy") copy();
    else if (b.dataset.act === "save") {
      b.disabled = true;
      try { await saveFile(); } catch (err) { note(err.message); } finally { b.disabled = false; }
    }
  });
  // 面板里打字时别触发页面自己的快捷键（Esc 关校对面板等）
  root.addEventListener("keydown", (e) => e.stopPropagation());
  mq.addEventListener("change", sync);

  /* ---------------------------------------------------------------- 导出与保存 */
  function diffCss() {
    const lines = (o, pad) => Object.entries(o).map(([n, v]) => `${pad}${n}: ${v};`).join("\n");
    let out = "";
    if (Object.keys(st.light).length) out += `:root {\n${lines(st.light, "  ")}\n}\n`;
    if (Object.keys(st.dark).length) out += `@media (prefers-color-scheme: dark) {\n  :root {\n${lines(st.dark, "    ")}\n  }\n}\n`;
    return out;
  }
  async function copy() {
    const text = diffCss();
    if (!text) return note("还没有改动");
    try {
      await navigator.clipboard.writeText(text);
      note("已复制。把这些值抄到 style.css 开头同名的那几行，或者直接贴给 Claude。");
    } catch {
      note("浏览器不让写剪贴板，请手动复制：");
      const ta = document.createElement("textarea");
      ta.value = text;
      $(".note").append(ta);
      ta.select();
    }
  }
  async function saveFile() {
    if (!changed()) return note("还没有改动");
    const r = await fetch("/__tune/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ light: st.light, dark: st.dark }),
    });
    const out = await r.json().catch(() => null);
    if (!r.ok) throw new Error((out && out.error) || `保存失败（${r.status}）`);
    await reloadSheet();
    readBase();
    render();
    apply();
    note(`已写进 style.css（${out.changed} 处）` + (out.missing.length ? `；这几项文件里没有，没写：${out.missing.join(" ")}` : ""));
  }
  /* 换一个新的 <link> 读改过的 style.css，载好了再去掉旧的，页面不闪 */
  function reloadSheet() {
    return new Promise((resolve) => {
      const old = document.querySelector('link[rel=stylesheet][href^="/static/style.css"]');
      if (!old) return resolve();
      const link = old.cloneNode();
      link.href = "/static/style.css?t=" + Date.now();
      link.onload = link.onerror = () => { old.remove(); resolve(); };
      old.after(link);
    });
  }

  readBase();
  document.body.append(host);
  render();
  layout();
  apply();
  // 只有 tools/preview.py 有写文件的接口；正式后端没有，就只能复制
  fetch("/__tune").then((r) => (r.ok ? r.json() : null)).then((j) => { if (j && j.ok) $("[data-act=save]").hidden = false; }).catch(() => {});
})();
