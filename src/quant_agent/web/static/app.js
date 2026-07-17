"use strict";

const $ = (sel) => document.querySelector(sel);

// ── Tab 切换 ──
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "reports") loadReports();
  });
});

// ── 轻量 Markdown 渲染 ──
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
// 行内：粗体 **x** / 代码 `x`
function inline(s) {
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}
function renderMarkdown(md) {
  const lines = String(md || "").split("\n");
  const out = [];
  let i = 0;
  let inList = false;
  const closeList = () => {
    if (inList) { out.push("</ul>"); inList = false; }
  };
  // 解析表格行：去掉首尾的 | 后按 | 切分
  const parseRow = (line) => {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map((c) => c.trim());
  };
  // 分隔符行判定：每格均为 --- / :-- / --: 形式
  const isSep = (cells) =>
    cells.length > 0 && cells.every((c) => /^:?-+:?$/.test(c));
  while (i < lines.length) {
    const line = lines[i].replace(/\s+$/, "");
    // 表格检测：当前行为表头，下一行为分隔符行
    if (line.startsWith("|") && i + 1 < lines.length) {
      const headerCells = parseRow(line);
      const sepCells = parseRow(lines[i + 1]);
      if (isSep(sepCells) && headerCells.length >= 1) {
        const ncol = headerCells.length;
        let html = '<table class="md-table"><thead><tr>';
        html += headerCells.map((c) => `<th>${inline(c)}</th>`).join("");
        html += "</tr></thead><tbody>";
        i += 2; // 跳过表头 + 分隔符
        while (i < lines.length) {
          const rline = lines[i].replace(/\s+$/, "");
          if (rline.trim() === "" || !rline.includes("|")) break;
          const cells = parseRow(rline);
          if (cells.length !== ncol) break;
          html += "<tr>" + cells.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
          i++;
        }
        html += "</tbody></table>";
        closeList();
        out.push(html);
        continue;
      }
    }
    // 普通块级渲染
    if (line.startsWith("### ")) {
      closeList(); out.push(`<h3>${inline(line.slice(4))}</h3>`);
    } else if (line.startsWith("## ")) {
      closeList(); out.push(`<h2>${inline(line.slice(3))}</h2>`);
    } else if (line.startsWith("# ")) {
      closeList(); out.push(`<h1>${inline(line.slice(2))}</h1>`);
    } else if (line.startsWith("> ")) {
      closeList(); out.push(`<blockquote>${inline(line.slice(2))}</blockquote>`);
    } else if (/^- /.test(line)) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(line.slice(2))}</li>`);
    } else if (line.trim() === "") {
      closeList();
    } else {
      closeList(); out.push(`<p>${inline(line)}</p>`);
    }
    i++;
  }
  closeList();
  return out.join("\n");
}

// ── 工具 ──
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 2600);
}
function setStatus(ok, text) {
  const el = $("#status-bar");
  el.className = "status " + (ok ? "ok" : "err");
  el.textContent = text;
}
async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}

// ── 健康检查 ──
async function health() {
  try {
    const d = await api("/api/health");
    const llm = d.llm_enabled ? "LLM 已启用" : "规则增强(无 LLM)";
    setStatus(true, `就绪 · ${llm}${d.offline_mode ? " · 离线" : ""}`);
  } catch (e) {
    setStatus(false, "服务未连接");
  }
}

// ── 个股分析 ──
$("#analyze-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const code = $("#stock-code").value.trim();
  if (!code) { toast("请输入股票代码"); return; }
  const days = +$("#analyze-days").value || 120;
  const offline = $("#analyze-offline").checked;
  const chart = $("#analyze-chart").checked;
  const box = $("#analyze-result");
  box.innerHTML = '<div class="loading">分析中…</div>';
  try {
    const d = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stock_code: code, days, offline, chart }),
    });
    if (d.error) throw new Error(d.error);
    const rep = d.report || {};
    const sig = (rep.signal || "HOLD");
    let md = d.markdown;
    if (!md) {
      if (rep && Object.keys(rep).length) {
        md = `# 量化分析报告 — ${rep.stock_code || ""}\n\n> Markdown 渲染内容为空，已用基础数据兜底。\n\n` +
             `## 综合结论\n\n- **最终信号**: ${sig}\n- **信心度**: ${rep.confidence ?? "?"}\n- **建议仓位**: ${rep.position_pct ?? "?"}%\n`;
      } else {
        md = "_（无报告内容）_";
      }
    }
    let html = "";
    if (d.data_warning) {
      html += `<div class="watermark">⚠️ <strong>数据可信警示</strong>：本报告基于受限/合成数据，不构成投资建议。</div>`;
    }
    html += `<div class="report-card">${renderMarkdown(md)}</div>`;
    if (d.chart_url) {
      html += `<img class="chart-img" src="${d.chart_url}" alt="走势图" />`;
    }
    box.innerHTML = html;
    toast(`分析完成：${sig}`);
  } catch (e) {
    box.innerHTML = `<div class="error-box">分析失败：${escapeHtml(e.message)}</div>`;
  }
});

document.querySelectorAll(".quick a").forEach((a) => {
  a.addEventListener("click", () => {
    $("#stock-code").value = a.dataset.code;
    $("#analyze-form").requestSubmit();
  });
});

// ── 股票智能搜索（自动补全）──
(function setupAutocomplete() {
  const input = $("#stock-code");
  const box = $("#stock-suggest");
  let items = [];       // 当前候选 [{code, name}]
  let active = -1;      // 高亮索引
  let timer = null;     // 防抖计时器
  let seq = 0;          // 请求序号，避免乱序覆盖

  const hide = () => { box.classList.add("hidden"); box.innerHTML = ""; items = []; active = -1; };

  function render() {
    if (!items.length) { hide(); return; }
    box.innerHTML = items
      .map((s, i) => `
        <li class="suggest-item${i === active ? " active" : ""}" data-i="${i}">
          <span class="s-code">${escapeHtml(s.code)}</span>
          <span class="s-name">${escapeHtml(s.name || "")}</span>
        </li>`)
      .join("");
    box.classList.remove("hidden");
  }

  function choose(i) {
    const s = items[i];
    if (!s) return;
    input.value = s.code;
    hide();
    $("#analyze-form").requestSubmit();
  }

  async function search(q) {
    const mySeq = ++seq;
    try {
      const d = await api("/api/search?q=" + encodeURIComponent(q) + "&limit=10");
      if (mySeq !== seq) return; // 已有更新的请求，丢弃旧结果
      items = d.results || [];
      active = -1;
      render();
    } catch (e) {
      hide();
    }
  }

  input.addEventListener("input", () => {
    const q = input.value.trim();
    clearTimeout(timer);
    if (!q) { hide(); return; }
    timer = setTimeout(() => search(q), 180);
  });

  input.addEventListener("keydown", (e) => {
    if (box.classList.contains("hidden") || !items.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault(); active = (active + 1) % items.length; render();
    } else if (e.key === "ArrowUp") {
      e.preventDefault(); active = (active - 1 + items.length) % items.length; render();
    } else if (e.key === "Enter") {
      if (active >= 0) { e.preventDefault(); choose(active); }
    } else if (e.key === "Escape") {
      hide();
    }
  });

  box.addEventListener("mousedown", (e) => {
    // mousedown 先于 blur，避免下拉在点击前消失
    const li = e.target.closest(".suggest-item");
    if (li) { e.preventDefault(); choose(+li.dataset.i); }
  });

  input.addEventListener("blur", () => setTimeout(hide, 120));
})();

// ── 智能选股 ──
$("#screen-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const top = +$("#screen-top").value || 10;
  const full = $("#screen-full").checked;
  const fund = $("#screen-fund").checked;
  const deep = $("#screen-deep").checked;
  const box = $("#screen-result");
  box.innerHTML = '<div class="loading">选股中…</div>';
  try {
    const q = new URLSearchParams({ top, full_scan: +full, fundamentals: +fund, deep: +deep });
    const d = await api("/api/screen?" + q.toString());
    const rows = (d.top_stocks || []).map((s, i) => `
      <tr>
        <td>${i + 1}</td>
        <td class="code">${s.stock_code}</td>
        <td class="name">${s.name ? escapeHtml(s.name) : "-"}</td>
        <td>${s.price != null ? Number(s.price).toFixed(2) : "-"}</td>
        <td class="score">${s.total_score != null ? Number(s.total_score).toFixed(1) : "-"}</td>
        <td>${s.technical_score ?? "-"}</td>
        <td>${s.momentum_score ?? "-"}</td>
        <td>${s.liquidity_score ?? "-"}</td>
        <td>${s.fundamental_score ?? "-"}</td>
      </tr>`).join("");
    if (!rows) {
      box.innerHTML = '<div class="loading">无结果（联网受限时选股池为空，属正常；请在联网环境使用）</div>';
      return;
    }
    let html = `<div class="report-card"><h2>智能选股 Top ${d.top_stocks.length}</h2>
      <table class="grid">
        <thead><tr><th>#</th><th>代码</th><th>名称</th><th>价格</th><th>评分</th><th>技术</th><th>动量</th><th>流动</th><th>基本</th></tr></thead>
        <tbody>${rows}</tbody></table></div>`;
    if (d.deep_reports && d.deep_reports.length) {
      html += '<h2 style="margin-top:1.5rem;color:var(--accent)">深度分析</h2>';
      d.deep_reports.forEach((r) => {
        html += `<div class="report-card" style="margin-top:.8rem">${renderMarkdown(r.report ? "" : "")}`;
        const sig = r.report && r.report.signal ? r.report.signal : "HOLD";
        const nm = r.name ? ` ${escapeHtml(r.name)}` : "";
        html += `<p><span class="badge ${sig}">${sig}</span> ${r.stock_code}${nm}</p></div>`;
      });
    }
    box.innerHTML = html;
    toast(`选股完成：${d.top_stocks.length} 只`);
  } catch (e) {
    box.innerHTML = `<div class="error-box">选股失败：${escapeHtml(e.message)}</div>`;
  }
});

// ── 历史报告 ──
async function loadReports() {
  const box = $("#reports-result");
  box.innerHTML = '<div class="loading">加载中…</div>';
  try {
    const d = await api("/api/reports");
    const list = d.reports || [];
    if (!list.length) {
      box.innerHTML = '<div class="loading">暂无历史报告。先进行一次分析即可生成。</div>';
      return;
    }
    box.innerHTML = list.map((e) => `
      <div class="report-item" data-file="${encodeURIComponent(e.file)}">
        <div>
          <strong>${e.stock_code}</strong>
          <span class="badge ${e.signal}">${e.signal}</span>
          <span class="meta"> · 信心 ${(e.confidence * 100).toFixed(0)}%</span>
        </div>
        <div class="meta">${e.timestamp}</div>
      </div>`).join("");
    box.querySelectorAll(".report-item").forEach((it) => {
      it.addEventListener("click", () => showReport(it.dataset.file));
    });
  } catch (e) {
    box.innerHTML = `<div class="error-box">加载失败：${escapeHtml(e.message)}</div>`;
  }
}

async function showReport(file) {
  const box = $("#reports-result");
  box.innerHTML = '<div class="loading">加载报告…</div>';
  try {
    const d = await api("/api/report?file=" + file);
    box.innerHTML = `<div class="report-card">${renderMarkdown(d.markdown)}</div>
      <button class="btn" style="margin-top:.8rem" onclick="loadReports()">← 返回列表</button>`;
  } catch (e) {
    box.innerHTML = `<div class="error-box">加载失败：${escapeHtml(e.message)}</div>`;
  }
}

$("#reports-refresh").addEventListener("click", loadReports);

// 初始化
health();
