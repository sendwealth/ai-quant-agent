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
  let inList = false;
  const closeList = () => {
    if (inList) { out.push("</ul>"); inList = false; }
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
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
    const llm = d.llm_enabled ? "LLM 已启用" : "离线规则增强";
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
    const sig = (d.report.signal || "HOLD");
    let html = `<div class="report-card">${renderMarkdown(d.markdown)}</div>`;
    if (d.chart_url) {
      html += `<img class="chart-img" src="${d.chart_url}" alt="走势图" />`;
    }
    box.innerHTML = html;
    box.querySelectorAll("h1").forEach((h) => {});
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
        <td>${s.stock_code}</td>
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
        <thead><tr><th>#</th><th>代码</th><th>价格</th><th>评分</th><th>技术</th><th>动量</th><th>流动</th><th>基本</th></tr></thead>
        <tbody>${rows}</tbody></table></div>`;
    if (d.deep_reports && d.deep_reports.length) {
      html += '<h2 style="margin-top:1.5rem;color:var(--accent)">深度分析</h2>';
      d.deep_reports.forEach((r) => {
        html += `<div class="report-card" style="margin-top:.8rem">${renderMarkdown(r.report ? "" : "")}`;
        const sig = r.report && r.report.signal ? r.report.signal : "HOLD";
        html += `<p><span class="badge ${sig}">${sig}</span> ${r.stock_code}</p></div>`;
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
