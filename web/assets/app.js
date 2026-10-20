/* 筑安云前端：原生 JS，无构建依赖 */
(function () {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) =>
    String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  const state = {
    sessionId: localStorage.getItem("zhuan_session") || "",
    category: "",
    cats: [],
    docs: [],
    streaming: false,
    msgSeq: 0,
    ws: null,
    wsReady: false,
    lastCites: [],
    msgIdBySeq: {},
  };

  /* ---------------- 工具 ---------------- */
  function toast(msg, ms = 2200) {
    const el = $("#toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove("show"), ms);
  }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let d = "";
      try {
        d = (await r.json()).detail || "";
      } catch (e) {}
      throw new Error(d || r.status + " " + r.statusText);
    }
    return r.json();
  }

  /* ---------------- Markdown ---------------- */
  function inline(s) {
    let t = esc(s);
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/\[(\d{1,2})\]/g, '<span class="cite-ref" data-idx="$1">$1</span>');
    return t;
  }

  function md(src) {
    const lines = String(src || "").replace(/\r/g, "").split("\n");
    const out = [];
    let list = null;
    let para = [];

    const flushPara = () => {
      if (para.length) {
        out.push("<p>" + para.map(inline).join("<br/>") + "</p>");
        para = [];
      }
    };
    const flushList = () => {
      if (list) {
        out.push(`<${list.tag}>` + list.items.map((x) => "<li>" + inline(x) + "</li>").join("") + `</${list.tag}>`);
        list = null;
      }
    };

    for (const raw of lines) {
      const line = raw.trim();
      if (!line) {
        flushPara();
        flushList();
        continue;
      }
      if (line.startsWith(">")) {
        flushPara();
        flushList();
        out.push("<blockquote>" + inline(line.replace(/^>\s?/, "")) + "</blockquote>");
        continue;
      }
      const h = line.match(/^(#{2,4})\s+(.*)$/);
      if (h) {
        flushPara();
        flushList();
        out.push(`<h4>${inline(h[2])}</h4>`);
        continue;
      }
      const ul = line.match(/^[-*]\s+(.*)$/);
      const ol = line.match(/^(\d+)[.、]\s*(.*)$/);
      if (ul || ol) {
        flushPara();
        const tag = ul ? "ul" : "ol";
        const text = ul ? ul[1] : ol[2];
        if (!list || list.tag !== tag) {
          flushList();
          list = { tag, items: [] };
        }
        list.items.push(text);
        continue;
      }
      flushList();
      para.push(line);
    }
    flushPara();
    flushList();
    return out.join("");
  }

  /* ---------------- 初始化 ---------------- */
  async function boot() {
    bindUI();
    try {
      const [h, s] = await Promise.all([api("/api/health"), api("/api/corpus/stats")]);
      renderHealth(h);
      renderCats(s);
    } catch (e) {
      toast("服务连接失败：" + e.message, 4000);
    }
    connectWS();
    loadSessions();
    pollHealth();

    // 支持 /?demo=问题 直接带问题进入（便于分享与演示）
    const params = new URLSearchParams(location.search);
    const demo = params.get("demo");
    if (demo) {
      const wait = setInterval(() => {
        if (state.wsReady) {
          clearInterval(wait);
          $("#q").value = demo;
          send();
        }
      }, 500);
      setTimeout(() => clearInterval(wait), 12000);
    }
  }

  function pollHealth() {
    setInterval(async () => {
      try {
        const h = await api("/api/health");
        renderHealth(h);
      } catch (e) {}
    }, 8000);
  }

  function renderHealth(h) {
    const st = $("#sysIndex");
    const dot = $("#dotIndex");
    const idx = h.index || {};
    dot.className = "dot";
    if (idx.state === "ready") {
      dot.classList.add("ok");
      st.textContent = `索引 ${idx.chunks} 子块`;
    } else if (idx.state === "error") {
      dot.classList.add("err");
      st.textContent = "索引异常";
    } else {
      dot.classList.add("warn");
      st.textContent = `索引 ${idx.progress || 0}%`;
    }
    const emb = (idx.embedder || {}).backend || "";
    const dm = $("#dotModel");
    dm.className = "dot " + (emb === "bge-m3" ? "ok" : emb ? "warn" : "");
    $("#sysModel").textContent = emb === "bge-m3" ? "BGE-M3" : emb === "bge-fallback" ? "BGE-small" : emb ? "本地向量" : "模型";
    $("#verText").textContent = "v" + h.version;
    if (h.corpus) {
      $("#kbDocs").textContent = h.corpus.documents;
      $("#kbClauses").textContent = h.corpus.clauses;
      $("#kbForce").textContent = (h.corpus.levels || {})["强制性"] || 0;
      $("#kbEnt").textContent = (h.corpus.levels || {})["企业制度"] || 0;
    }
    if (idx.embedder && idx.embedder.error) $("#sysModel").title = idx.embedder.error;
  }

  function renderCats(s) {
    state.cats = s.categories || [];
    const ul = $("#catList");
    const total = s.stats.clauses;
    ul.innerHTML = `<li class="cat active" data-cat=""><span>全部规范</span><b>${total}</b></li>`;
    for (const c of state.cats) {
      const li = document.createElement("li");
      li.className = "cat";
      li.dataset.cat = c.category;
      li.innerHTML = `<span>${esc(c.category)}</span><b>${c.clauses}</b>`;
      ul.appendChild(li);
    }
    const sel = $("#catSelect");
    const keep = sel.value;
    sel.innerHTML = '<option value="">自动判断</option>';
    for (const c of state.cats) {
      const o = document.createElement("option");
      o.value = c.category;
      o.textContent = c.category;
      sel.appendChild(o);
    }
    sel.value = keep || "";
  }

  function bindUI() {
    $$("#tabs .tab").forEach((b) =>
      b.addEventListener("click", () => {
        $$("#tabs .tab").forEach((x) => x.classList.toggle("active", x === b));
        const v = b.dataset.view;
        $$(".view").forEach((x) => x.classList.toggle("active", x.id === "view-" + v));
        if (v === "kb") loadKB();
        if (v === "eval") loadEval();
        if (v === "sessions") loadSessions();
      })
    );

    $("#catList").addEventListener("click", (e) => {
      const li = e.target.closest(".cat");
      if (!li) return;
      $$("#catList .cat").forEach((x) => x.classList.toggle("active", x === li));
      state.category = li.dataset.cat || "";
      $("#catSelect").value = state.category;
    });

    $("#catSelect").addEventListener("change", (e) => {
      state.category = e.target.value;
      $$("#catList .cat").forEach((x) => x.classList.toggle("active", (x.dataset.cat || "") === state.category));
    });

    $("#send").addEventListener("click", send);
    $("#q").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
    $("#q").addEventListener("input", (e) => {
      const t = e.target;
      t.style.height = "auto";
      t.style.height = Math.min(160, t.scrollHeight) + "px";
    });
    $("#chips").addEventListener("click", (e) => {
      const c = e.target.closest(".chip");
      if (!c) return;
      $("#q").value = c.textContent;
      send();
    });
    $("#btnNew").addEventListener("click", () => {
      state.sessionId = "";
      localStorage.removeItem("zhuan_session");
      $("#chat").innerHTML = "";
      $("#welcome").style.display = "";
      $("#sessTitle").textContent = "新会话";
      $("#sessMeta").textContent = "未开始";
      clearRight();
      toast("已新建会话");
    });
    $("#btnClearSessions").addEventListener("click", async () => {
      if (!confirm("确定清空全部会话记录？此操作不可撤销。")) return;
      const list = await api("/api/sessions");
      for (const s of list.items) await api("/api/sessions/" + s.id, { method: "DELETE" });
      loadSessions();
      toast("已清空会话");
    });
    $("#btnEval").addEventListener("click", runEval);
    $("#fileInput").addEventListener("change", uploadFile);
    $("#kbSearch").addEventListener("input", renderDocTable);
    $("#traceToggle").addEventListener("click", () => {
      const b = $("#traceBody");
      b.hidden = !b.hidden;
    });

    document.addEventListener("click", (e) => {
      const ref = e.target.closest(".cite-ref");
      if (ref) {
        const idx = ref.dataset.idx;
        const card = $(`.cite[data-idx="${idx}"]`);
        if (card) {
          card.scrollIntoView({ behavior: "smooth", block: "center" });
          card.classList.add("hl");
          setTimeout(() => card.classList.remove("hl"), 1400);
        }
      }
      const more = e.target.closest(".cite-more");
      if (more) {
        const card = more.closest(".cite");
        card.classList.toggle("open");
        more.textContent = card.classList.contains("open") ? "收起" : "展开原文";
      }
      const fb = e.target.closest("[data-fb]");
      if (fb) {
        const row = fb.closest(".msg-bot");
        const mid = row && row.dataset.mid;
        if (mid) {
          api("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message_id: Number(mid), value: Number(fb.dataset.fb) }),
          }).then(() => toast("感谢反馈"));
        } else toast("反馈已记录（本地）");
      }
    });
  }

  /* ---------------- WebSocket ---------------- */
  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/qa`);
    state.ws = ws;
    ws.onopen = () => {
      state.wsReady = true;
    };
    ws.onclose = () => {
      state.wsReady = false;
      setTimeout(connectWS, 3000);
    };
    ws.onerror = () => {
      state.wsReady = false;
    };
    ws.onmessage = (ev) => {
      let m;
      try {
        m = JSON.parse(ev.data);
      } catch (e) {
        return;
      }
      handleEvent(m);
    };
  }

  function resetStages() {
    $$("#stageBar .stage").forEach((s) => (s.className = "stage"));
  }
  function markStage(name, st) {
    const el = $(`#stageBar .stage[data-stage="${name}"]`);
    if (el) {
      el.classList.remove("running", "done");
      if (st) el.classList.add(st);
    }
  }

  function handleEvent(m) {
    const t = m.type;
    if (t === "stage") {
      const d = m.data || {};
      markStage(m.stage, d.state);
      return;
    }
    if (t === "intent") {
      state._intent = m.data;
      return;
    }
    if (t === "token") {
      state._buf = (state._buf || "") + (m.data || "");
      const a = $(".msg-bot.streaming .answer");
      if (a) {
        a.innerHTML = md(state._buf);
        scrollBottom();
      }
      return;
    }
    if (t === "citations") {
      state.lastCites = m.data || [];
      renderCites(state.lastCites);
      return;
    }
    if (t === "note") {
      const a = $(".msg-bot.streaming .answer");
      if (a) a.insertAdjacentHTML("beforeend", `<p class="muted" style="font-size:11.8px">${esc(m.data)}</p>`);
      return;
    }
    if (t === "session") {
      state.sessionId = (m.data || {}).session_id || "";
      if (state.sessionId) localStorage.setItem("zhuan_session", state.sessionId);
      return;
    }
    if (t === "saved") {
      const d = m.data || {};
      const row = $(".msg-bot.streaming");
      if (row && d.message_id) row.dataset.mid = d.message_id;
      if (d.session_id) {
        state.sessionId = d.session_id;
        localStorage.setItem("zhuan_session", d.session_id);
      }
      $("#sessMeta").textContent = `${d.latency_ms} ms · 已保存`;
      return;
    }
    if (t === "done") finishAnswer(m.data || {});
    if (t === "error") {
      finishWithError(m.data || "生成失败");
    }
  }

  function finishAnswer(d) {
    const row = $(".msg-bot.streaming");
    if (!row) return;
    row.classList.remove("streaming");
    const answer = d.answer || state._buf || "";
    const box = $(".answer", row);
    box.innerHTML = md(answer);
    const conf = typeof d.confidence === "number" ? d.confidence : 0;
    const head = $(".msg-head", row);
    if (head) {
      head.innerHTML = buildHead(d.intent, conf, d.latency_ms, d.generator);
    }
    const foot = document.createElement("div");
    foot.className = "msg-foot";
    const cites = d.citations || [];
    foot.innerHTML =
      `<span>引用 ${cites.length} 条 · ${d.latency_ms || 0} ms</span>` +
      `<button class="icon-btn" data-fb="1">有用</button>` +
      `<button class="icon-btn" data-fb="-1">不准确</button>`;
    row.appendChild(foot);
    state.streaming = false;
    $("#send").disabled = false;
    $("#send").textContent = "发送";
    if (d.citations) {
      state.lastCites = d.citations;
      renderCites(d.citations);
    }
    renderTrace(d.trace);
    const title = d.question || state._q || "";
    if (title) {
      $("#sessTitle").textContent = title.slice(0, 22) + (title.length > 22 ? "…" : "");
    }
    $$("#stageBar .stage").forEach((s) => s.classList.add("done"));
    scrollBottom();
  }

  function finishWithError(msg) {
    const row = $(".msg-bot.streaming");
    if (row) {
      row.classList.remove("streaming");
      $(".answer", row).innerHTML = `<p style="color:var(--red)">生成失败：${esc(msg)}</p>`;
    }
    state.streaming = false;
    $("#send").disabled = false;
    $("#send").textContent = "发送";
  }

  function buildHead(intent, conf, latency, gen) {
    const label = (intent && intent.intent) || "分析";
    const cls = label === "隐患判定" ? "red" : label === "整改要求" ? "amber" : "";
    const pct = Math.round((conf || 0) * 100);
    return (
      `<span class="tag ${cls}">${esc(label)}</span>` +
      `<span class="muted">${esc(gen || "")}</span>` +
      `<span class="conf">置信度 ${pct}%<span class="conf-bar"><i style="width:${pct}%"></i></span></span>`
    );
  }

  function scrollBottom() {
    const c = $("#chat");
    c.scrollTop = c.scrollHeight;
  }

  /* ---------------- 发送 ---------------- */
  function appendUser(text) {
    const w = $("#welcome");
    if (w) w.style.display = "none";
    const div = document.createElement("div");
    div.className = "msg msg-user";
    div.innerHTML = `<div class="bubble">${esc(text)}</div>`;
    $("#chat").appendChild(div);
  }

  function appendBotShell() {
    const div = document.createElement("div");
    div.className = "msg msg-bot streaming";
    div.innerHTML =
      `<div class="bubble"><div class="msg-head"><span class="tag">检索中…</span></div><div class="answer"><p class="muted">正在检索规范条文…</p></div></div>`;
    $("#chat").appendChild(div);
    scrollBottom();
    return div;
  }

  async function send() {
    if (state.streaming) return;
    const q = $("#q").value.trim();
    if (!q) return;
    if (q.length > 500) {
      toast("问题过长，请控制在 500 字以内");
      return;
    }
    $("#q").value = "";
    $("#q").style.height = "auto";
    state.streaming = true;
    state._buf = "";
    state._q = q;
    $("#send").disabled = true;
    $("#send").textContent = "生成中";
    resetStages();
    appendUser(q);
    appendBotShell();
    clearRight();

    if (state.wsReady && state.ws && state.ws.readyState === 1) {
      state.ws.send(JSON.stringify({ question: q, category: state.category || null, session_id: state.sessionId || null }));
    } else {
      // HTTP 回退
      try {
        const r = await api("/api/qa", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, category: state.category || null, session_id: state.sessionId || null }),
        });
        state._buf = r.answer;
        if (r.session_id) {
          state.sessionId = r.session_id;
          localStorage.setItem("zhuan_session", r.session_id);
        }
        finishAnswer(r);
      } catch (e) {
        finishWithError(e.message);
      }
    }
  }

  /* ---------------- 右栏 ---------------- */
  function clearRight() {
    $("#cites").innerHTML = `<div class="empty"><div class="empty-t">检索中</div><div class="empty-s">正在对规范条文做混合召回与精排。</div></div>`;
    $("#citeCount").textContent = "检索中";
    $("#traceBox").hidden = true;
  }

  function renderCites(cites) {
    const box = $("#cites");
    if (!cites || !cites.length) {
      box.innerHTML = `<div class="empty"><div class="empty-t">未命中条文</div><div class="empty-s">知识库中没有足够相关的条款，建议补充关键词或切换分类。</div></div>`;
      $("#citeCount").textContent = "0 条";
      return;
    }
    $("#citeCount").textContent = `${cites.length} 条 · 点击编号定位`;
    box.innerHTML = cites
      .map((c) => {
        const cls = c.level === "强制性" ? "force" : c.level === "企业制度" ? "ent" : "";
        const tagCls = c.level === "强制性" ? "red" : c.level === "企业制度" ? "amber" : "";
        const kp = (c.check_points || []).slice(0, 4);
        return `<div class="cite ${cls}" data-idx="${c.index}">
          <div class="cite-h">
            <span class="cite-idx">${c.index}</span>
            <div class="cite-t">《${esc(c.std_name)}》第 ${esc(c.clause_no)} 条</div>
          </div>
          <div class="cite-meta">
            <span class="tag ${tagCls}">${esc(c.level)}</span>
            <span>${esc(c.std_no)}</span>
            <span>${esc(c.chapter || "")}</span>
            <span>P${c.page || "-"}</span>
          </div>
          <div class="cite-text">${esc(c.text)}</div>
          <button class="cite-more">展开原文</button>
          ${
            kp.length
              ? `<div class="cite-kp"><b>检查要点</b><br/>${kp.map(esc).join("；")}</div>`
              : ""
          }
        </div>`;
      })
      .join("");
  }

  function renderTrace(t) {
    if (!t) return;
    const paths = (t.paths || [])[0] || {};
    const p = paths.paths || {};
    const rows = [];
    for (const k of ["dense", "sparse", "bm25", "rerank"]) {
      if (p[k]) rows.push(`<div class="trace-row"><span>${k} 召回</span><span>${p[k].count} 条 / ${p[k].ms} ms</span></div>`);
    }
    rows.push(`<div class="trace-row"><span>融合候选</span><span>${paths.fused_candidates || 0}</span></div>`);
    rows.push(`<div class="trace-row"><span>精排保留</span><span>${paths.kept || 0}</span></div>`);
    rows.push(`<div class="trace-row"><span>父块回溯</span><span>${paths.parents || 0}</span></div>`);
    rows.push(`<div class="trace-row"><span>阈值</span><span>${t.threshold || 0}</span></div>`);
    rows.push(`<div class="trace-row"><span>Top 分数</span><span>${(t.top_score || 0).toFixed(3)}</span></div>`);
    const v = t.version || (paths.version || {});
    if (v.embed) rows.push(`<div class="trace-row"><span>向量 / 精排</span><span>${esc(v.embed)} / ${esc(v.rerank)}</span></div>`);
    $("#traceBody").innerHTML = rows.join("");
    $("#traceBox").hidden = false;
  }

  /* ---------------- 知识库 ---------------- */
  async function loadKB() {
    try {
      const [s, d] = await Promise.all([api("/api/corpus/stats"), api("/api/corpus/docs")]);
      state.docs = d.items;
      const st = s.stats;
      $("#kbStats").innerHTML = [
        ["规范文件", st.documents, `标准 ${st.standards} · 法规 ${st.regulations} · 企业 ${st.enterprise}`],
        ["条文条目", st.clauses, `父块 ${st.parents}`],
        ["强制性条文", st.levels["强制性"] || 0, "必须严格执行"],
        ["平均条长", st.avg_chars, "字 / 条"],
      ]
        .map(
          ([l, v, sub]) =>
            `<div class="stat"><div class="stat-l">${l}</div><div class="stat-v">${v}</div><div class="stat-s">${sub}</div></div>`
        )
        .join("");
      renderDocTable();
    } catch (e) {
      toast("加载知识库失败：" + e.message);
    }
  }

  function renderDocTable() {
    const q = ($("#kbSearch").value || "").trim().toLowerCase();
    let items = state.docs;
    if (q)
      items = items.filter(
        (d) =>
          d.std_name.toLowerCase().includes(q) ||
          d.std_no.toLowerCase().includes(q) ||
          (d.alias || []).some((a) => a.toLowerCase().includes(q))
      );
    const head = `<div class="doc-row thead"><div>规范名称</div><div>分类</div><div class="num">条文</div><div class="num">状态</div></div>`;
    const rows = items
      .map(
        (d) => `<div class="doc-row" data-doc="${d.doc_id}">
          <div><div class="doc-name">${esc(d.std_name)}</div><div class="doc-sub">${esc(d.std_no)}</div></div>
          <div class="muted">${esc(d.category)}</div>
          <div class="num">${d.clause_count}</div>
          <div class="num">${esc(d.status)}</div>
        </div>`
      )
      .join("");
    $("#docTable").innerHTML = head + rows;
    $$("#docTable .doc-row[data-doc]").forEach((r) =>
      r.addEventListener("click", () => {
        $$("#docTable .doc-row").forEach((x) => x.classList.toggle("active", x === r));
        loadDocDetail(r.dataset.doc);
      })
    );
  }

  async function loadDocDetail(docId) {
    try {
      const d = await api("/api/corpus/docs/" + docId);
      $("#docMeta").textContent = `${d.clauses || d.clause_count} 条`;
      const meta = `
        <div class="detail-h">《${esc(d.std_name)}》</div>
        <div class="detail-meta">
          <span>编号 <b>${esc(d.std_no)}</b></span>
          <span>分类 <b>${esc(d.category)}</b></span>
          <span>性质 <b>${esc(d.doc_type)}</b></span>
          <span>实施 <b>${esc(d.effective_date || "-")}</b></span>
          <span>页数 <b>${d.pages || "-"}</b></span>
        </div>
        <p class="muted" style="margin:0 0 8px">${esc(d.scope || "")}</p>`;
      const chapters = (d.chapters || [])
        .map(
          (ch) => `<div class="chapter">
            <div class="chapter-h">${esc(ch.chapter)}</div>
            ${ch.clauses
              .map(
                (c) => `<div class="clause">
                  <div class="clause-h"><span>第 ${esc(c.clause_no)} 条</span>
                    <span class="tag ${c.level === "强制性" ? "red" : c.level === "企业制度" ? "amber" : ""}">${esc(c.level)}</span>
                    <span>P${c.page || "-"}</span></div>
                  <div class="clause-t">${esc(c.text)}</div>
                </div>`
              )
              .join("")}
          </div>`
        )
        .join("");
      $("#docDetail").innerHTML = meta + chapters;
    } catch (e) {
      toast("加载文档失败：" + e.message);
    }
  }

  async function uploadFile(e) {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    fd.append("category", state.category || "未分类");
    toast("正在解析并入库…", 4000);
    try {
      const r = await api("/api/kb/upload", { method: "POST", body: fd });
      toast(
        `入库完成：${r.chunks} 个子块${r.ocr_used ? "（OCR）" : ""}，耗时 ${r.elapsed_ms}ms`,
        3600
      );
      loadKB();
    } catch (err) {
      toast("上传失败：" + err.message, 4200);
    }
    e.target.value = "";
  }

  /* ---------------- 评测 ---------------- */
  async function loadEval() {
    try {
      const d = await api("/api/eval/latest");
      if (!d.available) {
        $("#evalState").textContent = "尚未运行";
        return;
      }
      renderEval(d);
    } catch (e) {}
  }

  async function runEval() {
    $("#btnEval").disabled = true;
    $("#evalState").textContent = "启动中…";
    try {
      await api("/api/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 60 }),
      });
      const timer = setInterval(async () => {
        try {
          const p = await api("/api/eval/progress");
          $("#evalState").textContent = p.running ? `评测中 ${p.done}/${p.total}` : p.message || "完成";
          if (!p.running) {
            clearInterval(timer);
            $("#btnEval").disabled = false;
            loadEval();
          }
        } catch (e) {
          clearInterval(timer);
          $("#btnEval").disabled = false;
        }
      }, 1500);
    } catch (e) {
      $("#btnEval").disabled = false;
      $("#evalState").textContent = "启动失败";
      toast(e.message);
    }
  }

  function renderEval(d) {
    const m = d.metrics || {};
    const names = [
      ["faithfulness", "Faithfulness"],
      ["answer_relevancy", "Answer Relevancy"],
      ["context_precision", "Context Precision"],
      ["context_recall", "Context Recall"],
      ["answer_correctness", "Answer Correctness"],
    ];
    $("#metricGrid").innerHTML = names
      .map(
        ([k, l]) =>
          `<div class="metric"><div class="metric-l">${l}</div><div class="metric-v">${
            m[k] == null ? "—" : Number(m[k]).toFixed(3)
          }</div><div class="metric-bar"><i style="width:${Math.round((m[k] || 0) * 100)}%"></i></div></div>`
      )
      .join("");
    $("#evalState").textContent = `${d.run_at} · ${d.evaluated}/${d.testset_size} 条 · ${d.elapsed_s}s`;
    $("#evalMeta").textContent = `整体 ${m.overall || "-"} · 意图准确率 ${d.intent_accuracy} · 平均 ${d.avg_latency_ms}ms · judge=${d.judge}`;
    const rows = Object.entries(d.by_intent || {})
      .map(
        ([k, v]) =>
          `<div class="doc-row"><div>${esc(k)}</div><div class="num">${v.n}</div><div class="num">${v.faithfulness}</div><div class="num">${v.context_recall}</div><div class="num">${v.answer_correctness}</div></div>`
      )
      .join("");
    $("#intentTable").innerHTML =
      `<div class="doc-row thead"><div>意图</div><div class="num">样本</div><div class="num">Faith.</div><div class="num">Recall</div><div class="num">Correct.</div></div>` +
      rows;

    const items = (d.samples || []).slice(0, 40);
    $("#evalList").innerHTML = items
      .map(
        (r) => `<div class="eval-row">
          <div class="eval-q">${esc(r.question)}</div>
          <div class="eval-a">${esc((r.answer || "").slice(0, 180))}</div>
          <div class="eval-m">
            <span>召回 <b>${(r.citations || []).length}</b></span>
            <span>P <b>${r.metrics.context_precision}</b></span>
            <span>R <b>${r.metrics.context_recall}</b></span>
            <span>F <b>${r.metrics.faithfulness}</b></span>
            <span>${r.latency_ms}ms</span>
          </div>
        </div>`
      )
      .join("");
  }

  /* ---------------- 会话 ---------------- */
  async function loadSessions() {
    try {
      const d = await api("/api/sessions");
      $("#sessCount").textContent = `${d.items.length} 个`;
      $("#sessList").innerHTML = d.items.length
        ? d.items
            .map(
              (s) => `<div class="sess-row" data-sid="${s.id}">
                <div class="sess-t">${esc(s.title)}</div>
                <div class="sess-m"><span>${new Date(s.updated_at * 1000).toLocaleString("zh-CN")}</span><span>${s.message_count} 条</span></div>
              </div>`
            )
            .join("")
        : `<div class="empty"><div class="empty-t">暂无会话</div><div class="empty-s">在问答页提出问题后会自动创建。</div></div>`;
      $$("#sessList .sess-row").forEach((r) =>
        r.addEventListener("click", () => {
          $$("#sessList .sess-row").forEach((x) => x.classList.toggle("active", x === r));
          loadSessionDetail(r.dataset.sid);
        })
      );
    } catch (e) {}
  }

  async function loadSessionDetail(sid) {
    try {
      const s = await api("/api/sessions/" + sid);
      $("#sessDetailMeta").textContent = `${s.messages.length} 条消息`;
      const html = s.messages
        .map((m) => {
          if (m.role === "user") return `<div class="dm"><div class="dm-u">${esc(m.content)}</div></div>`;
          const cites = (m.citations || [])
            .slice(0, 6)
            .map((c) => `<span class="dm-cite">[${c.index}] ${esc(c.std_no)} ${esc(c.clause_no)}</span>`)
            .join("");
          return `<div class="dm"><div class="dm-a">${esc((m.content || "").slice(0, 900))}</div><div class="dm-cites">${cites}</div></div>`;
        })
        .join("");
      $("#sessDetail").innerHTML = `<div class="detail-h">${esc(s.title)}</div>
        <div class="detail-meta"><span>创建 <b>${new Date(s.created_at * 1000).toLocaleString("zh-CN")}</b></span>
        <span>更新 <b>${new Date(s.updated_at * 1000).toLocaleString("zh-CN")}</b></span></div>` + html;
    } catch (e) {
      toast("加载会话失败：" + e.message);
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
