const API_BASE = "http://localhost:8000";
document.getElementById("api-base-display").textContent = API_BASE;

const summaryCardsEl = document.getElementById("summary-cards");
const documentsBody = document.getElementById("documents-tbody");
const chunksBody = document.getElementById("chunks-tbody");
const vectorsBody = document.getElementById("vectors-tbody");

const modalBackdrop = document.getElementById("modal-backdrop");
const modalTitle = document.getElementById("modal-title");
const modalBody = document.getElementById("modal-body");
document.getElementById("modal-close").addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", (e) => {
  if (e.target === modalBackdrop) closeModal();
});

function openModal(title, bodyHtml) {
  modalTitle.textContent = title;
  modalBody.innerHTML = bodyHtml;
  modalBackdrop.classList.remove("hidden");
}
function closeModal() {
  modalBackdrop.classList.add("hidden");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function truncate(str, len = 60) {
  if (!str) return "—";
  return str.length > len ? str.slice(0, len) + "…" : str;
}

// ---- Tabs ------------------------------------------------------------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---- Data fetch + render ----------------------------------------------

async function fetchJson(path) {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) throw new Error(`${path} failed: ${resp.status}`);
  return resp.json();
}

function renderSummary(summary) {
  const docs = summary.document_store;
  const vec = summary.vector_store;
  summaryCardsEl.innerHTML = `
    <div class="summary-card">
      <div class="label">bug_reports (SQLite)</div>
      <div class="number">${docs.bug_reports}</div>
      <div class="sub">PostgreSQL stand-in — storage/bug_analyzer.db</div>
    </div>
    <div class="summary-card">
      <div class="label">bug_chunks (SQLite)</div>
      <div class="number">${docs.bug_chunks}</div>
      <div class="sub">FK'd to bug_reports.bug_id</div>
    </div>
    <div class="summary-card">
      <div class="label">${vec.collection} (JSON)</div>
      <div class="number">${vec.points_count}</div>
      <div class="sub">Qdrant stand-in — ${vec.vector_size}d · ${vec.distance} · storage/vector_store.json</div>
    </div>
  `;
}

function renderDocuments(rows) {
  if (!rows.length) {
    documentsBody.innerHTML = `<tr><td colspan="7"><div class="empty-state">No rows yet — ingest a bug report on the main page.</div></td></tr>`;
    return;
  }
  documentsBody.innerHTML = rows
    .map((r) => {
      const s = r.structured;
      return `
      <tr class="clickable-row" data-bug-id="${r.bug_id}">
        <td class="mono">${truncate(r.bug_id, 12)}</td>
        <td><span class="truncate">${escapeHtml(s.metadata.title || s.message)}</span></td>
        <td>${severityBadge(s.metadata.severity)}</td>
        <td>${escapeHtml(s.metadata.component || "—")}</td>
        <td>${escapeHtml(r.source)}</td>
        <td class="mono">${new Date(r.ingested_at).toLocaleString()}</td>
        <td>${escapeHtml(r.validation_status)}</td>
      </tr>
    `;
    })
    .join("");

  documentsBody.querySelectorAll(".clickable-row").forEach((row) => {
    row.addEventListener("click", () => showDocumentDetail(row.dataset.bugId));
  });
}

function severityBadge(severity) {
  const s = severity || "MEDIUM";
  return `<span class="severity ${s}">${s}</span>`;
}

function renderChunks(rows) {
  if (!rows.length) {
    chunksBody.innerHTML = `<tr><td colspan="6"><div class="empty-state">No chunks yet.</div></td></tr>`;
    return;
  }
  chunksBody.innerHTML = rows
    .map(
      (c) => `
      <tr class="clickable-row" data-chunk='${escapeHtml(JSON.stringify(c))}'>
        <td class="mono">${truncate(c.chunk_id, 12)}</td>
        <td class="mono">${truncate(c.bug_id, 12)}</td>
        <td>${c.chunk_index + 1}</td>
        <td><span class="truncate">${escapeHtml(c.chunk_text)}</span></td>
        <td>${c.char_count}</td>
        <td class="mono">${new Date(c.created_at).toLocaleString()}</td>
      </tr>
    `
    )
    .join("");

  chunksBody.querySelectorAll(".clickable-row").forEach((row) => {
    row.addEventListener("click", () => {
      const c = JSON.parse(row.dataset.chunk);
      openModal(
        `Chunk ${c.chunk_index + 1} · ${c.chunk_id}`,
        `<div class="field">bug_id</div><div class="value mono" style="margin-bottom:10px;">${c.bug_id}</div>
         <div class="field">chunk_text (${c.char_count} chars)</div>
         <pre class="stack-trace">${escapeHtml(c.chunk_text)}</pre>`
      );
    });
  });
}

function renderVectors(rows) {
  if (!rows.length) {
    vectorsBody.innerHTML = `<tr><td colspan="6"><div class="empty-state">No vectors yet.</div></td></tr>`;
    return;
  }
  vectorsBody.innerHTML = rows
    .map(
      (v) => `
      <tr class="clickable-row" data-point-id="${v.id}">
        <td class="mono">${truncate(v.id, 12)}</td>
        <td class="mono">${truncate(v.payload.bug_id, 12)}</td>
        <td class="mono">${truncate(v.payload.chunk_id, 12)}</td>
        <td><span class="truncate">${escapeHtml(v.payload.text_preview)}</span></td>
        <td>
          <span class="vector-dots">
            ${v.vector_preview.map((n) => `<span>${n.toFixed(3)}</span>`).join("")}…
          </span>
        </td>
        <td>${v.vector_dim}</td>
      </tr>
    `
    )
    .join("");

  vectorsBody.querySelectorAll(".clickable-row").forEach((row) => {
    row.addEventListener("click", () => showVectorDetail(row.dataset.pointId));
  });
}

async function showDocumentDetail(bugId) {
  try {
    const doc = await fetchJson(`/admin/documents/${bugId}`);
    const s = doc.structured;
    openModal(
      `bug_reports row · ${bugId}`,
      `
      <div class="result-row">
        <div><div class="field">Severity</div><div class="value">${severityBadge(s.metadata.severity)}</div></div>
        <div><div class="field">Component</div><div class="value">${escapeHtml(s.metadata.component || "—")}</div></div>
        <div><div class="field">Error Code</div><div class="value">${escapeHtml(s.error_code || "—")}</div></div>
      </div>
      <div class="field">Message</div>
      <div class="value" style="margin-bottom:10px;">${escapeHtml(s.message)}</div>
      ${s.stack_trace ? `<div class="field">Stack Trace</div><pre class="stack-trace">${escapeHtml(s.stack_trace)}</pre>` : ""}
      <div class="field" style="margin-top:10px;">Chunks (${doc.chunks.length})</div>
      <div class="chunk-list">
        ${doc.chunks
          .map(
            (c) => `<div class="chunk-item"><div class="chunk-head"><span>Chunk ${c.chunk_index + 1}</span><span class="meta">${c.char_count} chars</span></div><pre class="stack-trace">${escapeHtml(c.chunk_text)}</pre></div>`
          )
          .join("")}
      </div>
      <div class="field" style="margin-top:10px;">Raw Text (original input)</div>
      <pre class="stack-trace">${escapeHtml(doc.raw_text)}</pre>
    `
    );
  } catch (err) {
    openModal("Error", `<div class="empty-state">${escapeHtml(err.message)}</div>`);
  }
}

async function showVectorDetail(pointId) {
  try {
    const point = await fetchJson(`/admin/vectors/${pointId}`);
    openModal(
      `bug_vectors point · ${pointId}`,
      `
      <div class="field">Payload</div>
      <pre class="stack-trace">${escapeHtml(JSON.stringify(point.payload, null, 2))}</pre>
      <div class="field" style="margin-top:10px;">Full vector (${point.vector.length} dims)</div>
      <pre class="stack-trace">${escapeHtml(JSON.stringify(point.vector))}</pre>
    `
    );
  } catch (err) {
    openModal("Error", `<div class="empty-state">${escapeHtml(err.message)}</div>`);
  }
}

async function loadAll() {
  try {
    const [summary, documents, chunks, vectors] = await Promise.all([
      fetchJson("/admin/summary"),
      fetchJson("/admin/documents"),
      fetchJson("/admin/chunks"),
      fetchJson("/admin/vectors"),
    ]);
    renderSummary(summary);
    renderDocuments(documents);
    renderChunks(chunks);
    renderVectors(vectors);
  } catch (err) {
    documentsBody.innerHTML = `<tr><td colspan="7"><div class="empty-state">Could not reach backend at ${API_BASE}. Is uvicorn running?</div></td></tr>`;
  }
}

loadAll();
