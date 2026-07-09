const API_BASE = "http://localhost:8000";

document.getElementById("api-base-display").textContent = API_BASE;

const form = document.getElementById("ingest-form");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");
const fileInput = document.getElementById("file-input");
const fileNameEl = document.getElementById("file-name");
const rawTextEl = document.getElementById("raw-text");
const sourceEl = document.getElementById("source");
const sourceHintEl = document.getElementById("source-hint");

const SOURCE_HINTS = {
  manual: "Manual / JIRA / CI-CD / Slack: your submission is always treated as exactly one bug report.",
  jira: "Manual / JIRA / CI-CD / Slack: your submission is always treated as exactly one bug report.",
  "ci-cd": "Manual / JIRA / CI-CD / Slack: your submission is always treated as exactly one bug report.",
  slack: "Manual / JIRA / CI-CD / Slack: your submission is always treated as exactly one bug report.",
  dataset: "Dataset: upload a .csv file and every row is imported as its own bug report. Requests are paced to stay within Groq's free-tier limit (~28-30/min), so large files can take a while — roughly 'row count ÷ 28' minutes. A few thousand rows can take a couple of hours; this is expected, not a hang.",
};

sourceEl.addEventListener("change", () => {
  sourceHintEl.textContent = SOURCE_HINTS[sourceEl.value] || "";
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    const f = fileInput.files[0];
    fileNameEl.textContent = `Selected: ${f.name} (${(f.size / 1024).toFixed(1)} KB) — this will be used instead of the pasted text.`;
    rawTextEl.disabled = true;
    rawTextEl.placeholder = "Using uploaded file instead — clear the file selection to type here.";
  } else {
    fileNameEl.textContent = "";
    rawTextEl.disabled = false;
    rawTextEl.placeholder = "Paste a raw stack trace, crash log, or bug description here...";
  }
});

function setStatus(message, type) {
  statusEl.textContent = message;
  statusEl.className = `status ${type || ""}`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function renderSummary(summary) {
  const hasFailures = summary.failed_count > 0;

  const failuresBlock = hasFailures
    ? `
      <div class="field" style="margin-top:16px; color:var(--err);">
        ${summary.failed_count} row(s) failed and were NOT imported (shown below${summary.failures.length < summary.failed_count ? `, first ${summary.failures.length}` : ""}):
      </div>
      <div class="chunk-list">
        ${summary.failures
          .map(
            (f) => `
          <div class="chunk-item" style="border-color:var(--err);">
            <div class="chunk-head"><span>Row ${f.row_index + 1}</span></div>
            <div class="value" style="color:var(--err); margin-bottom:6px;">${escapeHtml(f.error)}</div>
            <pre class="stack-trace">${escapeHtml(f.preview)}</pre>
          </div>
        `
          )
          .join("")}
      </div>
    `
    : "";

  resultEl.innerHTML = `
    <div class="result-card">
      <div class="field" style="font-size:1rem; color:${hasFailures ? "var(--warn)" : "var(--ok)"}; font-weight:700; margin-bottom:14px;">
        ${escapeHtml(summary.title)}
      </div>
      <div class="summary-cards" style="margin-bottom:0;">
        <div class="summary-card">
          <div class="label">Successfully Imported</div>
          <div class="number">${summary.successfully_imported}</div>
        </div>
        <div class="summary-card">
          <div class="label">SQLite Reports Added</div>
          <div class="number">${summary.sqlite_reports_added}</div>
        </div>
        <div class="summary-card">
          <div class="label">SQLite Chunks Added</div>
          <div class="number">${summary.sqlite_chunks_added}</div>
        </div>
        <div class="summary-card">
          <div class="label">Vector Embeddings Added</div>
          <div class="number">${summary.vector_embeddings_added}</div>
        </div>
        <div class="summary-card">
          <div class="label">Processing Time</div>
          <div class="number">${summary.processing_time_seconds}s</div>
        </div>
      </div>
      ${failuresBlock}
    </div>
  `;
}

async function ingestText(rawText, source) {
  const resp = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text: rawText, source }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || `Request failed with status ${resp.status}`);
  return data;
}

async function ingestFile(file, source) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source", source);

  const resp = await fetch(`${API_BASE}/ingest/file`, {
    method: "POST",
    body: formData,
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || `Request failed with status ${resp.status}`);
  return data;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const source = sourceEl.value;
  const hasFile = fileInput.files.length > 0;
  const rawText = rawTextEl.value.trim();

  if (!hasFile && !rawText) {
    setStatus("Paste some text or choose a file first.", "error");
    return;
  }

  submitBtn.disabled = true;
  setStatus(
    source === "dataset" && hasFile
      ? "Importing dataset — requests are paced to respect Groq's rate limit, so this can take a while for large files. Please keep this tab open..."
      : "Processing...",
    "loading"
  );

  try {
    const summary = hasFile
      ? await ingestFile(fileInput.files[0], source)
      : await ingestText(rawText, source);

    setStatus(
      `${summary.title} — ${summary.successfully_imported} imported${summary.failed_count ? `, ${summary.failed_count} failed` : ""} in ${summary.processing_time_seconds}s.`,
      summary.failed_count ? "error" : "ok"
    );
    renderSummary(summary);
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    submitBtn.disabled = false;
  }
});
