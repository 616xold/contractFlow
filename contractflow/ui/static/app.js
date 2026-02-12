const formEl = document.getElementById("extract-form");
const modeEl = document.getElementById("mode");
const backendEl = document.getElementById("retrieval_backend");
const verifierRowEl = document.getElementById("verifier-toggle-row");
const runBtnEl = document.getElementById("run-btn");
const statusEl = document.getElementById("status");
const emptyStateEl = document.getElementById("empty-state");
const resultAreaEl = document.getElementById("result-area");
const issuesBoxEl = document.getElementById("issues-box");
const fieldsWrapEl = document.getElementById("fields-table-wrap");
const riskCardEl = document.getElementById("risk-card");
const riskFactorsEl = document.getElementById("risk-factors");
const traceEl = document.getElementById("trace-json");
const rawEl = document.getElementById("raw-text");
const metaModeEl = document.getElementById("meta-mode");
const metaModelEl = document.getElementById("meta-model");
const metaLatencyEl = document.getElementById("meta-latency");
const metaTokensEl = document.getElementById("meta-tokens");

const tabButtons = Array.from(document.querySelectorAll(".tab"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setStatus(text, state = "idle") {
  statusEl.textContent = text;
  statusEl.className = `status ${state}`;
}

function modeUsesRetrieval(mode) {
  return mode === "retrieval" || mode === "field_agents" || mode === "orchestrated";
}

function syncFormByMode() {
  const mode = modeEl.value;
  const usesRetrieval = modeUsesRetrieval(mode);
  backendEl.disabled = !usesRetrieval;
  verifierRowEl.classList.toggle("hidden", mode !== "orchestrated");
}

function setActiveTab(tabId) {
  tabButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tab === tabId);
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === tabId);
  });
}

function formatValue(value) {
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function renderFieldsTable(fields) {
  const table = document.createElement("table");
  table.className = "kv-table";

  const body = document.createElement("tbody");
  Object.entries(fields).forEach(([field, value]) => {
    const row = document.createElement("tr");
    const th = document.createElement("th");
    const td = document.createElement("td");
    th.textContent = field;
    td.textContent = formatValue(value);
    row.appendChild(th);
    row.appendChild(td);
    body.appendChild(row);
  });
  table.appendChild(body);

  fieldsWrapEl.innerHTML = "";
  fieldsWrapEl.appendChild(table);
}

function renderIssues(issues) {
  if (!issues || issues.length === 0) {
    issuesBoxEl.classList.add("hidden");
    issuesBoxEl.innerHTML = "";
    return;
  }
  const html = `<strong>Validation / extraction issues</strong><ul>${issues
    .map((issue) => `<li>${escapeHtml(issue)}</li>`)
    .join("")}</ul>`;
  issuesBoxEl.innerHTML = html;
  issuesBoxEl.classList.remove("hidden");
}

function renderRiskPanel(risk) {
  const level = (risk && risk.risk_level ? String(risk.risk_level) : "unknown").toLowerCase();
  const explanation = risk && risk.risk_explanation ? risk.risk_explanation : "No risk explanation available.";
  const badgeClass = ["low", "medium", "high"].includes(level) ? level : "medium";
  const confidenceText =
    risk && typeof risk.confidence === "number"
      ? `Confidence ${(risk.confidence * 100).toFixed(1)}%`
      : "Confidence n/a";
  const arbitrationText =
    risk && risk.arbitration ? `Arbitration: ${risk.arbitration}` : "Arbitration: n/a";

  riskCardEl.innerHTML = `
    <div class="risk-card">
      <span class="badge ${badgeClass}">${escapeHtml(level.toUpperCase())}</span>
      <p>${escapeHtml(explanation)}</p>
      <div class="meta-bar">
        <span>${escapeHtml(confidenceText)}</span>
        <span>${escapeHtml(arbitrationText)}</span>
      </div>
    </div>
  `;

  const drivers = Array.isArray(risk?.drivers) ? risk.drivers : [];
  const protectors = Array.isArray(risk?.protectors) ? risk.protectors : [];
  const hardTriggers = Array.isArray(risk?.hard_triggers) ? risk.hard_triggers : [];
  const uncertainty = risk?.uncertainty && typeof risk.uncertainty === "object" ? risk.uncertainty : {};
  const orchestration = risk?.orchestration && typeof risk.orchestration === "object" ? risk.orchestration : {};

  riskFactorsEl.innerHTML = `
    <div class="risk-grid">
      <div class="factor-block">
        <h4>Top Risk Drivers</h4>
        <div class="factor-list">
          ${
            drivers.length
              ? drivers
                  .map(
                    (f) => `
              <div class="factor-item">
                <strong>${escapeHtml(f.label || f.factor_id || "factor")}</strong>
                <span>Contribution: ${escapeHtml(String(f.contribution))} | Confidence: ${escapeHtml(String(f.confidence ?? "n/a"))}</span>
              </div>
            `
                  )
                  .join("")
              : '<div class="factor-item"><span>No positive drivers surfaced.</span></div>'
          }
        </div>
      </div>
      <div class="factor-block">
        <h4>Top Protectors</h4>
        <div class="factor-list">
          ${
            protectors.length
              ? protectors
                  .map(
                    (f) => `
              <div class="factor-item">
                <strong>${escapeHtml(f.label || f.factor_id || "factor")}</strong>
                <span>Contribution: ${escapeHtml(String(f.contribution))} | Confidence: ${escapeHtml(String(f.confidence ?? "n/a"))}</span>
              </div>
            `
                  )
                  .join("")
              : '<div class="factor-item"><span>No protective factors surfaced.</span></div>'
          }
        </div>
      </div>
    </div>
    <div class="meta-bar">
      <span>Hard triggers: ${escapeHtml(hardTriggers.join(", ") || "none")}</span>
      <span>High uncertainty: ${escapeHtml(String(Boolean(uncertainty.high_uncertainty)))}</span>
      <span>Review triggered: ${escapeHtml(String(Boolean(orchestration.triggered)))}</span>
    </div>
  `;
}

function collectFormData() {
  const fileInput = document.getElementById("pdf");
  if (!fileInput.files || fileInput.files.length === 0) {
    throw new Error("Please upload a PDF file.");
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("pdf", file);

  const keys = [
    "mode",
    "retrieval_backend",
    "model",
    "top_k",
    "risk_review_top_k",
    "max_chunk_chars",
    "chunk_max_chars",
    "embedding_model",
    "reranker_model",
    "verifier_confidence_threshold",
    "verifier_max_repairs",
  ];
  keys.forEach((key) => {
    const el = document.getElementById(key);
    if (el && "value" in el) {
      formData.append(key, el.value);
    }
  });

  const checkboxKeys = ["enable_verifier", "enable_risk_review", "enable_risk_judge", "use_ocr", "structured_outputs"];
  checkboxKeys.forEach((key) => {
    const el = document.getElementById(key);
    formData.append(key, String(Boolean(el && el.checked)));
  });
  return formData;
}

async function runExtraction(event) {
  event.preventDefault();
  setStatus("Running extraction...", "running");
  runBtnEl.disabled = true;

  try {
    const formData = collectFormData();
    const response = await fetch("/api/extract", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : "Request failed";
      throw new Error(detail);
    }

    renderFieldsTable(payload.fields || {});
    renderIssues(payload.issues || []);
    renderRiskPanel(payload.risk || {});
    traceEl.textContent = JSON.stringify(payload.retrieval_trace || {}, null, 2);
    rawEl.textContent = payload.raw_text || "";

    const meta = payload.meta || {};
    metaModeEl.textContent = `mode: ${meta.mode || "n/a"}`;
    metaModelEl.textContent = `model: ${meta.model || "n/a"}`;
    metaLatencyEl.textContent = `latency: ${meta.latency_ms || "n/a"} ms`;
    metaTokensEl.textContent = `tokens: ${meta.total_tokens || "n/a"}`;

    emptyStateEl.classList.add("hidden");
    resultAreaEl.classList.remove("hidden");
    setStatus("Done", "idle");
  } catch (err) {
    setStatus(err.message || "Extraction failed", "error");
  } finally {
    runBtnEl.disabled = false;
  }
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

modeEl.addEventListener("change", syncFormByMode);
formEl.addEventListener("submit", runExtraction);
syncFormByMode();
