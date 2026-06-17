const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const dropzoneEmpty = document.getElementById("dropzone-empty");
const dropzonePreview = document.getElementById("dropzone-preview");
const dropzoneLoading = document.getElementById("dropzone-loading");
const previewImage = document.getElementById("preview-image");
const previewFilename = document.getElementById("preview-filename");
const resultsEmpty = document.getElementById("results-empty");
const resultsContent = document.getElementById("results-content");
const toast = document.getElementById("toast");
const howItWorksBtn = document.getElementById("how-it-works-btn");
const howItWorksModal = document.getElementById("modal-overlay");
const modalClose = document.getElementById("modal-close");
const useLlmToggle = document.getElementById("use-llm-toggle");
const llmKeyPanel = document.getElementById("llm-key-panel");
const openaiApiKeyInput = document.getElementById("openai-api-key");
const loadingText = document.getElementById("loading-text");

let selectedFile = null;

const RULE_LABELS = {
  axis_baseline: "Axis baseline",
  consistent_scaling: "Consistent scaling",
  zoom_requirement: "Zoom requirement",
  labelling: "Labelling",
};

const RULE_ORDER = [
  "axis_baseline",
  "consistent_scaling",
  "zoom_requirement",
  "labelling",
];

const RULE_SHORT_LABELS = {
  axis_baseline: "Baseline",
  consistent_scaling: "Scaling",
  zoom_requirement: "Zoom",
  labelling: "Labels",
};

const STATUS_COLORS = {
  pass: "#059669",
  fail: "#dc2626",
  unknown: "#9ca3af",
  not_applicable: "#d1d5db",
};

const STATUS_LABELS = {
  pass: "Pass",
  fail: "Fail",
  unknown: "Unknown",
  not_applicable: "Not applicable",
  compliant: "Compliant",
  non_compliant: "Non-compliant",
  skipped: "Skipped",
  ok: "OK",
};

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function formatStatus(status) {
  if (!status) return "Unknown";
  return STATUS_LABELS[status] || status.replace(/_/g, " ");
}

function formatRuleName(rule) {
  return RULE_LABELS[rule] || rule.replace(/_/g, " ");
}

function formatChartType(type) {
  if (!type) return "Chart";
  return type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatPosition(position) {
  if (!position) return "";
  return position.replace(/_/g, " ");
}

function formatOrientation(orientation) {
  if (!orientation) return "";
  const normalized = orientation.toLowerCase();
  if (normalized === "horizontal") return "Horizontal";
  if (normalized === "vertical") return "Vertical";
  return orientation.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function inferOrientationFromNotes(notes) {
  const text = (notes || "").toLowerCase();
  const hasHorizontal = /\bhorizontal\b/.test(text);
  const hasVertical = /\bvertical\b/.test(text);
  if (hasHorizontal && !hasVertical) return "horizontal";
  if (hasVertical && !hasHorizontal) return "vertical";
  return null;
}

function resolveChartOrientation(chart, compliance = {}) {
  const candidates = [
    chart?.orientation,
    chart?.detector_label,
    inferOrientationFromNotes(compliance.notes),
  ];

  for (const value of candidates) {
    const normalized = (value || "").toLowerCase();
    if (normalized === "horizontal" || normalized === "vertical") {
      return normalized;
    }
  }

  return null;
}

function renderOrientationTag(orientation) {
  if (!orientation) return "";
  const label = formatOrientation(orientation);
  return `<span class="chart-tag chart-tag--${escapeHtml(orientation)}">${escapeHtml(label)}</span>`;
}

function cleanEvidence(text) {
  if (!text) return "";
  return text
    .replace(/THINKING STEP \d+:\s*/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function statusBadgeClass(status) {
  const normalized = (status || "unknown").toLowerCase();
  if (["pass", "compliant", "ok"].includes(normalized)) return "badge--compliant";
  if (["fail", "non_compliant", "non-compliant"].includes(normalized)) return "badge--non-compliant";
  if (["not_applicable", "skipped"].includes(normalized)) return "badge--skipped";
  return "badge--unknown";
}

function overallBadgeClass(compliant) {
  if (compliant === true) return "badge--compliant";
  if (compliant === false) return "badge--non-compliant";
  return "badge--unknown";
}

function ruleStatusColor(status) {
  return STATUS_COLORS[(status || "unknown").toLowerCase()] || STATUS_COLORS.unknown;
}

function ruleStatusFill(status) {
  return (status || "unknown").toLowerCase() === "unknown" ? "url(#rule-hatch)" : ruleStatusColor(status);
}

function polarToCartesian(cx, cy, radius, angleRad) {
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad),
  };
}

function describeDonutSegment(cx, cy, outerR, innerR, startDeg, endDeg, gapDeg = 3) {
  const start = ((startDeg + gapDeg / 2) * Math.PI) / 180;
  const end = ((endDeg - gapDeg / 2) * Math.PI) / 180;
  const startOuter = polarToCartesian(cx, cy, outerR, start);
  const endOuter = polarToCartesian(cx, cy, outerR, end);
  const startInner = polarToCartesian(cx, cy, innerR, end);
  const endInner = polarToCartesian(cx, cy, innerR, start);
  const largeArc = end - start > Math.PI ? 1 : 0;

  return [
    `M ${startOuter.x.toFixed(2)} ${startOuter.y.toFixed(2)}`,
    `A ${outerR} ${outerR} 0 ${largeArc} 1 ${endOuter.x.toFixed(2)} ${endOuter.y.toFixed(2)}`,
    `L ${startInner.x.toFixed(2)} ${startInner.y.toFixed(2)}`,
    `A ${innerR} ${innerR} 0 ${largeArc} 0 ${endInner.x.toFixed(2)} ${endInner.y.toFixed(2)}`,
    "Z",
  ].join(" ");
}

const AXIS_BASELINE_ASSUMED_PASS_MESSAGE =
  "No value axis was detected in this chart. When the axis cannot be read, the chart is treated as starting at zero.";

const SAME_UNIT_NEGATIVE_PATTERNS = [
  /\bnot\s+the\s+same\s+unit\b/i,
  /\bnot\s+same\s+unit\b/i,
  /\bunits?\s+differ\b/i,
  /\bunit\s+does\s+not\s+match\b/i,
  /\bcannot\s+confirm\b[^.]{0,80}\bsame\s+unit\b/i,
  /\bunit\s+is\s+unreadable\b/i,
  /\bunit\s+unreadable\b/i,
];

const SAME_UNIT_POSITIVE_PATTERNS = [
  /\bsame\s+units?\b/i,
  /\bshare(?:s|d)?\s+the\s+same\s+unit\b/i,
  /\bshared\s+unit\b/i,
  /\btreat(?:ed)?\s+as\s+same\s+unit\b/i,
  /\ball\b[^.]{0,80}\bsame\s+unit\b/i,
  /\bunit\s+match\b[^.]{0,100}\bsame\s+unit\b/i,
  /\bsame\s+unit\s*\/\s*semantics\b/i,
  /\bcomparable\s+unit\b/i,
  /\bconsistent\s+unit\b/i,
];

function evidenceIndicatesSameUnit(evidence) {
  const text = evidence || "";
  if (SAME_UNIT_NEGATIVE_PATTERNS.some((pattern) => pattern.test(text))) {
    return false;
  }
  return SAME_UNIT_POSITIVE_PATTERNS.some((pattern) => pattern.test(text));
}

function normalizeRuleEntry(key, rawStatus, rawEvidence) {
  const status = rawStatus || "unknown";
  const evidence = cleanEvidence(rawEvidence || "");

  if (key === "axis_baseline" && status === "unknown") {
    return {
      status: "pass",
      evidence: AXIS_BASELINE_ASSUMED_PASS_MESSAGE,
    };
  }

  if (key === "consistent_scaling" && status !== "pass" && evidenceIndicatesSameUnit(evidence)) {
    return {
      status: "pass",
      evidence,
    };
  }

  return { status, evidence };
}

function getRuleEntries(ruleChecks) {
  return RULE_ORDER.map((key) => {
    const raw = ruleChecks?.[key] || {};
    const normalized = normalizeRuleEntry(key, raw.status, raw.evidence);
    return {
      key,
      label: formatRuleName(key),
      shortLabel: RULE_SHORT_LABELS[key] || formatRuleName(key),
      status: normalized.status,
      evidence: normalized.evidence,
    };
  });
}

function countPassingRules(rules) {
  const applicable = rules.filter((r) => r.status !== "not_applicable");
  const passed = applicable.filter((r) => r.status === "pass").length;
  return { passed, applicable: applicable.length };
}

function renderScoreRing(ruleChecks, compliant) {
  const rules = getRuleEntries(ruleChecks);
  const { passed, applicable } = countPassingRules(rules);
  const size = 168;
  const cx = size / 2;
  const cy = size / 2;
  const outerR = 74;
  const innerR = 52;
  const segmentSpan = 360 / rules.length;

  const segments = rules
    .map((rule, index) => {
      const startDeg = -90 + index * segmentSpan;
      const endDeg = startDeg + segmentSpan;
      const path = describeDonutSegment(cx, cy, outerR, innerR, startDeg, endDeg);
      return `
        <path
          class="score-segment"
          data-rule="${rule.key}"
          d="${path}"
          fill="${ruleStatusFill(rule.status)}"
          tabindex="0"
          role="button"
          aria-label="${escapeHtml(rule.label)}: ${escapeHtml(formatStatus(rule.status))}"
        />
      `;
    })
    .join("");

  const centerLabel =
    compliant === true ? "Compliant" : compliant === false ? "Non-compliant" : "Review";
  const centerScore = applicable ? `${passed}/${applicable}` : "—";

  const legend = rules
    .map(
      (rule) => `
        <button type="button" class="score-legend-item" data-rule="${rule.key}">
          <span class="score-legend-dot${rule.status === "unknown" ? " score-legend-dot--hatch" : ""}"${rule.status === "unknown" ? "" : ` style="background:${ruleStatusColor(rule.status)}"`}></span>
          <span class="score-legend-label">${escapeHtml(rule.shortLabel)}</span>
          <span class="score-legend-status">${escapeHtml(formatStatus(rule.status))}</span>
        </button>
      `
    )
    .join("");

  return `
    <div class="score-ring-wrap">
      <div class="score-ring-visual">
        <svg class="score-ring" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" aria-hidden="true">
          <defs>
            <pattern id="rule-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" fill="#f3f4f6"/>
              <line x1="0" y1="0" x2="0" y2="6" stroke="#d1d5db" stroke-width="1"/>
            </pattern>
          </defs>
          <circle cx="${cx}" cy="${cy}" r="${outerR}" fill="#f3f4f6"/>
          ${segments}
        </svg>
        <div class="score-ring-center">
          <span class="score-ring-value">${escapeHtml(centerScore)}</span>
          <span class="score-ring-label">${escapeHtml(centerLabel)}</span>
        </div>
      </div>
      <div class="score-legend">${legend}</div>
    </div>
  `;
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("toast--error", isError);
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 5000);
}

function setUiState(state) {
  dropzoneEmpty.classList.toggle("hidden", state !== "empty");
  dropzonePreview.classList.toggle("hidden", state !== "preview");
  dropzoneLoading.classList.toggle("hidden", state !== "loading");
  analyzeBtn.disabled = state === "loading";
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.removeAttribute("src");
  previewFilename.textContent = "";
  document.getElementById("results-card").classList.remove("results-card--populated");
  resultsContent.classList.add("hidden");
  resultsContent.innerHTML = "";
  resultsEmpty.classList.remove("hidden");
  setUiState("empty");
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showToast("Please choose a PNG, JPEG, or WebP image.", true);
    return;
  }
  selectedFile = file;
  previewFilename.textContent = file.name;
  previewImage.src = URL.createObjectURL(file);
  setUiState("preview");
}

function renderRuleChecks(ruleChecks) {
  if (!ruleChecks || typeof ruleChecks !== "object") return "";

  const items = getRuleEntries(ruleChecks)
    .map((rule) => `
      <article class="rule-check-card" id="rule-${rule.key}" data-rule="${rule.key}">
        <div class="rule-check-header">
          <span class="rule-check-name">
            <span class="rule-check-dot${rule.status === "unknown" ? " rule-check-dot--hatch" : ""}"${rule.status === "unknown" ? "" : ` style="background:${ruleStatusColor(rule.status)}"`}></span>
            ${escapeHtml(rule.label)}
          </span>
          <span class="badge ${statusBadgeClass(rule.status)}">${escapeHtml(formatStatus(rule.status))}</span>
        </div>
        ${
          rule.evidence
            ? `<p class="rule-check-evidence">${escapeHtml(rule.evidence).replace(/\n/g, "<br>")}</p>`
            : `<p class="rule-check-evidence rule-check-evidence--empty">No additional evidence recorded.</p>`
        }
      </article>
    `)
    .join("");

  return `
    <section class="dashboard-panel dashboard-panel--rules">
      <h3 class="section-title">Rule details</h3>
      <div class="rule-checks-grid">${items}</div>
    </section>
  `;
}

function renderViolations(violations) {
  if (!violations?.length) return "";

  const items = violations
    .map((v) => `
        <li class="violation-item">
          <div class="violation-header">
            <span class="badge badge--non-compliant">${escapeHtml(formatRuleName(v.rule))}</span>
          </div>
          <p class="violation-text">${escapeHtml(v.description || "")}</p>
        </li>
      `)
    .join("");

  return `
    <section class="dashboard-panel dashboard-panel--violations">
      <h3 class="section-title">Violations</h3>
      <ul class="violations-list">${items}</ul>
    </section>
  `;
}

function renderChartCard(chart, index) {
  const compliance = chart.compliance_chart || {};
  const chartLabel = compliance.id
    ? compliance.id.replace("_", " ").toUpperCase()
    : `Chart ${index + 1}`;

  const metaParts = [
    compliance.type ? formatChartType(compliance.type) : chart.detector_label,
    compliance.position ? formatPosition(compliance.position) : null,
    compliance.unit ? `Unit: ${compliance.unit}` : null,
  ].filter(Boolean);

  const range = compliance.estimated_range;
  const rangeText =
    range && (range.min != null || range.max != null)
      ? `Range: ${range.min ?? "?"} – ${range.max ?? "?"}`
      : "";

  const notes = compliance.notes || "";
  const heroUrl = chart.label_overlay_url || chart.bar_overlay_url || chart.crop_path || "";
  const orientation = resolveChartOrientation(chart, compliance);

  return `
    <article class="chart-card chart-card--visual">
      ${heroUrl ? `<img class="chart-card-hero" src="${escapeHtml(heroUrl)}" alt="${escapeHtml(chartLabel)} overlay" loading="lazy">` : ""}
      <div class="chart-card-body">
        <div class="chart-card-heading">
          <h4 class="chart-card-title">${escapeHtml(chartLabel)}</h4>
          ${renderOrientationTag(orientation)}
        </div>
        <p class="chart-card-meta">${escapeHtml(metaParts.join(" · "))}</p>
        ${rangeText ? `<p class="chart-range">${escapeHtml(rangeText)}</p>` : ""}
        ${notes ? `<p class="chart-notes">${escapeHtml(notes)}</p>` : ""}
      </div>
    </article>
  `;
}

function renderComplianceOnlyCharts(chartsDetected) {
  if (!chartsDetected?.length) return "";

  const cards = chartsDetected
    .map((chart, index) => {
      const metaParts = [
        chart.type ? formatChartType(chart.type) : null,
        chart.position ? formatPosition(chart.position) : null,
        chart.unit ? `Unit: ${chart.unit}` : null,
      ].filter(Boolean);

      const range = chart.estimated_range;
      const rangeText =
        range && (range.min != null || range.max != null)
          ? `Range: ${range.min ?? "?"} – ${range.max ?? "?"}`
          : "";
      const orientation = inferOrientationFromNotes(chart.notes);

      return `
        <article class="chart-card">
          <div class="chart-card-info">
            <div class="chart-card-heading">
              <h4 class="chart-card-title">${escapeHtml((chart.id || `chart_${index + 1}`).replace("_", " ").toUpperCase())}</h4>
              ${renderOrientationTag(orientation)}
            </div>
            <p class="chart-card-meta">${escapeHtml(metaParts.join(" · "))}</p>
            ${rangeText ? `<p class="chart-range">${escapeHtml(rangeText)}</p>` : ""}
            ${chart.notes ? `<p class="chart-notes">${escapeHtml(chart.notes)}</p>` : ""}
          </div>
        </article>
      `;
    })
    .join("");

  return `
    <section class="dashboard-panel dashboard-panel--charts">
      <h3 class="section-title">Charts in this dashboard</h3>
      <div class="charts-grid charts-grid--text-only">${cards}</div>
    </section>
  `;
}

function bindScoreInteractions(container) {
  const focusRule = (ruleKey) => {
    const target = container.querySelector(`#rule-${ruleKey}`);
    if (!target) return;
    target.classList.add("rule-check-card--highlight");
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setTimeout(() => target.classList.remove("rule-check-card--highlight"), 1200);
  };

  container.querySelectorAll(".score-segment, .score-legend-item").forEach((el) => {
    el.addEventListener("click", () => focusRule(el.dataset.rule));
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        focusRule(el.dataset.rule);
      }
    });
  });
}

function complianceSourceLabel(source) {
  if (source === "llm") return "Live LLM analysis";
  if (source === "bundled") return "Bundled compliance report";
  return null;
}

function renderResults(data) {
  resultsEmpty.classList.add("hidden");
  resultsContent.classList.remove("hidden");
  document.getElementById("results-card").classList.add("results-card--populated");

  const compliance = data.compliance;
  const matched = data.compliance_matched;
  const sourceLabel = complianceSourceLabel(data.compliance_source);
  const charts = data.results || [];
  const chartCountLabel = charts.length
    ? `${charts.length} chart${charts.length === 1 ? "" : "s"} detected`
    : matched && compliance?.charts_detected?.length
      ? `${compliance.charts_detected.length} chart${compliance.charts_detected.length === 1 ? "" : "s"} in report`
      : "";

  const headerHtml = matched && compliance
    ? `
      <header class="dashboard-header">
        <div class="dashboard-header-text">
          <h3 class="dashboard-filename">${escapeHtml(data.source_filename || "Dashboard")}</h3>
          ${chartCountLabel ? `<p class="dashboard-meta">${escapeHtml(chartCountLabel)}</p>` : ""}
          ${sourceLabel ? `<p class="dashboard-meta dashboard-meta--source">${escapeHtml(sourceLabel)}</p>` : ""}
        </div>
        <div class="dashboard-header-badges">
          <span class="badge ${overallBadgeClass(compliance.compliant)}">
            ${compliance.compliant ? "IBCS compliant" : "IBCS non-compliant"}
          </span>
        </div>
      </header>
    `
    : `
      <header class="dashboard-header dashboard-header--warning">
        <div class="dashboard-header-text">
          <h3 class="dashboard-filename">${escapeHtml(data.source_filename || "Dashboard")}</h3>
          <p class="dashboard-meta">Detection overlays only — no compliance report</p>
        </div>
        <span class="badge badge--unknown">No report</span>
      </header>
      <p class="compliance-hint">
        Enable <strong>Run live LLM compliance analysis</strong> and enter your OpenAI API key to get rule checks and explanations for any image.
        Otherwise, upload a dataset image with its original filename (e.g. <code>88.png</code>) to match a bundled report.
      </p>
    `;

  const heroVisualHtml = data.labeled_output_url
    ? `
      <section class="dashboard-hero-visual">
        <h3 class="section-title">Chart detection</h3>
        <div class="hero-visual-frame">
          <img class="labeled-output" src="${escapeHtml(data.labeled_output_url)}" alt="Detected chart bounding boxes" loading="lazy">
        </div>
      </section>
    `
    : `<section class="dashboard-hero-visual dashboard-hero-visual--empty"><p class="hero-visual-placeholder">No detection overlay available.</p></section>`;

  const scoreHtml =
    matched && compliance?.rule_checks
      ? `
        <section class="dashboard-score-panel">
          <h3 class="section-title">IBCS score</h3>
          ${renderScoreRing(compliance.rule_checks, compliance.compliant)}
        </section>
      `
      : "";

  let perChartHtml = "";
  if (charts.length) {
    perChartHtml = `
      <section class="dashboard-panel dashboard-panel--charts">
        <h3 class="section-title">Per-chart overlays</h3>
        <div class="charts-grid">${charts.map((c, i) => renderChartCard(c, i)).join("")}</div>
      </section>
    `;
  } else if (matched && compliance?.charts_detected?.length) {
    perChartHtml = renderComplianceOnlyCharts(compliance.charts_detected);
  }

  const explanationHtml =
    matched && compliance?.final_explanation
      ? `
        <section class="dashboard-panel dashboard-panel--summary">
          <h3 class="section-title">Summary</h3>
          <p class="final-explanation">${escapeHtml(cleanEvidence(compliance.final_explanation)).replace(/\n/g, "<br>")}</p>
        </section>
      `
      : "";

  const ruleChecksHtml = matched ? renderRuleChecks(compliance.rule_checks) : "";
  const violationsHtml = matched && compliance?.violations?.length ? renderViolations(compliance.violations) : "";

  const detailsHtml =
    explanationHtml || ruleChecksHtml || violationsHtml
      ? `<div class="dashboard-details">${explanationHtml}${violationsHtml}${ruleChecksHtml}</div>`
      : "";

  const topClass = scoreHtml ? "dashboard-top" : "dashboard-top dashboard-top--visual-only";

  resultsContent.innerHTML = `
    <div class="results-dashboard">
      ${headerHtml}
      <div class="${topClass}">
        ${heroVisualHtml}
        ${scoreHtml}
      </div>
      ${perChartHtml}
      ${detailsHtml}
      ${
        data.results_json_url
          ? `<footer class="dashboard-footer"><a class="download-link" href="${escapeHtml(data.results_json_url)}" download>Download full results JSON</a></footer>`
          : ""
      }
    </div>
  `;

  bindScoreInteractions(resultsContent);
}

async function analyzeDashboard() {
  if (!selectedFile) return;

  const useLlm = Boolean(useLlmToggle?.checked);
  const openaiApiKey = (openaiApiKeyInput?.value || "").trim();

  if (useLlm && !openaiApiKey) {
    showToast("Enter your OpenAI API key to run LLM analysis.", true);
    return;
  }

  setUiState("loading");
  if (loadingText) {
    loadingText.textContent = useLlm
      ? "Running ML detection and LLM compliance analysis…"
      : "Running IBCS analysis…";
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("use_llm", useLlm ? "true" : "false");
  if (useLlm) {
    formData.append("openai_api_key", openaiApiKey);
  }

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.detail || `Analysis failed (${response.status})`);
    }

    renderResults(data);
    setUiState("preview");
  } catch (err) {
    showToast(err.message || "Analysis failed. Please try again.", true);
    setUiState("preview");
  }
}

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) handleFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dropzone--active");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dropzone--active"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dropzone--active");
  if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
});

analyzeBtn.addEventListener("click", analyzeDashboard);
clearBtn.addEventListener("click", clearSelection);

function syncLlmPanel() {
  const enabled = Boolean(useLlmToggle?.checked);
  llmKeyPanel?.classList.toggle("hidden", !enabled);
  if (!enabled && openaiApiKeyInput) {
    openaiApiKeyInput.value = "";
  }
}

useLlmToggle?.addEventListener("change", syncLlmPanel);
syncLlmPanel();

function openModal() {
  howItWorksModal.classList.remove("hidden");
  howItWorksModal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  howItWorksModal.classList.add("hidden");
  howItWorksModal.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

howItWorksBtn.addEventListener("click", openModal);
modalClose.addEventListener("click", closeModal);
howItWorksModal.addEventListener("click", (e) => {
  if (e.target === howItWorksModal) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !howItWorksModal.classList.contains("hidden")) closeModal();
});
