const $ = (sel) => document.querySelector(sel);
const state = { last: null, loading: false, controller: null };

const SOURCE_LABEL = {
  ml_model: "ML model",
  dataset_evidence: "Dataset evidence",
  rule_based: "Curated rule",
  none: "No prediction",
};
const BADGE = {
  ok: (d) => d.predicted_label || "Prediction",
  rule: (d) => d.predicted_label || "Rule result",
  evidence: (d) => d.predicted_label || "Dataset result",
  uncertain: (d) => d.predicted_class === 1
    ? "Positive signal · low confidence"
    : (d.predicted_label || "Uncertain prediction"),
  not_supported: () => "Not supported",
};

function pct(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- health ---------- */
async function loadHealth() {
  const pill = $("#health");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("health");
    const data = await res.json();
    pill.className = "pill pill-ok";
    pill.replaceChildren(el("i"), el("span", null,
      `Ready · ${data.drugs.toLocaleString()} drugs · ${data.compounds.toLocaleString()} compounds`));
  } catch {
    pill.className = "pill pill-fail";
    pill.replaceChildren(el("i"), el("span", null, "API offline"));
  }
}

/* ---------- examples ---------- */
async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const { pairs } = await res.json();
    const box = $("#chips");
    pairs.forEach((p) => {
      const shortFood = p.food.length > 22 ? `${p.food.slice(0, 20)}…` : p.food;
      const chip = el("button", "chip", `${p.drug} + ${shortFood}`);
      chip.type = "button";
      chip.title = `${p.drug} + ${p.food}`;
      chip.append(el("small", null, p.note));
      chip.addEventListener("click", () => {
        $("#drug").value = p.drug;
        $("#food").value = p.food;
        runPrediction();
      });
      box.append(chip);
    });
  } catch {
    /* chips are optional */
  }
}

/* ---------- suggestions ---------- */
function wireSuggestions(inputId, listId, kind) {
  const input = $(`#${inputId}`);
  const list = $(`#${listId}`);
  let timer = null;
  let controller = null;

  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 1) {
      list.replaceChildren();
      return;
    }
    timer = setTimeout(async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const res = await fetch(`/api/suggest?kind=${kind}&q=${encodeURIComponent(q)}&limit=8`,
          { signal: controller.signal });
        if (!res.ok) return;
        const { suggestions } = await res.json();
        list.replaceChildren(...suggestions.map((s) => {
          const opt = document.createElement("option");
          opt.value = s;
          return opt;
        }));
      } catch (err) {
        if (err.name !== "AbortError") console.warn(err);
      }
    }, 150);
  });
}

/* ---------- rendering ---------- */
function highlight(parent, text, terms) {
  const hits = terms
    .filter(Boolean)
    .map((t) => ({ t, i: text.toLowerCase().indexOf(t.toLowerCase()) }))
    .filter((h) => h.i >= 0)
    .sort((a, b) => a.i - b.i);
  let cursor = 0;
  hits.forEach(({ t, i }) => {
    if (i < cursor) return;
    if (i > cursor) parent.append(document.createTextNode(text.slice(cursor, i)));
    parent.append(el("span", "em", text.slice(i, i + t.length)));
    cursor = i + t.length;
  });
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function severityFor(data) {
  if (data.status === "rule") return "rule";
  if (data.status === "not_supported") return "unsupported";
  return data.severity || "neutral";
}

function renderError(message) {
  const root = $("#result");
  root.replaceChildren();
  const box = el("div", "error-box");
  box.append(el("h3", null, "Could not run the prediction"));
  box.append(el("p", null, message));
  root.append(box);
}

function renderUnsupported(data) {
  const root = $("#result");
  const verdict = el("div", "verdict");
  verdict.dataset.sev = "unsupported";
  const top = el("div", "verdict-top");
  top.append(el("span", "badge", "Cannot score this pair"));
  top.append(el("span", "source-tag", "No prediction"));
  verdict.append(top);
  const pair = [data.drug || data.inputs?.drug, data.food || data.inputs?.food]
    .filter(Boolean).join(" + ");
  if (pair) verdict.append(el("p", "sentence", pair));
  verdict.append(el("p", "reason", data.reason || "This input is outside the supported range."));
  root.replaceChildren(verdict);
  renderShap(root, data);
  root.append(disclaimerStrip(data.disclaimer));
}

function disclaimerStrip(text) {
  const strip = el("div", "disclaimer-strip");
  strip.innerHTML = `<svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true"><path d="M10 1.8 18.2 16H1.8L10 1.8Zm0 4.6a.9.9 0 0 0-.9.9v3.4a.9.9 0 0 0 1.8 0V7.3a.9.9 0 0 0-.9-.9Zm0 6.2a.95.95 0 1 0 0 1.9.95.95 0 0 0 0-1.9Z" fill="currentColor"/></svg>`;
  strip.append(el("span", null, text || "Research tool; confirm with a physician or pharmacist."));
  return strip;
}

function renderResult(data) {
  const root = $("#result");
  root.replaceChildren();
  if (data.status === "not_supported") return renderUnsupported(data);

  const sev = severityFor(data);
  const verdict = el("div", "verdict");
  verdict.dataset.sev = sev;
  const top = el("div", "verdict-top");
  const badgeText = BADGE[data.status] ? BADGE[data.status](data) : "Prediction";
  const badge = el("span", "badge", badgeText);
  badge.dataset.sev = sev;
  top.append(badge);
  top.append(el("span", "source-tag", SOURCE_LABEL[data.prediction_source] || "Prediction"));
  if (typeof data.confidence === "number") {
    top.append(el("span", "meta-inline", `${pct(data.confidence)} confidence`));
  }
  verdict.append(top);

  const sentence = el("p", "sentence");
  highlight(sentence, data.sentence || data.predicted_label || "Prediction complete", [data.drug, data.food]);
  verdict.append(sentence);
  if (data.reason && data.status === "uncertain") verdict.append(el("p", "reason", data.reason));
  root.append(verdict);

  // Keep the three class probability bars as the primary model detail.
  if (Array.isArray(data.probability_rows) && data.probability_rows.length) {
    const best = data.probability_rows.reduce((a, b) => (b.probability > a.probability ? b : a));
    const probs = el("div", "probs");
    data.probability_rows.forEach((row, i) => {
      const line = el("div", "prob");
      line.dataset.class = String(row.class);
      if (row.class === best.class) line.classList.add("best");
      line.append(el("span", "prob-label", row.label));
      const track = el("div", "prob-track");
      const fill = el("div", "prob-fill");
      track.append(fill);
      line.append(track, el("span", "prob-value", pct(row.probability)));
      probs.append(line);
      requestAnimationFrame(() => {
        fill.style.width = `${Math.max(row.probability * 100, 1.5)}%`;
      });
    });
    root.append(probs);
  }

  // Show only the evidence provenance that helps users verify a known pair.
  if (data.evidence) {
    const evidence = data.evidence;
    const parts = [
      `Dataset evidence${evidence.record_id != null ? ` #${evidence.record_id}` : ""}`,
      [evidence.source_food, evidence.source_component].filter(Boolean).join(" → "),
      evidence.annotation,
      evidence.pmid ? `PMID ${evidence.pmid}` : null,
    ].filter(Boolean);
    root.append(el("p", "evidence-note", parts.join(" · ")));
  }

  if (data.extrapolation) {
    root.append(el("p", "result-note", "This drug was not represented in the model's training labels."));
  }

  // Every matched compound gets its own visible explanation, not just the selected one.
  renderCandidateExplanations(root, data);
  root.append(disclaimerStrip(data.disclaimer));
}

function shapRows(features, limit = 8) {
  const box = el("div", "shap-rows");
  const visible = features.slice(0, limit);
  const maxAbs = Math.max(...visible.map((f) => Math.abs(Number(f.shap_value) || 0)), 1e-9);
  visible.forEach((f) => {
    const value = Number(f.shap_value);
    if (!Number.isFinite(value)) return;
    const row = el("div", "shap-row");
    const name = el("span", "shap-name", f.feature || "Feature");
    name.title = `${f.feature || "Feature"} · input value: ${f.value ?? "not supplied"}`;
    row.append(name);
    const track = el("div", "shap-track");
    const fill = el("div", "shap-fill");
    fill.style.width = `${Math.max((Math.abs(value) / maxAbs) * 100, value === 0 ? 0 : 2)}%`;
    fill.dataset.direction = value >= 0 ? "increases" : "decreases";
    track.append(fill);
    row.append(track);
    row.append(el("span", "shap-value", `${value >= 0 ? "+" : ""}${value.toFixed(3)}`));
    box.append(row);
  });
  return box;
}

function shapFeatureList(explanation) {
  if (!explanation || typeof explanation !== "object") return [];
  if (Array.isArray(explanation.top_features) && explanation.top_features.length) {
    return explanation.top_features;
  }
  return Array.isArray(explanation.features) ? explanation.features : [];
}

function createShapPanel(explanation, options = {}) {
  const {
    title = "SHAP feature contributions",
    note = "",
    emptyMessage = "The model did not return SHAP attributions for this pair.",
    compact = false,
  } = options;
  const panel = el("section", compact ? "shap shap-compact" : "shap shap-panel");
  const header = el("div", "shap-heading");
  const heading = el("h4", null, title);
  header.append(heading);
  if (explanation?.method) header.append(el("span", "shap-method", "TreeSHAP"));
  panel.append(header);

  const features = shapFeatureList(explanation);
  if (features.length) {
    const label = explanation.predicted_label || explanation.label;
    if (label) panel.append(el("p", "shap-class-label", `Contributions for: ${label}`));
    panel.append(shapRows(features, compact ? 6 : 8));
    if (explanation.feature_count) {
      panel.append(el("p", "shap-scale-note", `Showing the ${Math.min(features.length, compact ? 6 : 8)} strongest of ${explanation.feature_count} features. Bar lengths are relative within this explanation; signed SHAP values are shown at right.`));
    }

    const byClass = explanation.by_class;
    if (!compact && byClass && typeof byClass === "object") {
      const otherClasses = Object.entries(byClass).filter(([key]) =>
        Number(key) !== Number(explanation.predicted_class)
      );
      if (otherClasses.length) {
        const more = el("details", "shap-other-classes");
        more.append(el("summary", null, "Compare SHAP contributions for the other classes"));
        otherClasses.forEach(([, classData]) => {
          const group = el("div", "shap-other-class");
          group.append(el("p", "shap-class-label", classData.label || "Other class"));
          const rows = Array.isArray(classData.contributions) ? classData.contributions : [];
          if (rows.length) group.append(shapRows(rows, 5));
          more.append(group);
        });
        panel.append(more);
      }
    }
  } else {
    panel.append(el("p", "shap-unavailable", emptyMessage));
  }

  if (note) panel.append(el("p", "shap-context-note", note));
  panel.append(el("p", "shap-caveat", explanation?.caveat ||
    "SHAP explains how input features influenced the model score. It does not establish a biological mechanism or prove a clinical interaction."));
  return panel;
}

function renderShap(root, data) {
  const explanation = data.explainability || data.model_support?.explainability || null;
  const support = data.model_support;
  let note = "";
  if (data.prediction_source === "dataset_evidence") {
    note = "Supporting model view only: the displayed verdict comes from an existing dataset annotation, not from this SHAP explanation.";
  } else if (data.prediction_source === "rule_based") {
    note = "Supporting model view only: the displayed verdict comes from a curated rule, not from this SHAP explanation.";
  } else if (support?.role === "diagnostic") {
    note = "Diagnostic explanation only: the system did not issue an accepted prediction for this pair.";
  }
  let emptyMessage;
  if (support?.available === false) {
    emptyMessage = support.note || "The model could not score this compound, so SHAP values cannot be calculated.";
  } else if (support?.shap_available === false) {
    emptyMessage = support.note || "The model returned a score, but SHAP feature attributions were unavailable.";
  } else if (data.shap_reason) {
    emptyMessage = "SHAP is unavailable because the model could not build a valid feature vector for this input.";
  } else {
    emptyMessage = "No SHAP values were returned. Check that the SHAP dependency and model explainer are available.";
  }
  root.append(createShapPanel(explanation, { note, emptyMessage }));
}

function renderCandidateExplanations(root, data) {
  if (!Array.isArray(data.candidates) || data.candidates.length < 2) {
    renderShap(root, data);
    return;
  }

  const section = el("section", "candidate-predictions");
  const header = el("div", "candidate-predictions-head");
  header.append(el("h3", null, "Compound-level explanations"));
  header.append(el("span", "candidate-count", `${data.candidates.length} compounds`));
  section.append(header);
  section.append(el("p", "candidate-intro", "The food name matched multiple compounds. Each compound below has its own model result and SHAP contribution bars."));

  data.candidates.forEach((candidate) => {
    const card = el("article", "candidate-card");
    const cardHead = el("div", "candidate-card-head");
    const title = el("h4", null, candidate.compound || "Matched compound");
    cardHead.append(title);
    if (candidate.compound === data.food) cardHead.append(el("span", "candidate-selected", "Selected result"));
    card.append(cardHead);

    const meta = el("p", "candidate-meta");
    const pieces = [candidate.label || "No class", candidate.status || null];
    if (typeof candidate.confidence === "number") pieces.push(`${pct(candidate.confidence)} confidence`);
    meta.textContent = pieces.filter(Boolean).join(" · ");
    card.append(meta);

    let contextNote = "";
    if (candidate.prediction_source === "dataset_evidence") {
      contextNote = "Supporting model view only; the class label comes from the dataset annotation.";
    } else if (candidate.prediction_source === "rule_based") {
      contextNote = "Supporting model view only; the class label comes from a curated rule.";
    }
    card.append(createShapPanel(candidate.explainability, {
      title: "SHAP feature contributions",
      note: contextNote,
      emptyMessage: candidate.shap_available === false
        ? "No SHAP attribution was returned for this compound. Check structure coverage and explainer availability."
        : "SHAP was not available for this compound. The model may not have been able to featurize its structure.",
      compact: true,
    }));
    section.append(card);
  });
  root.append(section);
}

function showLoading() {
  const root = $("#result");
  root.setAttribute("aria-busy", "true");
  root.replaceChildren();
  const skel = el("div", "skeleton");
  ["w40 tall", "w90", "w70", "w90", "w70"].forEach((c) => {
    const bar = el("div", `skel ${c}`);
    skel.append(bar);
  });
  root.append(skel);
}

/* ---------- prediction ---------- */
async function runPrediction() {
  const drug = $("#drug").value.trim();
  const food = $("#food").value.trim();
  const error = $("#form-error");
  error.textContent = "";
  if (!drug || !food) {
    error.textContent = "Enter both a drug and a food compound.";
    (!drug ? $("#drug") : $("#food")).focus();
    return;
  }

  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.loading = true;
  const btn = $("#submit");
  btn.classList.add("loading");
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Scoring…";
  showLoading();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drug, food }),
      signal: controller.signal,
    });
    const data = await res.json();
    if (!res.ok) {
      renderError(data.detail || `Request failed with status ${res.status}.`);
    } else {
      state.last = { drug, food };
      renderResult(data);
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    renderError("The API is unreachable. Start it with: uvicorn web.app:app --port 8000");
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      state.loading = false;
      btn.classList.remove("loading");
      btn.disabled = false;
      btn.querySelector(".btn-label").textContent = "Predict interaction";
      $("#result").removeAttribute("aria-busy");
    }
  }
}

/* ---------- wiring ---------- */
$("#form").addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});

$("#swap").addEventListener("click", () => {
  const drug = $("#drug");
  const food = $("#food");
  [drug.value, food.value] = [food.value, drug.value];
  drug.focus();
});

wireSuggestions("drug", "drug-list", "drug");
wireSuggestions("food", "food-list", "food");
loadHealth();
loadExamples();
