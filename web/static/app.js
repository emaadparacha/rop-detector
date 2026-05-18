const form = document.getElementById("predict-form");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("image-input");
const fileName = document.getElementById("file-name");
const predictBtn = document.getElementById("predict-btn");
const resultCard = document.getElementById("result-card");
const resultSummary = document.getElementById("result-summary");
const originalImage = document.getElementById("original-image");
const heatmapImage = document.getElementById("heatmap-image");
const reasonsFor = document.getElementById("reasons-for");
const reasonsAgainst = document.getElementById("reasons-against");
const errorBanner = document.getElementById("error-banner");

function showFileName(file) {
  fileName.textContent = file ? file.name : "";
}

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    showFileName(fileInput.files[0]);
  }
});

["dragenter", "dragover"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  });
});
dropZone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
    showFileName(files[0]);
  }
});

function renderReasons(listEl, items) {
  listEl.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "None identified.";
    li.style.color = "#94a3b8";
    listEl.appendChild(li);
    return;
  }
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    listEl.appendChild(li);
  });
}

function setError(msg) {
  if (!msg) {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
    return;
  }
  errorBanner.hidden = false;
  errorBanner.textContent = msg;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError(null);

  if (!fileInput.files || fileInput.files.length === 0) {
    setError("Please choose an image first.");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";

  try {
    const data = new FormData(form);
    const response = await fetch("/predict", { method: "POST", body: data });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Request failed: ${response.status}`);
    }
    const result = await response.json();

    const isRop = result.predicted_label === "rop";
    const summaryClass = isRop ? "rop" : "not-rop";
    const summaryText = isRop ? "ROP detected" : "No active ROP detected";

    resultSummary.innerHTML = "";
    const summary = document.createElement("div");
    summary.className = `summary ${summaryClass}`;
    summary.innerHTML = `
      <div>
        <div class="label">${summaryText}</div>
        <div>Confidence: <strong>${result.confidence_percent}%</strong></div>
        <div>ROP probability: ${result.probability_rop_percent}% &middot; Not-ROP probability: ${result.probability_not_rop_percent}%</div>
        ${
          result.metadata_used
            ? '<div style="color:#475569;font-size:12px;">Used clinical metadata.</div>'
            : '<div style="color:#94a3b8;font-size:12px;">No clinical metadata supplied. Image only.</div>'
        }
      </div>
      <div class="meter">
        <div class="bar"><span style="width:${result.probability_rop_percent}%"></span></div>
        <small>ROP probability</small>
      </div>
    `;
    resultSummary.appendChild(summary);

    originalImage.src = `data:image/png;base64,${result.original_png_base64}`;
    heatmapImage.src = `data:image/png;base64,${result.heatmap_png_base64}`;

    renderReasons(reasonsFor, result.reasons_for_rop);
    renderReasons(reasonsAgainst, result.reasons_against_rop);

    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setError(err.message || "Unknown error.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Analyze image";
  }
});
