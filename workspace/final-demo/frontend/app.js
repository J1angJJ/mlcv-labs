const form = document.querySelector("#upload-form");
const input = document.querySelector("#image-input");
const fileName = document.querySelector("#file-name");
const statusEl = document.querySelector("#status");
const submitButton = document.querySelector("#submit-button");
const inputPreview = document.querySelector("#input-preview");
const predictionPreview = document.querySelector("#prediction-preview");
const countValue = document.querySelector("#count-value");
const confidenceValue = document.querySelector("#confidence-value");
const elapsedValue = document.querySelector("#elapsed-value");
const detectionsValue = document.querySelector("#detections-value");
const boxesBody = document.querySelector("#boxes-body");

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    statusEl.textContent = data.model_loaded ? "Backend ready" : "Model missing";
  } catch {
    statusEl.textContent = "Backend unavailable";
  }
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function setImage(img, src) {
  if (!src) {
    img.removeAttribute("src");
    img.style.display = "none";
    return;
  }
  img.src = src;
  img.style.display = "block";
}

input.addEventListener("change", () => {
  const file = input.files[0];
  fileName.textContent = file ? file.name : "No file selected";
  setImage(inputPreview, file ? URL.createObjectURL(file) : "");
  setImage(predictionPreview, "");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) {
    statusEl.textContent = "Choose an image first";
    return;
  }

  const body = new FormData();
  body.append("file", file);
  submitButton.disabled = true;
  statusEl.textContent = "Running prediction...";

  try {
    const response = await fetch("/predict", { method: "POST", body });
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message);
    }
    const data = await response.json();
    countValue.textContent = data.count;
    confidenceValue.textContent = formatNumber(data.mean_confidence);
    elapsedValue.textContent = `${formatNumber(data.elapsed_ms, 1)} ms`;
    detectionsValue.textContent = data.all_detections;
    setImage(predictionPreview, data.figure_url ? `${data.figure_url}?t=${Date.now()}` : "");
    boxesBody.innerHTML = "";
    data.boxes.forEach((box, index) => {
      const row = document.createElement("tr");
      const coords = [box.x1, box.y1, box.x2, box.y2].map((v) => Math.round(v)).join(", ");
      row.innerHTML = `<td>${index + 1}</td><td>${formatNumber(box.confidence)}</td><td>${coords}</td>`;
      boxesBody.appendChild(row);
    });
    statusEl.textContent = "Prediction complete";
  } catch (error) {
    statusEl.textContent = `Prediction failed: ${error.message}`;
  } finally {
    submitButton.disabled = false;
  }
});

refreshHealth();
