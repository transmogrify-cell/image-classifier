import { loadClassifier, classifyImage, ModelLoadError } from './image-classifier.js';

const root = document.getElementById('image-classifier');
const fileInput = root.querySelector('.file-input');
const uploadButton = root.querySelector('[data-action="upload"]');
const cameraButton = root.querySelector('[data-action="camera"]');
const captureButton = root.querySelector('[data-action="capture"]');
const cancelCameraButton = root.querySelector('[data-action="camera-cancel"]');
const preview = root.querySelector('.preview');
const placeholder = root.querySelector('.placeholder');
const cameraControls = root.querySelector('.camera-controls');
const progressEl = root.querySelector('.progress');
const progressFill = root.querySelector('.progress-fill');
const progressDetail = root.querySelector('.progress-detail');
const statusEl = root.querySelector('.status');
const resultEl = root.querySelector('.result');
const captionEl = root.querySelector('.result-caption');
const labelEl = root.querySelector('.result-label');
const confidenceEl = root.querySelector('.result-confidence');
const listEl = root.querySelector('.result-list');

const TOP_K = 5;
const CONFIDENT = 0.6; // below this the answer is labelled a "best guess"
const MAX_SIDE = 1024; // photos are scaled down to this before classifying; the model works at 224 px

let modelReady = false;
let busy = false;
let stream = null;
let previewUrl = null;

/* ---------- Status and model loading ---------- */

function setStatus(message, kind = 'info') {
  statusEl.textContent = message;
  statusEl.dataset.kind = kind;
}

function setControlsDisabled(disabled) {
  uploadButton.disabled = disabled;
  cameraButton.disabled = disabled;
  captureButton.disabled = disabled;
}

function showDownloadProgress(loaded, total) {
  progressEl.hidden = false;
  progressFill.style.width = `${total ? (loaded / total) * 100 : 0}%`;
  progressDetail.textContent = `${Math.round(loaded / 1e6)} MB of ${Math.round(total / 1e6)} MB`;
}

async function ensureModel() {
  if (modelReady) return;
  root.classList.add('is-loading');
  progressEl.hidden = false;
  try {
    await loadClassifier(showDownloadProgress);
    modelReady = true;
  } finally {
    root.classList.remove('is-loading');
    progressEl.hidden = true;
  }
}

/* ---------- Preview ---------- */

function showInPreview(element) {
  preview.replaceChildren(element);
}

function resetPreview() {
  releasePreviewUrl();
  preview.replaceChildren(placeholder);
}

function releasePreviewUrl() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function decodeImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.alt = 'Your photo';
    image.onload = () => resolve({ image, url });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Unreadable image'));
    };
    image.src = url;
  });
}

// Draws an image or video frame onto a canvas, scaled down if it is very large.
function toCanvas(source, width, height) {
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext('2d');
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/* ---------- Results ---------- */

function formatPercent(score) {
  const percent = Math.round(score * 100);
  return percent < 1 ? '<1%' : `${percent}%`;
}

function clearResult() {
  resultEl.hidden = true;
  root.classList.remove('has-result');
}

function renderResult(results) {
  const [top] = results;
  captionEl.textContent = top.score >= CONFIDENT ? 'Looks like' : 'Best guess';
  labelEl.textContent = top.label;
  confidenceEl.textContent = `${formatPercent(top.score)} confident`;

  const fills = [];
  listEl.replaceChildren(
    ...results.map(({ label, score }) => {
      const item = document.createElement('li');
      item.className = 'result-item';

      const name = document.createElement('span');
      name.className = 'result-name';
      name.textContent = label;

      const value = document.createElement('span');
      value.className = 'result-score';
      value.textContent = formatPercent(score);

      const bar = document.createElement('span');
      bar.className = 'result-bar';
      bar.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('span');
      fill.className = 'result-bar-fill';
      bar.append(fill);
      fills.push([fill, score]);

      item.append(name, value, bar);
      return item;
    })
  );

  resultEl.hidden = false;
  root.classList.add('has-result');
  void listEl.offsetWidth; // start bars at 0 so they animate to their value
  fills.forEach(([fill, score]) => {
    fill.style.width = `${score * 100}%`;
  });

  // On narrow screens the answer sits below the photo; bring it into view.
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  resultEl.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'nearest' });
}

/* ---------- Classify ---------- */

async function classify(canvas) {
  busy = true;
  setControlsDisabled(true);
  clearResult();
  try {
    await ensureModel();
    setStatus('Analysing…', 'working');
    renderResult(await classifyImage(canvas, TOP_K));
    setStatus('');
  } catch (error) {
    console.error(error);
    setStatus(
      error instanceof ModelLoadError ? error.message : "Something went wrong analysing this photo. Try another one.",
      'error'
    );
  } finally {
    busy = false;
    setControlsDisabled(false);
  }
}

async function handleFile(file) {
  if (!file || busy) return;

  if (file.type && !file.type.startsWith('image/')) {
    setStatus(`"${file.name}" isn't an image. Choose a JPEG, PNG, WebP or GIF.`, 'error');
    return;
  }

  let decoded;
  try {
    decoded = await decodeImage(file);
  } catch {
    setStatus(`"${file.name}" couldn't be opened. It may be damaged or in a format this browser doesn't support.`, 'error');
    return;
  }

  stopCamera();
  releasePreviewUrl();
  previewUrl = decoded.url;
  showInPreview(decoded.image);

  const { image } = decoded;
  await classify(toCanvas(image, image.naturalWidth || 512, image.naturalHeight || 512));
}

/* ---------- Camera ---------- */

async function startCamera() {
  if (busy || stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
      audio: false,
    });
  } catch (error) {
    const messages = {
      NotAllowedError: 'Camera access is blocked. Allow it in your browser settings to take a photo.',
      NotFoundError: 'No camera was found on this device.',
    };
    setStatus(messages[error.name] || "The camera couldn't be started.", 'error');
    return;
  }

  const video = document.createElement('video');
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  releasePreviewUrl();
  showInPreview(video);
  clearResult();
  setStatus('');
  root.classList.add('is-camera');
  cameraControls.hidden = false;
  await video.play().catch(() => {});

  ensureModel().catch(() => {}); // get the model ready while the photo is framed
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
  root.classList.remove('is-camera');
  cameraControls.hidden = true;
}

async function capturePhoto() {
  const video = preview.querySelector('video');
  if (!video || !video.videoWidth || busy) return;
  const canvas = toCanvas(video, video.videoWidth, video.videoHeight);
  stopCamera();
  showInPreview(canvas);
  await classify(canvas);
}

/* ---------- Events ---------- */

uploadButton.addEventListener('click', () => fileInput.click());
preview.addEventListener('click', () => {
  if (!stream && !busy) fileInput.click();
});
fileInput.addEventListener('change', () => {
  handleFile(fileInput.files[0]);
  fileInput.value = ''; // allow choosing the same file again
});

cameraButton.addEventListener('click', startCamera);
captureButton.addEventListener('click', capturePhoto);
cancelCameraButton.addEventListener('click', () => {
  stopCamera();
  resetPreview();
});

const isFileDrag = (event) => Array.from(event.dataTransfer?.types || []).includes('Files');
window.addEventListener('dragover', (event) => {
  if (!isFileDrag(event)) return;
  event.preventDefault();
  root.classList.add('is-dragging');
});
window.addEventListener('dragleave', (event) => {
  if (!event.relatedTarget) root.classList.remove('is-dragging');
});
window.addEventListener('drop', (event) => {
  if (!isFileDrag(event)) return;
  event.preventDefault();
  root.classList.remove('is-dragging');
  handleFile(event.dataTransfer.files[0]);
});

window.addEventListener('paste', (event) => {
  const item = Array.from(event.clipboardData?.items || []).find((entry) => entry.type.startsWith('image/'));
  if (item) handleFile(item.getAsFile());
});

/* ---------- Start ---------- */

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  cameraButton.hidden = true;
}

if (!/Mac|iPhone|iPad/.test(navigator.platform)) {
  root.querySelector('.placeholder-hint kbd').textContent = 'Ctrl';
}

// Download the model straight away unless the visitor has asked to save data.
if (!navigator.connection?.saveData) {
  ensureModel().catch((error) => setStatus(error.message, 'error'));
}
