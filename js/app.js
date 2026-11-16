import { loadModel, classify, topPrediction } from './image-classifier.js';

const root = document.getElementById('image-classifier');
const modelUrl = root.dataset.modelUrl || './model/';

const statusEl = root.querySelector('.ic-status');
const fileInput = root.querySelector('.ic-file-input');
const uploadButton = root.querySelector('[data-action="upload"]');
const webcamButton = root.querySelector('[data-action="webcam"]');
const scanner = root.querySelector('.ic-scanner');
const preview = root.querySelector('.ic-preview');
const resultEl = root.querySelector('.ic-result');
const topLabelEl = root.querySelector('.ic-top-label');
const topScoreEl = root.querySelector('.ic-top-score');
const scoresEl = root.querySelector('.ic-scores');
const tickerTracks = root.querySelectorAll('.ic-ticker-track');

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

// The webcam setup follows the Teachable Machine image library's sample
// snippet (Apache-2.0), including its iOS workaround for frozen video.
const isIos = /iPhone|iPad/.test(navigator.userAgent);
const WEBCAM_SIZE = 224;
const CONFIDENT = 0.9;

let model = null;
let webcam = null;
let webcamFrame = 0;
let previewUrl = null;
let scoreRows = [];
let shownTopLabel = null;
let shownPercent = 0;
let counterFrame = 0;

function setStatus(message, kind = 'info') {
  statusEl.textContent = message;
  statusEl.dataset.kind = kind;
}

function setBusy(busy) {
  uploadButton.disabled = busy;
  webcamButton.disabled = busy;
  root.classList.toggle('is-busy', busy);
}

// Restarts a CSS animation by removing and re-adding its class.
function replayAnimation(element, className) {
  element.classList.remove(className);
  void element.offsetWidth;
  element.classList.add(className);
}

function fillTickers(words) {
  const run = words.map((word) => `${word} ✺ `).join('');
  const half = run.repeat(Math.max(2, Math.ceil(160 / run.length)));
  tickerTracks.forEach((track) => {
    track.textContent = half + half; // two identical halves loop seamlessly
    track.style.animationDuration = `${half.length * 0.22}s`;
  });
}

async function ensureModel() {
  if (!model) {
    setStatus('Loading model…', 'loading');
    model = await loadModel(modelUrl);
    buildScoreRows(model.getClassLabels());
    fillTickers(model.getClassLabels());
  }
  return model;
}

// One row per class, built once per model so live webcam updates only touch widths and text.
function buildScoreRows(labels) {
  scoresEl.replaceChildren();
  scoreRows = labels.map((label, index) => {
    const item = document.createElement('li');
    item.className = 'ic-score';

    const rank = document.createElement('span');
    rank.className = 'ic-score-rank';
    rank.textContent = String(index + 1).padStart(2, '0');

    const name = document.createElement('span');
    name.className = 'ic-score-name';
    name.textContent = label;

    const value = document.createElement('span');
    value.className = 'ic-score-value';

    const bar = document.createElement('span');
    bar.className = 'ic-score-bar';
    bar.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span');
    fill.className = 'ic-score-fill';
    bar.append(fill);

    item.append(rank, name, value, bar);
    scoresEl.append(item);
    return { item, fill, value };
  });
}

function formatPercent(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

// Counts the big percentage up to its new value; jumps straight there for live webcam frames.
function showTopPercent(probability, immediate) {
  const target = probability * 100;
  cancelAnimationFrame(counterFrame);
  topScoreEl.setAttribute('aria-label', formatPercent(probability));

  if (immediate || reducedMotion.matches) {
    shownPercent = target;
    topScoreEl.textContent = formatPercent(probability);
    return;
  }

  const from = shownPercent;
  topScoreEl.textContent = `${from.toFixed(1)}%`;
  const start = performance.now();
  const step = (now) => {
    const progress = Math.min(1, (now - start) / 700);
    shownPercent = from + (target - from) * (1 - (1 - progress) ** 3);
    topScoreEl.textContent = `${shownPercent.toFixed(1)}%`;
    if (progress < 1) counterFrame = requestAnimationFrame(step);
  };
  counterFrame = requestAnimationFrame(step);
}

function renderResults(predictions, { live = false } = {}) {
  const top = topPrediction(predictions);

  if (top.className !== shownTopLabel) {
    shownTopLabel = top.className;
    topLabelEl.textContent = top.className;
    topLabelEl.dataset.text = top.className;
    replayAnimation(topLabelEl, 'is-glitching');
    if (top.probability >= CONFIDENT) replayAnimation(resultEl, 'is-sure');
  }
  showTopPercent(top.probability, live);

  predictions.forEach((prediction, index) => {
    const row = scoreRows[index];
    row.fill.style.width = `${prediction.probability * 100}%`;
    row.value.textContent = formatPercent(prediction.probability);
    row.item.classList.toggle('is-top', prediction === top);
  });
  resultEl.hidden = false;
}

function clearResults() {
  resultEl.hidden = true;
  shownTopLabel = null;
  shownPercent = 0;
  cancelAnimationFrame(counterFrame);
}

function showInPreview(element) {
  preview.replaceChildren(element);
  preview.classList.add('has-content');
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
    image.alt = 'Uploaded image';
    image.onload = () => resolve({ image, url });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Unreadable image'));
    };
    image.src = url;
  });
}

async function handleFile(file) {
  if (!file) return;
  stopWebcam();

  if (file.type && !file.type.startsWith('image/')) {
    setStatus(`"${file.name}" is not an image. Choose a JPG, PNG, WebP or GIF file.`, 'error');
    return;
  }

  let decoded;
  try {
    decoded = await decodeImage(file);
  } catch {
    setStatus(`Couldn't read "${file.name}". The file may be damaged or in a format this browser can't open.`, 'error');
    return;
  }

  releasePreviewUrl();
  previewUrl = decoded.url;
  clearResults();
  // Preview a copy: the TM library crops using element.width/height, which become
  // the CSS size once an <img> is in the page, so the classified image stays off-DOM.
  showInPreview(decoded.image.cloneNode());
  replayAnimation(scanner, 'is-scanning');

  setBusy(true);
  try {
    const loadedModel = await ensureModel();
    setStatus('Classifying…', 'loading');
    renderResults(await classify(loadedModel, decoded.image));
    setStatus('');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function startWebcam() {
  setBusy(true);
  let loadedModel;
  try {
    loadedModel = await ensureModel();
  } catch (error) {
    setStatus(error.message, 'error');
    setBusy(false);
    return;
  }

  setStatus('Starting webcam…', 'loading');
  const camera = new window.tmImage.Webcam(WEBCAM_SIZE, WEBCAM_SIZE, true);
  try {
    await camera.setup();
    if (isIos) {
      camera.webcam.setAttribute('playsinline', '');
      camera.webcam.muted = true;
    }
    await camera.play();
  } catch (error) {
    if (camera.webcam && camera.webcam.srcObject) camera.stop();
    // The library rejects with plain strings for camera errors.
    setStatus(typeof error === 'string' ? error : 'Could not start the webcam.', 'error');
    setBusy(false);
    return;
  }

  releasePreviewUrl();
  clearResults();
  showInPreview(isIos ? camera.webcam : camera.canvas);
  webcam = camera;
  root.classList.add('is-live');
  webcamButton.textContent = 'Stop webcam';
  setStatus('Live');
  setBusy(false);

  const frameSource = isIos ? camera.webcam : camera.canvas;
  const loop = async () => {
    if (webcam !== camera) return;
    camera.update();
    const predictions = await classify(loadedModel, frameSource);
    if (webcam !== camera) return;
    renderResults(predictions, { live: true });
    webcamFrame = requestAnimationFrame(loop);
  };
  webcamFrame = requestAnimationFrame(loop);
}

function stopWebcam() {
  if (!webcam) return;
  cancelAnimationFrame(webcamFrame);
  webcam.stop();
  webcam = null;
  root.classList.remove('is-live');
  webcamButton.textContent = 'Use webcam';
  preview.replaceChildren();
  preview.classList.remove('has-content');
  clearResults();
  setStatus('');
}

uploadButton.addEventListener('click', () => fileInput.click());
scanner.addEventListener('click', () => {
  if (!webcam && !uploadButton.disabled) fileInput.click();
});
fileInput.addEventListener('change', () => {
  handleFile(fileInput.files[0]);
  fileInput.value = ''; // allow choosing the same file again
});
webcamButton.addEventListener('click', () => (webcam ? stopWebcam() : startWebcam()));

// Drop an image anywhere on the page.
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
  if (!uploadButton.disabled) handleFile(event.dataTransfer.files[0]);
});

// Background glow follows the pointer.
if (!reducedMotion.matches) {
  window.addEventListener(
    'pointermove',
    (event) => {
      document.body.style.setProperty('--mx', `${event.clientX}px`);
      document.body.style.setProperty('--my', `${event.clientY}px`);
    },
    { passive: true }
  );
}

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  webcamButton.hidden = true;
}

fillTickers(['Upload', 'Webcam', 'Classify']);

// Start loading straight away so the first prediction is fast.
setBusy(true);
ensureModel()
  .then((loadedModel) => setStatus(`Model ready: ${loadedModel.getClassLabels().length} classes`))
  .catch((error) => setStatus(error.message, 'error'))
  .finally(() => setBusy(false));
