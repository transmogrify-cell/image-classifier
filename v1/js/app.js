import { DEFAULT_SETTINGS, trainClassifier, predict, disposeClassifier, exportClassifier } from './image-classifier.js';
import { watchCamera, cameraSupported } from './camera.js';

const MIN_SAMPLES = 2; // per class; the library holds back 15% (at least one) of each class for validation
const RECORD_FPS = 24;
const SAMPLE_SIZE = 224;
const COLORS = ['#e8710a', '#e52592', '#9334e6', '#1a73e8', '#12b5cb', '#f9ab00', '#1e8e3e', '#d93025'];
const CLOSE_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';

const $ = (selector, root = document) => root.querySelector(selector);

const workspace = $('.workspace');
const connectors = $('.connectors');
const classList = $('.class-list');
const template = $('#class-template');
const addClassButton = $('[data-action="add-class"]');
const appMenu = $('.app-menu');

const trainingCard = $('.training');
const trainButton = $('[data-action="train"]');
const trainStatus = $('.train-status');
const trainFill = $('.train-progress-fill');
const trainDetail = $('.train-detail');
const trainMessage = $('.train-message');
const advancedToggle = $('.advanced-toggle');
const advanced = $('#advanced');
const epochsInput = $('#epochs');
const batchSelect = $('#batch-size');
const learningRateInput = $('#learning-rate');

const previewCard = $('.preview');
const exportButton = $('[data-action="export"]');
const previewEmpty = $('.preview-empty');
const previewLive = $('.preview-live');
const inputToggle = $('.input-toggle');
const switchText = $('.switch-text');
const inputSource = $('.input-source');
const previewCanvas = $('.preview-canvas');
const previewDrop = $('.preview-drop');
const previewFile = $('.preview-file');
const previewMessage = $('.preview-message');
const outputEl = $('.output');

const exportDialog = $('.export-dialog');
const snippetEl = $('.snippet code');

const state = {
  classes: [], // { id, name, samples: HTMLCanvasElement[], el }
  nextId: 1,
  model: null,
  modelClassIds: [],
  dirty: true,
  training: false,
};

let sampleCamera = null; // { cls, recording, last, unsubscribe }
let previewUnsubscribe = null;
let previewToken = null;
let previewImage = null;
let predicting = false;
let outputRows = [];

/* ---------- Helpers ---------- */

const isFileDrag = (event) => Array.from(event.dataTransfer?.types || []).includes('Files');

function decodeImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Unreadable image'));
    };
    image.src = url;
  });
}

// Centre-crops to a square and scales to 224×224, the model's input size.
function squareCanvas(source, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SAMPLE_SIZE;
  const side = Math.min(width, height);
  const context = canvas.getContext('2d');
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  return canvas;
}

function closeMenus(except) {
  document.querySelectorAll('.menu').forEach((menu) => {
    if (menu !== except) menu.hidden = true;
  });
}

function toggleMenu(menu) {
  closeMenus(menu);
  menu.hidden = !menu.hidden;
}

/* ---------- Classes ---------- */

function addClass() {
  if (state.training) return;
  const id = state.nextId++;
  const cls = { id, name: `Class ${id}`, samples: [] };
  cls.el = buildClassCard(cls);
  classList.append(cls.el.root);
  state.classes.push(cls);
  structureChanged();
}

function sizeNameInput(input) {
  input.size = Math.max(4, input.value.length + 1);
}

function buildClassCard(cls) {
  const root = template.content.firstElementChild.cloneNode(true);
  const el = {
    root,
    name: $('.class-name', root),
    menu: $('.class-menu', root),
    label: $('.samples-label', root),
    thumbs: $('.thumbs', root),
    message: $('.class-message', root),
    panel: $('.webcam-panel', root),
    view: $('.webcam-view', root),
    record: $('.record-button', root),
    input: $('.sample-input', root),
  };

  el.name.value = cls.name;
  sizeNameInput(el.name);
  el.name.addEventListener('input', () => sizeNameInput(el.name));
  el.name.addEventListener('change', () => renameClass(cls, el.name.value));
  el.name.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') el.name.blur();
    if (event.key === 'Escape') {
      el.name.value = cls.name;
      sizeNameInput(el.name);
      el.name.blur();
    }
  });

  root.addEventListener('click', (event) => {
    const command = event.target.closest('[data-cmd]')?.dataset.cmd;
    if (command === 'rename') {
      el.name.focus();
      el.name.select();
    } else if (command === 'menu') {
      toggleMenu(el.menu);
    } else if (command === 'clear') {
      closeMenus();
      clearSamples(cls);
    } else if (command === 'delete') {
      closeMenus();
      deleteClass(cls);
    } else if (command === 'webcam') {
      openSampleCamera(cls);
    } else if (command === 'close-webcam') {
      closeSampleCamera();
    } else if (command === 'upload') {
      el.input.click();
    }
  });

  el.input.addEventListener('change', () => {
    addFiles(cls, Array.from(el.input.files));
    el.input.value = '';
  });

  // Hold to Record: mouse, touch or keyboard (Space / Enter).
  const startRecording = () => {
    if (!sampleCamera || sampleCamera.cls !== cls || state.training) return;
    sampleCamera.recording = true;
    el.record.classList.add('is-recording');
    el.record.textContent = 'Recording…';
  };
  const stopRecording = () => {
    if (sampleCamera?.cls === cls) sampleCamera.recording = false;
    el.record.classList.remove('is-recording');
    el.record.textContent = 'Hold to Record';
  };
  el.record.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    el.record.setPointerCapture(event.pointerId);
    startRecording();
  });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach((type) => el.record.addEventListener(type, stopRecording));
  el.record.addEventListener('keydown', (event) => {
    if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) {
      event.preventDefault();
      startRecording();
    }
  });
  el.record.addEventListener('keyup', (event) => {
    if (event.key === ' ' || event.key === 'Enter') stopRecording();
  });
  el.stopRecording = stopRecording;

  root.addEventListener('dragover', (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    root.classList.add('is-drop-target');
  });
  root.addEventListener('dragleave', (event) => {
    if (!root.contains(event.relatedTarget)) root.classList.remove('is-drop-target');
  });
  root.addEventListener('drop', (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    root.classList.remove('is-drop-target');
    addFiles(cls, Array.from(event.dataTransfer.files));
  });

  return el;
}

function renameClass(cls, value) {
  const name = value.trim();
  if (!name) {
    cls.el.name.value = cls.name;
    sizeNameInput(cls.el.name);
    return;
  }
  cls.name = name;
  // Renaming only changes the label, so a trained model can keep working.
  const index = state.classes.indexOf(cls);
  if (modelIsCurrent()) {
    state.model.setLabel(index, name);
    if (outputRows[index]) outputRows[index].name.textContent = name;
  }
}

function deleteClass(cls) {
  if (state.training) return;
  if (state.classes.length <= 1) {
    cls.el.message.textContent = 'You need at least one class.';
    return;
  }
  if (cls.samples.length && !window.confirm(`Delete "${cls.name}" and its ${cls.samples.length} samples?`)) return;
  if (sampleCamera?.cls === cls) closeSampleCamera();
  cls.el.root.remove();
  state.classes.splice(state.classes.indexOf(cls), 1);
  structureChanged();
}

function updateSampleLabel(cls) {
  const count = cls.samples.length;
  cls.el.label.textContent = count ? `${count} Image Sample${count === 1 ? '' : 's'}` : 'Add Image Samples:';
}

function addSample(cls, canvas) {
  cls.samples.push(canvas);
  const thumb = document.createElement('div');
  thumb.className = 'thumb';
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'thumb-remove';
  remove.setAttribute('aria-label', 'Remove sample');
  remove.innerHTML = CLOSE_ICON;
  remove.addEventListener('click', () => {
    if (state.training) return;
    cls.samples.splice(cls.samples.indexOf(canvas), 1);
    thumb.remove();
    updateSampleLabel(cls);
    samplesChanged();
  });
  thumb.append(canvas, remove);
  cls.el.thumbs.append(thumb);
  cls.el.thumbs.scrollTop = cls.el.thumbs.scrollHeight;
  updateSampleLabel(cls);
  samplesChanged();
}

function clearSamples(cls) {
  if (state.training) return;
  cls.samples = [];
  cls.el.thumbs.replaceChildren();
  cls.el.message.textContent = '';
  updateSampleLabel(cls);
  samplesChanged();
}

async function addFiles(cls, files) {
  if (state.training || !files.length) return;
  let failed = 0;
  for (const file of files) {
    if (file.type && !file.type.startsWith('image/')) {
      failed++;
      continue;
    }
    try {
      const image = await decodeImage(file);
      addSample(cls, squareCanvas(image, image.naturalWidth || SAMPLE_SIZE, image.naturalHeight || SAMPLE_SIZE));
    } catch {
      failed++;
    }
  }
  cls.el.message.textContent = failed ? `${failed} file${failed === 1 ? " wasn't" : "s weren't"} added because ${failed === 1 ? "it isn't" : "they aren't"} a readable image.` : '';
}

/* ---------- Sample webcam ---------- */

async function openSampleCamera(cls) {
  if (sampleCamera?.cls === cls) return;
  closeSampleCamera();
  const { el } = cls;
  el.message.textContent = '';
  el.panel.hidden = false;
  el.root.classList.add('is-recording-panel');
  const context = el.view.getContext('2d');
  const session = { cls, recording: false, last: 0, unsubscribe: null };
  sampleCamera = session;

  try {
    const unsubscribe = await watchCamera((frame) => {
      context.drawImage(frame, 0, 0);
      if (!session.recording || state.training) return;
      const now = performance.now();
      if (now - session.last < 1000 / RECORD_FPS) return;
      session.last = now;
      const sample = document.createElement('canvas');
      sample.width = sample.height = SAMPLE_SIZE;
      sample.getContext('2d').drawImage(frame, 0, 0);
      addSample(cls, sample);
    });
    if (sampleCamera === session) session.unsubscribe = unsubscribe;
    else unsubscribe();
  } catch (error) {
    if (sampleCamera === session) {
      sampleCamera = null;
      el.panel.hidden = true;
      el.root.classList.remove('is-recording-panel');
    }
    el.message.textContent = `${error.message} You can still add images with Upload.`;
  }
}

function closeSampleCamera() {
  if (!sampleCamera) return;
  const { cls, unsubscribe } = sampleCamera;
  sampleCamera = null;
  cls.el.stopRecording();
  cls.el.panel.hidden = true;
  cls.el.root.classList.remove('is-recording-panel');
  unsubscribe?.();
}

/* ---------- Training ---------- */

function modelIsCurrent() {
  return Boolean(state.model) && state.modelClassIds.join() === state.classes.map((cls) => cls.id).join();
}

function trainingProblem() {
  if (state.classes.length < 2) return 'Add at least two classes to train a model.';
  const short = state.classes.find((cls) => cls.samples.length < MIN_SAMPLES);
  if (short) return `Add at least ${MIN_SAMPLES} image samples to "${short.name}".`;
  return null;
}

function setTrainMessage(message, kind = 'info') {
  trainMessage.textContent = message;
  trainMessage.dataset.kind = kind;
}

function updateTrainButton() {
  trainButton.classList.remove('is-ready');
  if (state.training) {
    trainButton.textContent = 'Training…';
    trainButton.disabled = true;
  } else if (modelIsCurrent() && !state.dirty) {
    trainButton.textContent = 'Model Trained';
    trainButton.disabled = true;
  } else {
    trainButton.textContent = 'Train Model';
    trainButton.disabled = false;
    if (!trainingProblem()) trainButton.classList.add('is-ready');
  }
  exportButton.disabled = !modelIsCurrent() || state.training;
  addClassButton.disabled = state.training;
}

function samplesChanged() {
  state.dirty = true;
  if (trainMessage.dataset.kind === 'error') setTrainMessage('');
  updateTrainButton();
}

function structureChanged() {
  if (state.model && !modelIsCurrent()) discardModel();
  samplesChanged();
  scheduleConnectors();
}

function setProgress(fraction, text) {
  trainFill.style.width = `${Math.round(fraction * 100)}%`;
  trainDetail.textContent = text;
}

function readSettings() {
  const epochs = Math.min(1000, Math.max(1, Math.round(Number(epochsInput.value)) || DEFAULT_SETTINGS.epochs));
  const learningRate = Math.min(1, Math.max(0.00001, Number(learningRateInput.value) || DEFAULT_SETTINGS.learningRate));
  epochsInput.value = epochs;
  learningRateInput.value = learningRate;
  return { epochs, batchSize: Number(batchSelect.value), learningRate };
}

function writeSettings(settings) {
  epochsInput.value = settings.epochs;
  batchSelect.value = String(settings.batchSize);
  learningRateInput.value = settings.learningRate;
}

async function train() {
  const problem = trainingProblem();
  if (problem) {
    setTrainMessage(problem, 'error');
    return;
  }

  const classes = state.classes.map((cls) => ({ id: cls.id, name: cls.name, samples: cls.samples.slice() }));
  const settings = readSettings();
  const total = classes.reduce((sum, cls) => sum + cls.samples.length, 0);

  state.training = true;
  closeSampleCamera();
  stopPreviewInput();
  updateTrainButton();
  setTrainMessage('');
  trainStatus.hidden = false;
  setProgress(0, 'Preparing training data…');

  let lastLogs = null;
  try {
    const model = await trainClassifier(classes, settings, {
      onPrepare: (done, count) => setProgress((done / count) * 0.2, `Preparing training data: ${done} / ${count}`),
      onEpoch: (epoch, epochs, logs) => {
        lastLogs = logs;
        setProgress(0.2 + (epoch / epochs) * 0.8, `Epoch ${epoch} / ${epochs}. Keep this tab open.`);
      },
    });
    if (state.model) disposeClassifier(state.model);
    state.model = model;
    state.modelClassIds = classes.map((cls) => cls.id);
    state.dirty = false;
    const accuracy = lastLogs && (lastLogs.val_acc ?? lastLogs.val_accuracy);
    setTrainMessage(
      `Trained on ${total} samples.` + (typeof accuracy === 'number' ? ` Accuracy on held-back samples: ${Math.round(accuracy * 100)}%.` : '')
    );
  } catch (error) {
    console.error(error);
    setTrainMessage(error.message || 'Training failed. Try again.', 'error');
  } finally {
    state.training = false;
    trainStatus.hidden = true;
    updateTrainButton();
  }

  if (modelIsCurrent()) startPreview();
}

/* ---------- Preview ---------- */

function discardModel() {
  stopPreviewInput();
  disposeClassifier(state.model);
  state.model = null;
  state.modelClassIds = [];
  previewLive.hidden = true;
  previewEmpty.hidden = false;
  scheduleConnectors();
}

function buildOutputRows() {
  outputRows = state.classes.map((cls, index) => {
    const row = document.createElement('div');
    row.className = 'output-row';
    row.style.setProperty('--class-color', COLORS[index % COLORS.length]);
    const name = document.createElement('span');
    name.className = 'output-name';
    name.textContent = cls.name;
    const bar = document.createElement('span');
    bar.className = 'output-bar';
    const fill = document.createElement('span');
    fill.className = 'output-fill';
    const value = document.createElement('span');
    value.className = 'output-value';
    value.textContent = '0%';
    bar.append(fill, value);
    row.append(name, bar);
    return { row, name, fill, value };
  });
  outputEl.replaceChildren(...outputRows.map(({ row }) => row));
}

function renderPredictions(predictions) {
  predictions.forEach(({ probability }, index) => {
    const row = outputRows[index];
    if (!row) return;
    row.fill.style.width = `${probability * 100}%`;
    row.value.textContent = `${Math.round(probability * 100)}%`;
    row.value.classList.toggle('is-low', probability < 0.15); // keep small values readable on the pale track
  });
}

function setPreviewMessage(message, kind = 'info') {
  previewMessage.textContent = message;
  previewMessage.dataset.kind = kind;
}

function clearPreviewCanvas() {
  previewCanvas.getContext('2d').clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function startPreview() {
  previewEmpty.hidden = true;
  previewLive.hidden = false;
  buildOutputRows();
  if (!cameraSupported) {
    inputSource.value = 'file';
    inputSource.querySelector('[value="webcam"]').disabled = true;
  }
  applyPreviewInput();
  scheduleConnectors();
}

function stopPreviewInput() {
  previewToken = null;
  previewUnsubscribe?.();
  previewUnsubscribe = null;
}

async function applyPreviewInput() {
  stopPreviewInput();
  switchText.textContent = inputToggle.checked ? 'ON' : 'OFF';
  setPreviewMessage('');
  previewDrop.hidden = true;
  if (!modelIsCurrent() || !inputToggle.checked) {
    clearPreviewCanvas();
    return;
  }

  if (inputSource.value === 'file') {
    if (previewImage) classifyPreviewImage(previewImage);
    else {
      clearPreviewCanvas();
      previewDrop.hidden = false;
    }
    return;
  }

  const token = {};
  previewToken = token;
  setPreviewMessage('Starting webcam…');
  const context = previewCanvas.getContext('2d');
  try {
    const unsubscribe = await watchCamera((frame) => {
      context.drawImage(frame, 0, 0);
      if (predicting || !modelIsCurrent() || state.training) return;
      predicting = true;
      predict(state.model, frame)
        .then(renderPredictions)
        .catch((error) => console.error(error))
        .finally(() => {
          predicting = false;
        });
    });
    if (previewToken !== token) {
      unsubscribe();
      return;
    }
    previewUnsubscribe = unsubscribe;
    setPreviewMessage('');
  } catch (error) {
    if (previewToken === token) setPreviewMessage(`${error.message} Switch the input to File to test with images.`, 'error');
  }
}

async function classifyPreviewImage(canvas) {
  previewImage = canvas;
  previewDrop.hidden = true;
  previewCanvas.getContext('2d').drawImage(canvas, 0, 0);
  if (!modelIsCurrent()) return;
  try {
    renderPredictions(await predict(state.model, canvas));
  } catch (error) {
    console.error(error);
    setPreviewMessage("That image couldn't be classified.", 'error');
  }
}

async function previewFromFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith('image/')) {
    setPreviewMessage(`"${file.name}" isn't an image.`, 'error');
    return;
  }
  try {
    const image = await decodeImage(file);
    setPreviewMessage('');
    await classifyPreviewImage(squareCanvas(image, image.naturalWidth || SAMPLE_SIZE, image.naturalHeight || SAMPLE_SIZE));
  } catch {
    setPreviewMessage(`"${file.name}" couldn't be opened.`, 'error');
  }
}

/* ---------- Export ---------- */

const SNIPPET = `<div id="webcam-container"></div>
<div id="label-container" style="white-space: pre"></div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8.5/dist/teachablemachine-image.min.js"></script>
<script>
  // Adapted from the Teachable Machine image library sample (Apache-2.0).
  const URL = './my_model/';
  let model, webcam, labels;

  async function init() {
    model = await tmImage.load(URL + 'model.json', URL + 'metadata.json');
    webcam = new tmImage.Webcam(224, 224, true); // width, height, flip
    await webcam.setup();
    await webcam.play();
    document.getElementById('webcam-container').appendChild(webcam.canvas);
    labels = document.getElementById('label-container');
    window.requestAnimationFrame(loop);
  }

  async function loop() {
    webcam.update();
    const prediction = await model.predict(webcam.canvas);
    labels.textContent = prediction
      .map((p) => p.className + ': ' + p.probability.toFixed(2))
      .join('\\n');
    window.requestAnimationFrame(loop);
  }

  init();
</script>`;

async function downloadModel(button) {
  if (!modelIsCurrent()) return;
  button.disabled = true;
  button.textContent = 'Preparing…';
  try {
    const blob = await exportClassifier(state.model);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'my-image-model.zip';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (error) {
    console.error(error);
    window.alert("The model couldn't be exported. Try again.");
  } finally {
    button.disabled = false;
    button.textContent = 'Download my model';
  }
}

/* ---------- Connectors between cards ---------- */

let connectorFrame = 0;
function scheduleConnectors() {
  cancelAnimationFrame(connectorFrame);
  connectorFrame = requestAnimationFrame(drawConnectors);
}

function drawConnectors() {
  if (getComputedStyle(connectors).display === 'none') return;
  const base = workspace.getBoundingClientRect();
  const training = trainingCard.getBoundingClientRect();
  const preview = previewCard.getBoundingClientRect();
  const targetX = training.left - base.left;
  const targetY = training.top - base.top + training.height / 2;
  const paths = state.classes.map(({ el }) => {
    const card = el.root.getBoundingClientRect();
    const x = card.right - base.left;
    const y = card.top - base.top + card.height / 2;
    const bend = (targetX - x) / 2;
    return `M${x},${y} C${x + bend},${y} ${targetX - bend},${targetY} ${targetX},${targetY}`;
  });
  paths.push(`M${training.right - base.left},${targetY} L${preview.left - base.left},${preview.top - base.top + preview.height / 2}`);
  connectors.innerHTML = paths.map((d) => `<path d="${d}"/>`).join('');
}

/* ---------- Events ---------- */

addClassButton.addEventListener('click', addClass);
trainButton.addEventListener('click', train);

advancedToggle.addEventListener('click', () => {
  const expanded = advancedToggle.getAttribute('aria-expanded') !== 'true';
  advancedToggle.setAttribute('aria-expanded', String(expanded));
  advanced.hidden = !expanded;
});

document.querySelectorAll('[data-step]').forEach((button) => {
  button.addEventListener('click', () => {
    const [id, step] = button.dataset.step.split(':');
    const input = document.getElementById(id);
    const value = Number(input.value) + Number(step);
    input.value = Number(Math.max(Number(input.min), Math.min(Number(input.max), value)).toFixed(5));
  });
});

$('[data-action="reset-settings"]').addEventListener('click', () => writeSettings(DEFAULT_SETTINGS));

inputToggle.addEventListener('change', applyPreviewInput);
inputSource.addEventListener('change', applyPreviewInput);
previewDrop.addEventListener('click', () => previewFile.click());
previewCanvas.addEventListener('click', () => {
  if (inputSource.value === 'file' && inputToggle.checked) previewFile.click();
});
previewFile.addEventListener('change', () => {
  previewFromFile(previewFile.files[0]);
  previewFile.value = '';
});
previewCard.addEventListener('dragover', (event) => {
  if (!isFileDrag(event) || !modelIsCurrent()) return;
  event.preventDefault();
  previewCard.classList.add('is-drop-target');
});
previewCard.addEventListener('dragleave', (event) => {
  if (!previewCard.contains(event.relatedTarget)) previewCard.classList.remove('is-drop-target');
});
previewCard.addEventListener('drop', (event) => {
  if (!isFileDrag(event) || !modelIsCurrent()) return;
  event.preventDefault();
  event.stopPropagation();
  previewCard.classList.remove('is-drop-target');
  if (inputSource.value !== 'file' || !inputToggle.checked) {
    inputSource.value = 'file';
    inputToggle.checked = true;
    stopPreviewInput();
    switchText.textContent = 'ON';
  }
  previewFromFile(event.dataTransfer.files[0]);
});

// Don't let a file dropped elsewhere replace the page.
window.addEventListener('dragover', (event) => {
  if (isFileDrag(event)) event.preventDefault();
});
window.addEventListener('drop', (event) => {
  if (isFileDrag(event)) event.preventDefault();
});

exportButton.addEventListener('click', () => {
  snippetEl.textContent = SNIPPET;
  exportDialog.showModal();
});
exportDialog.addEventListener('click', (event) => {
  const action = event.target.closest('[data-action]')?.dataset.action;
  if (action === 'close-export' || event.target === exportDialog) exportDialog.close();
  if (action === 'download-model') downloadModel(event.target.closest('button'));
  if (action === 'copy-snippet') {
    const button = event.target.closest('button');
    navigator.clipboard?.writeText(SNIPPET).then(() => {
      button.textContent = 'Copied';
      setTimeout(() => (button.textContent = 'Copy'), 1500);
    });
  }
});

$('[data-action="menu"]').addEventListener('click', () => toggleMenu(appMenu));
$('[data-action="new-project"]').addEventListener('click', () => {
  closeMenus();
  if (state.training) return;
  const hasWork = state.classes.some((cls) => cls.samples.length) || state.model;
  if (hasWork && !window.confirm('Start a new project? Your classes, samples and trained model will be removed.')) return;
  closeSampleCamera();
  if (state.model) discardModel();
  state.classes.forEach((cls) => cls.el.root.remove());
  state.classes = [];
  state.nextId = 1;
  previewImage = null;
  setTrainMessage('');
  addClass();
  addClass();
});

document.addEventListener('click', (event) => {
  if (!event.target.closest('.menu, [data-action="menu"], [data-cmd="menu"]')) closeMenus();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeMenus();
});

new ResizeObserver(scheduleConnectors).observe(workspace);
[classList, trainingCard, previewCard].forEach((element) => new ResizeObserver(scheduleConnectors).observe(element));
window.addEventListener('resize', scheduleConnectors);

/* ---------- Start ---------- */

writeSettings(DEFAULT_SETTINGS);
addClass();
addClass();
