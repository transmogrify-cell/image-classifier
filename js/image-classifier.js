/**
 * Image classification engine for Teachable Machine image models.
 *
 * A thin wrapper around the official Teachable Machine image library
 * (@teachablemachine/image, from github.com/googlecreativelab/teachablemachine-community,
 * Apache-2.0), loaded as the global `tmImage` by index.html.
 *
 * The library does the model-specific preprocessing inside `model.predict()`:
 * it centre-crops the input to a square, resizes it to the model's image size
 * (224px), optionally converts it to grayscale, and normalises pixels to [-1, 1].
 */

export class ModelLoadError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ModelLoadError';
  }
}

// One load per model URL; every caller shares the same promise.
const modelCache = new Map();

function withTrailingSlash(url) {
  return url.endsWith('/') ? url : `${url}/`;
}

async function fetchMetadata(url) {
  let response;
  try {
    response = await fetch(url);
  } catch {
    throw new ModelLoadError(
      `Could not reach ${url}. If you opened this page straight from the file system, serve it from a web server instead.`
    );
  }
  if (!response.ok) {
    throw new ModelLoadError(
      `No model found: ${url} returned HTTP ${response.status}. Add your exported model.json, metadata.json and weights.bin to the model folder.`
    );
  }

  let metadata;
  try {
    metadata = await response.json();
  } catch {
    throw new ModelLoadError('metadata.json is not valid JSON.');
  }
  if (!Array.isArray(metadata.labels) || metadata.labels.length === 0) {
    throw new ModelLoadError('metadata.json lists no class labels. Is this a Teachable Machine image model?');
  }
  return metadata;
}

async function loadFromUrl(baseUrl) {
  if (!window.tf || !window.tmImage) {
    throw new ModelLoadError('The TensorFlow.js libraries did not load. Check your internet connection and reload.');
  }

  const metadata = await fetchMetadata(`${baseUrl}metadata.json`);

  let model;
  try {
    model = await window.tmImage.load(`${baseUrl}model.json`, metadata);
  } catch (error) {
    throw new ModelLoadError(
      `Could not load the model from ${baseUrl}. Check that model.json and weights.bin are both there and come from the same export. (${error.message})`
    );
  }

  const outputCount = model.getTotalClasses();
  if (outputCount !== metadata.labels.length) {
    model.model.dispose();
    throw new ModelLoadError(
      `The model has ${outputCount} outputs but metadata.json lists ${metadata.labels.length} labels. Re-export both files together.`
    );
  }

  // Warm-up run: the first prediction compiles GPU shaders, so do it while "Loading model…" is shown.
  const blank = document.createElement('canvas');
  blank.width = blank.height = 224;
  await model.predict(blank);

  return model;
}

/**
 * Loads a Teachable Machine image model from a folder URL that contains
 * model.json, metadata.json and weights.bin. Cached: repeated calls reuse the
 * loaded model, and a failed load is retried on the next call.
 */
export function loadModel(modelUrl) {
  const baseUrl = withTrailingSlash(modelUrl);
  if (!modelCache.has(baseUrl)) {
    const loading = loadFromUrl(baseUrl);
    loading.catch(() => modelCache.delete(baseUrl));
    modelCache.set(baseUrl, loading);
  }
  return modelCache.get(baseUrl);
}

/**
 * Classifies an image, canvas or video frame.
 * Resolves to one { className, probability } entry per class, in label order.
 */
export function classify(model, source) {
  return model.predict(source);
}

export function topPrediction(predictions) {
  return predictions.reduce((best, current) => (current.probability > best.probability ? current : best));
}
