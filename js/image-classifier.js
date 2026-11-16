/**
 * Image classification engine.
 *
 * Runs ConvNeXt V2 Base (Meta AI, Apache-2.0), fine-tuned on ImageNet-22k then
 * ImageNet-1k (1,000 everyday categories), entirely in the browser through
 * Transformers.js (Hugging Face, Apache-2.0). The 8-bit model (~91 MB) is
 * downloaded once from the Hugging Face Hub and then cached by the browser.
 *
 * Preprocessing (resize, centre crop to 224 px, ImageNet normalisation) comes
 * from the model's own preprocessor_config.json and is applied by the pipeline.
 */
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

export const MODEL = {
  id: 'Xenova/convnextv2-base-22k-224',
  name: 'ConvNeXt V2 Base',
  dtype: 'q8',
};

export class ModelLoadError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ModelLoadError';
  }
}

let classifierPromise = null;
const progressListeners = new Set();

async function createClassifier() {
  let classifier;
  try {
    classifier = await pipeline('image-classification', MODEL.id, {
      dtype: MODEL.dtype,
      progress_callback: (info) => {
        if (info.status === 'progress_total') {
          progressListeners.forEach((listener) => listener(info.loaded, info.total));
        }
      },
    });
  } catch (error) {
    console.error(error);
    throw new ModelLoadError("Couldn't download the model. Check your internet connection and try again.");
  }

  // Warm-up run so the first real photo isn't slowed by one-off setup work.
  const blank = document.createElement('canvas');
  blank.width = blank.height = 224;
  await classifier(blank, { top_k: 1 });
  return classifier;
}

/**
 * Loads the model once and shares it. onProgress(loadedBytes, totalBytes) is
 * called while files download. A failed load is retried on the next call.
 */
export function loadClassifier(onProgress) {
  if (onProgress) progressListeners.add(onProgress);
  if (!classifierPromise) {
    classifierPromise = createClassifier();
    classifierPromise.catch(() => {
      classifierPromise = null;
    });
  }
  return classifierPromise;
}

// ImageNet labels list synonyms ("Pembroke, Pembroke Welsh corgi"). Use the first,
// or a longer synonym that contains it when that is clearer ("Pembroke Welsh corgi").
function tidyLabel(label) {
  const [first, ...synonyms] = label.split(',').map((part) => part.trim());
  const fuller = synonyms.find((name) => name.length > first.length && name.toLowerCase().includes(first.toLowerCase()));
  const name = fuller || first;
  return name.charAt(0).toUpperCase() + name.slice(1);
}

/**
 * Classifies a Blob/File or canvas. Resolves to the top `topK`
 * { label, score } results, highest score first.
 */
export async function classifyImage(source, topK = 5) {
  const classifier = await loadClassifier();
  const results = await classifier(source, { top_k: topK });
  return results.map(({ label, score }) => ({ label: tidyLabel(label), score }));
}
