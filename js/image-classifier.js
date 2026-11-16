/**
 * Training, prediction and export, using the Teachable Machine image library
 * (@teachablemachine/image, github.com/googlecreativelab/teachablemachine-community,
 * Apache-2.0), loaded as the global `tmImage`, on TensorFlow.js (global `tf`).
 *
 * Architecture (the library's own TeachableMobileNet, unchanged):
 *   image → centre crop + resize to 224×224, pixels scaled to [-1, 1]
 *         → MobileNet V2 (alpha 0.35, ImageNet weights, frozen), cut at `out_relu`
 *         → global average pooling → 1,280 features
 *         → Dense(100, ReLU) → Dense(classes, softmax)   ← the only layers trained
 *
 * The library's code defaults to MobileNet V1; V2/0.35 is used here because
 * that is what Teachable Machine's own exported image models contain, so
 * models exported from this page load anywhere a Teachable Machine model does.
 */

export const BASE_MODEL = { version: 2, alpha: 0.35 };
export const DEFAULT_SETTINGS = { epochs: 50, batchSize: 16, learningRate: 0.001 };
const DENSE_UNITS = 100; // hidden layer size Teachable Machine uses

let basePromise = null;

// The frozen MobileNet feature extractor, downloaded once (~2 MB) and reused for every training run.
function loadBaseModel() {
  if (!basePromise) {
    basePromise = window.tmImage.loadTruncatedMobileNet(BASE_MODEL);
    basePromise.catch(() => {
      basePromise = null;
    });
  }
  return basePromise;
}

/**
 * Trains a classifier on the given classes.
 * classes: [{ name, samples: HTMLCanvasElement[] }] (224×224 canvases)
 * hooks: onPrepare(done, total), onEpoch(epoch, totalEpochs, logs)
 * Resolves to a trained TeachableMobileNet.
 */
export async function trainClassifier(classes, settings, hooks = {}) {
  if (!window.tf || !window.tmImage) {
    throw new Error('The TensorFlow.js libraries did not load. Check your internet connection and reload.');
  }

  let base;
  try {
    base = await loadBaseModel();
  } catch {
    throw new Error("Couldn't download MobileNet. Check your internet connection and try again.");
  }

  const labels = classes.map((item) => item.name);
  const model = new window.tmImage.TeachableMobileNet(base, {
    labels,
    modelName: 'my-image-model',
    tfjsVersion: window.tf.version.tfjs,
    imageSize: 224,
  });
  model.setLabels(labels);

  // Feature extraction: each sample goes through MobileNet once; only its 1,280 features are kept.
  const total = classes.reduce((sum, item) => sum + item.samples.length, 0);
  let done = 0;
  for (let index = 0; index < classes.length; index++) {
    for (const sample of classes[index].samples) {
      await model.addExample(index, sample);
      done++;
      if (done % 10 === 0 || done === total) {
        hooks.onPrepare?.(done, total);
        await new Promise((resolve) => setTimeout(resolve, 0)); // let the page repaint; unlike tf.nextFrame this keeps going in background tabs
      }
    }
  }

  await model.train(
    {
      denseUnits: DENSE_UNITS,
      epochs: settings.epochs,
      learningRate: settings.learningRate,
      batchSize: settings.batchSize,
    },
    {
      onEpochEnd: (epoch, logs) => hooks.onEpoch?.(epoch + 1, settings.epochs, logs),
    }
  );
  return model;
}

/** Confidence for every class, in class order: [{ className, probability }]. */
export function predict(model, source) {
  return model.predict(source);
}

/** Frees the trained head. The shared MobileNet base is kept for the next run. */
export function disposeClassifier(model) {
  model?.trainingModel?.dispose();
}

/**
 * Packs the trained model the way Teachable Machine's "Download" export does:
 * a zip with model.json, weights.bin and metadata.json.
 */
export async function exportClassifier(model) {
  let artifacts;
  await model.save(
    window.tf.io.withSaveHandler(async (saved) => {
      artifacts = saved;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    })
  );

  const modelJson = {
    modelTopology: artifacts.modelTopology,
    format: artifacts.format,
    generatedBy: artifacts.generatedBy,
    convertedBy: artifacts.convertedBy,
    weightsManifest: [{ paths: ['./weights.bin'], weights: artifacts.weightSpecs }],
  };

  const zip = new window.JSZip();
  zip.file('model.json', JSON.stringify(modelJson));
  zip.file('weights.bin', artifacts.weightData);
  zip.file('metadata.json', JSON.stringify(model.getMetadata()));
  return zip.generateAsync({ type: 'blob' });
}
