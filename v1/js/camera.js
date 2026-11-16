/**
 * One shared webcam for sample recording and the live preview.
 * Uses the Teachable Machine library's Webcam (224×224, mirrored), so samples
 * and predictions see the same square, flipped frame Teachable Machine uses.
 */
const SIZE = 224;

let webcam = null;
let starting = null;
let frame = 0;
const subscribers = new Set();

function tick() {
  if (!webcam) return;
  webcam.update();
  subscribers.forEach((callback) => callback(webcam.canvas));
  frame = requestAnimationFrame(tick);
}

async function start() {
  if (webcam) return webcam;
  if (!starting) {
    starting = (async () => {
      const camera = new window.tmImage.Webcam(SIZE, SIZE, true);
      try {
        await camera.setup();
      } catch (error) {
        // The library rejects with plain strings.
        throw new Error(typeof error === 'string' ? error : 'The webcam could not be started.');
      }
      camera.webcam.setAttribute('playsinline', '');
      camera.webcam.muted = true;
      camera.webcam.className = 'camera-source'; // kept in the page (hidden) so iOS keeps updating frames
      document.body.append(camera.webcam);
      await camera.play();
      webcam = camera;
      frame = requestAnimationFrame(tick);
      return camera;
    })();
    starting.finally(() => {
      starting = null;
    }).catch(() => {});
  }
  return starting;
}

function stop() {
  if (!webcam) return;
  cancelAnimationFrame(frame);
  webcam.stop();
  webcam.webcam.remove();
  webcam = null;
}

/**
 * Starts the webcam if needed and calls callback(canvas) on every frame.
 * Resolves to an unsubscribe function; the camera turns off when nobody is subscribed.
 */
export async function watchCamera(callback) {
  subscribers.add(callback);
  try {
    await start();
  } catch (error) {
    subscribers.delete(callback);
    throw error;
  }
  return () => {
    subscribers.delete(callback);
    if (subscribers.size === 0) stop();
  };
}

export const cameraSupported = Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
