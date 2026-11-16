# Image Classifier

Find out what's in a photo. Choose, drop or paste an image, or take one with the camera, and the page names it with a confidence score, plus the next four likely matches.

Recognition runs entirely in the browser with **ConvNeXt V2 Base** (Meta AI), which scores about 87% top-1 on ImageNet. Photos never leave the device. The model knows the 1,000 ImageNet categories (animals, food, vehicles, everyday objects); it doesn't identify people.

**Live:** <https://transmogrify-cell.github.io/image-classifier/> (served by GitHub Pages from the `main` branch; every push redeploys it).

## Run it locally

The page uses JavaScript modules, so serve the folder over HTTP rather than opening `index.html` directly:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## How it works

1. On first visit the page downloads the 8-bit model (about 91 MB) from the Hugging Face Hub. The browser caches it, so later visits load it in a few seconds.
2. A photo is drawn to a canvas (scaled to at most 1,024 px) and passed to the model.
3. Transformers.js applies the model's own preprocessing (resize, centre crop to 224 px, ImageNet normalisation), runs it on the CPU with WebAssembly, and returns the top five ImageNet labels with scores.

## Files

| File | Purpose |
| --- | --- |
| `js/image-classifier.js` | Loads the model once and classifies images. No UI code. |
| `js/app.js` | Photo picker, drag and drop, paste, camera and results. |
| `css/image-classifier.css` | Styles, with automatic light and dark mode. |

To use a different Transformers.js image-classification model, change `MODEL.id` in `js/image-classifier.js`.

## Credits

ConvNeXt V2 (Meta AI) and Transformers.js (Hugging Face), both Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Implemented with the assistance of an AI assistant.
