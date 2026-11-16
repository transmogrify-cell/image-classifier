# Image Trainer

Train your own image classifier in the browser, the way Teachable Machine does: add classes, collect image samples from the webcam or by uploading, train, test it live, and export it.

**Live:** <https://transmogrify-cell.github.io/image-classifier/> (served by GitHub Pages from the `main` branch; every push redeploys it).

## Versions

| Version | Where | Notes |
| --- | --- | --- |
| v2 (current) | `/` | Simplified three-step layout (Collect, Train, Test) with light and dark mode. |
| v1.0 | [`/v1/`](https://transmogrify-cell.github.io/image-classifier/v1/), git tag `v1.0` | The Teachable Machine-style layout, kept as a frozen copy in `v1/`. |

## Using it

1. **Collect:** name each class (click its name) and add more with **Add a class**. Give every class at least 2 images: **Camera** then **Hold to record**, or **Upload** (you can also drop images on a class).
2. **Train:** click **Train model**. Keep the tab open while it trains, because browsers pause background tabs. Epochs, batch size and learning rate are under **Settings**.
3. **Test:** try it with **Camera** or **Image**.
4. **Export model** downloads `my-image-model.zip` (`model.json`, `weights.bin`, `metadata.json`), the same format as a Teachable Machine image export.

Everything runs on your device; images are never uploaded.

## How the model works

It uses the Teachable Machine image library's own training class, `TeachableMobileNet`, unchanged:

```
image → centre crop, 224×224, pixels scaled to [-1, 1]
      → MobileNet V2 (alpha 0.35, pre-trained on ImageNet, frozen), cut at out_relu
      → global average pooling → 1,280 features
      → Dense(100, ReLU) → Dense(number of classes, softmax)   ← only these are trained
```

- **Feature extraction:** each sample goes through MobileNet once and only its 1,280 features are kept.
- **Training:** the two dense layers train with Adam (default learning rate 0.001, batch size 16, 50 epochs) and categorical cross-entropy. 15% of each class is held back to measure accuracy.
- **Confidence scores:** these are the softmax outputs, one per class, adding up to 100%.
- **MobileNet version:** the library's code defaults to MobileNet V1, but Teachable Machine's own exports use V2 with alpha 0.35. This page uses V2 so its exports load anywhere a Teachable Machine model does, including `tmImage.load()`.

## Run it locally

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Files

| File | Purpose |
| --- | --- |
| `js/image-classifier.js` | Training, prediction and export (wraps the Teachable Machine library). |
| `js/camera.js` | One shared webcam for recording samples and the preview. |
| `js/app.js` | Classes, samples, training controls, preview and export dialog. |
| `css/image-classifier.css` | Styles, with automatic light and dark mode. |
| `v1/` | Frozen copy of version 1.0. |

## Credits

Teachable Machine image library and TensorFlow.js (Apache-2.0), JSZip (MIT), Plus Jakarta Sans font (OFL); v1 also uses Material icons (Apache-2.0) and the Poppins and Roboto fonts (OFL). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project is not affiliated with Google.

Implemented with the assistance of an AI assistant.
