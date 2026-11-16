# Image Trainer

Train your own image classifier in the browser, the way Teachable Machine does: add classes, collect image samples from the webcam or by uploading, train, test it live, and export it.

**Live:** <https://transmogrify-cell.github.io/image-classifier/> (served by GitHub Pages from the `main` branch; every push redeploys it).

## Using it

1. Name your classes (click the pencil) and add more with **Add a class**.
2. Add at least 2 image samples to every class: **Webcam** then **Hold to Record**, or **Upload** (you can also drop images on a class).
3. Click **Train Model**. Keep the tab open while it trains, because browsers pause background tabs.
4. Test it in **Preview** with the webcam or an image file.
5. **Export Model** downloads `my-image-model.zip` (`model.json`, `weights.bin`, `metadata.json`), the same format as a Teachable Machine image export.

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
| `css/image-classifier.css` | Styles. |

## Credits

Teachable Machine image library and TensorFlow.js (Apache-2.0), JSZip (MIT), Material icons (Apache-2.0), Poppins and Roboto fonts (OFL). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project is not affiliated with Google.

Implemented with the assistance of an AI assistant.
