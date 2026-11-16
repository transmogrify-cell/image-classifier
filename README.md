# Image classifier

Classifies images with a model trained in [Teachable Machine](https://teachablemachine.withgoogle.com/) (image projects only). Upload or drop a picture anywhere on the page, or use the webcam, to see the predicted class and the confidence for every class.

Animations switch off when the operating system's "reduce motion" setting is on.

**Live:** <https://transmogrify-cell.github.io/image-classifier/> (served by GitHub Pages from the `main` branch; every push redeploys it).

## Run it

The page loads model files with `fetch`, so serve the folder over HTTP rather than opening `index.html` directly:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Use your own model

1. Train an **Image Project** in Teachable Machine.
2. **Export Model → Tensorflow.js**, then either:
   - **Download**: unzip `model.json`, `metadata.json` and `weights.bin` into [`model/`](model/), or
   - **Upload (shareable link)**: copy the link (`https://teachablemachine.withgoogle.com/models/<id>/`) into `data-model-url` in [`index.html`](index.html).
3. Reload the page. For the live site, commit the files and push to `main`.

## Files

| File | Purpose |
| --- | --- |
| `js/image-classifier.js` | Loads and caches the model, runs predictions. No UI code. |
| `js/app.js` | Upload, webcam, preview and result display. |
| `css/image-classifier.css` | Styles, all scoped under `.ic`. |
| `model/` | Your exported model goes here. |

To add the feature to another page, copy the `#image-classifier` section and the three `<script>` tags from `index.html`, plus the `js/` and `css/` files.

## Credits

Uses the Teachable Machine image library and TensorFlow.js (both Apache-2.0) and the Unbounded and JetBrains Mono fonts (OFL). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Implemented with the assistance of an AI assistant.
