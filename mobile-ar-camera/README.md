# Mobile AR Camera

A mobile-first Snapchat-style web camera built with:

- HTML/CSS/Vanilla JS
- Three.js shader filters
- MediaPipe Tasks Vision face/hand/pose tracking
- Canvas 2D AR overlays

## Files

- `index.html` — app shell, import map, stacked render layers
- `style.css` — mobile layout and controls
- `app.js` — camera, rendering loop, capture/share, gestures
- `shaders.js` — full-screen shader filters
- `ar-filters.js` — AR overlay drawing and particles
- `mediapipe-init.js` — lazy MediaPipe model setup and detection cache

## Run locally

Because camera access requires a secure context, use `localhost`:

```bash
cd "/Users/lgandolfi/Desktop/AI/fcc-ML/Titanic-Survival-Prediction-ML/mobile-ar-camera"
python3 -m http.server 8080
```

Then open:

- `http://localhost:8080`

## Notes

- On iPhone Safari, add the page to the home screen for a more app-like experience.
- Torch support depends on device and selected camera.
- AR models are loaded lazily when an AR filter is first selected.
