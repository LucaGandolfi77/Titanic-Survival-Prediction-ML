/*
 * app.js
 * Main application controller: camera lifecycle, Three.js rendering, filter UI,
 * MediaPipe lazy loading, mobile gestures, capture/share flow, and app state.
 */
import * as THREE from 'three';
import { SHADER_FILTERS, createShaderMaterial, disposeMaterial } from './shaders.js';
import { AR_FILTERS, createARState, drawActiveARFilter } from './ar-filters.js';
import { ensureForFilter, detectForFilter, getCachedResults } from './mediapipe-init.js';

const FILTERS = [...SHADER_FILTERS, ...AR_FILTERS];
const appState = {
  selectedFilter: FILTERS[0],
  facingMode: 'user',
  zoom: 1,
  torchEnabled: false,
  stream: null,
  lastBlob: null,
  lastObjectUrl: null,
  previewing: false,
  webglSupported: true,
  renderer: null,
  scene: null,
  camera: null,
  mesh: null,
  material: null,
  videoTexture: null,
  arState: createARState(),
  detectionBusy: false,
  arUnavailable: false,
  gesture: { mode: null, startX: 0, startDistance: 0, startZoom: 1 },
  loopHandle: 0,
  lastFrameAt: performance.now(),
};

const els = {
  camera: document.getElementById('camera'),
  fallbackVideo: document.getElementById('fallback-video'),
  fallbackLayer: document.getElementById('fallback-video-layer'),
  renderStack: document.getElementById('render-stack'),
  glCanvas: document.getElementById('gl-canvas'),
  overlayCanvas: document.getElementById('overlay-canvas'),
  shaderCarousel: document.getElementById('shader-carousel'),
  arCarousel: document.getElementById('ar-carousel'),
  effectsStrip: document.getElementById('effects-strip'),
  effectsPrev: document.getElementById('effects-prev'),
  effectsNext: document.getElementById('effects-next'),
  loadingPill: document.getElementById('loading-pill'),
  statusBanner: document.getElementById('status-banner'),
  permissionScreen: document.getElementById('permission-screen'),
  httpsWarning: document.getElementById('https-warning'),
  retryCamera: document.getElementById('retry-camera'),
  flipCamera: document.getElementById('flip-camera'),
  shutter: document.getElementById('shutter'),
  shareLast: document.getElementById('share-last'),
  torchToggle: document.getElementById('torch-toggle'),
  zoomPill: document.getElementById('zoom-pill'),
  capturePreview: document.getElementById('capture-preview'),
  previewImage: document.getElementById('preview-image'),
  savePhoto: document.getElementById('save-photo'),
  discardPhoto: document.getElementById('discard-photo'),
};
const overlayCtx = els.overlayCanvas.getContext('2d');

boot().catch((error) => {
  console.error(error);
  showBanner('Failed to start app. Reload and try again.');
});

async function boot() {
  if (!isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
    els.httpsWarning.classList.remove('hidden');
  }

  buildFilterCarousels();
  bindUi();
  initThree();
  await startCamera();
  resize();
  window.addEventListener('resize', resize);
  startLoop();
}

function buildFilterCarousels() {
  // Populate a single combined, scrollable effects strip (shaders + AR)
  els.effectsStrip.innerHTML = FILTERS.map(renderChip).join('');
  // Attach click handlers to chips
  els.effectsStrip.querySelectorAll('.filter-chip').forEach((button) => {
    button.addEventListener('click', () => selectFilter(button.dataset.filterId));
  });
  // Prev/Next arrows
  if (els.effectsPrev) els.effectsPrev.addEventListener('click', () => {
    els.effectsStrip.scrollBy({ left: -240, behavior: 'smooth' });
  });
  if (els.effectsNext) els.effectsNext.addEventListener('click', () => {
    els.effectsStrip.scrollBy({ left: 240, behavior: 'smooth' });
  });
  updateActiveChip();
}

function renderChip(filter) {
  return `<button class="filter-chip${filter.id === appState.selectedFilter.id ? ' active' : ''}" data-filter-id="${filter.id}">
    <span>${filter.name}</span>
    <small>${filter.category.toUpperCase()}</small>
  </button>`;
}

function updateActiveChip() {
  // Toggle active class only within the effects strip
  if (!els.effectsStrip) return;
  els.effectsStrip.querySelectorAll('.filter-chip').forEach((chip) => {
    chip.classList.toggle('active', chip.dataset.filterId === appState.selectedFilter.id);
  });
  // Center the active chip in the strip for better visibility
  centerActiveChip();
}

function centerActiveChip() {
  const strip = els.effectsStrip;
  if (!strip) return;
  const active = strip.querySelector('.filter-chip.active');
  if (!active) return;
  const target = active.offsetLeft + (active.offsetWidth / 2) - (strip.clientWidth / 2);
  const maxLeft = Math.max(0, strip.scrollWidth - strip.clientWidth);
  let left = Math.max(0, target);
  left = Math.min(left, maxLeft);
  strip.scrollTo({ left, behavior: 'smooth' });
}

function bindUi() {
  els.retryCamera.addEventListener('click', () => startCamera());
  els.flipCamera.addEventListener('click', async () => {
    appState.facingMode = appState.facingMode === 'user' ? 'environment' : 'user';
    await startCamera();
    updateMirrorMode();
  });
  els.shutter.addEventListener('click', capturePhoto);
  els.savePhoto.addEventListener('click', saveOrSharePhoto);
  els.discardPhoto.addEventListener('click', discardPreview);
  els.shareLast.addEventListener('click', saveOrSharePhoto);
  els.torchToggle.addEventListener('click', toggleTorch);

  bindGestures();
}

function bindGestures() {
  els.renderStack.addEventListener('touchstart', (event) => {
    if (event.touches.length === 2) {
      appState.gesture.mode = 'pinch';
      appState.gesture.startDistance = touchDistance(event.touches[0], event.touches[1]);
      appState.gesture.startZoom = appState.zoom;
    } else if (event.touches.length === 1) {
      appState.gesture.mode = 'swipe';
      appState.gesture.startX = event.touches[0].clientX;
    }
  }, { passive: true });

  els.renderStack.addEventListener('touchmove', (event) => {
    if (appState.gesture.mode === 'pinch' && event.touches.length === 2) {
      const distance = touchDistance(event.touches[0], event.touches[1]);
      const nextZoom = clamp(appState.gesture.startZoom * (distance / appState.gesture.startDistance), 1, 3);
      setZoom(nextZoom);
    }
  }, { passive: true });

  els.renderStack.addEventListener('touchend', (event) => {
    if (appState.gesture.mode === 'swipe' && event.changedTouches.length === 1) {
      const delta = event.changedTouches[0].clientX - appState.gesture.startX;
      if (Math.abs(delta) > 40) cycleFilter(delta < 0 ? 1 : -1);
    }
    appState.gesture.mode = null;
  });
}

async function startCamera() {
  stopCamera();
  hidePermission();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: appState.facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    appState.stream = stream;
    els.camera.srcObject = stream;
    els.fallbackVideo.srcObject = stream;
    await Promise.all([els.camera.play(), els.fallbackVideo.play()]);
    updateMirrorMode();
    showBanner('');
  } catch (error) {
    console.error(error);
    els.permissionScreen.classList.remove('hidden');
    showBanner('Camera permission denied. Retry to continue.');
  }
}

function stopCamera() {
  if (!appState.stream) return;
  appState.stream.getTracks().forEach((track) => track.stop());
  appState.stream = null;
}

function hidePermission() {
  els.permissionScreen.classList.add('hidden');
}

function initThree() {
  try {
    appState.renderer = new THREE.WebGLRenderer({ canvas: els.glCanvas, alpha: false, antialias: true, preserveDrawingBuffer: true });
    appState.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    appState.scene = new THREE.Scene();
    appState.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    appState.videoTexture = new THREE.VideoTexture(els.camera);
    appState.videoTexture.colorSpace = THREE.SRGBColorSpace;
    appState.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), createShaderMaterial('normal', appState.videoTexture, { width: innerWidth, height: innerHeight }));
    appState.material = appState.mesh.material;
    appState.scene.add(appState.mesh);
  } catch (error) {
    console.warn('WebGL unavailable, using fallback video layer.', error);
    appState.webglSupported = false;
    els.fallbackLayer.classList.remove('hidden');
    showBanner('WebGL unavailable. Shader filters disabled.');
  }
}

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  els.overlayCanvas.width = w;
  els.overlayCanvas.height = h;
  if (appState.renderer) {
    appState.renderer.setSize(w, h, false);
    if (appState.material?.uniforms?.uResolution) appState.material.uniforms.uResolution.value.set(w, h);
  }
}

async function selectFilter(filterId) {
  const filter = FILTERS.find((item) => item.id === filterId);
  if (!filter || filter.id === appState.selectedFilter.id) return;

  if (filter.category === 'shader' && !appState.webglSupported) {
    showBanner('Shader filters are unavailable on this device.');
    return;
  }

  appState.selectedFilter = filter;
  updateActiveChip();

  if (filter.category === 'shader') {
    applyShaderFilter(filter.id);
    clearOverlay();
    return;
  }

  applyShaderFilter('normal');
  els.loadingPill.classList.remove('hidden');
  try {
    await ensureForFilter(filter);
    els.loadingPill.classList.add('hidden');
  } catch (error) {
    console.error(error);
    appState.arUnavailable = true;
    els.loadingPill.classList.add('hidden');
    showBanner('AR unavailable. Shader filters still work.');
  }
}

function applyShaderFilter(filterId) {
  if (!appState.webglSupported || !appState.mesh) return;
  disposeMaterial(appState.material);
  appState.material = createShaderMaterial(filterId, appState.videoTexture, { width: innerWidth, height: innerHeight });
  appState.mesh.material = appState.material;
  updateMirrorMode();
}

function updateMirrorMode() {
  const mirrored = appState.facingMode === 'user';
  globalThis.__AR_CAMERA_MIRRORED__ = mirrored;
  if (appState.material?.uniforms?.uMirror) appState.material.uniforms.uMirror.value = mirrored ? 1 : 0;
  els.fallbackVideo.style.transform = mirrored ? 'scaleX(-1)' : 'scaleX(1)';
}

function startLoop() {
  const loop = async (time) => {
    appState.loopHandle = requestAnimationFrame(loop);
    const dt = (time - appState.lastFrameAt) / 1000;
    appState.lastFrameAt = time;
    if (appState.previewing) return;
    if (!els.camera.videoWidth) return;

    if (appState.material?.uniforms?.uTime) appState.material.uniforms.uTime.value = time / 1000;
    if (appState.webglSupported) appState.renderer.render(appState.scene, appState.camera);

    clearOverlay();
    if (appState.selectedFilter.category === 'ar' && !appState.arUnavailable) {
      if (!appState.detectionBusy) {
        appState.detectionBusy = true;
        detectForFilter(els.camera, appState.selectedFilter, performance.now())
          .catch((error) => {
            console.error(error);
            appState.arUnavailable = true;
            showBanner('AR tracking failed.');
          })
          .finally(() => { appState.detectionBusy = false; });
      }
      const results = getCachedResults();
      const payload = appState.selectedFilter.tracker === 'face'
        ? results.face
        : appState.selectedFilter.tracker === 'hand'
          ? results.hands
          : results.pose;
      if (payload) drawActiveARFilter(overlayCtx, appState.selectedFilter.id, payload, appState.arState, dt);
    }
  };
  appState.loopHandle = requestAnimationFrame(loop);
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, els.overlayCanvas.width, els.overlayCanvas.height);
}

function cycleFilter(direction) {
  const currentIndex = FILTERS.findIndex((item) => item.id === appState.selectedFilter.id);
  const nextIndex = (currentIndex + direction + FILTERS.length) % FILTERS.length;
  selectFilter(FILTERS[nextIndex].id);
}

function setZoom(value) {
  appState.zoom = value;
  els.renderStack.style.transform = `scale(${value})`;
  els.zoomPill.textContent = `${value.toFixed(1)}x`;
}

async function toggleTorch() {
  const track = appState.stream?.getVideoTracks?.()[0];
  if (!track) return;
  const caps = track.getCapabilities?.();
  if (!caps?.torch) {
    showBanner('Torch not available on this camera.');
    return;
  }
  appState.torchEnabled = !appState.torchEnabled;
  await track.applyConstraints({ advanced: [{ torch: appState.torchEnabled }] });
}

async function capturePhoto() {
  if (appState.previewing) return;
  const canvas = document.createElement('canvas');
  canvas.width = els.glCanvas.width || innerWidth;
  canvas.height = els.glCanvas.height || innerHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(appState.webglSupported ? els.glCanvas : els.fallbackVideo, 0, 0, canvas.width, canvas.height);
  ctx.drawImage(els.overlayCanvas, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.75));
  if (!blob) return;

  if (appState.lastObjectUrl) URL.revokeObjectURL(appState.lastObjectUrl);
  appState.lastBlob = blob;
  appState.lastObjectUrl = URL.createObjectURL(blob);
  els.previewImage.src = appState.lastObjectUrl;
  els.capturePreview.classList.remove('hidden');
  appState.previewing = true;
}

function discardPreview() {
  els.capturePreview.classList.add('hidden');
  appState.previewing = false;
}

async function saveOrSharePhoto() {
  if (!appState.lastBlob) return;
  const file = new File([appState.lastBlob], 'snap.jpg', { type: 'image/jpeg' });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: 'AR Snap Camera' });
      discardPreview();
      return;
    } catch (error) {
      console.warn('Share cancelled or failed.', error);
    }
  }
  const link = document.createElement('a');
  link.href = appState.lastObjectUrl;
  link.download = 'snap.jpg';
  link.click();
  discardPreview();
}

function showBanner(message) {
  els.statusBanner.textContent = message;
  els.statusBanner.classList.toggle('hidden', !message);
}

function touchDistance(a, b) {
  return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
