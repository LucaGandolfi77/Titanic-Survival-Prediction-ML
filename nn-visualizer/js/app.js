import { createInitialState, clampScale } from './state.js';
import { renderModel, applyTransform, focusNode } from './renderer.js';
import { setupPanZoom, setupTouchGestures } from './interaction.js';
import { createSampleModel, initModelWeights, buildNodeDetailHtml } from './model-loader.js';
import { renderSidebarList, updateNodeDetail } from './ui/sidebar.js';
import { createAudioEngine } from './audio.js';
import { vibrate } from './haptics.js';
import { shareModel } from './share.js';
import { createTrainingSimulator } from './training.js';
import { createUndoSystem } from './undo.js';
import { createVoiceEngine } from './voice.js';
import { createGestureEngine } from './gestures.js';
import { createCosmicRenderer } from './cosmic.js';
import { createRainbowFilter } from './filters/rainbow.js';
import { createStoryFilter } from './filters/story.js';
import { createMusicMinigame } from './filters/music.js';

const THEME_KEY = 'nnv-theme';
const MUSIC_KEY = 'nnv-music';

const model = createSampleModel();
initModelWeights(model);

const state = createInitialState(model);
const undo = createUndoSystem();
const audio = createAudioEngine();
const training = createTrainingSimulator(model);
let voiceEngine = null;
let rainbowFilter = null;
let storyFilter = null;
let musicGame = null;

const svgRoot = document.getElementById('svgroot');
const nodesList = document.getElementById('nodes-list');
const nodeDetail = document.getElementById('node-detail');
const resetBtn = document.getElementById('reset');
const fitBtn = document.getElementById('fit');
const modelsSelect = document.getElementById('models');
const themeToggle = document.getElementById('theme-toggle');
const cosmicToggle = document.getElementById('cosmic-toggle');
const musicToggle = document.getElementById('music-toggle');
const voiceToggle = document.getElementById('voice-toggle');
const shareBtn = document.getElementById('share-btn');
const trainingToggle = document.getElementById('training-toggle');
const undoBtn = document.getElementById('undo-btn');
const redoBtn = document.getElementById('redo-btn');
const trainingStatus = document.getElementById('training-status');
const voiceStatus = document.getElementById('voice-status');
const voiceTranscript = document.getElementById('voice-transcript');
const voiceToolbar = document.getElementById('voice-toolbar');
const langToggle = document.getElementById('lang-toggle');
const speakToggle = document.getElementById('speak-toggle');
const gestureToggle = document.getElementById('gesture-toggle');
const gestureOverlay = document.getElementById('gesture-overlay');
const gestureLabel = document.getElementById('gesture-label');
const gestureBadge = document.getElementById('gesture-badge');
const cosmicContainer = document.getElementById('cosmic-container');
const cosmicLoading = document.getElementById('cosmic-loading');
const rainbowToggle = document.getElementById('rainbow-toggle');
const storyToggle = document.getElementById('story-toggle');
const musicGameToggle = document.getElementById('music-game-toggle');

let panZoomInitialized = false;
let gesturesInitialized = false;
let musicEnabled = false;
let themeLight = false;
let cosmicMode = false;
let isTrainingRender = false;
let gestureActive = false;
let gestureEngine = null;
let voiceSpeaking = true;
let currentLang = 'en';
let confidenceDisplay = 0;
let cosmicRenderer = null;
let cosmicInitialized = false;

rainbowFilter = createRainbowFilter(svgRoot, state);
storyFilter = createStoryFilter(svgRoot, state);
musicGame = createMusicMinigame(svgRoot, state, audio);

loadTheme();
loadMusicPref();
loadCosmicPref();
initGestures();
initVoice();
initCosmic();
renderApp();
function initCosmic() {
  if (!window.THREE) {
    cosmicContainer.style.display = 'none';
    return;
  }
  cosmicRenderer = createCosmicRenderer(cosmicContainer);
  if (cosmicRenderer) {
    cosmicRenderer.init(window.THREE, window.OrbitControls);
    cosmicInitialized = true;
  }
}
function initVoice() {
  voiceEngine = createVoiceEngine(
    (action, transcript, confidence) => {
      handleVoiceCommand(action, transcript, confidence);
      if (voiceSpeaking && action !== 'unknown') {
        voiceEngine.speakAction(action);
      }
    },
    (transcript, confidence) => {
      if (voiceTranscript) {
        voiceTranscript.textContent = `"${transcript}" — ${Math.round(confidence * 100)}%`;
        voiceTranscript.classList.add('active');
        confidenceDisplay = confidence;
        clearTimeout(voiceTranscript._timeout);
        voiceTranscript._timeout = setTimeout(() => {
          voiceTranscript.classList.remove('active');
        }, 2000);
      }
    }
  );
}

function saveSnapshot() {
  if (isTrainingRender) return;
  undo.push({
    tx: state.tx,
    ty: state.ty,
    scale: state.scale,
    cosmic: cosmicMode,
    model: JSON.parse(JSON.stringify(state.model)),
  });
}

function renderApp() {
  if (cosmicMode) {
    startCosmicMode();
    return;
  }
  isTrainingRender = training.isRunning();
  saveSnapshot();
  const positions = renderModel(svgRoot, state, (layer, idx) => {
    showNodeDetail(layer, idx);
    focusNode(svgRoot, state, positions[layer][idx].x, positions[layer][idx].y);
    vibrate('click');
    updateSidebarSelection(layer, idx);
    if (musicEnabled) audio.playNote(layer, idx, 0.5);
  });

  renderSidebarList(nodesList, state.model, positions, (layer, idx) => {
    showNodeDetail(layer, idx);
    focusNode(svgRoot, state, positions[layer][idx].x, positions[layer][idx].y);
    vibrate('click');
    updateSidebarSelection(layer, idx);
    if (musicEnabled) audio.playNote(layer, idx, 0.5);
  });

  if (!panZoomInitialized) {
    setupPanZoom(svgRoot, state, () => applyTransform(svgRoot, state));
    panZoomInitialized = true;
  }
  if (!gesturesInitialized) {
    setupTouchGestures(svgRoot, state, () => applyTransform(svgRoot, state));
    gesturesInitialized = true;
  }

  undoBtn.disabled = !undo.canUndo();
  redoBtn.disabled = !undo.canRedo();
}

function updateSidebarSelection(layer, idx) {
  const items = nodesList.querySelectorAll('.node-item');
  items.forEach((item) => item.classList.remove('selected'));
  let index = 0;
  for (let i = 0; i < state.model.layers.length; i++) {
    for (let n = 0; n < state.model.layers[i].size; n++) {
      if (i === layer && n === idx && items[index]) {
        items[index].classList.add('selected');
        items[index].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      index++;
    }
  }
}

function showNodeDetail(layer, idx) {
  const html = buildNodeDetailHtml(state.model, layer, idx);
  updateNodeDetail(nodeDetail, html);
  state.selected = { layer, idx };
}

function loadTheme() {
  themeLight = localStorage.getItem(THEME_KEY) === 'light';
  if (themeLight) document.body.classList.add('light');
}

function loadMusicPref() {
  musicEnabled = localStorage.getItem(MUSIC_KEY) === 'true';
  updateMusicButton();
}

function loadCosmicPref() {
  cosmicMode = localStorage.getItem('nnv-cosmic') === 'true';
  cosmicToggle.classList.toggle('toolbar-btn-active', cosmicMode);
}

function initGestures() {
  gestureEngine = createGestureEngine(handleGestureCommand);
  gestureEngine.checkMediaPipe();
}

function updateMusicButton() {
  musicToggle.textContent = musicEnabled ? '🔊' : '🔇';
}

function updateTrainingButton() {
  trainingToggle.classList.toggle('toolbar-btn-active', training.isRunning());
  trainingStatus.style.display = training.isRunning() ? 'block' : 'none';
  if (training.isRunning()) {
    trainingStatus.textContent = `Epoch: ${training.getEpoch()} | Loss: ${training.getLoss().toFixed(4)}`;
  }
}

function startTraining() {
  training.start();
  updateTrainingButton();
  vibrate('success');
  animationLoop();
}

function stopTraining() {
  training.stop();
  updateTrainingButton();
  isTrainingRender = false;
}

let animFrameId = null;
function animationLoop() {
  if (!training.isRunning()) return;
  if (cosmicMode) return;
  const result = training.step();
  if (result) {
    trainingStatus.textContent = `Epoch: ${result.epoch} | Loss: ${result.loss.toFixed(4)}`;
    if (musicEnabled && result.epoch % 3 === 0) {
      audio.playChord(state.model.layers.flatMap((_, i) =>
        Array.from({ length: Math.min(state.model.layers[i].size, 3) }, (_, j) => 0.3 + Math.random() * 0.5)
      ));
    }
    renderApp();
  }
  animFrameId = requestAnimationFrame(() => {
    if (training.isRunning()) animationLoop();
  });
}

resetBtn.addEventListener('click', () => {
  saveSnapshot();
  state.scale = 1;
  state.tx = 0;
  state.ty = 0;
  applyTransform(svgRoot, state);
  vibrate('click');
});

fitBtn.addEventListener('click', () => {
  saveSnapshot();
  state.scale = 1;
  state.tx = 0;
  state.ty = 0;
  applyTransform(svgRoot, state);
  vibrate('click');
});

themeToggle.addEventListener('click', () => {
  saveSnapshot();
  themeLight = !themeLight;
  document.body.classList.toggle('light', themeLight);
  localStorage.setItem(THEME_KEY, themeLight ? 'light' : 'dark');
  vibrate('click');
});

musicToggle.addEventListener('click', () => {
  musicEnabled = audio.toggle();
  updateMusicButton();
  localStorage.setItem(MUSIC_KEY, musicEnabled ? 'true' : 'false');
  vibrate('click');
});

cosmicToggle.addEventListener('click', () => {
  saveSnapshot();
  cosmicMode = !cosmicMode;
  cosmicToggle.classList.toggle('toolbar-btn-active', cosmicMode);
  localStorage.setItem('nnv-cosmic', cosmicMode ? 'true' : 'false');
  vibrate('click');
  if (cosmicMode) {
    startCosmicMode();
  } else {
    stopCosmicMode();
  }
});

rainbowToggle.addEventListener('click', () => {
  rainbowFilter.toggle();
  vibrate('click');
});

storyToggle.addEventListener('click', () => {
  storyFilter.toggle();
  vibrate('click');
});

musicGameToggle.addEventListener('click', () => {
  musicGame.toggle();
  vibrate('click');
});

function startCosmicMode() {
  svgRoot.style.display = 'none';
  cosmicContainer.style.display = 'block';
  if (cosmicLoading) cosmicLoading.style.display = 'block';

  if (cosmicRenderer && cosmicInitialized) {
    const positions = computeCosmicPositions(state.model);
    cosmicRenderer.start(state.model, positions, (layer, idx) => {
      showNodeDetail(layer, idx);
      vibrate('click');
    });
    if (cosmicLoading) cosmicLoading.style.display = 'none';
  } else {
    setTimeout(() => {
      if (cosmicMode && cosmicRenderer && cosmicInitialized) {
        const positions = computeCosmicPositions(state.model);
        cosmicRenderer.start(state.model, positions, (layer, idx) => {
          showNodeDetail(layer, idx);
          vibrate('click');
        });
      }
      if (cosmicLoading) cosmicLoading.style.display = 'none';
    }, 2000);
  }
}

function stopCosmicMode() {
  svgRoot.style.display = 'block';
  cosmicContainer.style.display = 'none';
  if (cosmicRenderer) {
    cosmicRenderer.stop();
  }
}

function computeCosmicPositions(model) {
  const positions = [];
  const width = 12;
  const height = 8;
  const depth = 4;
  const layers = model.layers.length;
  for (let i = 0; i < layers; i++) {
    const size = model.layers[i].size;
    positions[i] = [];
    const layerX = (i / Math.max(layers - 1, 1) - 0.5) * width;
    for (let n = 0; n < size; n++) {
      const layerY = ((n / Math.max(size, 1)) - 0.5) * height;
      const jitter = (Math.random() - 0.5) * depth;
      positions[i].push({
        x: layerX,
        y: layerY,
        z: jitter,
      });
    }
  }
  return positions;
}

voiceToggle.addEventListener('click', () => {
  if (!voiceEngine) {
    vibrate('error');
    return;
  }
  const listening = voiceEngine.toggle();
  voiceToggle.classList.toggle('toolbar-btn-active', listening);
  voiceStatus.style.display = listening ? 'block' : 'none';
  if (voiceToolbar) voiceToolbar.style.display = listening ? 'flex' : 'none';
  vibrate('click');
});

langToggle.addEventListener('click', () => {
  currentLang = currentLang === 'en' ? 'it' : 'en';
  langToggle.textContent = currentLang === 'en' ? '🇬🇧' : '🇮🇹';
  langToggle.classList.toggle('lang-it', currentLang === 'it');
  langToggle.classList.toggle('lang-en', currentLang === 'en');
  if (voiceEngine) {
    voiceEngine.setLanguage(currentLang === 'it' ? 'it-IT' : 'en-US');
  }
  vibrate('click');
  if (voiceEngine && voiceEngine.isListening()) {
    voiceEngine.stop();
    setTimeout(() => {
      if (voiceEngine && !voiceEngine.isListening()) {
        voiceEngine.start();
      }
    }, 300);
  }
});

speakToggle.addEventListener('click', () => {
  voiceSpeaking = !voiceSpeaking;
  speakToggle.textContent = voiceSpeaking ? '🔊' : '🔇';
  speakToggle.classList.toggle('toolbar-btn-active', voiceSpeaking);
  vibrate('click');
});

gestureToggle.addEventListener('click', async () => {
  if (!gestureEngine) {
    vibrate('error');
    return;
  }
  const started = await gestureEngine.toggle();
  gestureActive = started;
  gestureToggle.classList.toggle('toolbar-btn-active', started);
  gestureOverlay.classList.toggle('no-active', !started);
  if (started) {
    const video = gestureEngine.getVideo();
    if (video && gestureOverlay) {
      const label = gestureOverlay.querySelector('.gesture-label');
      gestureOverlay.insertBefore(video, label);
      label.textContent = '🖐️ Watching...';
    }
    vibrate('success');
  } else {
    const video = gestureOverlay.querySelector('video');
    if (video) video.remove();
    const label = gestureOverlay.querySelector('.gesture-label');
    if (label) label.textContent = '🖐️ Gesture';
    gestureActive = false;
    gestureToggle.classList.remove('toolbar-btn-active');
    gestureOverlay.classList.add('no-active');
  }
  updateGestureBadge();
});

function updateGestureBadge() {
  if (gestureActive) {
    gestureBadge.textContent = '✋ Gestures ON';
    gestureBadge.classList.add('visible');
  } else {
    gestureBadge.classList.remove('visible');
  }
}

function handleGestureCommand(action) {
  if (!gestureActive) return;
  vibrate('click');
  switch (action) {
    case 'noHand':
      if (gestureLabel) gestureLabel.textContent = '🖐️ No hand';
      break;
    case 'reset':
      saveSnapshot();
      state.scale = 1; state.tx = 0; state.ty = 0;
      applyTransform(svgRoot, state);
      if (gestureLabel) gestureLabel.textContent = '✨ Reset';
      break;
    case 'zoomIn':
      saveSnapshot();
      state.scale = clampScale(state.scale * 1.3);
      applyTransform(svgRoot, state);
      if (gestureLabel) gestureLabel.textContent = '🔍 Zoom In';
      break;
    case 'zoomOut':
      saveSnapshot();
      state.scale = clampScale(state.scale / 1.3);
      applyTransform(svgRoot, state);
      if (gestureLabel) gestureLabel.textContent = '🔎 Zoom Out';
      break;
    case 'fit':
      saveSnapshot();
      state.scale = 1; state.tx = 0; state.ty = 0;
      applyTransform(svgRoot, state);
      if (gestureLabel) gestureLabel.textContent = '⊡ Fit';
      break;
    case 'play':
      if (!training.isRunning()) startTraining();
      if (gestureLabel) gestureLabel.textContent = '👍 Play';
      break;
    case 'stop':
      if (training.isRunning()) {
        stopTraining();
        if (animFrameId) cancelAnimationFrame(animFrameId);
      }
      if (gestureLabel) gestureLabel.textContent = '👎 Stop';
      break;
    case 'cosmic':
      cosmicToggle.click();
      if (gestureLabel) gestureLabel.textContent = '🌌 Cosmic';
      break;
    case 'nextModel':
      const sel = modelsSelect;
      sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
      sel.dispatchEvent(new Event('change'));
      if (gestureLabel) gestureLabel.textContent = '🔄 Next';
      break;
    case 'undo':
      undoBtn.click();
      if (gestureLabel) gestureLabel.textContent = '↩ Undo';
      break;
  }
  setTimeout(() => {
    if (gestureActive && gestureLabel) {
      gestureLabel.textContent = '🖐️ Watching...';
    }
  }, 1200);
}

function handleVoiceCommand(action, transcript, confidence) {
  vibrate('click');
  updateVoiceStatus(`🎤 "${transcript}"`, 3000);
  switch (action) {
    case 'reset':
      saveSnapshot();
      state.scale = 1; state.tx = 0; state.ty = 0;
      applyTransform(svgRoot, state);
      break;
    case 'zoom in':
    case 'zoom in more':
      saveSnapshot();
      state.scale = clampScale(state.scale * 1.3);
      applyTransform(svgRoot, state);
      break;
    case 'zoom out':
    case 'zoom out more':
      saveSnapshot();
      state.scale = clampScale(state.scale / 1.3);
      applyTransform(svgRoot, state);
      break;
    case 'fit':
      saveSnapshot();
      state.scale = 1; state.tx = 0; state.ty = 0;
      applyTransform(svgRoot, state);
      break;
    case 'play':
      if (!training.isRunning()) startTraining();
      break;
    case 'stop':
      if (training.isRunning()) {
        stopTraining();
        if (animFrameId) cancelAnimationFrame(animFrameId);
      }
      break;
    case 'cosmic':
      cosmicToggle.click();
      return;
    case 'rainbow':
      rainbowFilter.toggle();
      return;
    case 'story':
      storyFilter.toggle();
      return;
    case 'music game':
      musicGame.toggle();
      return;
    case 'music on':
      musicEnabled = true;
      audio.init();
      updateMusicButton();
      localStorage.setItem(MUSIC_KEY, 'true');
      break;
    case 'music off':
      musicEnabled = false;
      updateMusicButton();
      localStorage.setItem(MUSIC_KEY, 'false');
      break;
    case 'next model':
      const sel = modelsSelect;
      sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
      sel.dispatchEvent(new Event('change'));
      break;
    case 'undo':
      undoBtn.click();
      break;
    case 'redo':
      redoBtn.click();
      break;
    case 'unknown':
      if (confidence < 0.5) return;
      console.log('Voice unknown:', transcript);
      vibrate('error');
      break;
  }
  if (confidence > 0.7) updateVoiceStatus(`✅ ${action}`, 2000);
}

function updateVoiceStatus(text, duration = 3000) {
  if (!voiceStatus) return;
  voiceStatus.textContent = text;
  voiceStatus.style.display = 'block';
  clearTimeout(voiceStatus._timeout);
  voiceStatus._timeout = setTimeout(() => {
    if (voiceEngine && voiceEngine.isListening()) return;
    voiceStatus.style.display = 'none';
  }, duration);
}

shareBtn.addEventListener('click', async () => {
  saveSnapshot();
  if (ok) {
    vibrate('success');
    if (musicEnabled) audio.playChord([0.3, 0.5, 0.7]);
  } else {
    vibrate('error');
  }
});

trainingToggle.addEventListener('click', () => {
  saveSnapshot();
  if (training.isRunning()) {
    stopTraining();
    if (animFrameId) cancelAnimationFrame(animFrameId);
  } else {
    startTraining();
  }
  vibrate('click');
});

undoBtn.addEventListener('click', () => {
  const prev = undo.undo({
    tx: state.tx,
    ty: state.ty,
    scale: state.scale,
    model: JSON.parse(JSON.stringify(state.model)),
  });
  if (prev) {
    state.tx = prev.tx;
    state.ty = prev.ty;
    state.scale = prev.scale;
    cosmicMode = prev.cosmic ?? cosmicMode;
    cosmicToggle.classList.toggle('toolbar-btn-active', cosmicMode);
    if (prev.model) {
      state.model = prev.model;
      while (svgRoot.firstChild) svgRoot.removeChild(svgRoot.firstChild);
      nodesList.innerHTML = '';
      renderApp();
    }
    applyTransform(svgRoot, state);
    vibrate('click');
  }
});

redoBtn.addEventListener('click', () => {
  const next = undo.redo({
    tx: state.tx,
    ty: state.ty,
    scale: state.scale,
    cosmic: cosmicMode,
    model: JSON.parse(JSON.stringify(state.model)),
  });
  if (next) {
    state.tx = next.tx;
    state.ty = next.ty;
    state.scale = next.scale;
    cosmicMode = next.cosmic ?? cosmicMode;
    cosmicToggle.classList.toggle('toolbar-btn-active', cosmicMode);
    if (next.model) {
      state.model = next.model;
      while (svgRoot.firstChild) svgRoot.removeChild(svgRoot.firstChild);
      nodesList.innerHTML = '';
      renderApp();
    }
    applyTransform(svgRoot, state);
    vibrate('click');
  }
});

modelsSelect.addEventListener('change', async (e) => {
  vibrate('click');
  const value = e.target.value;
  if (value === 'sample') {
    state.model = createSampleModel();
    initModelWeights(state.model);
  }
  state.scale = 1;
  state.tx = 0;
  state.ty = 0;
  state.selected = null;
  training.stop();
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (voiceEngine) voiceEngine.stop();
  voiceToggle.classList.remove('toolbar-btn-active');
  voiceStatus.style.display = 'none';
  if (voiceToolbar) voiceToolbar.style.display = 'none';
  if (gestureActive) {
    gestureEngine.stopCamera();
    gestureActive = false;
    gestureToggle.classList.remove('toolbar-btn-active');
    gestureOverlay.classList.add('no-active');
    gestureOverlay.innerHTML = '<div class="gesture-label" id="gesture-label">🖐️ Gesture</div>';
    updateGestureBadge();
  }
  if (cosmicMode) stopCosmicMode();
  if (musicGame && musicGame.isActive()) musicGame.destroy();
  updateTrainingButton();
  undo.clear();
  while (svgRoot.firstChild) svgRoot.removeChild(svgRoot.firstChild);
  nodesList.innerHTML = '';
  renderApp();
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch((err) => {
    console.error('SW registration failed:', err);
  });
}

let deferredPrompt = null;
const installBtn = document.getElementById('pwa-install');

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  installBtn.classList.add('visible');
});

installBtn.addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  const result = await deferredPrompt.userChoice;
  if (result.outcome === 'accepted') {
    vibrate('success');
  }
  deferredPrompt = null;
  installBtn.classList.remove('visible');
});
