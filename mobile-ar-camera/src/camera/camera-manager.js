import { APP_CONFIG } from '../core/constants.js';

export function createCameraManager(state, lifecycle) {
  const { set, get } = state;

  async function start() {
    stop();
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: get('facingMode') },
        width: { ideal: APP_CONFIG.CAMERA_WIDTH },
        height: { ideal: APP_CONFIG.CAMERA_HEIGHT },
      },
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      set('stream', stream);
      set('lastFrameAt', performance.now());
      return stream;
    } catch (error) {
      console.error('Camera start failed:', error);
      set('stream', null);
      throw error;
    }
  }

  function stop() {
    const stream = get('stream');
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
    set('stream', null);
  }

  async function flip() {
    const current = get('facingMode');
    set('facingMode', current === 'user' ? 'environment' : 'user');
    await start();
    return get('facingMode');
  }

  async function toggleTorch() {
    const stream = get('stream');
    const track = stream?.getVideoTracks?.[0];
    if (!track) return false;
    const caps = track.getCapabilities?.();
    if (!caps?.torch) return false;

    const next = !get('torchEnabled');
    set('torchEnabled', next);
    await track.applyConstraints({ advanced: [{ torch: next }] });
    return next;
  }

  async function setZoom(value) {
    const clamped = Math.max(APP_CONFIG.MIN_ZOOM, Math.min(APP_CONFIG.MAX_ZOOM, value));
    set('zoom', clamped);
    return clamped;
  }

  function getSupportedFormats() {
    if (!navigator.mediaDevices?.getSupportedConstraints) return {};
    return {
      facingMode: navigator.mediaDevices.getSupportedConstraints().facingMode,
      torch: navigator.mediaDevices.getSupportedConstraints().torch,
      zoom: navigator.mediaDevices.getSupportedConstraints().zoom,
    };
  }

  lifecycle.on('shutdown', () => {
    stop();
  });

  return {
    start,
    stop,
    flip,
    toggleTorch,
    setZoom,
    getSupportedFormats,
  };
}
