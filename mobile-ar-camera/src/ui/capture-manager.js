import { capture as captureImpl } from '../media/capture.js';
import { share as shareImpl } from '../media/share.js';

export function createCaptureManager(appState, i18n, lifecycle) {
  const { set, get } = appState;

  lifecycle.on('shutdown', () => {
    cleanup();
  });

  async function capture(canvasSource, overlayCanvas) {
    try {
      const blob = await captureImpl(canvasSource, overlayCanvas);
      if (!blob) return null;

      cleanupURL();
      set('lastBlob', blob);
      const url = URL.createObjectURL(blob);
      set('lastObjectURL', url);
      set('previewing', true);

      if ('vibrate' in navigator) {
        navigator.vibrate(APP_CONFIG.SHUTTER_HAPTIC);
      }

      return { blob, url };
    } catch (error) {
      console.error('Capture failed:', error);
      lifecycle.emit('error', error);
      return null;
    }
  }

  function getPreviewData() {
    return {
      blob: get('lastBlob'),
      url: get('lastObjectURL'),
      previewing: get('previewing'),
    };
  }

  function discardPreview() {
    set('previewing', false);
  }

  async function saveOrShare() {
    if (!get('lastBlob')) return;

    const file = new File([get('lastBlob')], 'snap.jpg', { type: 'image/jpeg' });

    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      try {
        await shareImpl({ files: [file], title: 'AR Snap Camera' });
        discardPreview();
        return { shared: true };
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.warn('Share failed:', error);
        }
      }
    }

    const url = get('lastObjectURL');
    if (url) {
      const link = document.createElement('a');
      link.href = url;
      link.download = 'snap.jpg';
      link.click();
    }

    discardPreview();
    return { shared: false, downloaded: true };
  }

  function cleanupURL() {
    const url = get('lastObjectURL');
    if (url) {
      URL.revokeObjectURL(url);
      set('lastObjectURL', null);
    }
  }

  function cleanup() {
    cleanupURL();
    set('lastBlob', null);
  }

  return {
    capture,
    getPreviewData,
    discardPreview,
    saveOrShare,
    cleanup,
  };
}

const APP_CONFIG = {
  SHUTTER_HAPTIC: 15,
};
