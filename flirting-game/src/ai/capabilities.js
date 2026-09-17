// Browser capability detection for the AI + device-API layer.

export async function detectCapabilities() {
  const caps = {
    webnn: false, // Web Neural Network API (experimental)
    webgpu: false,
    wasm: false,
    speechSynthesis: false,
    speechRecognition: false,
    passkeys: false,
    backgroundSync: false
  };
  try {
    caps.webnn = typeof navigator !== 'undefined' && !!navigator.ml;
  } catch { /* noop */ }
  try {
    caps.webgpu = typeof navigator !== 'undefined' && !!navigator.gpu;
  } catch { /* noop */ }
  try {
    caps.wasm = typeof WebAssembly === 'object';
  } catch { /* noop */ }
  try {
    caps.speechSynthesis = typeof window !== 'undefined' && 'speechSynthesis' in window;
  } catch { /* noop */ }
  try {
    const SR =
      typeof window !== 'undefined'
        ? window.SpeechRecognition || window.webkitSpeechRecognition
        : null;
    caps.speechRecognition = typeof SR === 'function';
  } catch { /* noop */ }
  try {
    caps.passkeys =
      typeof window !== 'undefined' && typeof window.PublicKeyCredential === 'function';
  } catch { /* noop */ }
  try {
    caps.backgroundSync = typeof window !== 'undefined' && 'SyncManager' in window;
  } catch { /* noop */ }
  return caps;
}

/** Preferred inference device for local micro-models. */
export function devicePreference(caps) {
  if (caps?.webnn) return 'webnn'; // hardware-accelerated, native
  if (caps?.webgpu) return 'webgpu';
  return 'wasm';
}
