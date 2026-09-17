// DeviceMotion spike detection: a sudden acceleration (phone picked up,
// pocket movement) wakes the ambient "night out" mode. Feature-detected;
// on iOS the permission is requested from a user gesture by the caller.

let spikeCallbacks = new Set();
let listening = false;
let lastMagnitude = 0;
let lastSpikeAt = 0;
const DEBOUNCE_MS = 3000;
const SPIKE_THRESHOLD = 12;

export function isSupported() {
  return (
    typeof window !== 'undefined' && typeof window.DeviceMotionEvent === 'function'
  );
}

/** iOS 13+ requires an explicit permission request from a user gesture. */
export async function requestPermission() {
  try {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      return (await DeviceMotionEvent.requestPermission()) === 'granted';
    }
    return true;
  } catch {
    return false;
  }
}

export function start(onSpike) {
  if (!isSupported()) return false;
  spikeCallbacks.add(onSpike);
  if (!listening) {
    window.addEventListener('devicemotion', handler);
    listening = true;
  }
  return true;
}

export function stop(onSpike) {
  spikeCallbacks.delete(onSpike);
  if (!spikeCallbacks.size && listening) {
    window.removeEventListener('devicemotion', handler);
    listening = false;
  }
}

function handler(event) {
  const acc = event.accelerationIncludingGravity;
  if (!acc || acc.x == null) return;
  const magnitude = Math.sqrt((acc.x || 0) ** 2 + (acc.y || 0) ** 2 + (acc.z || 0) ** 2);
  const delta = Math.abs(magnitude - lastMagnitude);
  lastMagnitude = magnitude;
  const now = performance.now();
  if (delta > SPIKE_THRESHOLD && now - lastSpikeAt > DEBOUNCE_MS) {
    lastSpikeAt = now;
    spikeCallbacks.forEach((cb) => cb(delta));
  }
}
