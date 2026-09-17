// Structured haptic feedback with robust feature detection.

export const patterns = {
  tap: 8,
  good: 15,
  fast: [10, 30, 10],
  risky: 40,
  timeout: 80,
  ending: [30, 50, 30, 50]
};

export function haptic(pattern) {
  try {
    if (typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function') {
      navigator.vibrate(pattern);
    }
  } catch {
    /* Vibration API unsupported — silently ignore */
  }
}
