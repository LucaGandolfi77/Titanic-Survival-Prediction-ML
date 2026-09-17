export const HAPTIC_PATTERNS = {
  click: [15],
  success: [20, 30, 20],
  error: [100, 30, 100],
  discovery: [10, 20, 10, 20, 50],
  swipe: [5],
  longPress: [50, 30, 50],
};

export function vibrate(pattern) {
  if (navigator.vibrate) {
    const p = Array.isArray(pattern) ? pattern : HAPTIC_PATTERNS.click;
    navigator.vibrate(p);
  }
}
