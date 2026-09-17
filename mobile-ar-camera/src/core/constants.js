export const FILTERS = [
  { id: 'normal', name: 'Normal', category: 'shader' },
  { id: 'vintage', name: 'Vintage', category: 'shader' },
  { id: 'neon', name: 'Neon', category: 'shader' },
  { id: 'glitch', name: 'Glitch', category: 'shader' },
  { id: 'pixelate', name: 'Pixelate', category: 'shader' },
  { id: 'sketch', name: 'Sketch', category: 'shader' },
  { id: 'blur', name: 'Blur', category: 'shader' },
  { id: 'warm-glow', name: 'Warm Glow', category: 'shader' },
  { id: 'cool-bleach', name: 'Cool Bleach', category: 'shader' },
  { id: 'film-grain', name: 'Film Grain', category: 'shader' },
];

export const AR_FILTERS = [
  { id: 'dog-face', name: 'Dog Face', category: 'ar', tracker: 'face' },
  { id: 'sunglasses', name: 'Sunglasses', category: 'ar', tracker: 'face' },
  { id: 'rainbow', name: 'Rainbow', category: 'ar', tracker: 'face' },
  { id: 'devil-horns', name: 'Devil Horns', category: 'ar', tracker: 'face' },
  { id: 'face-mesh', name: 'Face Mesh', category: 'ar', tracker: 'face' },
  { id: 'sparkle-crown', name: 'Sparkle Crown', category: 'ar', tracker: 'face' },
  { id: 'laser-eyes', name: 'Laser Eyes', category: 'ar', tracker: 'face' },
  { id: 'butterfly-wings', name: 'Butterfly Wings', category: 'ar', tracker: 'face' },
  { id: 'flower-halo', name: 'Flower Halo', category: 'ar', tracker: 'face' },
  { id: 'fire-hands', name: 'Fire Hands', category: 'ar', tracker: 'hand' },
  { id: 'magic-wand', name: 'Magic Wand', category: 'ar', tracker: 'hand' },
  { id: 'ghost-trail', name: 'Ghost Trail', category: 'ar', tracker: 'hand' },
  { id: 'fireworks', name: 'Fireworks', category: 'ar', tracker: 'hand' },
  { id: 'neon-trails', name: 'Neon Trails', category: 'ar', tracker: 'hand' },
  { id: 'hand-skeleton', name: 'Hand Skeleton', category: 'ar', tracker: 'hand' },
  { id: 'body-skeleton', name: 'Body Skeleton', category: 'ar', tracker: 'pose' },
  { id: 'aura-field', name: 'Aura Field', category: 'ar', tracker: 'pose' },
  { id: 'jetpack', name: 'Jetpack', category: 'ar', tracker: 'pose' },
];

export const ALL_FILTERS = [...FILTERS, ...AR_FILTERS];

export const MEDIAPIPE = {
  WASM_ROOT: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
  MODELS: {
    face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  },
  DETECT_THROTTLE_MS: 100,
};

export const APP_CONFIG = {
  CAMERA_WIDTH: 1280,
  CAMERA_HEIGHT: 720,
  MAX_ZOOM: 3,
  MIN_ZOOM: 1,
  SWIPE_THRESHOLD: 60,
  SHUTTER_HAPTIC: 15,
  FILTER_CHANGE_HAPTIC: [20, 30, 20],
  RESIZE_DEBOUNCE_MS: 100,
  UNLOAD_CLEANUP_DELAY_MS: 500,
};
