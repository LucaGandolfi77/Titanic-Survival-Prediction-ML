/*
 * mediapipe-init.js
 * Lazy MediaPipe Tasks Vision loader with GPU-first initialization, CPU fallback,
 * result caching, and timestamp-gated video detection for face, hand, and pose.
 */
import {
  FilesetResolver,
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
} from 'mediapipe-tasks';

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODELS = {
  face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
};

const cache = {
  vision: null,
  face: null,
  hand: null,
  pose: null,
  lastDetectAt: 0,
  results: {
    face: null,
    hands: null,
    pose: null,
  },
};

async function getVision() {
  if (!cache.vision) cache.vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  return cache.vision;
}

async function createWithFallback(factory) {
  try {
    return await factory('GPU');
  } catch (error) {
    console.warn('GPU delegate unavailable, falling back to CPU.', error);
    return factory('CPU');
  }
}

export async function ensureFaceLandmarker() {
  if (cache.face) return cache.face;
  const vision = await getVision();
  cache.face = await createWithFallback((delegate) => FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODELS.face, delegate },
    runningMode: 'VIDEO',
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    numFaces: 1,
  }));
  return cache.face;
}

export async function ensureHandLandmarker() {
  if (cache.hand) return cache.hand;
  const vision = await getVision();
  cache.hand = await createWithFallback((delegate) => HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODELS.hand, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
  }));
  return cache.hand;
}

export async function ensurePoseLandmarker() {
  if (cache.pose) return cache.pose;
  const vision = await getVision();
  cache.pose = await createWithFallback((delegate) => PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODELS.pose, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
  }));
  return cache.pose;
}

export async function ensureForFilter(filter) {
  if (!filter || filter.category !== 'ar') return null;
  if (filter.tracker === 'face') return ensureFaceLandmarker();
  if (filter.tracker === 'hand') return ensureHandLandmarker();
  if (filter.tracker === 'pose') return ensurePoseLandmarker();
  return null;
}

export async function detectForFilter(video, filter, timestamp) {
  if (!filter || filter.category !== 'ar' || !video.videoWidth || !video.videoHeight) return cache.results;
  // Throttle detection to reduce CPU/GPU load. Previously 33ms (~30fps),
  // raising to 100ms reduces work to ~10fps which is sufficient for AR overlays
  // and avoids overwhelming the WASM/TFLite pipeline on mobile devices.
  if (timestamp - cache.lastDetectAt < 100) return cache.results;
  cache.lastDetectAt = timestamp;

  if (filter.tracker === 'face') {
    const landmarker = await ensureFaceLandmarker();
    cache.results.face = landmarker.detectForVideo(video, timestamp);
  } else if (filter.tracker === 'hand') {
    const landmarker = await ensureHandLandmarker();
    cache.results.hands = landmarker.detectForVideo(video, timestamp);
  } else if (filter.tracker === 'pose') {
    const landmarker = await ensurePoseLandmarker();
    cache.results.pose = landmarker.detectForVideo(video, timestamp);
  }

  return cache.results;
}

export function getCachedResults() {
  return cache.results;
}
