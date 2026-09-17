const GESTURE_COOLDOWN = 1000;

export function createGestureEngine(onGesture) {
  let hands = null;
  let camera = null;
  let video = null;
  let active = false;
  let prevHandCenter = null;
  let prevPinchDist = null;
  let lastGestureTime = 0;
  let swipeBuffer = [];
  let isMediaPipeReady = false;
  let loadingFailed = false;

  function checkMediaPipe() {
    isMediaPipeReady = !!(window.Hands && window.Camera);
    return isMediaPipeReady;
  }

  function isReady() {
    return checkMediaPipe();
  }

  function isLoadingFailed() {
    return loadingFailed;
  }

  function isPinch(landmarks) {
    const thumb = landmarks[4];
    const index = landmarks[8];
    return Math.sqrt(
      (thumb.x - index.x) ** 2 +
      (thumb.y - index.y) ** 2 +
      (thumb.z - index.z) ** 2
    ) < 0.06;
  }

  function pinchDistance(landmarks) {
    const thumb = landmarks[4];
    const index = landmarks[8];
    return Math.sqrt(
      (thumb.x - index.x) ** 2 +
      (thumb.y - index.y) ** 2 +
      (thumb.z - index.z) ** 2
    );
  }

  function handCenter(landmarks) {
    return {
      x: (landmarks[0].x + landmarks[9].x) / 2,
      y: (landmarks[0].y + landmarks[9].y) / 2,
    };
  }

  function fingerExtended(landmarks, fingerMcp, fingerTip) {
    return landmarks[fingerTip].y < landmarks[fingerMcp].y;
  }

  function countExtendedFingers(landmarks) {
    let count = 0;
    if (fingerExtended(landmarks, 2, 4)) count++;
    if (fingerExtended(landmarks, 5, 8)) count++;
    if (fingerExtended(landmarks, 9, 12)) count++;
    if (fingerExtended(landmarks, 13, 16)) count++;
    if (fingerExtended(landmarks, 17, 20)) count++;
    return count;
  }

  function isThumbsUp(landmarks) {
    return fingerExtended(landmarks, 2, 4) &&
      !fingerExtended(landmarks, 5, 8) &&
      !fingerExtended(landmarks, 9, 12) &&
      !fingerExtended(landmarks, 13, 16) &&
      !fingerExtended(landmarks, 17, 20);
  }

  function isThumbsDown(landmarks) {
    return !fingerExtended(landmarks, 2, 4) &&
      !fingerExtended(landmarks, 5, 8) &&
      !fingerExtended(landmarks, 9, 12) &&
      !fingerExtended(landmarks, 13, 16) &&
      !fingerExtended(landmarks, 17, 20) &&
      landmarks[4].y > landmarks[3].y;
  }

  function isPeace(landmarks) {
    return fingerExtended(landmarks, 5, 8) &&
      fingerExtended(landmarks, 9, 12) &&
      !fingerExtended(landmarks, 2, 4) &&
      !fingerExtended(landmarks, 13, 16) &&
      !fingerExtended(landmarks, 17, 20);
  }

  function isOpenPalm(landmarks) {
    return countExtendedFingers(landmarks) >= 4;
  }

  function isFist(landmarks) {
    return countExtendedFingers(landmarks) === 0;
  }

  function detectGesture(landmarks) {
    const now = Date.now();
    if (now - lastGestureTime < GESTURE_COOLDOWN) return null;

    if (isThumbsUp(landmarks)) {
      lastGestureTime = now;
      return 'play';
    }
    if (isThumbsDown(landmarks)) {
      lastGestureTime = now;
      return 'stop';
    }
    if (isPeace(landmarks)) {
      lastGestureTime = now;
      return 'cosmic';
    }
    if (isOpenPalm(landmarks)) {
      lastGestureTime = now;
      return 'reset';
    }
    if (isFist(landmarks)) {
      lastGestureTime = now;
      return 'fit';
    }

    if (isPinch(landmarks)) {
      const dist = pinchDistance(landmarks);
      if (prevPinchDist !== null) {
        const delta = prevPinchDist - dist;
        if (Math.abs(delta) > 0.005) {
          lastGestureTime = now;
          prevPinchDist = dist;
          return delta > 0 ? 'zoomIn' : 'zoomOut';
        }
      }
      prevPinchDist = dist;
    } else {
      prevPinchDist = null;
    }

    const center = handCenter(landmarks);
    if (prevHandCenter && center.x > 0 && center.y > 0) {
      const dx = center.x - prevHandCenter.x;
      if (Math.abs(dx) > 0.12) {
        swipeBuffer.push({ dx, time: Date.now() });
        if (swipeBuffer.length > 5) swipeBuffer.shift();

        if (swipeBuffer.length >= 2) {
          const recent = swipeBuffer.filter(s => Date.now() - s.time < 800);
          if (recent.length >= 2) {
            const totalDx = recent.reduce((sum, s) => sum + s.dx, 0);
            if (Math.abs(totalDx) > 0.25) {
              lastGestureTime = now;
              swipeBuffer = [];
              prevHandCenter = center;
              return totalDx > 0 ? 'nextModel' : 'undo';
            }
          }
        }
      }
    }
    prevHandCenter = center;

    return null;
  }

  function onResults(results) {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      prevHandCenter = null;
      prevPinchDist = null;
      onGesture('noHand', null);
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const gesture = detectGesture(landmarks);
    if (gesture) {
      onGesture(gesture, landmarks);
    }
  }

  async function startCamera() {
    if (!checkMediaPipe()) {
      loadingFailed = true;
      console.warn('MediaPipe Hands not loaded — check CDN');
      return false;
    }

    try {
      video = document.createElement('video');
      video.playsInline = true;
      video.muted = true;

      hands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5,
      });

      hands.onResults(onResults);

      camera = new window.Camera(video, {
        onFrame: async () => {
          if (hands && active) {
            try {
              await hands.send({ image: video });
            } catch (e) { /* frame skip */ }
          }
        },
        width: 640,
        height: 480,
      });

      await camera.start();
      active = true;
      return true;
    } catch (err) {
      console.error('Gesture engine failed:', err);
      stopCamera();
      return false;
    }
  }

  function stopCamera() {
    active = false;
    if (camera) {
      try { camera.stop(); } catch (e) { /* ok */ }
      camera = null;
    }
    if (hands) {
      try { hands.close(); } catch (e) { /* ok */ }
      hands = null;
    }
    video = null;
    prevHandCenter = null;
    prevPinchDist = null;
    swipeBuffer = [];
  }

  async function toggle() {
    if (active) {
      stopCamera();
      return false;
    }
    return await startCamera();
  }

  function isActive() {
    return active;
  }

  function getVideo() {
    return video;
  }

  return { toggle, isActive, getVideo, startCamera, stopCamera, checkMediaPipe, isReady, isLoadingFailed };
}
