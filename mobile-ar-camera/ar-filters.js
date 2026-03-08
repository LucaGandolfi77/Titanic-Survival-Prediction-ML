/*
 * ar-filters.js
 * MediaPipe-driven 2D AR overlay filters, drawing helpers, gesture logic,
 * particle systems, and overlay filter metadata.
 */

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
  { id: 'gravity-bubble', name: 'Gravity Bubble', category: 'ar', tracker: 'pose' },
  { id: 'jetpack', name: 'Jetpack', category: 'ar', tracker: 'pose' },
];

export const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

export const POSE_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
  [9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],
  [12,14],[14,16],[16,18],[16,20],[16,22],[11,23],[12,24],
  [23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]
];

// Reduced but visually dense mesh lines for 478 face points.
export const FACE_TESSELATION = Array.from({ length: 477 }, (_, i) => [i, i + 1]);

export function projectLandmark(lm, canvasW, canvasH, mirrored = globalThis.__AR_CAMERA_MIRRORED__ ?? true) {
  const x = mirrored ? (1 - lm.x) * canvasW : lm.x * canvasW;
  const y = lm.y * canvasH;
  return { x, y, z: lm.z ?? 0 };
}

export function getFaceAngle(matrixLike) {
  const matrix = matrixLike?.data || matrixLike || [];
  if (!matrix.length) return { roll: 0, pitch: 0, yaw: 0 };
  const m = matrix;
  const sy = Math.sqrt(m[0] * m[0] + m[1] * m[1]);
  const singular = sy < 1e-6;
  const pitch = Math.atan2(m[6], m[10]);
  const yaw = Math.atan2(-m[2], sy);
  const roll = singular ? Math.atan2(-m[4], m[5]) : Math.atan2(m[1], m[0]);
  return { roll, pitch, yaw };
}

export function getFaceBoundingBox(landmarks, width, height, mirrored = true) {
  const points = landmarks.map((lm) => projectLandmark(lm, width, height, mirrored));
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

export function isFingerExtended(handLandmarks, fingerIndex) {
  const chains = {
    thumb: [1, 2, 3, 4],
    index: [5, 6, 7, 8],
    middle: [9, 10, 11, 12],
    ring: [13, 14, 15, 16],
    pinky: [17, 18, 19, 20],
  };
  const key = ['thumb', 'index', 'middle', 'ring', 'pinky'][fingerIndex];
  const [mcp, pip, dip, tip] = chains[key].map((i) => handLandmarks[i]);
  if (fingerIndex === 0) return Math.abs(tip.x - mcp.x) > Math.abs(dip.x - pip.x);
  return tip.y < dip.y && dip.y < pip.y && pip.y < mcp.y;
}

export function drawGlowCircle(ctx, x, y, r, color) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, color);
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

export class ParticleSystem {
  constructor() {
    this.particles = [];
  }

  spawn(x, y, config = {}) {
    this.particles.push({
      x,
      y,
      vx: config.vx ?? (Math.random() - 0.5) * 2,
      vy: config.vy ?? (Math.random() - 0.5) * 2,
      life: config.life ?? 0.8,
      age: 0,
      size: config.size ?? 8,
      color: config.color ?? 'rgba(255,160,60,0.8)',
    });
  }

  update(dt) {
    this.particles = this.particles.filter((p) => {
      p.age += dt;
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.05;
      return p.age < p.life;
    });
  }

  draw(ctx) {
    for (const p of this.particles) {
      const alpha = 1 - p.age / p.life;
      drawGlowCircle(ctx, p.x, p.y, p.size * alpha, withAlpha(p.color, alpha));
    }
  }
}

function withAlpha(color, alpha) {
  if (color.startsWith('rgba(')) {
    const parts = color.slice(5, -1).split(',').map((part) => part.trim());
    return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  if (color.startsWith('rgb(')) {
    const parts = color.slice(4, -1).split(',').map((part) => part.trim());
    return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  return color;
}

export function createARState() {
  return {
    particles: {
      fire: new ParticleSystem(),
      sparkles: new ParticleSystem(),
      jet: new ParticleSystem(),
    },
    wandTrail: [],
    ghostTrails: [],
    previousHipY: null,
    hue: 0,
  };
}

function getBlendshapeValue(faceResult, name) {
  const categories = faceResult?.faceBlendshapes?.[0]?.categories || [];
  const category = categories.find((c) => c.categoryName === name);
  return category?.score ?? 0;
}

function drawPolyline(ctx, points, color, width = 2) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x, points[i].y);
  ctx.stroke();
}

function drawDogFace(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;
  const leftEar = projectLandmark(landmarks[127], width, height);
  const rightEar = projectLandmark(landmarks[356], width, height);
  const nose = projectLandmark(landmarks[1], width, height);
  const jawOpen = getBlendshapeValue(faceResult, 'jawOpen');

  ctx.save();
  ctx.fillStyle = '#8b5a2b';
  ctx.beginPath();
  ctx.moveTo(leftEar.x - 18, leftEar.y - 12);
  ctx.quadraticCurveTo(leftEar.x - 54, leftEar.y - 76, leftEar.x - 16, leftEar.y - 96);
  ctx.quadraticCurveTo(leftEar.x + 8, leftEar.y - 62, leftEar.x + 12, leftEar.y - 18);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(rightEar.x + 18, rightEar.y - 12);
  ctx.quadraticCurveTo(rightEar.x + 54, rightEar.y - 76, rightEar.x + 16, rightEar.y - 96);
  ctx.quadraticCurveTo(rightEar.x - 8, rightEar.y - 62, rightEar.x - 12, rightEar.y - 18);
  ctx.fill();

  ctx.fillStyle = '#1f1b1a';
  ctx.beginPath();
  ctx.ellipse(nose.x, nose.y + 6, 16, 10, 0, 0, Math.PI * 2);
  ctx.fill();

  if (jawOpen > 0.4) {
    ctx.fillStyle = '#ff6b9c';
    ctx.beginPath();
    ctx.moveTo(nose.x - 10, nose.y + 22);
    ctx.quadraticCurveTo(nose.x, nose.y + 54, nose.x + 10, nose.y + 22);
    ctx.fill();
  }
  ctx.restore();
}

function drawSunglasses(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;
  const leftEye = projectLandmark(landmarks[33], width, height);
  const rightEye = projectLandmark(landmarks[263], width, height);
  const center = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const angle = Math.atan2(dy, dx);
  const dist = Math.hypot(dx, dy);

  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(angle);
  ctx.strokeStyle = '#10131a';
  ctx.lineWidth = Math.max(4, dist * 0.06);
  ctx.fillStyle = 'rgba(14,18,22,0.72)';
  const lensW = dist * 0.48;
  const lensH = dist * 0.22;
  const gap = dist * 0.07;
  roundRect(ctx, -gap / 2 - lensW, -lensH / 2, lensW, lensH, 12, true, true);
  roundRect(ctx, gap / 2, -lensH / 2, lensW, lensH, 12, true, true);
  ctx.beginPath();
  ctx.moveTo(-gap / 2, 0);
  ctx.lineTo(gap / 2, 0);
  ctx.stroke();
  ctx.restore();
}

function drawRainbow(ctx, faceResult, width, height, arState) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;
  const jawOpen = getBlendshapeValue(faceResult, 'jawOpen');
  if (jawOpen <= 0.5) return;
  const forehead = projectLandmark(landmarks[10], width, height);
  const box = getFaceBoundingBox(landmarks, width, height);
  const radius = box.w * 0.42;
  const colors = ['#ff4f81', '#ff9f1a', '#ffe66d', '#4ecdc4', '#4d96ff'];
  ctx.save();
  ctx.lineWidth = 10;
  colors.forEach((color, i) => {
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(forehead.x, forehead.y - 26, radius - i * 10, Math.PI, Math.PI * 2);
    ctx.stroke();
  });
  for (let i = 0; i < 3; i += 1) {
    arState.particles.sparkles.spawn(forehead.x + (Math.random() - 0.5) * radius, forehead.y - 30 - Math.random() * 40, {
      life: 0.7,
      size: 12,
      color: 'rgba(255,255,180,0.9)',
      vy: -0.4,
    });
  }
  ctx.restore();
}

function drawDevilHorns(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;
  const top = projectLandmark(landmarks[10], width, height);
  ctx.save();
  ctx.fillStyle = '#ff2e43';
  ctx.shadowColor = '#ff2e43';
  ctx.shadowBlur = 18;
  [[-28, -46], [28, -46]].forEach(([dx, dy]) => {
    ctx.beginPath();
    ctx.moveTo(top.x + dx, top.y + dy + 24);
    ctx.quadraticCurveTo(top.x + dx - 8, top.y + dy, top.x + dx + 12, top.y + dy - 14);
    ctx.quadraticCurveTo(top.x + dx + 18, top.y + dy + 6, top.x + dx + 10, top.y + dy + 24);
    ctx.fill();
  });
  ctx.restore();
}

function drawSparkleCrown(ctx, faceResult, width, height, arState) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;
  const forehead = projectLandmark(landmarks[10], width, height);
  const box = getFaceBoundingBox(landmarks, width, height);
  const radius = box.w * 0.45;
  // spawn sparkles intermittently
  if (Math.random() < 0.12) {
    const angle = Math.random() * Math.PI - Math.PI / 2;
    const x = forehead.x + Math.cos(angle) * (radius * 0.9);
    const y = forehead.y - 28 + Math.sin(angle) * (radius * 0.4);
    arState.particles.sparkles.spawn(x, y, { life: 0.7, size: 10, color: 'rgba(255,255,200,0.95)', vy: -0.6 });
  }
  // small crown gems
  ctx.save();
  const gemCount = 5;
  for (let i = 0; i < gemCount; i += 1) {
    const gx = forehead.x - radius * 0.8 + (i / (gemCount - 1)) * radius * 1.6;
    const gy = forehead.y - 40 - Math.sin((i / gemCount) * Math.PI) * 18;
    drawGlowCircle(ctx, gx, gy, 8, 'rgba(255,220,120,0.95)');
  }
  ctx.restore();
}

function drawLaserEyes(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;
  const leftEye = projectLandmark(landmarks[33], width, height);
  const rightEye = projectLandmark(landmarks[263], width, height);
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(255,20,100,0.95)';
  ctx.beginPath();
  ctx.moveTo(leftEye.x, leftEye.y);
  ctx.lineTo(leftEye.x + (leftEye.x - width * 0.5) * 2, leftEye.y - 120);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(rightEye.x, rightEye.y);
  ctx.lineTo(rightEye.x + (rightEye.x - width * 0.5) * 2, rightEye.y - 120);
  ctx.stroke();
  // small glow at eye center
  drawGlowCircle(ctx, leftEye.x, leftEye.y, 6, 'rgba(255,80,140,0.95)');
  drawGlowCircle(ctx, rightEye.x, rightEye.y, 6, 'rgba(255,80,140,0.95)');
  ctx.restore();
}

function drawGhostTrail(ctx, handResult, width, height, arState) {
  const hands = handResult?.handLandmarks || handResult?.hands || [];
  if (!Array.isArray(hands) || hands.length === 0) return;
  // store tips as trails
  hands.forEach((hand, idx) => {
    const tip = hand[8] || hand[4] || { x: 0.5, y: 0.5 };
    const p = projectLandmark(tip, width, height);
    if (!arState.ghostTrails[idx]) arState.ghostTrails[idx] = [];
    arState.ghostTrails[idx].unshift({ x: p.x, y: p.y, t: performance.now() });
    arState.ghostTrails[idx] = arState.ghostTrails[idx].slice(0, 20);
    // draw fading polyline
    const trail = arState.ghostTrails[idx];
    for (let i = 0; i < trail.length - 1; i += 1) {
      const a = trail[i];
      const b = trail[i + 1];
      const alpha = 1 - i / trail.length;
      ctx.strokeStyle = `rgba(180,220,255,${alpha * 0.8})`;
      ctx.lineWidth = 8 * (1 - i / trail.length);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  });
}

function drawFaceMesh(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(45, 255, 155, 0.45)';
  ctx.lineWidth = 1;
  for (const [a, b] of FACE_TESSELATION) {
    const p1 = projectLandmark(landmarks[a], width, height);
    const p2 = projectLandmark(landmarks[b], width, height);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawFireHands(ctx, handResult, width, height, arState) {
  // Support multiple result shapes returned by different Mediapipe runtimes
  const hands = handResult?.handLandmarks || handResult?.landmarks || handResult?.hands || [];
  if (!Array.isArray(hands) || hands.length === 0) return;

  hands.forEach((hand) => {
    if (!hand || hand.length < 21) return;
    const tipIndices = [4, 8, 12, 16, 20];
    const tips = tipIndices.map((i) => {
      const lm = hand[i] || { x: 0.5, y: 0.5, z: 0 };
      return projectLandmark(lm, width, height);
    });

    // Compute a safe spread estimate between index and pinky
    const idx = tips[1] || tips[0];
    const pinky = tips[4] || tips[tips.length - 1];
    const spread = Math.hypot(idx.x - pinky.x, idx.y - pinky.y) || 20;
    const burst = Math.max(1, Math.round(spread / 80));

    tips.forEach((tip) => {
      // spawn a small burst per tip
      for (let i = 0; i < burst; i += 1) {
        arState.particles.fire.spawn(tip.x + (Math.random() - 0.5) * 6, tip.y + (Math.random() - 0.5) * 6, {
          size: 12 + Math.random() * 10,
          color: 'rgba(255,140,50,0.92)',
          vx: (Math.random() - 0.5) * 2.2,
          vy: -0.6 - Math.random() * 2.2,
          life: 0.45 + Math.random() * 0.6,
        });
      }
      drawGlowCircle(ctx, tip.x, tip.y, 14, 'rgba(255,120,0,0.6)');
    });
  });
}

function drawMagicWand(ctx, handResult, width, height, arState) {
  const hands = handResult.handLandmarks || [];
  hands.forEach((hand) => {
    const pointing = isFingerExtended(hand, 1) && !isFingerExtended(hand, 2) && !isFingerExtended(hand, 3) && !isFingerExtended(hand, 4);
    if (!pointing) return;
    const tip = projectLandmark(hand[8], width, height);
    arState.wandTrail.unshift(tip);
    arState.wandTrail = arState.wandTrail.slice(0, 20);
    drawPolyline(ctx, arState.wandTrail, 'rgba(145, 255, 255, 0.85)', 4);
    arState.particles.sparkles.spawn(tip.x, tip.y, { color: 'rgba(180,255,255,0.9)', size: 14, vy: -0.6, life: 0.5 });
    ctx.save();
    ctx.strokeStyle = '#fff6c0';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(tip.x - 24, tip.y + 10);
    ctx.lineTo(tip.x + 8, tip.y - 10);
    ctx.stroke();
    ctx.restore();
  });
}

function drawHandSkeleton(ctx, handResult, width, height) {
  const hands = handResult.handLandmarks || [];
  ctx.save();
  ctx.strokeStyle = 'rgba(90,255,255,0.9)';
  ctx.fillStyle = 'rgba(90,255,255,0.95)';
  ctx.lineWidth = 2;
  hands.forEach((hand) => {
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const p1 = projectLandmark(hand[a], width, height);
      const p2 = projectLandmark(hand[b], width, height);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    });
    hand.forEach((lm) => {
      const p = projectLandmark(lm, width, height);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  ctx.restore();
}

function drawBodySkeleton(ctx, poseResult, width, height) {
  const pose = poseResult.poseLandmarks?.[0] || poseResult.landmarks?.[0];
  if (!pose) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(123, 246, 255, 0.95)';
  ctx.fillStyle = 'rgba(123, 246, 255, 0.95)';
  ctx.lineWidth = 3;
  POSE_CONNECTIONS.forEach(([a, b]) => {
    const p1 = projectLandmark(pose[a], width, height);
    const p2 = projectLandmark(pose[b], width, height);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  });
  pose.forEach((lm) => {
    const p = projectLandmark(lm, width, height);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawAuraField(ctx, poseResult, width, height, arState, dt) {
  const pose = poseResult.poseLandmarks?.[0] || poseResult.landmarks?.[0];
  if (!pose) return;
  const points = [11, 12, 15, 16, 23, 24].map((i) => projectLandmark(pose[i], width, height));
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const radius = Math.max(120, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  arState.hue = (arState.hue + dt * 60) % 360;
  const grad = ctx.createRadialGradient(cx, cy, radius * 0.15, cx, cy, radius);
  grad.addColorStop(0, `hsla(${arState.hue}, 100%, 65%, 0.32)`);
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawJetpack(ctx, poseResult, width, height, arState) {
  const pose = poseResult.poseLandmarks?.[0] || poseResult.landmarks?.[0];
  if (!pose) return;
  const leftShoulder = projectLandmark(pose[11], width, height);
  const rightShoulder = projectLandmark(pose[12], width, height);
  const leftHip = projectLandmark(pose[23], width, height);
  const rightHip = projectLandmark(pose[24], width, height);
  const center = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 + 18 };
  const hipY = (leftHip.y + rightHip.y) / 2;
  const velocity = arState.previousHipY == null ? 0 : arState.previousHipY - hipY;
  arState.previousHipY = hipY;

  ctx.save();
  ctx.fillStyle = '#5d6b86';
  ctx.fillRect(center.x - 24, center.y - 22, 48, 50);
  ctx.fillStyle = '#8897b2';
  ctx.fillRect(center.x - 18, center.y - 14, 14, 34);
  ctx.fillRect(center.x + 4, center.y - 14, 14, 34);
  ctx.fillStyle = '#ffd166';
  ctx.fillRect(center.x - 16, center.y + 28, 10, 8);
  ctx.fillRect(center.x + 6, center.y + 28, 10, 8);
  ctx.restore();

  const flamePower = Math.max(1, velocity > 6 ? 4 : 2);
  for (let i = 0; i < flamePower; i += 1) {
    arState.particles.jet.spawn(center.x - 10 + Math.random() * 20, center.y + 34, {
      color: 'rgba(255,170,60,0.88)',
      size: 16,
      vy: 1.5 + Math.random() * 1.4,
      vx: (Math.random() - 0.5) * 1.2,
      life: 0.5,
    });
  }
}

export function drawActiveARFilter(ctx, filterId, mpState, arState, deltaTime) {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const dt = Math.max(0.016, deltaTime || 0.016);

  arState.particles.fire.update(dt);
  arState.particles.sparkles.update(dt);
  arState.particles.jet.update(dt);

  switch (filterId) {
    case 'dog-face': drawDogFace(ctx, mpState, width, height); break;
    case 'sunglasses': drawSunglasses(ctx, mpState, width, height); break;
    case 'rainbow': drawRainbow(ctx, mpState, width, height, arState); break;
    case 'sparkle-crown': drawSparkleCrown(ctx, mpState, width, height, arState); break;
    case 'laser-eyes': drawLaserEyes(ctx, mpState, width, height); break;
    case 'devil-horns': drawDevilHorns(ctx, mpState, width, height); break;
    case 'face-mesh': drawFaceMesh(ctx, mpState, width, height); break;
    case 'fire-hands': drawFireHands(ctx, mpState, width, height, arState); break;
    case 'ghost-trail': drawGhostTrail(ctx, mpState, width, height, arState); break;
    case 'magic-wand': drawMagicWand(ctx, mpState, width, height, arState); break;
    case 'hand-skeleton': drawHandSkeleton(ctx, mpState, width, height); break;
    case 'body-skeleton': drawBodySkeleton(ctx, mpState, width, height); break;
    case 'aura-field': drawAuraField(ctx, mpState, width, height, arState, dt); break;
    case 'jetpack': drawJetpack(ctx, mpState, width, height, arState); break;
    default: break;
  }

  arState.particles.fire.draw(ctx);
  arState.particles.sparkles.draw(ctx);
  arState.particles.jet.draw(ctx);
}

function roundRect(ctx, x, y, w, h, r, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}
