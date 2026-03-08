/*
 * ar-filters-extended.js
 * 13 nuovi filtri AR MediaPipe (face × 7, hand × 3, pose × 3)
 * Da usare insieme ad ar-filters.js
 *
 * INTEGRAZIONE:
 *  1. Aggiungere NEW_AR_FILTERS a AR_FILTERS nel file originale.
 *  2. Chiamare extendARState(state) subito dopo createARState().
 *  3. Nel loop di rendering, dopo drawActiveARFilter(...) aggiungere:
 *       drawExtendedARFilter(ctx, filterId, mpState, arState, deltaTime);
 *  4. Assicurarsi che le funzioni non-esportate (vedi sotto) siano
 *     esportate dall'originale, oppure usare le ridefinizioni locali incluse qui.
 */

import {
  projectLandmark,
  getFaceBoundingBox,
  isFingerExtended,
  drawGlowCircle,
  ParticleSystem,
} from './ar-filters.js';


// ─── Nuovi filtri da aggiungere ad AR_FILTERS ─────────────────────────────────

export const NEW_AR_FILTERS = [
  // face
  { id: 'cat-face',     name: 'Cat Face',       category: 'ar', tracker: 'face' },
  { id: 'halo',         name: 'Angel Halo',      category: 'ar', tracker: 'face' },
  { id: 'star-eyes',    name: 'Star Eyes',       category: 'ar', tracker: 'face' },
  { id: 'neon-paint',   name: 'Neon War Paint',  category: 'ar', tracker: 'face' },
  { id: 'pixel-crown',  name: 'Pixel Crown',     category: 'ar', tracker: 'face' },
  { id: 'vhs-glitch',   name: 'VHS Glitch',      category: 'ar', tracker: 'face' },
  { id: 'tear-drops',   name: 'Crying Laugh',    category: 'ar', tracker: 'face' },
  // hand
  { id: 'confetti-burst', name: 'Confetti Burst', category: 'ar', tracker: 'hand' },
  { id: 'electric-arc',   name: 'Electric Arc',   category: 'ar', tracker: 'hand' },
  { id: 'bubble-hands',   name: 'Bubble Hands',   category: 'ar', tracker: 'hand' },
  // pose
  { id: 'angel-wings',  name: 'Angel Wings',  category: 'ar', tracker: 'pose' },
  { id: 'matrix-rain',  name: 'Matrix Rain',  category: 'ar', tracker: 'pose' },
  { id: 'force-shield', name: 'Force Shield', category: 'ar', tracker: 'pose' },
];


// ─── Estendi arState con i sistemi di particelle aggiuntivi ───────────────────

/**
 * Chiama questa funzione dopo createARState() per aggiungere
 * i sistemi di particelle necessari ai nuovi filtri.
 * @param {object} state – il valore restituito da createARState()
 * @returns {object} lo stesso state, mutato in-place
 */
export function extendARState(state) {
  state.particles.bubbles   = new ParticleSystem();
  state.particles.confetti  = new ParticleSystem();
  state.particles.tears     = new ParticleSystem();
  state._vhsFrame    = 0;
  state._shieldAngle = 0;
  state._matrixCols  = null;
  return state;
}


// ─── Helper locali (non esportati dall'originale) ─────────────────────────────

function getBlendshapeValue(faceResult, name) {
  const cats = faceResult?.faceBlendshapes?.[0]?.categories ?? [];
  return cats.find((c) => c.categoryName === name)?.score ?? 0;
}

function drawStar(ctx, cx, cy, outerR, innerR, points, rotation = 0) {
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const angle = (i * Math.PI) / points + rotation - Math.PI / 2;
    const r = i % 2 === 0 ? outerR : innerR;
    if (i === 0) ctx.moveTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
    else ctx.lineTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
  }
  ctx.closePath();
}

function lerp(a, b, t) { return a + (b - a) * t; }



// ═════════════════════════════════════════════════════════════════════════════=
// FACE FILTERS
// ═════════════════════════════════════════════════════════════════════════════=

// ── 1. Cat Face ───────────────────────────────────────────────────────────────
function drawCatFace(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;

  const lEar   = projectLandmark(landmarks[127], width, height);
  const rEar   = projectLandmark(landmarks[356], width, height);
  const nose   = projectLandmark(landmarks[1],   width, height);
  const lCheek = projectLandmark(landmarks[205], width, height);
  const rCheek = projectLandmark(landmarks[425], width, height);

  ctx.save();

  // Orecchie esterne
  const drawEar = (tip, side) => {
    const sx = side === 'left' ? -1 : 1;
    ctx.fillStyle = '#f4a0c0';
    ctx.strokeStyle = '#d06090';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tip.x + sx * 8,  tip.y);
    ctx.lineTo(tip.x + sx * 28, tip.y - 62);
    ctx.lineTo(tip.x - sx * 12, tip.y - 22);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    // Interno
    ctx.fillStyle = '#ffcce0';
    ctx.beginPath();
    ctx.moveTo(tip.x + sx * 8,  tip.y - 4);
    ctx.lineTo(tip.x + sx * 20, tip.y - 50);
    ctx.lineTo(tip.x - sx * 6,  tip.y - 20);
    ctx.closePath();
    ctx.fill();
  };

  drawEar(lEar, 'left');
  drawEar(rEar, 'right');

  // Nasino triangolare
  ctx.fillStyle = '#ff9eb5';
  ctx.shadowColor = '#ff70a0';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.moveTo(nose.x,      nose.y - 3);
  ctx.lineTo(nose.x - 9,  nose.y + 9);
  ctx.lineTo(nose.x + 9,  nose.y + 9);
  ctx.closePath();
  ctx.fill();

  // Baffi
  ctx.shadowBlur = 0;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.lineWidth = 1.5;
  [
    [lCheek, -1],
    [rCheek,  1],
  ].forEach(([cheek, s]) => {
    [[-6, -3], [-8, 4], [-6, 11]].forEach(([dy1, dy2]) => {
      ctx.beginPath();
      ctx.moveTo(cheek.x,              cheek.y + dy1);
      ctx.lineTo(cheek.x + s * 52,     cheek.y + dy2);
      ctx.stroke();
    });
  });

  ctx.restore();
}


// ── 2. Angel Halo ─────────────────────────────────────────────────────────────
function drawHalo(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;

  const top = projectLandmark(landmarks[10], width, height);
  const box = getFaceBoundingBox(landmarks, width, height);
  const rx  = box.w * 0.40;
  const ry  = rx * 0.24;
  const cy  = top.y - 36;

  ctx.save();
  ctx.shadowColor = '#ffe066';
  ctx.shadowBlur  = 22;

  // Alone luminoso
  const grad = ctx.createRadialGradient(top.x, cy, rx * 0.4, top.x, cy, rx * 1.4);
  grad.addColorStop(0, 'rgba(255, 230, 80, 0.15)');
  grad.addColorStop(1, 'rgba(255, 230, 80, 0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(top.x, cy, rx * 1.4, ry * 1.8, 0, 0, Math.PI * 2);
  ctx.fill();

  // Anello principale
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth   = 9;
  ctx.beginPath();
  ctx.ellipse(top.x, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Riflesso interno
  ctx.strokeStyle = 'rgba(255, 255, 220, 0.55)';
  ctx.lineWidth   = 3;
  ctx.beginPath();
  ctx.ellipse(top.x, cy, rx * 0.85, ry * 0.72, 0, 0, Math.PI);
  ctx.stroke();

  ctx.restore();
}


// ── 3. Star Eyes ──────────────────────────────────────────────────────────────
function drawStarEyes(ctx, faceResult, width, height, arState) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;

  const lEye = projectLandmark(landmarks[33],  width, height);
  const rEye = projectLandmark(landmarks[263], width, height);
  const dist = Math.hypot(rEye.x - lEye.x, rEye.y - lEye.y);
  const r    = dist * 0.24;

  arState.hue = (arState.hue + 2.5) % 360;

  ctx.save();
  [lEye, rEye].forEach((eye, idx) => {
    ctx.save();
    ctx.translate(eye.x, eye.y);
    ctx.rotate((arState.hue + idx * 45) * (Math.PI / 180));

    // Stella a 6 punte
    drawStar(ctx, 0, 0, r, r * 0.42, 6);
    ctx.fillStyle   = `hsl(${(arState.hue + idx * 120) % 360}, 100%, 65%)`;
    ctx.shadowColor = `hsl(${arState.hue}, 100%, 70%)`;
    ctx.shadowBlur  = 16;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    // Piccolo nucleo brillante
    ctx.fillStyle  = '#fff';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.18, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  });
  ctx.restore();
}


// ── 4. Neon War Paint ─────────────────────────────────────────────────────────
function drawNeonPaint(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;

  const lCheek  = projectLandmark(landmarks[205], width, height);
  const rCheek  = projectLandmark(landmarks[425], width, height);
  const forehead = projectLandmark(landmarks[10],  width, height);
  const lEye    = projectLandmark(landmarks[33],  width, height);
  const rEye    = projectLandmark(landmarks[263], width, height);
  const nose    = projectLandmark(landmarks[1],   width, height);

  ctx.save();
  ctx.lineWidth = 3;
  ctx.lineCap   = 'round';

  const stroke = (color, lines) => {
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 12;
    lines.forEach(([x1, y1, x2, y2]) => {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });
  };

  // Guancia sinistra – barre diagonali magenta
  stroke('#ff00ff', [
    [lCheek.x,      lCheek.y - 6,  lCheek.x - 32, lCheek.y + 10],
    [lCheek.x - 4,  lCheek.y + 6,  lCheek.x - 32, lCheek.y + 22],
  ]);

  // Guancia destra – barre diagonali cyan
  stroke('#00ffff', [
    [rCheek.x,      rCheek.y - 6,  rCheek.x + 32, rCheek.y + 10],
    [rCheek.x + 4,  rCheek.y + 6,  rCheek.x + 32, rCheek.y + 22],
  ]);

  // Linee sotto gli occhi – verde
  stroke('#00ff88', [
    [lEye.x - 10, lEye.y + 14, lEye.x + 6,  lEye.y + 30],
    [rEye.x + 10, rEye.y + 14, rEye.x - 6,  rEye.y + 30],
  ]);

  // Diamante sulla fronte – giallo
  ctx.lineWidth = 2;
  const d = 16;
  stroke('#ffff00', [
    [forehead.x,     forehead.y - d - 8, forehead.x + d, forehead.y - 8],
    [forehead.x + d, forehead.y - 8,     forehead.x,     forehead.y + d - 8],
    [forehead.x,     forehead.y + d - 8, forehead.x - d, forehead.y - 8],
    [forehead.x - d, forehead.y - 8,     forehead.x,     forehead.y - d - 8],
  ]);

  // Linea verticale sul naso – arancione
  ctx.lineWidth = 2.5;
  stroke('#ff8800', [
    [nose.x, nose.y - 28, nose.x, nose.y + 8],
  ]);

  ctx.restore();
}


// ── 5. Pixel Crown ────────────────────────────────────────────────────────────
function drawPixelCrown(ctx, faceResult, width, height) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;

  const top = projectLandmark(landmarks[10], width, height);
  const box = getFaceBoundingBox(landmarks, width, height);
  const bw  = Math.max(7, box.w / 10);
  const cols = 9;

  // Pattern altezze (in blocchi) e colori per colonna
  const heights = [3, 5, 4, 7, 6, 7, 4, 5, 3];
  const palette = ['#ff6b6b','#ffd166','#06d6a0','#118ab2','#e040fb','#ff9f1c','#00e5ff','#69f0ae','#ff4081'];
  const startX  = top.x - (cols / 2) * bw;
  const baseY   = top.y - 14;

  ctx.save();
  heights.forEach((h, col) => {
    ctx.fillStyle = palette[col % palette.length];
    for (let row = 0; row < h; row++) {
      ctx.fillRect(
        Math.round(startX + col * bw),
        Math.round(baseY  - row * bw),
        bw - 1,
        bw - 1,
      );
    }
    // Gemma in cima
    ctx.fillStyle = '#fff';
    ctx.fillRect(
      Math.round(startX + col * bw + bw * 0.3),
      Math.round(baseY  - (h - 1) * bw + 1),
      bw * 0.4,
      bw * 0.3,
    );
  });
  ctx.restore();
}


// ── 6. VHS Glitch ─────────────────────────────────────────────────────────────
function drawVHSGlitch(ctx, faceResult, width, height, arState) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;

  const box = getFaceBoundingBox(landmarks, width, height);
  const pad = 28;
  const bx  = Math.round(box.x - pad);
  const by  = Math.round(box.y - pad);
  const bw  = Math.round(box.w + pad * 2);
  const bh  = Math.round(box.h + pad * 2);

  arState._vhsFrame = (arState._vhsFrame ?? 0) + 1;

  ctx.save();

  // Scanlines
  ctx.fillStyle = 'rgba(0, 0, 0, 0.18)';
  for (let y = by; y < by + bh; y += 4) ctx.fillRect(bx, y, bw, 2);

  // Bande glitch orizzontali (casuali, intermittenti)
  if (arState._vhsFrame % 7 < 3) {
    const bands = 1 + Math.floor(Math.random() * 3);
    for (let i = 0; i < bands; i++) {
      const bandY = by + Math.random() * bh;
      const bandH = 4 + Math.random() * 16;
      const shift = (Math.random() - 0.5) * 26;
      ctx.fillStyle = `rgba(255, 0,  80, 0.14)`;
      ctx.fillRect(bx + shift - 2, bandY, bw, bandH);
      ctx.fillStyle = `rgba(0,  80, 255, 0.11)`;
      ctx.fillRect(bx + shift + 3, bandY, bw, bandH);
    }
  }

  // Bordo RGB-split
  ctx.strokeStyle = 'rgba(255, 0,   0, 0.30)';
  ctx.lineWidth = 2;
  ctx.strokeRect(bx - 2, by, bw, bh);
  ctx.strokeStyle = 'rgba(0,   0, 255, 0.22)';
  ctx.strokeRect(bx + 2, by, bw, bh);

  // Indicatore REC lampeggiante
  if (arState._vhsFrame % 50 < 30) {
    ctx.fillStyle = '#ff2200';
    ctx.font      = 'bold 13px monospace';
    ctx.fillText('● REC', bx + 8, by + 20);
  }

  ctx.restore();
}


// ── 7. Crying Laugh (Tear Drops) ──────────────────────────────────────────────
function drawTearDrops(ctx, faceResult, width, height, arState) {
  const landmarks = faceResult.faceLandmarks?.[0];
  if (!landmarks) return;

  const jawOpen = getBlendshapeValue(faceResult, 'jawOpen');
  const lEye    = projectLandmark(landmarks[33],  width, height);
  const rEye    = projectLandmark(landmarks[263], width, height);

  // Spawna lacrime solo con la bocca aperta
  if (jawOpen > 0.28) {
    [lEye, rEye].forEach((eye) => {
      if (Math.random() < 0.25) {
        arState.particles.tears.spawn(eye.x + (Math.random() - 0.5) * 8, eye.y + 10, {
          color: 'rgba(130, 190, 255, 0.9)',
          size : 7 + Math.random() * 5,
          vx   : (Math.random() - 0.5) * 0.6,
          vy   : 0.9 + Math.random() * 1.0,
          life : 0.9,
        });
      }
    });
  }

  // Disegna lacrime come gocce (cerchio + punta)
  for (const p of arState.particles.tears.particles) {
    const alpha = (1 - p.age / p.life) * 0.85;
    ctx.save();
    ctx.fillStyle   = `rgba(140, 195, 255, ${alpha})`;
    ctx.shadowColor = `rgba(80, 150, 255, ${alpha})`;
    ctx.shadowBlur  = 7;
    // Corpo circolare
    ctx.beginPath();
    ctx.arc(p.x, p.y + p.size * 0.8, p.size * 0.7, 0, Math.PI * 2);
    ctx.fill();
    // Punta superiore
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - p.size * 0.4);
    ctx.lineTo(p.x - p.size * 0.5, p.y + p.size * 0.5);
    ctx.lineTo(p.x + p.size * 0.5, p.y + p.size * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}


// ═════════════════════════════════════════════════════════════════════════════=
// HAND FILTERS
// ═════════════════════════════════════════════════════════════════════════════=

// ── 8. Confetti Burst ─────────────────────────────────────────────────────────
function drawConfettiBurst(ctx, handResult, width, height, arState) {
  const hands = handResult?.handLandmarks || handResult?.hands || [];
  if (!Array.isArray(hands) || hands.length === 0) return;

  const palette = [
    'rgba(255,107,107,0.9)', 'rgba(255,209,102,0.9)',
    'rgba(6,214,160,0.9)',   'rgba(17,138,178,0.9)',
    'rgba(224,64,251,0.9)',  'rgba(0,229,255,0.9)',
    'rgba(255,153,0,0.9)',   'rgba(105,240,174,0.9)',
  ];

  hands.forEach((hand) => {
    [4, 8, 12, 16, 20].forEach((i) => {
      if (Math.random() < 0.35) {
        const tip = projectLandmark(hand[i], width, height);
        arState.particles.confetti.spawn(tip.x, tip.y, {
          color: palette[Math.floor(Math.random() * palette.length)],
          size : 5 + Math.random() * 7,
          vx   : (Math.random() - 0.5) * 4.5,
          vy   : -1.5 - Math.random() * 2.5,
          life : 0.7 + Math.random() * 0.4,
        });
      }
    });
  });

  // Disegna come rettangolini ruotati (coriandoli)
  for (const p of arState.particles.confetti.particles) {
    const alpha = 1 - p.age / p.life;
    const rot   = p.age * 6;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(rot);
    ctx.globalAlpha = alpha;
    ctx.fillStyle   = p.color;
    ctx.fillRect(-p.size / 2, -p.size / 3, p.size, p.size * 0.55);
    ctx.restore();
  }
}


// ── 9. Electric Arc ───────────────────────────────────────────────────────────
function drawElectricArc(ctx, handResult, width, height) {
  const hands = handResult?.handLandmarks || handResult?.hands || [];
  if (!Array.isArray(hands) || hands.length === 0) return;

  hands.forEach((hand) => {
    const tips = [4, 8, 12, 16, 20].map((i) => projectLandmark(hand[i], width, height));

    ctx.save();
    ctx.shadowBlur  = 14;
    ctx.shadowColor = '#88ccff';

    for (let i = 0; i < tips.length - 1; i++) {
      const a    = tips[i];
      const b    = tips[i + 1];
      const dist = Math.hypot(b.x - a.x, b.y - a.y);
      if (dist > 110) continue; // traccia archi solo tra dita vicine

      // Filamento principale
      ctx.strokeStyle = 'rgba(120, 210, 255, 0.95)';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      for (let s = 1; s < 12; s++) {
        const t = s / 12;
        ctx.lineTo(
          lerp(a.x, b.x, t) + (Math.random() - 0.5) * 20,
          lerp(a.y, b.y, t) + (Math.random() - 0.5) * 20,
        );
      }
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      // Filamento secondario (più sottile)
      ctx.strokeStyle = 'rgba(200, 235, 255, 0.45)';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      for (let s = 1; s < 12; s++) {
        const t = s / 12;
        ctx.lineTo(
          lerp(a.x, b.x, t) + (Math.random() - 0.5) * 14,
          lerp(a.y, b.y, t) + (Math.random() - 0.5) * 14,
        );
      }
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // Bagliore su ogni punta
    tips.forEach((t) => drawGlowCircle(ctx, t.x, t.y, 7, 'rgba(100, 200, 255, 0.85)'));
    ctx.restore();
  });
}


// ── 10. Bubble Hands ──────────────────────────────────────────────────────────
function drawBubbleHands(ctx, handResult, width, height, arState) {
  const hands = handResult?.handLandmarks || handResult?.hands || [];

  // Spawn
  if (Array.isArray(hands) && hands.length > 0) {
    hands.forEach((hand) => {
      [4, 8, 12, 16, 20].forEach((i) => {
        if (Math.random() < 0.14) {
          const tip = projectLandmark(hand[i], width, height);
          arState.particles.bubbles.spawn(tip.x, tip.y, {
            size : 14 + Math.random() * 22,
            vx   : (Math.random() - 0.5) * 0.9,
            vy   : -0.4 - Math.random() * 0.7,
            life : 2.0 + Math.random() * 0.5,
            color: 'rgba(180, 220, 255, 0.5)',
          });
        }
      });
    });
  }

  // Render personalizzato: contorno traslucido + riflesso speculare
  for (const p of arState.particles.bubbles.particles) {
    const alpha = (1 - p.age / p.life) * 0.75;
    ctx.save();

    ctx.strokeStyle = `rgba(180, 220, 255, ${alpha})`;
    ctx.lineWidth   = 1.6;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.stroke();

    // Colore interno tenue
    ctx.fillStyle = `rgba(200, 230, 255, ${alpha * 0.12})`;
    ctx.fill();

    // Riflesso speculare
    ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.4})`;
    ctx.beginPath();
    ctx.arc(
      p.x - p.size * 0.30,
      p.y - p.size * 0.30,
      p.size * 0.27,
      0, Math.PI * 2,
    );
    ctx.fill();

    ctx.restore();
  }
}


// ═════════════════════════════════════════════════════════════════════════════=
// POSE FILTERS
// ═════════════════════════════════════════════════════════════════════════════=

// ── 11. Angel Wings ───────────────────────────────────────────────────────────
function drawAngelWings(ctx, poseResult, width, height) {
  const pose = poseResult.poseLandmarks?.[0] || poseResult.landmarks?.[0];
  if (!pose) return;

  const ls   = projectLandmark(pose[11], width, height);
  const rs   = projectLandmark(pose[12], width, height);
  const span = Math.hypot(rs.x - ls.x, rs.y - ls.y);

  ctx.save();
  ctx.shadowColor = 'rgba(210, 225, 255, 0.85)';
  ctx.shadowBlur  = 24;

  const drawWing = (anchor, side) => {
    const s = side === 'left' ? -1 : 1;

    // Ala principale
    ctx.fillStyle   = 'rgba(240, 246, 255, 0.88)';
    ctx.strokeStyle = 'rgba(200, 218, 255, 0.65)';
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    ctx.moveTo(anchor.x, anchor.y);
    ctx.bezierCurveTo(
      anchor.x + s * span * 1.0, anchor.y - span * 0.45,
      anchor.x + s * span * 1.25, anchor.y + span * 0.18,
      anchor.x + s * span * 0.82, anchor.y + span * 0.52,
    );
    ctx.bezierCurveTo(
      anchor.x + s * span * 0.42, anchor.y + span * 0.38,
      anchor.x + s * span * 0.18, anchor.y + span * 0.10,
      anchor.x, anchor.y,
    );
    ctx.fill();
    ctx.stroke();

    // Pennine (linee curvate di dettaglio)
    ctx.strokeStyle = 'rgba(170, 195, 240, 0.45)';
    ctx.lineWidth   = 1;
    for (let f = 1; f <= 5; f++) {
      const t  = f / 6;
      const fx = anchor.x + s * span * t * 1.1;
      const fy = anchor.y - span * t * 0.3 + span * t * t * 0.6;
      ctx.beginPath();
      ctx.moveTo(fx, fy - span * 0.08);
      ctx.quadraticCurveTo(fx + s * 10, fy + span * 0.12, fx, fy + span * 0.24);
      ctx.stroke();
    }
  };

  drawWing(ls, 'left');
  drawWing(rs, 'right');
  ctx.restore();
}


// ── 12. Matrix Rain ───────────────────────────────────────────────────────────
const MATRIX_CHARS = 'アカサタナハマヤラワアイウエオカキクケコ01ﾊﾐﾋｰｳｼﾅﾓﾆｻﾜﾂｵﾘｱﾎﾃﾏｹﾒﾝ';

function drawMatrixRain(ctx, poseResult, width, height, arState) {
  const pose = poseResult.poseLandmarks?.[0] || poseResult.landmarks?.[0];
  if (!pose) return;

  const pts  = pose.map((lm) => projectLandmark(lm, width, height));
  const xs   = pts.map((p) => p.x);
  const ys   = pts.map((p) => p.y);
  const minX = Math.max(0, Math.min(...xs) - 50);
  const maxX = Math.min(width,  Math.max(...xs) + 50);
  const minY = Math.max(0, Math.min(...ys) - 50);
  const maxY = Math.min(height, Math.max(...ys) + 60);
  const colW = 14;
  const numCols = Math.max(1, Math.floor((maxX - minX) / colW));

  // Inizializza o reimposta le colonne se cambiano dimensioni
  if (!arState._matrixCols || arState._matrixCols.length !== numCols) {
    arState._matrixCols = Array.from({ length: numCols }, () => ({
      y    : minY + Math.random() * (maxY - minY),
      speed: 7 + Math.random() * 11,
      chars: Array.from({ length: 14 }, () =>
        MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)],
      ),
    }));
  }

  ctx.save();
  ctx.font = `${colW - 1}px monospace`;

  arState._matrixCols.forEach((col, i) => {
    // Aggiorna caratteri casualmente
    if (Math.random() < 0.1) {
      col.chars[Math.floor(Math.random() * col.chars.length)] =
        MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)];
    }

    const x = minX + i * colW;

    // Testa brillante
    ctx.fillStyle = '#ccffcc';
    ctx.fillText(col.chars[0], x, col.y);

    // Scia in dissolvenza
    for (let t = 1; t < col.chars.length; t++) {
      const alpha = (1 - t / col.chars.length) * 0.85;
      ctx.fillStyle = `rgba(0, 200, 70, ${alpha})`;
      ctx.fillText(col.chars[t], x, col.y - t * colW);
    }

    col.y += col.speed;
    if (col.y > maxY + colW * col.chars.length) col.y = minY - colW * 4;
  });

  ctx.restore();
}


// ── 13. Force Shield ──────────────────────────────────────────────────────────
function drawForceShield(ctx, poseResult, width, height, arState, dt) {
  const pose = poseResult.poseLandmarks?.[0] || poseResult.landmarks?.[0];
  if (!pose) return;

  const anchors = [11, 12, 23, 24].map((i) => projectLandmark(pose[i], width, height));
  const xs  = anchors.map((p) => p.x);
  const ys  = anchors.map((p) => p.y);
  const cx  = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy  = (Math.min(...ys) + Math.max(...ys)) / 2;
  const rx  = (Math.max(...xs) - Math.min(...xs)) * 0.68 + 48;
  const ry  = (Math.max(...ys) - Math.min(...ys)) * 0.82 + 48;

  arState.hue          = (arState.hue + (dt ?? 0.016) * 80) % 360;
  arState._shieldAngle = (arState._shieldAngle ?? 0) + (dt ?? 0.016) * 1.4;

  ctx.save();

  // Alone radiale
  const grad = ctx.createRadialGradient(cx, cy, Math.min(rx, ry) * 0.5, cx, cy, Math.max(rx, ry) * 1.05);
  grad.addColorStop(0,   `hsla(${arState.hue}, 80%, 65%, 0)`);
  grad.addColorStop(0.72,`hsla(${arState.hue}, 90%, 65%, 0.10)`);
  grad.addColorStop(1,   `hsla(${arState.hue}, 100%, 72%, 0.40)`);
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx * 1.05, ry * 1.05, 0, 0, Math.PI * 2);
  ctx.fill();

  // Anello esterno tratteggiato (rotante)
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(arState._shieldAngle);
  ctx.strokeStyle = `hsla(${arState.hue}, 100%, 78%, 0.88)`;
  ctx.lineWidth   = 3;
  ctx.shadowColor = `hsl(${arState.hue}, 100%, 72%)`;
  ctx.shadowBlur  = 20;
  ctx.setLineDash([20, 10]);
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // Anello interno (contro-rotante, colore complementare)
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-arState._shieldAngle * 0.65);
  ctx.strokeStyle = `hsla(${(arState.hue + 150) % 360}, 100%, 78%, 0.55)`;
  ctx.lineWidth   = 1.5;
  ctx.setLineDash([10, 16]);
  ctx.beginPath();
  ctx.ellipse(0, 0, rx * 0.80, ry * 0.80, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // Nodi di impatto (piccoli punti lucenti agli angoli dell'ellisse)
  const nodeAngles = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];
  nodeAngles.forEach((a) => {
    const nx = cx + Math.cos(a + arState._shieldAngle) * rx;
    const ny = cy + Math.sin(a + arState._shieldAngle) * ry;
    drawGlowCircle(ctx, nx, ny, 6, `hsla(${arState.hue}, 100%, 80%, 0.9)`);
  });

  ctx.restore();
}


// ═════════════════════════════════════════════════════════════════════════════=
// DISPATCHER principale per i nuovi filtri
// ═════════════════════════════════════════════════════════════════════════════=

/**
 * Chiama questa funzione nel loop di rendering, dopo drawActiveARFilter().
 * Gestisce aggiornamento e disegno delle particelle aggiuntive.
 */
export function drawExtendedARFilter(ctx, filterId, mpState, arState, deltaTime) {
  const width  = ctx.canvas.width;
  const height = ctx.canvas.height;
  const dt     = Math.max(0.016, deltaTime || 0.016);

  // Aggiorna sistemi di particelle extra
  arState.particles.bubbles?.update(dt);
  arState.particles.confetti?.update(dt);
  arState.particles.tears?.update(dt);

  switch (filterId) {
    // face
    case 'cat-face':    drawCatFace(ctx, mpState, width, height);                   break;
    case 'halo':        drawHalo(ctx, mpState, width, height);                      break;
    case 'star-eyes':   drawStarEyes(ctx, mpState, width, height, arState);         break;
    case 'neon-paint':  drawNeonPaint(ctx, mpState, width, height);                 break;
    case 'pixel-crown': drawPixelCrown(ctx, mpState, width, height);                break;
    case 'vhs-glitch':  drawVHSGlitch(ctx, mpState, width, height, arState);        break;
    case 'tear-drops':  drawTearDrops(ctx, mpState, width, height, arState);        break;
    // hand
    case 'confetti-burst': drawConfettiBurst(ctx, mpState, width, height, arState); break;
    case 'electric-arc':   drawElectricArc(ctx, mpState, width, height);            break;
    case 'bubble-hands':   drawBubbleHands(ctx, mpState, width, height, arState);   break;
    // pose
    case 'angel-wings':  drawAngelWings(ctx, mpState, width, height);               break;
    case 'matrix-rain':  drawMatrixRain(ctx, mpState, width, height, arState);      break;
    case 'force-shield': drawForceShield(ctx, mpState, width, height, arState, dt); break;
    default: break;
  }

  // Disegna particelle extra (solo i sistemi che non hanno rendering custom)
  // bubbles e tears hanno rendering custom inline; confetti viene disegnato
  // dentro drawConfettiBurst. Il seguente garantisce cleanup:
  arState.particles.tears?.draw(ctx);   // fallback se draw() è definito
}
