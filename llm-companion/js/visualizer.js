const Visualizer = (() => {
  let canvas, ctx, bgCanvas, bgCtx;
  let particles = [], fireParticles = [], rainDrops = [], neuralNodes = [], neuralConnections = [];
  let lossChartData = [], trainingParticles = [];
  let lumina = { x: 0.5, y: 0.5, targetX: 0.5, targetY: 0.5, glow: 0.5, size: 30 };
  let time = 0, isTraining = false, trainingProgress = 0;
  let needsBgRedraw = true, cachedW = 0, cachedH = 0;
  const FRAME_TIME = 1000 / 60;
  let lastFrame = 0, accumulator = 0;

  function init(canvasEl) {
    canvas = canvasEl;
    ctx = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', () => { resize(); });
    initParticles(); initFire(); initRain(); initNeural();
    requestAnimationFrame(loop);
  }

  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    if (w === cachedW && h === cachedH) return;
    cachedW = w; cachedH = h;
    canvas.width = w; canvas.height = h;
    bgCanvas = document.createElement('canvas');
    bgCanvas.width = w; bgCanvas.height = h;
    bgCtx = bgCanvas.getContext('2d');
    needsBgRedraw = true;
    initNeural();
  }

  function initParticles() {
    particles = [];
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * cachedW, y: Math.random() * cachedH,
        size: Math.random() * 2 + 0.5, speedX: (Math.random() - 0.5) * 0.3, speedY: (Math.random() - 0.5) * 0.3 - 0.1,
        opacity: Math.random() * 0.5 + 0.2, life: Math.random() * 200, maxLife: 200 + Math.random() * 100,
        hue: Math.random() > 0.7 ? 35 : 220, currentOpacity: 0.3,
      });
    }
  }

  function initFire() {
    fireParticles = [];
    for (let i = 0; i < 25; i++) {
      fireParticles.push({ x: 0.2 + Math.random() * 0.1, y: 0.9 + Math.random() * 0.05, vx: (Math.random() - 0.5) * 0.003, vy: -Math.random() * 0.008 - 0.004, size: Math.random() * 4 + 2, life: Math.random() * 100, maxLife: 80 + Math.random() * 60 });
    }
  }

  function initRain() {
    rainDrops = [];
    for (let i = 0; i < 60; i++) {
      rainDrops.push({ x: Math.random(), y: Math.random(), speed: Math.random() * 0.01 + 0.005, length: Math.random() * 20 + 10, opacity: Math.random() * 0.3 + 0.1 });
    }
  }

  function initNeural() {
    neuralNodes = [];
    const layers = [3, 5, 5, 4, 2];
    const w = cachedW || window.innerWidth, h = cachedH || window.innerHeight;
    layers.forEach((count, layer) => {
      for (let i = 0; i < count; i++) {
        neuralNodes.push({ x: 0.1 + (layer / (layers.length - 1)) * 0.8, y: (i + 1) / (count + 1), activation: 0, targetActivation: Math.random() * 0.3, layer });
      }
    });
    neuralConnections = [];
    for (let l = 0; l < layers.length - 1; l++) {
      neuralNodes.filter(n => n.layer === l).forEach(f => {
        neuralNodes.filter(n => n.layer === l + 1).forEach(t => {
          neuralConnections.push({ from: f, to: t, weight: Math.random() * 0.5 + 0.2 });
        });
      });
    }
  }

  function startTraining() { isTraining = true; trainingProgress = 0; trainingParticles = []; lossChartData = []; }
  function stopTraining() { isTraining = false; }
  function setTrainingProgress(p) { trainingProgress = p; }
  function setLuminaTarget(x, y) { lumina.targetX = x; lumina.targetY = y; }
  function setNeedsBgRedraw() { needsBgRedraw = true; }

  function tick(dt) {
    time += dt / 1000;
    const w = cachedW, h = cachedH;
    if (!w || !h) return;

    // Update particles
    particles.forEach(p => {
      p.x += p.speedX; p.y += p.speedY; p.life++;
      if (p.life > p.maxLife || p.x < -10 || p.x > w + 10 || p.y < -10 || p.y > h + 10) {
        p.x = Math.random() * w; p.y = h + 10; p.life = 0; p.opacity = Math.random() * 0.5 + 0.2;
      }
      const fadeIn = Math.min(1, p.life / 30);
      const fadeOut = Math.max(0, 1 - (p.life - p.maxLife + 50) / 50);
      p.currentOpacity = p.opacity * fadeIn * fadeOut;
    });

    // Rain
    rainDrops.forEach(d => { d.y += d.speed; if (d.y > 1) d.y = -0.05; });

    // Fire
    fireParticles.forEach(p => { p.x += p.vx; p.y += p.vy; p.life++; if (p.life > p.maxLife) { p.life = 0; p.x = 0.2 + Math.random() * 0.1; p.y = 0.9 + Math.random() * 0.05; } });

    // Neural
    neuralNodes.forEach(n => {
      n.activation += (n.targetActivation - n.activation) * 0.02;
      n.targetActivation = isTraining ? Math.random() * trainingProgress + 0.1 : Math.sin(time * 0.5 + n.x * 10 + n.y * 5) * 0.3 + 0.2;
    });
    neuralConnections.forEach(c => { c.weight = c.from.activation * c.to.activation * 0.8; });

    // Lumina pos
    lumina.x += (lumina.targetX - lumina.x) * 0.03;
    lumina.y += (lumina.targetY - lumina.y) * 0.03;
    lumina.glow = 0.5 + Math.sin(time * 2) * 0.2;

    // Training particles
    if (isTraining) {
      if (Math.random() < 0.3 && neuralNodes.length > 2) {
        const from = neuralNodes[Math.floor(Math.random() * neuralNodes.length)];
        const to = neuralNodes[Math.floor(Math.random() * neuralNodes.length)];
        if (from !== to) trainingParticles.push({ x: from.x, y: from.y, targetX: to.x, targetY: to.y, progress: 0, speed: 0.01 + Math.random() * 0.02, size: 2 + Math.random() * 2, hue: Math.random() > 0.5 ? 40 : 340 });
      }
      trainingParticles = trainingParticles.filter(p => {
        const dx = p.targetX - p.x, dy = p.targetY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        p.x += dx * p.speed; p.y += dy * p.speed; p.progress += p.speed;
        return dist > 0.01;
      });
      trainingProgress = Math.min(1, trainingProgress + 0.001);
      if (lossChartData.length < 60) {
        lossChartData.push(Math.max(0.01, 2.5 * Math.pow(1 - trainingProgress, 2) + Math.random() * 0.05 * (1 - trainingProgress)));
      }
    }

    // Draw
    if (needsBgRedraw) {
      const grad = bgCtx.createRadialGradient(w * 0.5, h * 0.6, 0, w * 0.5, h * 0.6, w * 0.7);
      grad.addColorStop(0, '#1a1a2e'); grad.addColorStop(0.5, '#16213e'); grad.addColorStop(1, '#0a0e1a');
      bgCtx.fillStyle = grad; bgCtx.fillRect(0, 0, w, h);
      needsBgRedraw = false;
    }
    ctx.drawImage(bgCanvas, 0, 0);

    // Rain
    rainDrops.forEach(d => {
      ctx.beginPath(); ctx.moveTo(d.x * w, d.y * h); ctx.lineTo(d.x * w, d.y * h + d.length);
      ctx.strokeStyle = `rgba(180, 200, 255, ${d.opacity})`; ctx.lineWidth = 1; ctx.stroke();
    });

    // Particles
    particles.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.hue === 35 ? `rgba(255, 212, 111, ${p.currentOpacity})` : `rgba(160, 180, 255, ${p.currentOpacity * 0.5})`;
      ctx.fill();
    });

    // Fire
    const px = w * 0.2, py = h * 0.88;
    fireParticles.forEach(p => {
      const x = px + p.x * w * 0.6, y = py + p.y * h * 0.1;
      const alpha = 1 - (p.life / p.maxLife);
      ctx.beginPath(); ctx.arc(x, y, p.size * alpha, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, ${Math.floor(100 + alpha * 100)}, ${Math.floor(alpha * 50)}, ${alpha * 0.6})`;
      ctx.fill();
    });
    const fireGlow = ctx.createRadialGradient(px + w * 0.3, py, 0, px + w * 0.3, py, w * 0.3);
    fireGlow.addColorStop(0, 'rgba(255, 107, 53, 0.06)'); fireGlow.addColorStop(1, 'rgba(255, 107, 53, 0)');
    ctx.fillStyle = fireGlow; ctx.fillRect(0, 0, w, h);

    // Neural connections
    neuralConnections.forEach(c => {
      if (c.weight > 0.05) {
        const x1 = c.from.x * w, y1 = c.from.y * h, x2 = c.to.x * w, y2 = c.to.y * h;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
        ctx.strokeStyle = `rgba(255, 212, 111, ${c.weight * 0.4})`; ctx.lineWidth = c.weight * 2; ctx.stroke();
      }
    });

    // Neural nodes
    neuralNodes.forEach(n => {
      const x = n.x * w, y = n.y * h, r = 4 + n.activation * 6;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 212, 111, ${0.3 + n.activation * 0.7})`; ctx.fill();
      if (n.activation > 0.3) {
        const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
        g.addColorStop(0, `rgba(255, 212, 111, ${n.activation * 0.2})`); g.addColorStop(1, 'rgba(255, 212, 111, 0)');
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 3, 0, Math.PI * 2); ctx.fill();
      }
    });

    // Training particles
    trainingParticles.forEach(p => {
      const x = p.x * w, y = p.y * h;
      ctx.beginPath(); ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, 0.9)`; ctx.fill();
    });

    // Lumina
    const lx = lumina.x * w, ly = lumina.y * h, s = lumina.size, glow = lumina.glow;
    const lg = ctx.createRadialGradient(lx, ly, 0, lx, ly, s * 4);
    lg.addColorStop(0, `rgba(255, 212, 111, ${glow * 0.3})`); lg.addColorStop(0.3, `rgba(255, 212, 111, ${glow * 0.1})`); lg.addColorStop(1, 'rgba(255, 212, 111, 0)');
    ctx.fillStyle = lg; ctx.beginPath(); ctx.arc(lx, ly, s * 4, 0, Math.PI * 2); ctx.fill();
    const bg2 = ctx.createRadialGradient(lx, ly, 0, lx, ly, s * 1.5);
    bg2.addColorStop(0, `rgba(255, 230, 170, ${glow})`); bg2.addColorStop(0.5, `rgba(255, 212, 111, ${glow * 0.7})`); bg2.addColorStop(1, 'rgba(255, 212, 111, 0)');
    ctx.fillStyle = bg2; ctx.beginPath(); ctx.arc(lx, ly, s * 1.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(255, 250, 240, ${glow * 0.9})`; ctx.beginPath(); ctx.arc(lx, ly, s * 0.3, 0, Math.PI * 2); ctx.fill();
    const eo = s * 0.25, es = s * 0.06;
    ctx.fillStyle = `rgba(233, 69, 96, ${glow * 0.8})`;
    ctx.beginPath(); ctx.arc(lx - eo, ly - eo * 0.3, es, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(lx + eo, ly - eo * 0.3, es, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(255, 255, 255, ${glow * 0.5})`;
    ctx.beginPath(); ctx.arc(lx - eo + 1, ly - eo * 0.3 - 1, es * 0.4, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(lx + eo + 1, ly - eo * 0.3 - 1, es * 0.4, 0, Math.PI * 2); ctx.fill();

    // Loss chart
    if (isTraining && lossChartData.length > 2) {
      const cw = w * 0.8, ch = 100, cx = w * 0.1, cy = h - ch - 20;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'; ctx.beginPath(); ctx.roundRect(cx, cy, cw, ch, 8); ctx.fill();
      ctx.beginPath();
      lossChartData.forEach((val, i) => {
        const x = cx + (i / (lossChartData.length - 1)) * cw, y = cy + (1 - val / 3) * ch;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = '#ffd46f'; ctx.lineWidth = 2; ctx.stroke();
      const last = lossChartData.length - 1;
      ctx.lineTo(cx + cw, cy + ch); ctx.lineTo(cx, cy + ch); ctx.closePath();
      ctx.fillStyle = 'rgba(255, 212, 111, 0.08)'; ctx.fill();
    }
  }

  function loop(timestamp) {
    const elapsed = Math.min(timestamp - lastFrame, 50);
    lastFrame = timestamp;
    accumulator += elapsed;
    while (accumulator >= FRAME_TIME) {
      tick(FRAME_TIME);
      accumulator -= FRAME_TIME;
    }
    requestAnimationFrame(loop);
  }

  return { init, startTraining, stopTraining, setTrainingProgress, setLuminaTarget, setNeedsBgRedraw };
})();
