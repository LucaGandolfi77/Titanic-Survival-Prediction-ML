const Visualizer = (() => {
  let canvas, ctx, bgCanvas, bgCtx;
  let particles = [];
  let lumina = { x: 0, y: 0, targetX: 0, targetY: 0, glow: 0, size: 30 };
  let fireParticles = [];
  let rainDrops = [];
  let neuralNodes = [];
  let neuralConnections = [];
  let lossChartData = [];
  let lossChartAnim = 0;
  let time = 0;
  let isTraining = false;
  let trainingProgress = 0;
  let trainingParticles = [];
  let bgGradient = null;
  let needsBgRedraw = true;
  let cachedWidth = 0;
  let cachedHeight = 0;
  const FPS = 60;
  let lastFrame = 0;
  let accumulator = 0;
  const FRAME_TIME = 1000 / FPS;

  function init(canvasEl) {
    canvas = canvasEl;
    ctx = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', () => { resize(); needsBgRedraw = true; });
    initParticles();
    initFire();
    initRain();
    initNeural();
    requestAnimationFrame(loop);
  }

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (w === cachedWidth && h === cachedHeight) return;
    cachedWidth = w; cachedHeight = h;
    canvas.width = w; canvas.height = h;
    bgCanvas = document.createElement('canvas');
    bgCanvas.width = w; bgCanvas.height = h;
    bgCtx = bgCanvas.getContext('2d');
    needsBgRedraw = true;
  }

  function initParticles() {
    particles = [];
    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * (cachedWidth || window.innerWidth),
        y: Math.random() * (cachedHeight || window.innerHeight),
        size: Math.random() * 2 + 0.5,
        speedX: (Math.random() - 0.5) * 0.3,
        speedY: (Math.random() - 0.5) * 0.3 - 0.1,
        opacity: Math.random() * 0.5 + 0.2,
        life: Math.random() * 200,
        maxLife: 200 + Math.random() * 100,
        hue: Math.random() > 0.7 ? 35 : 220,
      });
    }
  }

  function initFire() {
    fireParticles = [];
    for (let i = 0; i < 30; i++) {
      fireParticles.push({
        x: 0.2 + Math.random() * 0.1, y: 0.9 + Math.random() * 0.05,
        vx: (Math.random() - 0.5) * 0.003, vy: -Math.random() * 0.008 - 0.004,
        size: Math.random() * 4 + 2, life: Math.random() * 100,
        maxLife: 80 + Math.random() * 60,
      });
    }
  }

  function initRain() {
    rainDrops = [];
    for (let i = 0; i < 80; i++) {
      rainDrops.push({
        x: Math.random(), y: Math.random(),
        speed: Math.random() * 0.01 + 0.005,
        length: Math.random() * 20 + 10,
        opacity: Math.random() * 0.3 + 0.1,
      });
    }
  }

  function initNeural() {
    neuralNodes = [];
    const layers = [3, 5, 5, 4, 2];
    const w = cachedWidth || window.innerWidth;
    const h = cachedHeight || window.innerHeight;
    layers.forEach((count, layer) => {
      for (let i = 0; i < count; i++) {
        neuralNodes.push({
          x: 0.1 + (layer / (layers.length - 1)) * 0.8,
          y: (i + 1) / (count + 1),
          activation: 0,
          targetActivation: Math.random() * 0.3,
          layer,
        });
      }
    });
    neuralConnections = [];
    for (let l = 0; l < layers.length - 1; l++) {
      const from = neuralNodes.filter(n => n.layer === l);
      const to = neuralNodes.filter(n => n.layer === l + 1);
      from.forEach(f => {
        to.forEach(t => {
          neuralConnections.push({ from, to, weight: Math.random() * 0.5 + 0.2 });
        });
      });
    }
  }

  function startTraining() { isTraining = true; trainingProgress = 0; trainingParticles = []; lossChartData = []; lossChartAnim = 0; }
  function stopTraining() { isTraining = false; }
  function setTrainingProgress(p) { trainingProgress = p; }
  function setLuminaTarget(x, y) { lumina.targetX = x; lumina.targetY = y; }
  function setLuminaSize(s) { lumina.size = s; }
  function setNeedsBgRedraw() { needsBgRedraw = true; }

  function drawBackground(w, h) {
    if (!needsBgRedraw && bgGradient) return;
    const grad = bgCtx.createRadialGradient(w * 0.5, h * 0.6, 0, w * 0.5, h * 0.6, w * 0.7);
    grad.addColorStop(0, '#1a1a2e');
    grad.addColorStop(0.5, '#16213e');
    grad.addColorStop(1, '#0a0e1a');
    bgCtx.fillStyle = grad;
    bgCtx.fillRect(0, 0, w, h);
    needsBgRedraw = false;
  }

  function updateParticles() {
    particles.forEach(p => {
      p.x += p.speedX;
      p.y += p.speedY;
      p.life++;
      if (p.life > p.maxLife || p.x < -10 || p.x > cachedWidth + 10 || p.y < -10 || p.y > cachedHeight + 10) {
        p.x = Math.random() * cachedWidth;
        p.y = cachedHeight + 10;
        p.life = 0;
        p.opacity = Math.random() * 0.5 + 0.2;
      }
      const fadeIn = Math.min(1, p.life / 30);
      const fadeOut = Math.max(0, 1 - (p.life - p.maxLife + 50) / 50);
      p.currentOpacity = p.opacity * fadeIn * fadeOut;
    });
  }

  function updateFire() {
    const w = cachedWidth, h = cachedHeight;
    fireParticles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      p.life++;
      if (p.life > p.maxLife) {
        p.life = 0;
        p.x = 0.2 + Math.random() * 0.1;
        p.y = 0.9 + Math.random() * 0.05;
      }
    });
  }

  function updateNeural() {
    neuralNodes.forEach(n => {
      n.activation += (n.targetActivation - n.activation) * 0.02;
      if (isTraining) {
        n.targetActivation = Math.random() * trainingProgress + 0.1;
      } else {
        n.targetActivation = Math.sin(time * 0.5 + n.x * 10 + n.y * 5) * 0.3 + 0.2;
      }
    });
    neuralConnections.forEach(c => {
      c.weight = c.from.activation * c.to.activation * 0.8;
    });
  }

  function updateLuminaPos() {
    lumina.x += (lumina.targetX - lumina.x) * 0.03;
    lumina.y += (lumina.targetY - lumina.y) * 0.03;
    lumina.glow = 0.5 + Math.sin(time * 2) * 0.2;
  }

  function updateTrainingParticles() {
    if (!isTraining) { trainingParticles = []; return; }
    if (Math.random() < 0.4) {
      const fromIdx = Math.floor(Math.random() * neuralNodes.length);
      const toIdx = Math.floor(Math.random() * neuralNodes.length);
      const from = neuralNodes[fromIdx];
      const to = neuralNodes[toIdx];
      if (from && to && from !== to) {
        trainingParticles.push({
          x: from.x, y: from.y,
          targetX: to.x, targetY: to.y,
          progress: 0, speed: 0.01 + Math.random() * 0.02,
          size: 2 + Math.random() * 2,
          hue: Math.random() > 0.5 ? 40 : 340,
        });
      }
    }
    trainingParticles = trainingParticles.filter(p => {
      const dx = p.targetX - p.x;
      const dy = p.targetY - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      p.x += dx * p.speed;
      p.y += dy * p.speed;
      p.progress += p.speed;
      return dist > 0.01;
    });
    if (lossChartData.length < 60) {
      const loss = Math.max(0.01, 2.5 * Math.pow(1 - trainingProgress, 2) + Math.random() * 0.05 * (1 - trainingProgress));
      lossChartData.push(loss);
    }
  }

  function drawParticles() {
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.hue === 35
        ? `rgba(255, 212, 111, ${p.currentOpacity})`
        : `rgba(160, 180, 255, ${p.currentOpacity * 0.5})`;
      ctx.fill();
    });
  }

  function drawRain(w, h) {
    rainDrops.forEach(d => {
      d.y += d.speed;
      if (d.y > 1) d.y = -0.05;
      ctx.beginPath();
      ctx.moveTo(d.x * w, d.y * h);
      ctx.lineTo(d.x * w, d.y * h + d.length);
      ctx.strokeStyle = `rgba(180, 200, 255, ${d.opacity})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  function drawFire(w, h) {
    const px = w * 0.2; const py = h * 0.88;
    fireParticles.forEach(p => {
      const x = px + p.x * w * 0.6;
      const y = py + p.y * h * 0.1;
      const alpha = 1 - (p.life / p.maxLife);
      ctx.beginPath();
      ctx.arc(x, y, p.size * alpha, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, ${Math.floor(100 + alpha * 100)}, ${Math.floor(alpha * 50)}, ${alpha * 0.6})`;
      ctx.fill();
    });
    if (state) {
      const glow = ctx.createRadialGradient(px + w * 0.3, py, 0, px + w * 0.3, py, w * 0.3);
      glow.addColorStop(0, 'rgba(255, 107, 53, 0.08)');
      glow.addColorStop(1, 'rgba(255, 107, 53, 0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, w, h);
    }
  }

  function drawLumina(w, h) {
    const x = lumina.x * w; const y = lumina.y * h; const s = lumina.size;
    const glow = lumina.glow;
    // Outer glow
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, s * 4);
    gradient.addColorStop(0, `rgba(255, 212, 111, ${glow * 0.3})`);
    gradient.addColorStop(0.3, `rgba(255, 212, 111, ${glow * 0.1})`);
    gradient.addColorStop(1, 'rgba(255, 212, 111, 0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, s * 4, 0, Math.PI * 2);
    ctx.fill();
    // Body
    const bodyGrad = ctx.createRadialGradient(x, y, 0, x, y, s * 1.5);
    bodyGrad.addColorStop(0, `rgba(255, 230, 170, ${glow})`);
    bodyGrad.addColorStop(0.5, `rgba(255, 212, 111, ${glow * 0.7})`);
    bodyGrad.addColorStop(1, 'rgba(255, 212, 111, 0)');
    ctx.fillStyle = bodyGrad;
    ctx.beginPath();
    ctx.arc(x, y, s * 1.5, 0, Math.PI * 2);
    ctx.fill();
    // Core
    ctx.fillStyle = `rgba(255, 250, 240, ${glow * 0.9})`;
    ctx.beginPath();
    ctx.arc(x, y, s * 0.3, 0, Math.PI * 2);
    ctx.fill();
    // "Eyes"
    const eyeOff = s * 0.25;
    const eyeSize = s * 0.06;
    ctx.fillStyle = `rgba(233, 69, 96, ${glow * 0.8})`;
    ctx.beginPath();
    ctx.arc(x - eyeOff, y - eyeOff * 0.3, eyeSize, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x + eyeOff, y - eyeOff * 0.3, eyeSize, 0, Math.PI * 2);
    ctx.fill();
    // Eye highlights
    ctx.fillStyle = `rgba(255, 255, 255, ${glow * 0.5})`;
    ctx.beginPath();
    ctx.arc(x - eyeOff + 1, y - eyeOff * 0.3 - 1, eyeSize * 0.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x + eyeOff + 1, y - eyeOff * 0.3 - 1, eyeSize * 0.4, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawNeural() {
    // Connections
    neuralConnections.forEach(c => {
      if (c.weight > 0.05) {
        const x1 = c.from.x * cachedWidth, y1 = c.from.y * cachedHeight;
        const x2 = c.to.x * cachedWidth, y2 = c.to.y * cachedHeight;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `rgba(255, 212, 111, ${c.weight * 0.4})`;
        ctx.lineWidth = c.weight * 2;
        ctx.stroke();
      }
    });
    // Nodes
    neuralNodes.forEach(n => {
      const x = n.x * cachedWidth, y = n.y * cachedHeight;
      const r = 4 + n.activation * 6;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      const alpha = 0.3 + n.activation * 0.7;
      ctx.fillStyle = `rgba(255, 212, 111, ${alpha})`;
      ctx.fill();
      if (n.activation > 0.3) {
        const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
        g.addColorStop(0, `rgba(255, 212, 111, ${n.activation * 0.2})`);
        g.addColorStop(1, 'rgba(255, 212, 111, 0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, r * 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  function drawTrainingParticles() {
    trainingParticles.forEach(p => {
      const x = p.x * cachedWidth; const y = p.y * cachedHeight;
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, 0.9)`;
      ctx.fill();
    });
  }

  function drawLossChart(w, h) {
    if (!lossChartData || lossChartData.length < 2) return;
    const chartW = w * 0.8; const chartH = 120;
    const chartX = w * 0.1; const chartY = h - chartH - 30;
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.roundRect(chartX, chartY, chartW, chartH, 8);
    ctx.fill();
    // Line
    ctx.beginPath();
    lossChartData.forEach((val, i) => {
      const x = chartX + (i / (lossChartData.length - 1)) * chartW;
      const y = chartY + (1 - val / 3) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#ffd46f';
    ctx.lineWidth = 2;
    ctx.stroke();
    // Fill under
    const last = lossChartData.length - 1;
    ctx.lineTo(chartX + chartW, chartY + chartH);
    ctx.lineTo(chartX, chartY + chartH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 212, 111, 0.08)';
    ctx.fill();
  }

  let state = null;
  function setGameStateRef(s) { state = s; }

  function loop(timestamp) {
    const elapsed = timestamp - lastFrame;
    lastFrame = timestamp;
    accumulator += elapsed;

    while (accumulator >= FRAME_TIME) {
      time += 1 / FPS;
      const w = cachedWidth, h = cachedHeight;

      updateParticles();
      updateFire();
      updateNeural();
      updateLuminaPos();
      if (isTraining) {
        updateTrainingParticles();
        trainingProgress = Math.min(1, trainingProgress + 0.001);
      }

      // Draw
      drawBackground(w, h);
      ctx.drawImage(bgCanvas, 0, 0);
      drawRain(w, h);
      drawParticles();
      drawFire(w, h);
      drawNeural();
      drawTrainingParticles();
      drawLumina(w, h);
      if (isTraining && trainingProgress > 0.1) {
        drawLossChart(w, h);
      }

      accumulator -= FRAME_TIME;
    }

    requestAnimationFrame(loop);
  }

  return { init, startTraining, stopTraining, setTrainingProgress, setLuminaTarget, setLuminaSize, setNeedsBgRedraw, setGameStateRef };
})();
