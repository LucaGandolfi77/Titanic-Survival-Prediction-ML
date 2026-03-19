/* ============================================================
   Daisy Field — confetti.js
   Lightweight canvas-based confetti in pastel nature colours.
   
   Usage:
     Confetti.start(canvasElement)   — begin raining confetti
     Confetti.stop()                 — fade out and clean up
   ============================================================ */

const Confetti = (() => {
  const COLORS = ['#FDD835', '#F48FB1', '#FFFDE7', '#A5D6A7', '#FF80AB', '#FFF176'];
  const COUNT  = 120;
  const GRAVITY = 0.06;

  let canvas, ctx, pieces, raf, running;

  /** Create a single confetti piece */
  function _piece() {
    return {
      x:    Math.random() * canvas.width,
      y:    -10 - Math.random() * canvas.height * 0.5,
      w:    4 + Math.random() * 6,
      h:    6 + Math.random() * 8,
      rot:  Math.random() * 360,
      dRot: (Math.random() - 0.5) * 8,
      dx:   (Math.random() - 0.5) * 1.5,
      dy:   1.5 + Math.random() * 2.5,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      opacity: 1
    };
  }

  /** Animation loop */
  function _draw() {
    if (!running) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let alive = 0;
    for (const p of pieces) {
      p.x   += p.dx;
      p.y   += p.dy;
      p.dy  += GRAVITY;
      p.rot += p.dRot;

      /* Fade out when nearing bottom */
      if (p.y > canvas.height * 0.85) {
        p.opacity -= 0.015;
      }
      if (p.opacity <= 0) continue;
      alive++;

      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate((p.rot * Math.PI) / 180);
      ctx.globalAlpha = p.opacity;
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    }

    if (alive > 0) {
      raf = requestAnimationFrame(_draw);
    } else {
      _cleanup();
    }
  }

  function _cleanup() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /** Start confetti on the given <canvas> */
  function start(cvs) {
    canvas = cvs;
    ctx = canvas.getContext('2d');
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    pieces  = Array.from({ length: COUNT }, _piece);
    running = true;
    _draw();
  }

  /** Trigger fade-out */
  function stop() {
    if (!pieces) return;
    for (const p of pieces) p.opacity = Math.min(p.opacity, 0.4);
  }

  return { start, stop };
})();

window.Confetti = Confetti;
