/* ============================================================
   Daisy Field — daisy.js
   SVG daisy factory with spatial grid collision awareness.
   
   Each daisy is an inline SVG with:
     • 8 radial ellipse petals (white/cream, occasional pink/yellow)
     • Yellow center with radial gradient
     • Green stem + small leaf
     • Random size/rotation/sway variation
   ============================================================ */

/* ---- Spatial grid for collision-aware placement ---- */
const _grid = {
  cells: new Map(),
  cellSize: 50,                         // px per grid cell

  /** Return the cell key for a point */
  key(x, y) {
    return `${Math.floor(x / this.cellSize)},${Math.floor(y / this.cellSize)}`;
  },

  /** Register a placed daisy's center */
  add(x, y, r) {
    const k = this.key(x, y);
    if (!this.cells.has(k)) this.cells.set(k, []);
    this.cells.get(k).push({ x, y, r });
  },

  /**
   * Check whether (x, y) with radius r overlaps any existing daisy
   * by more than 60 % of the smaller radius (allows partial overlap
   * for density, prevents exact stacking).
   */
  overlaps(x, y, r) {
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const entries = this.cells.get(`${cx + dx},${cy + dy}`);
        if (!entries) continue;
        for (const e of entries) {
          const dist = Math.hypot(e.x - x, e.y - y);
          const minR = Math.min(r, e.r);
          if (dist < minR * 0.6) return true;
        }
      }
    }
    return false;
  },

  /** Reset grid for a new session */
  clear() { this.cells.clear(); }
};

/* ---- Color variants ---- */
const PETAL_COLORS = {
  white:  { petal: '#FFFDE7', tint: '#F8E0E0' },
  pink:   { petal: '#F48FB1', tint: '#F06292' },
  yellow: { petal: '#FFF176', tint: '#FDD835' }
};

/**
 * Pick a random petal colour:
 * 80 % white, 10 % pink, 10 % yellow.
 */
function _pickColor() {
  const r = Math.random();
  if (r < 0.10) return PETAL_COLORS.pink;
  if (r < 0.20) return PETAL_COLORS.yellow;
  return PETAL_COLORS.white;
}

/* ---- Unique IDs for SVG gradients ---- */
let _gradId = 0;

/* ============================================================
   createDaisy(options)
   Returns an absolutely-positioned <div class="daisy"> wrapping
   an inline SVG.

   options:
     x, y        — CSS left / top in px (required)
     size        — base scale multiplier, default 1
     rotation    — base rotation in deg, default random
     color       — { petal, tint } or auto-picked
   ============================================================ */
function createDaisy(options = {}) {
  const size     = options.size     ?? (0.8 + Math.random() * 0.4);   // ±20 %
  const rotation = options.rotation ?? (Math.random() * 30 - 15);     // ±15 deg
  const color    = options.color    ?? _pickColor();
  const x        = options.x ?? 0;
  const y        = options.y ?? 0;

  const id = `dg${_gradId++}`;

  /* SVG viewBox is 60×80 — petals centred at (30, 32), stem below */
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 80" width="60" height="80">
  <defs>
    <radialGradient id="${id}">
      <stop offset="0%"   stop-color="#FDD835"/>
      <stop offset="100%" stop-color="#F9A825"/>
    </radialGradient>
  </defs>

  <!-- Stem -->
  <line x1="30" y1="44" x2="30" y2="74" stroke="#388E3C" stroke-width="3" stroke-linecap="round"/>
  <!-- Leaf -->
  <ellipse cx="38" cy="60" rx="7" ry="3.5" fill="#43A047" transform="rotate(-25 38 60)"/>

  <!-- 8 petals arranged radially -->
  ${Array.from({ length: 8 }, (_, i) => {
    const angle = i * 45;
    return `<ellipse cx="30" cy="14" rx="7" ry="15"
      fill="${color.petal}" stroke="${color.tint}" stroke-width="0.5"
      transform="rotate(${angle} 30 32)"/>`;
  }).join('\n  ')}

  <!-- Center -->
  <circle cx="30" cy="32" r="8" fill="url(#${id})"/>
</svg>`;

  /* Wrapper element */
  const el = document.createElement('div');
  el.className = 'daisy';
  el.innerHTML = svg;
  el.setAttribute('aria-hidden', 'true');

  /* Position & variation */
  const pxSize = 60 * size;
  el.style.left   = `${x - pxSize / 2}px`;
  el.style.top    = `${y - pxSize * 1.1}px`;   // anchor at stem base
  el.style.width  = `${pxSize}px`;
  el.style.height = `${pxSize * (80 / 60)}px`;

  el.style.setProperty('--base-rot', `${rotation}deg`);
  el.style.setProperty('--sway-dur', `${3 + Math.random() * 3}s`);
  el.style.setProperty('--sway-delay', `${-Math.random() * 4}s`);

  /* Scale the inner SVG to fill wrapper */
  const svgEl = el.querySelector('svg');
  svgEl.style.width  = '100%';
  svgEl.style.height = '100%';

  return { el, cx: x, cy: y, r: pxSize / 2 };
}

/**
 * spawnDaisy(container)
 * Creates a daisy at a random non-overlapping position inside `container`,
 * injects it into the DOM, and plays a GSAP bloom animation.
 * Returns null if no suitable position could be found after a few retries.
 */
function spawnDaisy(container) {
  const W = container.clientWidth;
  const H = container.clientHeight;
  const size = 0.8 + Math.random() * 0.4;
  const r = (60 * size) / 2;

  /* Try up to 8 random positions to avoid heavy overlap */
  let x, y, tries = 0;
  do {
    x = r + Math.random() * (W - 2 * r);
    y = r + Math.random() * (H - 2 * r);
    tries++;
  } while (_grid.overlaps(x, y, r) && tries < 8);

  _grid.add(x, y, r);

  const daisy = createDaisy({ x, y, size });
  container.appendChild(daisy.el);

  /* Bloom animation via GSAP */
  if (typeof gsap !== 'undefined') {
    gsap.from(daisy.el, {
      scale: 0,
      duration: 0.6,
      ease: 'elastic.out(1, 0.5)',
      onComplete() { daisy.el.classList.add('sway'); }
    });
  } else {
    daisy.el.classList.add('sway');
  }

  return daisy.el;
}

/**
 * createBouquet()
 * Returns a self-contained SVG element depicting a gathered bouquet
 * of ~8 daisies with leaves and a pink ribbon.
 */
function createBouquet() {
  const W = 240, H = 320;
  const stems = [];
  const flowers = [];

  /* Place ~8 daisy heads in a fan arrangement */
  const count = 8;
  for (let i = 0; i < count; i++) {
    const angle = -35 + (70 / (count - 1)) * i;      // fan span
    const rad   = (angle * Math.PI) / 180;
    const dist  = 80 + Math.random() * 30;
    const fx    = W / 2 + Math.sin(rad) * dist;
    const fy    = H / 2 - 50 - Math.cos(rad) * dist + Math.random() * 20;
    const col   = Math.random() < 0.15 ? PETAL_COLORS.pink
                : Math.random() < 0.15 ? PETAL_COLORS.yellow
                : PETAL_COLORS.white;
    const sz    = 18 + Math.random() * 6;
    const rot   = angle + (Math.random() * 10 - 5);

    stems.push(`<line x1="${W / 2}" y1="${H - 60}" x2="${fx}" y2="${fy + sz}"
      stroke="#388E3C" stroke-width="4" stroke-linecap="round"/>`);

    const petals = Array.from({ length: 8 }, (_, j) => {
      const a = j * 45;
      return `<ellipse cx="0" cy="${-sz * 0.65}" rx="${sz * 0.3}" ry="${sz * 0.55}"
        fill="${col.petal}" stroke="${col.tint}" stroke-width="0.5"
        transform="rotate(${a})"/>`;
    }).join('');

    flowers.push(`<g transform="translate(${fx},${fy}) rotate(${rot})">
      ${petals}
      <circle r="${sz * 0.3}" fill="#FDD835"/>
    </g>`);
  }

  /* A few leaves along the stems */
  const leaves = [
    `<ellipse cx="${W / 2 - 18}" cy="${H - 110}" rx="14" ry="6" fill="#43A047" transform="rotate(-30 ${W / 2 - 18} ${H - 110})"/>`,
    `<ellipse cx="${W / 2 + 20}" cy="${H - 130}" rx="14" ry="6" fill="#66BB6A" transform="rotate(25 ${W / 2 + 20} ${H - 130})"/>`,
    `<ellipse cx="${W / 2 - 10}" cy="${H - 160}" rx="12" ry="5" fill="#43A047" transform="rotate(-20 ${W / 2 - 10} ${H - 160})"/>`
  ];

  /* Ribbon */
  const ribbon = `
    <path d="M${W / 2 - 20},${H - 65} Q${W / 2},${H - 45} ${W / 2 + 20},${H - 65}
             L${W / 2 + 24},${H - 50} Q${W / 2},${H - 30} ${W / 2 - 24},${H - 50} Z"
          fill="#FF80AB"/>
    <path d="M${W / 2 - 8},${H - 48} Q${W / 2 - 18},${H - 20} ${W / 2 - 28},${H - 10}" 
          stroke="#FF80AB" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M${W / 2 + 8},${H - 48} Q${W / 2 + 18},${H - 20} ${W / 2 + 28},${H - 10}" 
          stroke="#FF80AB" stroke-width="5" fill="none" stroke-linecap="round"/>`;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  svg.setAttribute('aria-label', 'A bouquet of daisies');
  svg.innerHTML = `
    ${stems.join('\n')}
    ${leaves.join('\n')}
    ${ribbon}
    ${flowers.join('\n')}`;

  return svg;
}

/**
 * clearMeadow()
 * Reset the spatial grid so the next session starts fresh.
 */
function clearMeadow() {
  _grid.clear();
  _gradId = 0;
}

/* ---- Public API ---- */
window.DaisyFactory = { createDaisy, spawnDaisy, createBouquet, clearMeadow };
