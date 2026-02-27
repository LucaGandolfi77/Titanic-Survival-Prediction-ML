/* ================================================================
   THREE.JS — Particle Neural Network Background
   ================================================================ */

export function initThreeBackground() {
  if (typeof THREE === 'undefined') {
    console.warn('Three.js not loaded, skipping 3D background.');
    return;
  }

  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;

  const isMobile = window.innerWidth < 768;
  const PARTICLE_COUNT = isMobile ? 80 : 200;
  const CONNECTION_DIST = isMobile ? 120 : 150;
  const MAX_CONNECTIONS = isMobile ? 300 : 800;

  /* ── Renderer ─────────────────────────────────────────────── */
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  /* ── Scene & Camera ──────────────────────────────────────── */
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.z = 300;

  /* ── Particles ────────────────────────────────────────────── */
  const positions = new Float32Array(PARTICLE_COUNT * 3);
  const velocities = new Float32Array(PARTICLE_COUNT * 3);
  const halfW = window.innerWidth * 0.6;
  const halfH = window.innerHeight * 0.6;

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const i3 = i * 3;
    positions[i3]     = (Math.random() - 0.5) * halfW * 2;
    positions[i3 + 1] = (Math.random() - 0.5) * halfH * 2;
    positions[i3 + 2] = (Math.random() - 0.5) * 100;
    velocities[i3]     = (Math.random() - 0.5) * 0.4;
    velocities[i3 + 1] = (Math.random() - 0.5) * 0.4;
    velocities[i3 + 2] = (Math.random() - 0.5) * 0.2;
  }

  const pointsGeometry = new THREE.BufferGeometry();
  pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const pointsMaterial = new THREE.PointsMaterial({
    color: 0x6366F1,
    size: isMobile ? 2.5 : 3,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true,
  });

  const points = new THREE.Points(pointsGeometry, pointsMaterial);
  scene.add(points);

  /* ── Connection Lines ─────────────────────────────────────── */
  const linePositions = new Float32Array(MAX_CONNECTIONS * 6);
  const lineColors = new Float32Array(MAX_CONNECTIONS * 6);
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  lineGeometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

  const lineMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.3,
  });

  const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
  scene.add(lines);

  /* ── Mouse tracking ──────────────────────────────────────── */
  let mouseX = 0, mouseY = 0;
  const onPointerMove = (e) => {
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    mouseX = (clientX / window.innerWidth - 0.5) * 30;
    mouseY = (clientY / window.innerHeight - 0.5) * -30;
  };
  window.addEventListener('mousemove', onPointerMove, { passive: true });
  window.addEventListener('touchmove', onPointerMove, { passive: true });

  /* ── Animation ────────────────────────────────────────────── */
  let lastTime = 0;
  const FPS_INTERVAL = 1000 / 60;

  function animate(time) {
    requestAnimationFrame(animate);

    const delta = time - lastTime;
    if (delta < FPS_INTERVAL) return;
    lastTime = time - (delta % FPS_INTERVAL);

    const posAttr = pointsGeometry.getAttribute('position');
    const arr = posAttr.array;

    /* Move particles */
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;
      arr[i3]     += velocities[i3]     + Math.sin(time * 0.0005 + i) * 0.15;
      arr[i3 + 1] += velocities[i3 + 1] + Math.cos(time * 0.0005 + i * 0.7) * 0.15;
      arr[i3 + 2] += velocities[i3 + 2];

      /* Bounce at boundaries */
      if (Math.abs(arr[i3])     > halfW) velocities[i3]     *= -1;
      if (Math.abs(arr[i3 + 1]) > halfH) velocities[i3 + 1] *= -1;
      if (Math.abs(arr[i3 + 2]) > 50)    velocities[i3 + 2] *= -1;
    }
    posAttr.needsUpdate = true;

    /* Update connection lines */
    let lineIdx = 0;
    for (let i = 0; i < PARTICLE_COUNT && lineIdx < MAX_CONNECTIONS; i++) {
      for (let j = i + 1; j < PARTICLE_COUNT && lineIdx < MAX_CONNECTIONS; j++) {
        const i3 = i * 3, j3 = j * 3;
        const dx = arr[i3] - arr[j3];
        const dy = arr[i3 + 1] - arr[j3 + 1];
        const dz = arr[i3 + 2] - arr[j3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < CONNECTION_DIST) {
          const alpha = 1 - dist / CONNECTION_DIST;
          const l6 = lineIdx * 6;
          linePositions[l6]     = arr[i3];
          linePositions[l6 + 1] = arr[i3 + 1];
          linePositions[l6 + 2] = arr[i3 + 2];
          linePositions[l6 + 3] = arr[j3];
          linePositions[l6 + 4] = arr[j3 + 1];
          linePositions[l6 + 5] = arr[j3 + 2];
          /* Purple-ish vertex colors */
          lineColors[l6]     = 0.545 * alpha;
          lineColors[l6 + 1] = 0.361 * alpha;
          lineColors[l6 + 2] = 0.965 * alpha;
          lineColors[l6 + 3] = 0.545 * alpha;
          lineColors[l6 + 4] = 0.361 * alpha;
          lineColors[l6 + 5] = 0.965 * alpha;
          lineIdx++;
        }
      }
    }

    /* Clear unused segments */
    for (let k = lineIdx * 6; k < linePositions.length; k++) {
      linePositions[k] = 0;
      lineColors[k] = 0;
    }

    lineGeometry.attributes.position.needsUpdate = true;
    lineGeometry.attributes.color.needsUpdate = true;
    lineGeometry.setDrawRange(0, lineIdx * 2);

    /* Camera parallax */
    camera.position.x += (mouseX - camera.position.x) * 0.05;
    camera.position.y += (mouseY - camera.position.y) * 0.05;

    renderer.render(scene, camera);
  }

  animate(0);

  /* ── Resize ───────────────────────────────────────────────── */
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    }, 200);
  }, { passive: true });
}
