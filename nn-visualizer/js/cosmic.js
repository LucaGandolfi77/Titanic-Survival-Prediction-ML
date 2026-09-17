const NODE_RADIUS = 0.15;
const STAR_COUNT = 2000;
const GLOW_SCALE = 2.5;

export function createCosmicRenderer(container) {
  let scene = null;
  let camera = null;
  let renderer = null;
  let controls = null;
  let animationId = null;
  let active = false;
  let nodes = [];
  let connections = [];
  let stars = null;
  let raycaster = null;
  let mouse = null;
  let onNodeClick = null;
  let THREE = null;

  function init(THREE_lib, OrbitControlsLib) {
    THREE = THREE_lib;
    const OrbitControls = OrbitControlsLib;

    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x071028, 0.002);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 2, 8);

    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    try {
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.3;
      controls.enableZoom = true;
      controls.minDistance = 2;
      controls.maxDistance = 30;
    } catch (e) {
      console.warn('OrbitControls failed:', e);
    }

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    createStars();

    renderer.domElement.addEventListener('click', onClick);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    window.addEventListener('resize', onResize);

    return true;
  }

  function createStars() {
    if (!THREE) return;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(STAR_COUNT * 3);
    const colors = new Float32Array(STAR_COUNT * 3);

    for (let i = 0; i < STAR_COUNT; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 100;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 100;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 100;

      const c = Math.random();
      colors[i * 3] = c > 0.7 ? 0.4 : 0.8;
      colors[i * 3 + 1] = c > 0.7 ? 0.6 : 0.9;
      colors[i * 3 + 2] = 1.0;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
    });

    stars = new THREE.Points(geometry, material);
    scene.add(stars);
  }

  function createGlowTexture(color = '#60a5fa') {
    if (!THREE) return null;
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.3, color);
    gradient.addColorStop(0.6, 'rgba(96, 165, 250, 0.3)');
    gradient.addColorStop(1, 'rgba(96, 165, 250, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
  }

  function clearScene() {
    if (!THREE) return;
    while (scene.children.length > 1) {
      const child = scene.children[scene.children.length - 1];
      scene.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    }
    nodes = [];
    connections = [];
  }

  function build(model, positions) {
    if (!THREE || !scene) return;
    clearScene();
    createStars();

    if (!positions || positions.length === 0) return;

    const accentColor = 0x60a5fa;
    const pinkColor = 0xfb7185;
    const glowTex = createGlowTexture('#60a5fa');
    const glowTexPink = createGlowTexture('#fb7185');

    // Draw connections first (behind nodes)
    for (let L = 1; L < positions.length; L++) {
      const W = model.weights[L - 1];
      if (!W) continue;
      for (let to = 0; to < W.length; to++) {
        for (let from = 0; from < W[to].length; from++) {
          const w = W[to][from];
          const p1 = positions[L - 1][from];
          const p2 = positions[L][to];
          if (!p1 || !p2) continue;

          const geometry = new THREE.BufferGeometry();
          const vertices = new Float32Array([
            p1.x / 50, -p1.y / 50, 0,
            p2.x / 50, -p2.y / 50, 0,
          ]);
          geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

          const color = w > 0 ? accentColor : pinkColor;
          const intensity = Math.min(Math.abs(w) * 2, 1);

          const material = new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: 0.2 + intensity * 0.6,
            linewidth: 1,
          });

          const line = new THREE.Line(geometry, material);
          scene.add(line);
          connections.push(line);
        }
      }
    }

    // Draw nodes
    for (let i = 0; i < positions.length; i++) {
      for (let n = 0; n < positions[i].length; n++) {
        const p = positions[i][n];
        const isInput = i === 0;
        const color = isInput ? 0x111827 : (i % 2 === 0 ? accentColor : pinkColor);
        const glowTexNode = isInput ? glowTex : (i % 2 === 0 ? glowTex : glowTexPink);

        // Core sphere
        const sphereGeo = new THREE.SphereGeometry(NODE_RADIUS, 16, 16);
        const sphereMat = new THREE.MeshBasicMaterial({
          color: isInput ? 0x0f1724 : color,
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.set(p.x / 50, -p.y / 50, 0);
        sphere.userData = { layer: i, idx: n, isNode: true };
        scene.add(sphere);
        nodes.push(sphere);

        // Glow sprite
        if (glowTexNode) {
          const spriteMat = new THREE.SpriteMaterial({
            map: glowTexNode,
            color: color,
            transparent: true,
            opacity: isInput ? 0.3 : 0.7,
            blending: THREE.AdditiveBlending,
          });
          const sprite = new THREE.Sprite(spriteMat);
          sprite.scale.set(GLOW_SCALE, GLOW_SCALE, 1);
          sprite.position.copy(sphere.position);
          scene.add(sprite);
        }
      }
    }
  }

  function onClick(event) {
    if (!raycaster || !camera || !mouse || !onNodeClick) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(nodes);

    if (intersects.length > 0) {
      const node = intersects[0].object;
      onNodeClick(node.userData.layer, node.userData.idx);
    }
  }

  function onMouseMove(event) {
    if (!raycaster || !camera || !mouse) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(nodes);
    renderer.domElement.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
  }

  function onResize() {
    if (!camera || !renderer) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  function animate(time) {
    if (!active) return;
    animationId = requestAnimationFrame(animate);

    if (controls) controls.update();
    if (stars) stars.rotation.y += 0.0002;

    // Subtle node floating animation
    const t = time * 0.001;
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const offset = i * 0.7;
      node.position.y += Math.sin(t * 2 + offset) * 0.0003;
    }

    renderer.render(scene, camera);
  }

  function start(model, positions, nodeClickCallback) {
    if (active) return;
    onNodeClick = nodeClickCallback;
    active = true;
    build(model, positions);
    animate(0);
  }

  function stop() {
    active = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  function dispose() {
    stop();
    clearScene();
    if (renderer && renderer.domElement && renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
    if (renderer) {
      renderer.dispose();
      renderer = null;
    }
    camera = null;
    scene = null;
    controls = null;
    stars = null;
    raycaster = null;
    mouse = null;
    onNodeClick = null;
    THREE = null;
  }

  function isActive() {
    return active;
  }

  return {
    init, start, stop, dispose, isActive, onResize,
  };
}
