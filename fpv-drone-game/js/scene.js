/**
 * Three.js scene setup — DRONE STRIKE
 * @module scene
 */

/**
 * Create renderer, scene, camera. Attach to #game-canvas.
 * @returns {{ renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera }}
 */
const THREE = globalThis.THREE;

export function initScene() {
  const canvas = document.getElementById('game-canvas');

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);
  scene.fog = new THREE.FogExp2(0x1a1a2e, 0.002);

  const camera = new THREE.PerspectiveCamera(
    90,
    window.innerWidth / window.innerHeight,
    0.05,
    2000
  );
  camera.position.set(0, 20, 0);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  return { renderer, scene, camera };
}

/**
 * Add lights to the scene.
 */
export function initLights(scene) {
  // Hemisphere light — sky / ground
  const hemi = new THREE.HemisphereLight(0x87ceeb, 0x3d5a2a, 0.4);
  scene.add(hemi);

  // Directional (sun)
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(100, 200, 50);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 500;
  sun.shadow.camera.left = -200;
  sun.shadow.camera.right = 200;
  sun.shadow.camera.top = 200;
  sun.shadow.camera.bottom = -200;
  scene.add(sun);

  // Ambient fill
  const amb = new THREE.AmbientLight(0x404060, 0.3);
  scene.add(amb);
}
