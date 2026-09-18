import * as THREE from 'three';

let renderer, scene, camera;

export function initScene(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x87ceeb);
  scene.fog = new THREE.Fog(0x87ceeb, 80, 250);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.5, 500);
  camera.position.set(0, 12, 18);
  camera.lookAt(0, 0, 5);

  const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x556b2f, 0.6);
  scene.add(hemiLight);

  const sunLight = new THREE.DirectionalLight(0xfff4e0, 1.2);
  sunLight.position.set(20, 40, 30);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 2048;
  sunLight.shadow.mapSize.height = 2048;
  sunLight.shadow.camera.near = 0.5;
  sunLight.shadow.camera.far = 150;
  sunLight.shadow.camera.left = -40;
  sunLight.shadow.camera.right = 40;
  sunLight.shadow.camera.top = 40;
  sunLight.shadow.camera.bottom = -40;
  sunLight.shadow.bias = -0.0005;
  scene.add(sunLight);

  const fillLight = new THREE.DirectionalLight(0xb4d8f0, 0.3);
  fillLight.position.set(-10, 15, -10);
  scene.add(fillLight);

  const ambientLight = new THREE.AmbientLight(0x404060, 0.2);
  scene.add(ambientLight);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  return { renderer, scene, camera };
}

export function getScene() { return scene; }
export function getCamera() { return camera; }
export function getRenderer() { return renderer; }
