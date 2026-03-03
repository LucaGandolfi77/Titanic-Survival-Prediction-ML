import * as THREE from 'three';

export class SceneManager {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.scene = new THREE.Scene();
    
    // Setup renderer with stencil buffer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      stencil: true,
      alpha: false
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowShadowMap;
    this.renderer.autoClear = false;
    this.renderer.setClearColor(0x0a0a14, 1);

    // Setup camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 1.8, 0);

    // Setup lighting
    this.setupLighting();

    // Fog
    this.scene.fog = new THREE.FogExp2(0x0a0a14, 0.08);

    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());

    // Stencil state
    this.currentStencilID = 0;
  }

  setupLighting() {
    // Ambient light
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambient);

    // Directional light (simulating sun/ceiling)
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(10, 15, 10);
    dir.target.position.set(0, 0, 0);
    dir.castShadow = true;
    dir.shadow.mapSize.width = 2048;
    dir.shadow.mapSize.height = 2048;
    dir.shadow.camera.near = 0.5;
    dir.shadow.camera.far = 50;
    dir.shadow.camera.left = -30;
    dir.shadow.camera.right = 30;
    dir.shadow.camera.top = 30;
    dir.shadow.camera.bottom = -30;
    dir.shadow.bias = -0.001;
    this.scene.add(dir);
    this.directionalLight = dir;

    // Fluorescent tube lights (multiple points)
    const fluorColors = [0xffffff, 0xfffacd]; // white, light yellow
    for (let i = -2; i <= 2; i++) {
      const point = new THREE.PointLight(
        fluorColors[Math.abs(i) % fluorColors.length],
        0.5,
        20
      );
      point.position.set(i * 6, 10, 0);
      point.castShadow = true;
      this.scene.add(point);
    }

    // Store lights for flickering
    this.pointLights = this.scene.children.filter(child => child instanceof THREE.PointLight);
  }

  onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  // Render the scene
  render() {
    this.renderer.clear(true, true, true);
    this.renderer.render(this.scene, this.camera);
  }

  // Clear with stencil
  clearWithStencil() {
    this.renderer.clear(true, true, true);
  }

  // Add flickering effect to lights (called from main loop)
  updateLighting(time) {
    // Occasional flicker
    if (Math.random() < 0.03) {
      this.pointLights.forEach(light => {
        light.intensity *= (0.8 + Math.random() * 0.4);
      });
    }
    
    // Directional light subtle flicker
    this.directionalLight.intensity = 0.8 + Math.sin(time * 3) * 0.05;
  }

  // Get stencil ID for portal
  getStencilID() {
    this.currentStencilID = (this.currentStencilID + 1) % 255;
    return Math.max(1, this.currentStencilID); // Never return 0
  }
}